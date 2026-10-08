"""Coordinate saved configuration independently of active service resources."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Literal, cast

from aide.config.config import (
    ConfigError,
    ConfigFieldError,
    ConfigLoader,
    ConfigRevisionConflict,
    ReasoningEffort,
    UserConfiguration,
)
from aide.service.contracts import ServiceStopDTO
from aide.service.errors import service_error


@dataclass(frozen=True, slots=True)
class ConfigurationEdit:
    action: Literal["patch", "repair"]
    request_id: str
    expected_revision: str
    fields: Mapping[str, object]
    secrets: Mapping[str, object] | None = None
    client_id: str | None = None
    baseline: Mapping[str, object] | None = None
    baseline_secrets: Mapping[str, object] | None = None
    overwrite_conflicts: bool = False
    editor_id: str | None = None
    edit_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class ConfigurationSave:
    view: dict[str, object]
    changed: bool


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshot:
    """One complete validated configuration eligible for a subsequent Run."""

    revision: str
    configuration: UserConfiguration
    generation: int = 0


def _configuration_request_fingerprint(
    client_id: str | None,
    action: str,
    payload: object,
) -> str:
    serialized = json.dumps(
        [client_id, action, payload],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


_CONFIG_EDIT_MISSING = object()


def _advance_configuration_baseline(
    baseline: object, previous_baseline: object, before: object, saved: object
) -> object:
    """Recognize committed edits from this editor while retaining external conflicts."""
    if all(isinstance(value, Mapping) for value in (baseline, previous_baseline, before, saved)):
        base = cast(Mapping[str, object], baseline)
        old_base = cast(Mapping[str, object], previous_baseline)
        prior = cast(Mapping[str, object], before)
        accepted = cast(Mapping[str, object], saved)
        result: dict[str, object] = {}
        for key in base.keys() | prior.keys() | accepted.keys():
            value = _advance_configuration_baseline(
                base.get(key, _CONFIG_EDIT_MISSING),
                old_base.get(key, _CONFIG_EDIT_MISSING),
                prior.get(key, _CONFIG_EDIT_MISSING),
                accepted.get(key, _CONFIG_EDIT_MISSING),
            )
            if value is not _CONFIG_EDIT_MISSING:
                result[key] = value
        return result
    baseline, previous_baseline, before, saved = (
        list(value) if isinstance(value, tuple) else value
        for value in (baseline, previous_baseline, before, saved)
    )
    if before != saved and (baseline == previous_baseline or baseline == before):
        return saved
    return baseline


@dataclass(frozen=True, slots=True)
class _CommittedEdit:
    sequence: int
    baseline: object
    before: object
    saved: object
    baseline_secrets: Mapping[str, object]
    saved_secrets: Mapping[str, object]
    before_secrets: Mapping[str, object]
    changed_secrets: frozenset[str]


class ConfigurationEditor:
    """Own saved projections, ordered edits, and serialized configuration writes."""

    def __init__(
        self, loader: ConfigLoader, configuration: UserConfiguration | None = None
    ) -> None:
        self._loader = loader
        self._lock = asyncio.Lock()
        self._request_results: dict[str, dict[str, object]] = {}
        self._request_fingerprints: dict[str, str] = {}
        self._edits: dict[tuple[str | None, str, str], deque[_CommittedEdit]] = {}
        self._saved_configuration = configuration
        self._saved_revision: str | None = None
        self._active_revision: str | None = None
        self._fields: dict[str, dict[str, object]] | None = None
        self._secret_revisions: dict[str, str | None] = {}
        self._status = "active" if configuration is not None else "pending-repair"
        self._state = "active" if configuration is not None else "missing"
        self._repair_required = configuration is None
        self._backup_required = False
        self._requires_secret_reentry = configuration is None
        self._projection_error: dict[str, str] | None = None
        self._startup_diagnostics: tuple[str, ...] = ()
        self._snapshot_revision: str | None = None
        self._snapshot_generation = 0
        self._active_generation = 0

    @property
    def ready(self) -> bool:
        return self._active_revision is not None or self._saved_configuration is not None

    def capture(self) -> ConfigurationSnapshot:
        """Check external edits before capturing a complete candidate for execution."""
        self.view()
        if self._saved_configuration is None or self._saved_revision is None:
            raise service_error(
                "config_invalid", "Repair User Configuration before starting new work.", status=422
            )
        if self._snapshot_revision != self._saved_revision:
            self._snapshot_revision = self._saved_revision
            self._snapshot_generation += 1
        return ConfigurationSnapshot(
            self._saved_revision, self._saved_configuration, self._snapshot_generation,
        )

    def activate(self, snapshot: ConfigurationSnapshot) -> bool:
        """Publish the version whose execution resources were prepared successfully."""
        if snapshot.generation < self._active_generation:
            return False
        changed = self._active_revision != snapshot.revision
        self._active_revision = snapshot.revision
        self._active_generation = snapshot.generation
        self._status = (
            "pending-repair" if self._repair_required else
            "active" if self._saved_revision == snapshot.revision else "next-run-required"
        )
        return changed

    def start(self) -> UserConfiguration | None:
        """Capture the startup revision once and return its eligible configuration."""
        snapshot = self._loader.web_snapshot()
        self._startup_diagnostics = tuple(
            diagnostic.message for diagnostic in self._loader.diagnostics
        )
        self._saved_revision = snapshot.revision
        self._fields = {section: dict(values) for section, values in snapshot.fields.items()}
        self._secret_revisions = dict(self._loader.secret_revisions(snapshot.configuration))
        self._state = snapshot.state
        self._repair_required = snapshot.repair_required
        self._backup_required = snapshot.backup_required
        self._requires_secret_reentry = snapshot.requires_secret_reentry
        self._projection_error = None if snapshot.error is None else dict(snapshot.error)
        if snapshot.state == "active":
            self._saved_configuration = snapshot.configuration
            self._active_revision = snapshot.revision
            self._status = "active"
            return snapshot.configuration
        self._saved_configuration = None
        self._active_revision = None
        self._status = "pending-repair"
        return None

    def view(self) -> dict[str, object]:
        """Read saved settings and report whether later Runs need a new snapshot."""
        snapshot = self._loader.web_snapshot()
        self._saved_revision = snapshot.revision
        self._saved_configuration = snapshot.configuration if snapshot.state == "active" else None
        self._state = snapshot.state
        self._repair_required = snapshot.repair_required
        self._backup_required = snapshot.backup_required
        self._requires_secret_reentry = snapshot.requires_secret_reentry
        self._projection_error = None if snapshot.error is None else dict(snapshot.error)
        self._fields = {section: dict(values) for section, values in snapshot.fields.items()}
        self._secret_revisions = dict(self._loader.secret_revisions(snapshot.configuration))
        self._status = (
            "pending-repair"
            if snapshot.repair_required
            else "active"
            if snapshot.revision == self._active_revision
            else "next-run-required"
        )
        return self._response()

    def text_view(self) -> dict[str, object]:
        """Return the redacted text view used by the CLI config command."""
        try:
            self._loader.ensure_default()
            self.view()
            view = self._loader.view()
            view = replace(view, service_status_text=self.status_text())
        except (OSError, UnicodeError) as error:
            raise service_error(
                "persistence_error",
                f"User Configuration could not be read or written.\nPath: {self._loader.path}",
                status=500,
            ) from error
        return {
            "header_text": view.header_text(),
            "redacted_content": view.redacted_content,
            "error_code": None if view.error is None else view.error.code,
        }

    def startup_view(self) -> dict[str, object]:
        """Prepare the CLI configuration template and report Service startup eligibility."""
        result = self.view()
        available = self.ready and not self._repair_required
        error = None if available else self._projection_error
        if not available and error is None:
            error = {
                "code": "config_invalid",
                "message": "Repair User Configuration before starting a conversation.",
            }
        if not available and self._state == "missing":
            self.text_view()
            error = {
                "code": "config_missing",
                "message": "A default User Configuration was created; edit it before starting Aide.\n"
                f"Path: {self._loader.path}",
            }
        return {
            **result,
            "startup": {
                "available": available,
                "diagnostics": list(self._startup_diagnostics),
                "error": error,
            },
        }

    def _prepare_configuration_edit(
        self,
        client_id: str | None,
        editor_id: str | None,
        edit_sequence: int | None,
        fields: Mapping[str, object],
        secrets: Mapping[str, object] | None,
        baseline: Mapping[str, object] | None,
        baseline_secrets: Mapping[str, object] | None,
    ) -> tuple[Mapping[str, object] | None, Mapping[str, object] | None]:
        if editor_id is None and edit_sequence is None:
            return baseline, baseline_secrets
        if (
            not isinstance(editor_id, str)
            or not editor_id
            or isinstance(edit_sequence, bool)
            or not isinstance(edit_sequence, int)
            or edit_sequence < 1
            or baseline is None
        ):
            raise service_error(
                "validation_error", "Configuration edit identity is invalid.", status=422
            )
        coordinated = dict(baseline)
        coordinated_secrets = dict(baseline_secrets or {})
        sections = set(fields) | {path.split(".")[0] for path in (secrets or {})}
        for section in sections:
            history = self._edits.get((client_id, editor_id, section))
            if not history:
                continue
            if edit_sequence <= history[-1].sequence:
                raise service_error(
                    "config_edit_superseded",
                    "A newer configuration edit was already saved.",
                    status=409,
                )
            value = baseline.get(section, _CONFIG_EDIT_MISSING)
            for previous in history:
                value = _advance_configuration_baseline(
                    value, previous.baseline, previous.before, previous.saved
                )
                for path in previous.changed_secrets:
                    if coordinated_secrets.get(path) in (
                        previous.baseline_secrets.get(path),
                        previous.before_secrets.get(path),
                    ):
                        coordinated_secrets[path] = previous.saved_secrets.get(path)
            if value is not _CONFIG_EDIT_MISSING:
                coordinated[section] = value
        return coordinated, coordinated_secrets

    def _record_configuration_edit(
        self,
        client_id: str | None,
        editor_id: str | None,
        edit_sequence: int | None,
        fields: Mapping[str, object],
        secrets: Mapping[str, object] | None,
        baseline: Mapping[str, object] | None,
        baseline_secrets: Mapping[str, object] | None,
        previous_fields: Mapping[str, Mapping[str, object]],
        previous_secret_revisions: Mapping[str, str | None],
    ) -> None:
        if editor_id is None or edit_sequence is None or baseline is None:
            return
        sections = set(fields) | {path.split(".")[0] for path in (secrets or {})}
        for section in sections:
            history = self._edits.setdefault((client_id, editor_id, section), deque(maxlen=32))
            history.append(
                _CommittedEdit(
                    edit_sequence,
                    baseline.get(section, {}),
                    previous_fields.get(section, {}),
                    (self._fields or {}).get(section, {}),
                    baseline_secrets or {},
                    dict(self._secret_revisions),
                    previous_secret_revisions,
                    frozenset(
                        path
                        for path in previous_secret_revisions.keys() | self._secret_revisions.keys()
                        if path.split(".")[0] == section
                        and previous_secret_revisions.get(path) != self._secret_revisions.get(path)
                    ),
                )
            )
        while len(self._edits) > 256:
            self._edits.pop(next(iter(self._edits)))

    def _configuration_request_result(
        self, request_id: str, fingerprint: str
    ) -> dict[str, object] | None:
        previous = self._request_fingerprints.get(request_id)
        if previous is not None and previous != fingerprint:
            raise service_error(
                "request_conflict",
                "Request ID was already used for a different operation.",
                status=409,
            )
        return self._request_results.get(request_id)

    def _response(self) -> dict[str, object]:
        saved_revision = self._saved_revision or ConfigLoader.revision_from_bytes(b"")
        fields = {} if self._fields is None else self._fields
        return {
            "revision": saved_revision,
            "fields": {section: dict(values) for section, values in fields.items()},
            "secret_revisions": dict(self._secret_revisions),
            "configuration": {
                "state": self._state,
                "repair_required": self._repair_required,
                "backup_required": self._backup_required,
                "requires_secret_reentry": self._requires_secret_reentry,
                "error": (None if self._projection_error is None else dict(self._projection_error)),
            },
            "application": {
                "status": self._status,
                "saved_revision": saved_revision,
                "active_revision": self._active_revision,
                "restart_required": False,
            },
        }

    def status_text(self) -> str:
        """Render the version available to subsequent Runs for Command-line management."""
        application = cast(dict[str, object], self.view()["application"])
        return (
            f"Saved version: {application['saved_revision']}\n"
            f"Active version: {application['active_revision'] or '-'}\n"
            "Configuration changes apply to the next Agent Run.\n"
        )

    async def save(self, edit: ConfigurationEdit) -> ConfigurationSave:
        """Save one edit, returning the original receipt for an identical retry."""
        if not edit.request_id:
            raise service_error("validation_error", "Request ID is required.", status=422)
        action = edit.action
        request_id = edit.request_id
        expected_revision = edit.expected_revision
        fields = edit.fields
        secrets = edit.secrets
        client_id = edit.client_id
        baseline = edit.baseline
        baseline_secrets = edit.baseline_secrets
        overwrite_conflicts = edit.overwrite_conflicts
        editor_id = edit.editor_id
        edit_sequence = edit.edit_sequence
        fingerprint = _configuration_request_fingerprint(
            client_id,
            action,
            {
                "revision": expected_revision,
                "baseline": baseline,
                "baseline_secrets": baseline_secrets,
                "overwrite_conflicts": overwrite_conflicts,
                "editor_id": editor_id,
                "edit_sequence": edit_sequence,
                "fields": fields,
                "secrets": secrets or {},
            },
        )
        async with self._lock:
            existing = self._configuration_request_result(request_id, fingerprint)
            if existing is not None:
                return ConfigurationSave(existing, changed=False)
            coordinated, coordinated_secrets = self._prepare_configuration_edit(
                client_id, editor_id, edit_sequence, fields, secrets, baseline, baseline_secrets
            )
            try:
                persist = (
                    self._loader.patch_editable_fields
                    if action == "patch"
                    else self._loader.repair_editable_fields
                )
                result = persist(
                    expected_revision,
                    fields,
                    secrets,
                    coordinated,
                    coordinated_secrets,
                    overwrite_conflicts,
                )
            except ConfigRevisionConflict as error:
                raise service_error(
                    "config_revision_conflict",
                    error.error.message,
                    status=409,
                    retryable=True,
                    field_errors=error.field_errors,
                ) from error
            except ConfigFieldError as error:
                raise service_error(
                    error.error.code,
                    error.error.message,
                    status=422,
                    field_errors=error.field_errors,
                ) from error
            except ConfigError as error:
                raise service_error(
                    error.error.code,
                    "The complete User Configuration is invalid.",
                    status=422,
                    field_errors=error.field_errors,
                ) from error
            except OSError as error:
                raise service_error(
                    "persistence_error",
                    "User Configuration could not be written.",
                    status=500,
                    retryable=True,
                ) from error

            self._saved_configuration = result.configuration
            self._saved_revision = result.revision
            self._fields = {section: dict(values) for section, values in result.fields.items()}
            self._secret_revisions = dict(self._loader.secret_revisions(result.configuration))
            self._record_configuration_edit(
                client_id,
                editor_id,
                edit_sequence,
                fields,
                secrets,
                baseline,
                baseline_secrets,
                result.previous_fields,
                result.previous_secret_revisions,
            )
            self._state = "active"
            self._repair_required = False
            self._backup_required = False
            self._requires_secret_reentry = False
            self._projection_error = None
            self._status = (
                "active" if result.revision == self._active_revision else "next-run-required"
            )
            response = self._response()
            if action == "repair":
                response = {"backup_id": result.backup_id, **response}
            self._request_results[request_id] = response
            self._request_fingerprints[request_id] = fingerprint
            if len(self._request_results) > 256:
                oldest = next(iter(self._request_results))
                self._request_results.pop(oldest, None)
                self._request_fingerprints.pop(oldest, None)
        return ConfigurationSave(response, changed=True)

    async def persist_reasoning_effort(self, effort: ReasoningEffort) -> None:
        async with self._lock:
            self._loader.update_reasoning_effort(effort)
            self.view()

    async def coordinate_restart(self, restart: Callable[[], ServiceStopDTO]) -> ServiceStopDTO:
        """Make a restart decision atomically with saved-setting writes."""
        async with self._lock:
            return restart()

    def default_chat_workspace(self) -> str:
        return self._loader.load().web.default_chat_workspace

    def application(self) -> dict[str, object] | None:
        if self._saved_revision is None:
            return None
        return cast(dict[str, object], self._response()["application"])
