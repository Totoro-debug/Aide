"""Authenticated confirmation presentation."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING
from uuid import uuid4

from aide.agent.confirmation import (
    ConfirmationDecision,
    ConfirmationEnvelope,
    ConfirmationPresenter,
    ForegroundConfirmationOwner,
    SubAgentConfirmationOwner,
)
from aide.service.errors import service_error
from aide.service.runtime.records import _consume_task_result

if TYPE_CHECKING:
    from aide.service.runtime.service import AgentService


class ServiceConfirmationPresenter(ConfirmationPresenter):
    """Bridge the existing one-shot coordinator to authenticated clients."""

    def __init__(self, service: AgentService) -> None:
        self._service = service
        self._wire_tokens: dict[object, str] = {}
        self._wire_sources: dict[object, tuple[str | None, str | None, str | None]] = {}
        self._wire_requests: dict[object, asyncio.Task[None]] = {}
        self._wire_payloads: dict[object, dict[str, object]] = {}

    def present_confirmation(
        self,
        envelope: ConfirmationEnvelope,
        token: object,
        respond: Callable[[object, ConfirmationDecision], bool],
    ) -> None:
        wire_token = str(uuid4())
        self._wire_tokens[token] = wire_token
        workspace_id, session_id, run_id = self._service.confirmation_source(envelope.owner)
        self._wire_sources[token] = (workspace_id, session_id, run_id)
        payload: dict[str, object] = {
            "token": wire_token,
            "origin": envelope.origin,
            "request": envelope.request.to_dict(),
        }
        if envelope.job_id is not None:
            payload["job_id"] = envelope.job_id
        if envelope.title is not None:
            payload["title"] = envelope.title
        if isinstance(envelope.owner, ForegroundConfirmationOwner):
            payload["owner"] = {
                "kind": "foreground",
                "generation_id": str(envelope.owner.generation_id),
                "run_id": str(envelope.owner.run_id),
            }
        elif isinstance(envelope.owner, SubAgentConfirmationOwner):
            payload["owner"] = {
                "kind": "subagent",
                "generation_id": str(envelope.owner.generation_id),
                "workspace_id": envelope.owner.workspace_id,
                "session_id": envelope.owner.session_id,
                "agent_id": envelope.owner.agent_id,
            }
        else:
            payload["owner"] = {
                "kind": "background",
                "generation_id": str(envelope.owner.generation_id),
                "job_id": envelope.owner.job_id,
                "occurrence_id": str(envelope.owner.occurrence_id),
            }
        self._wire_payloads[token] = payload
        task = asyncio.create_task(
            self._service.emit(
                "confirmation.requested",
                workspace_id=workspace_id,
                session_id=session_id,
                run_id=run_id,
                payload=payload,
                target_client_ids=self._audience(workspace_id, session_id),
            )
        )
        self._wire_requests[token] = task
        task.add_done_callback(_consume_task_result)

    async def dismiss_confirmation(self, token: object) -> None:
        self._wire_payloads.pop(token, None)
        wire_token = self._wire_tokens.pop(token, None)
        if wire_token is None:
            return
        workspace_id, session_id, run_id = self._wire_sources.pop(token, (None, None, None))
        await self._emit_resolved(
            self._wire_requests.pop(token),
            wire_token,
            workspace_id,
            session_id,
            run_id,
        )

    async def _emit_resolved(
        self,
        requested: asyncio.Task[None],
        wire_token: str,
        workspace_id: str | None,
        session_id: str | None,
        run_id: str | None,
    ) -> None:
        await asyncio.gather(requested, return_exceptions=True)
        await self._service.emit(
            "confirmation.resolved",
            workspace_id=workspace_id,
            session_id=session_id,
            run_id=run_id,
            payload={"token": wire_token},
            target_client_ids=self._audience(workspace_id, session_id),
        )

    def _audience(self, workspace_id: str | None, session_id: str | None) -> tuple[str, ...]:
        if workspace_id is None:
            return ()
        audience = set(self._service.workspace_audience(workspace_id))
        workspace = self._service._workspaces.get(workspace_id)
        if workspace is not None:
            workspace_key = os.path.normcase(str(workspace.workspace_path))
            registered = any(
                os.path.normcase(str(record.path.resolve(strict=False))) == workspace_key
                for record in self._service.projects.list()
            )
            if registered:
                # Web clients can inspect the account-global Project catalog, while
                # CLI clients only receive confirmations for attached Workspaces.
                audience.update(
                    client.client_id
                    for client in self._service._clients.values()
                    if client.kind == "web"
                )
        return tuple(
            client.client_id
            for client in self._service._clients.values()
            if client.client_id in audience
        )

    def decide(self, client_id: str, wire_token: str, decision: ConfirmationDecision) -> bool:
        for token, candidate in tuple(self._wire_tokens.items()):
            if candidate == wire_token:
                workspace_id, session_id, _ = self._wire_sources[token]
                if client_id not in self._audience(workspace_id, session_id):
                    raise service_error(
                        "forbidden", "Confirmation does not belong to this Client.", status=403
                    )
                accepted = self._service.confirmation.decide(token, decision)
                if accepted:
                    self._wire_payloads.pop(token, None)
                    self._wire_tokens.pop(token, None)
                    workspace_id, session_id, run_id = self._wire_sources.pop(
                        token,
                        (None, None, None),
                    )
                    task = asyncio.create_task(
                        self._emit_resolved(
                            self._wire_requests.pop(token),
                            wire_token,
                            workspace_id,
                            session_id,
                            run_id,
                        )
                    )
                    task.add_done_callback(_consume_task_result)
                return accepted
        return False

    def snapshot(self, client_id: str) -> dict[str, object] | None:
        """Project the current display slot only for an authorized Client."""
        token = next((candidate for candidate in self._wire_payloads
                      if self._service.confirmation.is_pending(candidate)), None)
        if token is None:
            return None
        payload = self._wire_payloads[token]
        workspace_id, session_id, run_id = self._wire_sources[token]
        if client_id not in self._audience(workspace_id, session_id):
            return None
        workspace = self._service._workspaces.get(workspace_id or "")
        project_id = None
        if workspace is not None:
            key = os.path.normcase(str(workspace.workspace_path))
            project_id = next((record.project_id for record in self._service.projects.list()
                               if os.path.normcase(str(record.path.resolve(strict=False))) == key), None)
        return {"workspace_id": workspace_id, "project_id": project_id,
                "session_id": session_id, "run_id": run_id, "payload": deepcopy(payload)}
