"""Public data shapes exchanged with local service clients."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from omni.service.errors import ServiceError


class ConversationClaimDTO(TypedDict):
    workspace_id: str
    session_id: str
    claim_version: int
    reconnect_credential: str


class ConversationOpenDTO(TypedDict):
    project_id: str | None
    workspace_id: str
    directory: str
    session_id: str
    claim: ConversationClaimDTO
    snapshot: dict[str, object]


class ConversationOpenFailureDTO(TypedDict):
    target: dict[str, object]
    current_context: ConversationClaimDTO | None
    current_conversation: NotRequired[dict[str, object]]


class ConversationOpenError(ServiceError):
    """A failed navigation with the remaining authoritative Claim."""

    def __init__(self, error: ServiceError, conversation: ConversationOpenFailureDTO) -> None:
        super().__init__(
            error.code, error.message, error.status, error.retryable, error.field_errors
        )
        self.conversation = conversation

    def to_dict(self, request_id: str) -> dict[str, object]:
        return {**super().to_dict(request_id), "conversation": self.conversation}


class ProjectJobSummaryDTO(TypedDict):
    job_id: str
    title: str
    schedule: dict[str, object]
    due_at: str | None
    review_status: str


class ProjectSummaryDTO(TypedDict, total=False):
    project_id: str
    path: str
    name: str
    schedule_state: str
    available: bool
    saved_jobs: list[ProjectJobSummaryDTO]
    schedule_status: dict[str, object] | None
    removal_operation_id: str
    removal_error: str


class ProjectListDTO(TypedDict):
    projects: list[ProjectSummaryDTO]


class ProjectRegistrationDTO(TypedDict):
    project_id: str
    workspace_id: str
    schedule_state: str
    saved_jobs: list[ProjectJobSummaryDTO]


class ServiceStatusDTO(TypedDict):
    service_instance_id: str
    protocol_version: int
    state: str
    active_workspace_count: int


class MemoryViewDTO(TypedDict):
    request_id: str
    workspace_id: str
    content: str


class DreamRunDTO(TypedDict):
    request_id: str
    workspace_id: str
    result: dict[str, object]


class SkillMetadataDTO(TypedDict):
    name: str
    description: str
    path: str


class SkillReloadDTO(TypedDict):
    request_id: str
    workspace_id: str
    skills: list[SkillMetadataDTO]


class RuntimeStatusDTO(TypedDict):
    request_id: str
    workspace_id: str
    status: dict[str, object]


class ServiceStopDTO(TypedDict):
    request_id: str
    accepted: bool
    operation_id: str


class WebLaunchTicketDTO(TypedDict):
    request_id: str
    ticket: str
    expires_in: int
