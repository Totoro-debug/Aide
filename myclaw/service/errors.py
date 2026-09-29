"""Typed, presentation-safe errors used by the local service boundary."""

from __future__ import annotations

from collections.abc import Mapping

from myclaw.errors import ErrorInfo


class ServiceError(Exception):
    """A safe service failure with an HTTP/WebSocket status mapping."""

    __slots__ = ("code", "field_errors", "message", "retryable", "status")

    def __init__(
        self,
        code: str,
        message: str,
        status: int = 409,
        retryable: bool = False,
        field_errors: Mapping[str, str] | None = None,
    ) -> None:
        if not code or not message:
            raise ValueError("service error code and message must be non-empty")
        if status not in {400, 401, 403, 404, 409, 422, 500}:
            raise ValueError("service error status is invalid")
        self.code = code
        self.message = message
        self.status = status
        self.retryable = retryable
        self.field_errors = {} if field_errors is None else dict(field_errors)
        Exception.__init__(self, self.message)

    @classmethod
    def from_error_info(cls, error: ErrorInfo, *, status: int = 409) -> ServiceError:
        return cls(error.code, error.message, status=status)

    def to_dict(self, request_id: str) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "field_errors": dict(self.field_errors),
            "retryable": self.retryable,
            "request_id": request_id,
        }


def service_error(
    code: str,
    message: str,
    *,
    status: int = 409,
    retryable: bool = False,
    field_errors: Mapping[str, str] | None = None,
) -> ServiceError:
    return ServiceError(
        code,
        message,
        status=status,
        retryable=retryable,
        field_errors={} if field_errors is None else field_errors,
    )


__all__ = ["ServiceError", "service_error"]
