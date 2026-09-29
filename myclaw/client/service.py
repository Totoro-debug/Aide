"""Compatibility import path for the local service client adapters."""

from myclaw.service.client import (
    RemoteConfirmationCoordinator,
    RemoteControl,
    RemoteManagementCommandDispatcher,
    RemoteMessageBus,
    ServiceClient,
    ServiceStartupError,
)

__all__ = [
    "RemoteConfirmationCoordinator",
    "RemoteControl",
    "RemoteManagementCommandDispatcher",
    "RemoteMessageBus",
    "ServiceClient",
    "ServiceStartupError",
]
