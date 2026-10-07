"""Per-Agent-Home discovery and startup coordination for the local service."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from omni.config.agent_home import AgentHome
from omni.utils.host_filesystem import HOST_FILESYSTEM

SERVICE_PROTOCOL_VERSION = 1
DEFAULT_SERVICE_HOST = "127.0.0.1"
DEFAULT_SERVICE_PORT = 8765
DISCOVERY_FILENAME = "service.json"
CREDENTIAL_FILENAME = "service.token"
LOCK_FILENAME = "service.lock"


@dataclass(frozen=True, slots=True)
class ServiceDiscovery:
    """Non-secret information needed to locate one service instance."""

    service_instance_id: str
    protocol_version: int
    host: str
    port: int
    pid: int

    def __post_init__(self) -> None:
        if not self.service_instance_id:
            raise ValueError("service instance id must be non-empty")
        if self.protocol_version != SERVICE_PROTOCOL_VERSION:
            raise ValueError("unsupported service protocol version")
        if self.host != DEFAULT_SERVICE_HOST:
            raise ValueError("local service must bind IPv4 loopback")
        if not 1 <= self.port <= 65535:
            raise ValueError("service port is invalid")
        if self.pid < 0:
            raise ValueError("service pid is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "service_instance_id": self.service_instance_id,
            "protocol_version": self.protocol_version,
            "host": self.host,
            "port": self.port,
            "pid": self.pid,
        }

    @classmethod
    def from_dict(cls, value: object) -> ServiceDiscovery:
        if not isinstance(value, dict):
            raise ValueError("service discovery must be an object")
        if set(value) != {"service_instance_id", "protocol_version", "host", "port", "pid"}:
            raise ValueError("service discovery contains unknown fields")
        try:
            return cls(
                service_instance_id=value["service_instance_id"],
                protocol_version=value["protocol_version"],
                host=value["host"],
                port=value["port"],
                pid=value["pid"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("service discovery is invalid") from error


def discovery_path(agent_home: AgentHome) -> Path:
    return agent_home.path / DISCOVERY_FILENAME


def credential_path(agent_home: AgentHome) -> Path:
    return agent_home.path / CREDENTIAL_FILENAME


def startup_lock_path(agent_home: AgentHome) -> Path:
    return agent_home.path / LOCK_FILENAME


def read_discovery(agent_home: AgentHome) -> ServiceDiscovery | None:
    path = discovery_path(agent_home)
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return ServiceDiscovery.from_dict(json.loads(content))


def write_discovery(agent_home: AgentHome, discovery: ServiceDiscovery) -> None:
    agent_home.initialize()
    target = discovery_path(agent_home)
    content = json.dumps(discovery.to_dict(), ensure_ascii=True, separators=(",", ":")) + "\n"
    HOST_FILESYSTEM.atomic_replace_text(target, content)


def remove_discovery(agent_home: AgentHome, *, instance_id: str | None = None) -> None:
    path = discovery_path(agent_home)
    if instance_id is not None:
        try:
            current = read_discovery(agent_home)
        except (OSError, ValueError, json.JSONDecodeError):
            current = None
        if current is not None and current.service_instance_id != instance_id:
            return
    path.unlink(missing_ok=True)


def create_credential(agent_home: AgentHome) -> str:
    agent_home.initialize()
    path = credential_path(agent_home)
    token = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as stream:
            fd = -1
            stream.write(token)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    HOST_FILESYSTEM.restrict_private_file(path)
    return token


def read_credential(agent_home: AgentHome) -> str:
    value = credential_path(agent_home).read_text(encoding="ascii").strip()
    if not value or any(character.isspace() for character in value):
        raise ValueError("service credential is invalid")
    return value


def identity_proof(token: str, challenge: str, instance_id: str, protocol_version: int) -> str:
    """Prove possession of the private token without sending it to an unverified port."""
    message = f"{challenge}:{instance_id}:{protocol_version}".encode("ascii")
    return hmac.new(token.encode("ascii"), message, hashlib.sha256).hexdigest()


def remove_credential(agent_home: AgentHome) -> None:
    credential_path(agent_home).unlink(missing_ok=True)


@contextmanager
def startup_lock(agent_home: AgentHome) -> Iterator[None]:
    """Hold an OS-level lock until one starter has published discovery."""
    agent_home.initialize()
    with HOST_FILESYSTEM.exclusive_lock(startup_lock_path(agent_home), timeout=30.0):
        yield


__all__ = [
    "DEFAULT_SERVICE_HOST",
    "DEFAULT_SERVICE_PORT",
    "SERVICE_PROTOCOL_VERSION",
    "ServiceDiscovery",
    "create_credential",
    "credential_path",
    "discovery_path",
    "identity_proof",
    "read_credential",
    "read_discovery",
    "remove_credential",
    "remove_discovery",
    "startup_lock",
    "startup_lock_path",
    "write_discovery",
]
