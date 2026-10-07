"""Shared network target safety assessment for Core Tools."""

from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Protocol

from aide.agent.tools.base import is_public_ip
from aide.agent.tools.permission import NetworkTargetRisk


class DNSResolver(Protocol):
    """Resolve every TCP address for one network target."""

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]: ...


class SocketDNSResolver:
    """Resolve every TCP address through the host event loop."""

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        records = await asyncio.get_running_loop().getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
        return tuple(dict.fromkeys(record[4][0] for record in records))


@dataclass(frozen=True, slots=True)
class TargetResolution:
    """One resolver result, including the exact addresses used for one hop."""

    addresses: tuple[str, ...]
    risk: NetworkTargetRisk | None = None
    error_message: str | None = None


async def resolve_target(
    hostname: str,
    port: int,
    resolver: DNSResolver,
) -> TargetResolution:
    """Resolve one target once and retain the audited address set."""
    try:
        literal = ip_address(hostname)
    except ValueError:
        try:
            answers = await resolver.resolve(hostname, port)
        except Exception as error:
            message = str(error).strip() or "DNS resolution failed."
            return TargetResolution(
                addresses=(),
                risk="dns_failure",
                error_message=message,
            )
        addresses = tuple(dict.fromkeys(answers))
        if not addresses:
            return TargetResolution(addresses=(), risk="dns_empty")
        if all(is_public_ip(answer) for answer in addresses):
            return TargetResolution(addresses=addresses)
        return TargetResolution(addresses=addresses, risk="dns_non_global")

    address = str(literal)
    return TargetResolution(
        addresses=(address,),
        risk=None if is_public_ip(address) else "literal_non_global",
    )


__all__ = [
    "DNSResolver",
    "SocketDNSResolver",
    "TargetResolution",
    "resolve_target",
]
