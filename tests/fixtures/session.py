"""Raw Session state construction for tests with known persisted history."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from aide.agent.session.session import Session


def seed_session_state(
    session: Session,
    *,
    messages: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    last_compacted: int,
) -> None:
    """Replace public mutable Session state with detached explicit test data."""
    session.messages = deepcopy([dict(message) for message in messages])
    session.metadata = deepcopy(dict(metadata))
    session.last_compacted = last_compacted
