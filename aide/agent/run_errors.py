"""Internal Agent Run failures with terminal-state disposition."""

from aide.provider.errors import ModelCallError


class CommittableAgentRunError(ModelCallError):
    """A run failure whose formed messages and staged state must be committed."""
