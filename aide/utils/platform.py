"""Runtime platform check kept outside module import and build paths."""

import os
from typing import Final

WINDOWS_REQUIRED_ERROR: Final = "Aide requires Windows."


def is_windows_host() -> bool:
    return os.name == "nt"
