"""Small helpers shared by the lessons (so each lesson stays focused)."""

import sys
from pathlib import Path

# Make "import rag" work no matter where the lesson is run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"


def _friendly_errors(exc_type, exc, tb):
    """Show setup problems (missing API key, no internet...) as a short message, not a traceback."""
    if issubclass(exc_type, RuntimeError):
        print(f"\nProblem: {exc}")
    else:
        sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = _friendly_errors


def title(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def pause(interactive: bool = True) -> None:
    """Wait for Enter so students can read. Skipped when not in a terminal."""
    if interactive and sys.stdin.isatty():
        input("\n[press Enter to continue] ")


def short(text: str, length: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[: length - 3] + "..."
