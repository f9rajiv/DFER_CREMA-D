"""Shared constants and helpers.

Everything that both the Mac and Roihu need to agree on lives here: the label
order of the DFEW-trained model, how CREMA-D codes map onto it, where the
authors' code is, and which device to run on.
"""

import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DFER_CLIP_DIR = REPO_ROOT / "third_party" / "DFER-CLIP"

# Order of the 7 output classes in the DFEW checkpoints (models/Text.py in DFER-CLIP).
CLASS_NAMES_7 = ["happiness", "sadness", "neutral", "anger", "surprise", "disgust", "fear"]
SURPRISE_INDEX = 4

# CREMA-D filename emotion code -> DFEW class index. CREMA-D has no "surprise".
CREMA_EMOTION_TO_INDEX = {"HAP": 0, "SAD": 1, "NEU": 2, "ANG": 3, "DIS": 5, "FEA": 6}

# CREMA-D human rating letters (summaryTable.csv FaceVote column) -> CREMA-D codes.
VOTE_LETTER_TO_EMOTION = {"H": "HAP", "S": "SAD", "N": "NEU", "A": "ANG", "D": "DIS", "F": "FEA"}

INTENSITY_ORDER = ["LO", "MD", "HI", "XX"]


def data_dir() -> Path:
    """Root for data, frames and results. Override with DFER_DATA_DIR (Roihu: scratch)."""
    return Path(os.environ.get("DFER_DATA_DIR", REPO_ROOT / "data"))


def add_dfer_clip_to_path() -> None:
    """Make the authors' code importable exactly as their main.py expects.

    Their modules use both `from models.X import ...` (needs the repo root on the
    path) and `from clip import clip` (needs models/ on the path).
    """
    if not DFER_CLIP_DIR.exists():
        raise FileNotFoundError(
            f"{DFER_CLIP_DIR} not found. Run: git submodule update --init"
        )
    for path in (DFER_CLIP_DIR, DFER_CLIP_DIR / "models"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # Their clip.py does `from pkg_resources import packaging`. pkg_resources is
    # gone from recent setuptools: either missing entirely (ImportError), or left
    # behind as an empty folder that imports but has no `packaging` (seen on
    # Roihu-GPU). In both cases provide a stand-in with the one attribute it uses.
    try:
        import pkg_resources  # noqa: F401  # type: ignore
        usable = hasattr(pkg_resources, "packaging")
    except ImportError:
        usable = False
    if not usable:
        import packaging
        import packaging.version  # noqa: F401

        shim = types.ModuleType("pkg_resources")
        shim.packaging = packaging
        sys.modules["pkg_resources"] = shim


def pick_device(requested: str = "auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")