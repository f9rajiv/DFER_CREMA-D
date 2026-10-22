""" demo figure. Shows the 16 frames the model saw, with its top-3 answer.

Use it on webcam clips (or any CREMA-D clip) after they went through
make_annotations -> extract_frames -> zero_shot like any other data.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common import CLASS_NAMES_7, CREMA_EMOTION_TO_INDEX, data_dir

SIX_CLASS = sorted(CREMA_EMOTION_TO_INDEX.values())


def softmax(values: np.ndarray) -> np.ndarray:
    exp = np.exp(values - values.max())
    return exp / exp.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_csv", type=Path)
    parser.add_argument("--clip_ids", nargs="*", default=None, help="default: first 4 clips")
    parser.add_argument("--frames_dir", type=Path, default=data_dir() / "frames" / "crema_d")
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    results = pd.read_csv(args.result_csv).set_index("clip_id")
    clip_ids = args.clip_ids or list(results.index[:4])
    out_dir = args.out_dir or args.result_csv.parent / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    for clip_id in clip_ids:
        row = results.loc[clip_id]
        logits = np.array([row[f"logit_{CLASS_NAMES_7[i]}"] for i in SIX_CLASS])
        probs = softmax(logits)
        top = np.argsort(probs)[::-1][:3]
        lines = [f"{CLASS_NAMES_7[SIX_CLASS[i]]} {probs[i] * 100:.0f}%" for i in top]
        truth = CLASS_NAMES_7[int(row["label"])]
        verdict = "correct" if SIX_CLASS[top[0]] == int(row["label"]) else "wrong"

        frames = sorted((args.frames_dir / clip_id).glob("*.jpg"))
        fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))
        for ax, frame in zip(axes.flat, frames):
            ax.imshow(Image.open(frame))
            ax.axis("off")
        fig.suptitle(f"{clip_id}   true: {truth}   predicted: {' | '.join(lines)}   ({verdict})",
                     fontsize=12)
        fig.tight_layout()
        out_png = out_dir / f"{clip_id}.png"
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f"{out_png}  {verdict}")


if __name__ == "__main__":
    main()