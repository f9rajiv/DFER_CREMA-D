import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CLASS_NAMES_7, CREMA_EMOTION_TO_INDEX, SURPRISE_INDEX, data_dir

TRUE_INDICES = sorted(CREMA_EMOTION_TO_INDEX.values())  # the 6 emotions CREMA-D has


def confusion(results: pd.DataFrame, pred_column: str, predicted_indices: list) -> np.ndarray:
    """Rows: true emotion (6). Columns: predicted class. Values: % of each true emotion."""
    matrix = np.zeros((len(TRUE_INDICES), len(predicted_indices)))
    for r, true_index in enumerate(TRUE_INDICES):
        subset = results[results["label"] == true_index]
        for c, pred_index in enumerate(predicted_indices):
            matrix[r, c] = (subset[pred_column] == pred_index).mean() * 100 if len(subset) else np.nan
    return matrix


def score(results: pd.DataFrame, mode: str) -> dict:
    pred_column = "pred_7" if mode == "7-class" else "pred_6"
    correct = results[pred_column] == results["label"]
    per_class = {CLASS_NAMES_7[i]: correct[results["label"] == i].mean() * 100 for i in TRUE_INDICES}
    row = {
        "prompt_set": results["prompt_set"].iloc[0],
        "mode": mode,
        "clips": len(results),
        "WAR": correct.mean() * 100,
        "UAR": np.nanmean(list(per_class.values())),
    }
    row.update({f"recall_{k}": v for k, v in per_class.items()})
    if mode == "7-class":
        row["predicted_surprise_%"] = (results["pred_7"] == SURPRISE_INDEX).mean() * 100
    return row


def plot_confusion(results: pd.DataFrame, mode: str, out_png: Path) -> None:
    predicted = list(range(7)) if mode == "7-class" else TRUE_INDICES
    pred_column = "pred_7" if mode == "7-class" else "pred_6"
    matrix = confusion(results, pred_column, predicted)
    fig, ax = plt.subplots(figsize=(1.0 * len(predicted) + 2, 5.5))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(predicted)), [CLASS_NAMES_7[i] for i in predicted], rotation=45, ha="right")
    ax.set_yticks(range(len(TRUE_INDICES)), [CLASS_NAMES_7[i] for i in TRUE_INDICES])
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            ax.text(c, r, f"{value:.0f}", ha="center", va="center",
                    color="white" if value > 55 else "black", fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (CREMA-D)")
    ax.set_title(f"{results['prompt_set'].iloc[0]} ({mode}, row %)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results",
        type=Path,
        nargs="*",
        help="result CSVs from zero_shot.py (default: all in results dir)",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    files = args.results or sorted((data_dir() / "results").glob("*.csv"))
    files = [f for f in files if "summary" not in f.name]
    if not files:
        raise SystemExit("no result CSVs found")

    out_dir = args.out_dir or files[0].parent / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        results = pd.read_csv(path)
        for mode in ("7-class", "6-class"):
            rows.append(score(results, mode) | {"run": path.stem})
            plot_confusion(results, mode, out_dir / f"{path.stem}__{mode}.png")

    summary = pd.DataFrame(rows).sort_values(["mode", "UAR"], ascending=[True, False])
    summary.to_csv(out_dir / "summary.csv", index=False)

    with pd.option_context("display.width", 200, "display.precision", 1):
        print(
            summary[
                ["prompt_set", "mode", "clips", "WAR", "UAR"]
                + [c for c in summary.columns if c.startswith("recall_")]
            ].to_string(index=False)
        )

    # Several checkpoints (DFEW folds) per prompt set -> mean and spread across folds.
    if summary.groupby(["prompt_set", "mode"]).size().max() > 1:
        by_set = (
            summary.groupby(["mode", "prompt_set"])[["WAR", "UAR"]]
            .agg(["mean", "std", "count"])
            .round(2)
        )
        by_set.to_csv(out_dir / "summary_by_prompt_set.csv")
        print("\nacross checkpoints (mean, std, number of folds):\n", by_set.to_string())

    print(f"\nsummary table and confusion matrices in {out_dir}")


if __name__ == "__main__":
    main()
