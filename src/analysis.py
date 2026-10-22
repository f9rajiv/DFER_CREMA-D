"""deeper analysis of one run (default: the baseline prompt set).

"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import CLASS_NAMES_7, INTENSITY_ORDER, data_dir


def accuracy_by(results: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = results.groupby(column)
    return pd.DataFrame({"clips": grouped.size(), "accuracy_%": grouped["correct"].mean() * 100})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_csv", type=Path)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    results = pd.read_csv(args.result_csv)
    results["correct"] = results["pred_6"] == results["label"]
    results["emotion_name"] = results["label"].map(lambda i: CLASS_NAMES_7[i])
    out_dir = args.out_dir or args.result_csv.parent / "analysis" / args.result_csv.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Intensity: only meaningful for the IEO sentence, the one recorded at LO/MD/HI.
    intensity = accuracy_by(results[results["intensity"] != "XX"], "intensity")
    intensity = intensity.reindex([i for i in INTENSITY_ORDER if i in intensity.index])
    intensity_by_emotion = (results[results["intensity"] != "XX"]
                            .pivot_table(index="emotion_name", columns="intensity",
                                         values="correct", aggfunc="mean") * 100)

    actors = accuracy_by(results, "actor")
    sentences = accuracy_by(results, "sentence").sort_values("accuracy_%")

    print("accuracy by intensity (sentence IEO only):\n", intensity.round(1).to_string(), "\n")
    print("by intensity and emotion:\n", intensity_by_emotion.round(1).to_string(), "\n")
    print(f"per actor: mean {actors['accuracy_%'].mean():.1f}%, "
          f"min {actors['accuracy_%'].min():.1f}%, max {actors['accuracy_%'].max():.1f}%\n")
    print("by sentence:\n", sentences.round(1).to_string(), "\n")

    if "human_face_vote" in results.columns and results["human_face_vote"].notna().any():
        rated = results[results["human_face_vote"].notna()].copy()
        rated["human_correct"] = [emotion in str(vote).split(":")
                                  for emotion, vote in zip(rated["emotion"], rated["human_face_vote"])]
        both = pd.crosstab(rated["human_correct"].map({True: "human right", False: "human wrong"}),
                           rated["correct"].map({True: "model right", False: "model wrong"}),
                           normalize="all") * 100
        per_emotion = rated.groupby("emotion_name")[["human_correct", "correct"]].mean() * 100
        per_emotion.columns = ["human_face_only_%", "model_%"]
        print(f"humans (silent video): {rated['human_correct'].mean() * 100:.1f}%   "
              f"model: {rated['correct'].mean() * 100:.1f}%")
        print(per_emotion.round(1).to_string(), "\n")
        print("agreement (% of clips):\n", both.round(1).to_string(), "\n")
        per_emotion.to_csv(out_dir / "human_vs_model.csv")
        both.to_csv(out_dir / "human_model_agreement.csv")

        fig, ax = plt.subplots(figsize=(7, 4))
        per_emotion.plot.bar(ax=ax, rot=0)
        ax.set_ylabel("accuracy (%)")
        ax.set_title("Model vs human raters on the same silent video")
        fig.tight_layout()
        fig.savefig(out_dir / "human_vs_model.png", dpi=150)
        plt.close(fig)

    intensity.to_csv(out_dir / "by_intensity.csv")
    intensity_by_emotion.to_csv(out_dir / "by_intensity_and_emotion.csv")
    actors.to_csv(out_dir / "by_actor.csv")
    sentences.to_csv(out_dir / "by_sentence.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    intensity["accuracy_%"].plot.bar(ax=axes[0], rot=0, color="#4c72b0")
    axes[0].set_title("Accuracy by intensity")
    axes[0].set_ylabel("accuracy (%)")
    axes[1].boxplot(actors["accuracy_%"])
    axes[1].set_xticks([1], ["91 actors"])
    axes[1].set_title("Accuracy per actor")
    fig.tight_layout()
    fig.savefig(out_dir / "intensity_and_actors.png", dpi=150)
    plt.close(fig)
    print(f"tables and figures in {out_dir}")


if __name__ == "__main__":
    main()