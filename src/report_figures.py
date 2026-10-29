"""Figures for the report, made from the committed result files (no GPU, no frames needed).

1. fig_per_fold_war   - change in WAR vs the original descriptions for every text set on each of
                        the 5 DFEW folds, marking differences that are not significant. Shows that rewording effects
                        change from model to model.
2. fig_three_step     - plain CLIP -> DFER-CLIP with wrong text -> DFER-CLIP with correct text,
                        next to humans on the same silent video. What training vs words contribute.
3. fig_plain_vs_dfer  - plain CLIP vs DFER-CLIP (5-fold mean) for every text set.

Each figure is saved as PNG (slides) and PDF (report). All numbers are 6-class WAR on CREMA-D.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import REPO_ROOT

# Display order and readable names for the text sets.
TEXT_SETS = {
    "class_descriptor": "Original\ndescriptions",
    "class_names": "Class name\nonly (e.g.\n\u201canger.\u201d)",
    "class_names_with_context": "\u201cAn expression\nof [emotion].\u201d",
    "other_llm": "Other-LLM\ndescriptions",
    "facs": "FACS\ndescriptions",
    "shuffled": "Shuffled (wrong)\ndescriptions",
}
CHANCE = 100 / 6

# Validated categorical slots (light surface) + neutral ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HUMAN_GRAY = "#9a998f"


def style_axes(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)


def chance_line(ax, x_right=None) -> None:
    """Dashed chance line; labelled in place, or (x_right=None) via the legend."""
    ax.axhline(CHANCE, color=INK_MUTED, linewidth=1, linestyle="--",
               label="chance (16.7%)" if x_right is None else None)
    if x_right is not None:
        ax.text(x_right, CHANCE + 0.8, "chance (16.7%)", ha="right", va="bottom",
                fontsize=8, color=INK_MUTED)


def save(fig, out_dir: Path, name: str) -> None:
    for suffix in (".png", ".pdf"):
        fig.savefig(out_dir / f"{name}{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_dir / name}.png / .pdf")


def load_folds(results_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(results_dir / "evaluation" / "summary.csv")
    summary = summary[(summary["mode"] == "6-class") & summary["run"].str.startswith("dfew_fold")].copy()
    summary["fold"] = summary["run"].str.extract(r"fold(\d)").astype(int)
    return summary[summary["prompt_set"].isin(TEXT_SETS)]


def human_accuracy(results_dir: Path):
    path = results_dir / "analysis" / "dfew_fold1__class_descriptor" / "human_model_agreement.csv"
    if not path.exists():
        print(f"  note: {path} not found, the human bar is left out")
        return None
    table = pd.read_csv(path, index_col=0)
    return float(table.loc["human right"].sum())


def fig_per_fold(folds: pd.DataFrame, results_dir: Path, out_dir: Path) -> None:
    """Plain black-and-white table: accuracy with the original descriptions, then the change
    caused by each other wording, for each of the 5 models and their average."""
    original = folds[folds["prompt_set"] == "class_descriptor"].set_index("fold")["WAR"]
    other_sets = [name for name in TEXT_SETS if name != "class_descriptor"]
    fold_ids = sorted(original.index)
    change = (folds[folds["prompt_set"].isin(other_sets)]
              .pivot(index="prompt_set", columns="fold", values="WAR")
              .sub(original, axis="columns")
              .reindex(other_sets))
    change["average"] = change[fold_ids].mean(axis=1)

    not_significant = pd.DataFrame(False, index=change.index, columns=change.columns)
    significance_path = results_dir / "evaluation" / "significance_vs_class_descriptor_6class.csv"
    if significance_path.exists():
        significance = pd.read_csv(significance_path)
        significance["fold"] = significance["checkpoint"].str.extract(r"fold(\d)").astype(int)
        for row in significance.itertuples():
            if row.prompt_set in not_significant.index:
                not_significant.loc[row.prompt_set, row.fold] = row.p_value >= 0.05

    column_titles = [f"Model {fold}" for fold in fold_ids] + ["Average"]
    first_row = [f"{original[fold]:.1f}%" for fold in fold_ids] + [f"{original.mean():.1f}%"]
    table_rows = []
    for prompt_set in other_sets:
        cells = []
        for column in change.columns:
            value = f"{change.loc[prompt_set, column]:+.1f}"
            cells.append(f"({value})" if not_significant.loc[prompt_set, column] else value)
        table_rows.append((TEXT_SETS[prompt_set].replace("\n", " "), cells))

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    label_x = 0.0
    column_x = [0.42 + index * 0.105 for index in range(len(column_titles))]
    row_height = 0.095
    y = 0.95

    def rule(y_position, width=1.0):
        ax.plot([0, 1], [y_position, y_position], color="black", linewidth=width)

    rule(y + 0.04, 1.2)
    ax.text(label_x, y, "Text given to the model", fontsize=9, fontweight="bold", va="center")
    for x, title in zip(column_x, column_titles):
        ax.text(x, y, title, fontsize=9, fontweight="bold", ha="center", va="center")
    y -= row_height
    rule(y + 0.045, 0.6)
    ax.text(label_x, y, "Original descriptions (accuracy)", fontsize=9, va="center")
    for x, cell in zip(column_x, first_row):
        ax.text(x, y, cell, fontsize=9, ha="center", va="center")
    y -= row_height
    rule(y + 0.045, 0.6)
    ax.text(label_x, y, "Change compared with the original:", fontsize=9, style="italic", va="center")
    for label, cells in table_rows:
        y -= row_height
        ax.text(label_x + 0.02, y, label, fontsize=9, va="center")
        for x, cell in zip(column_x, cells):
            ax.text(x, y, cell, fontsize=9, ha="center", va="center")
    rule(y - 0.05, 1.2)
    ax.text(label_x, y - 0.1,
            "Accuracy = WAR (% of the 7,442 CREMA-D clips classified correctly). Changes are in percentage "
            "points:\n+ = better, \u2212 = worse than the original descriptions on the same model. "
            "Numbers in brackets are not statistically\nsignificant (McNemar test, p \u2265 0.05). "
            "Models 1\u20135: the five DFER-CLIP models trained on DFEW (one per fold).\n"
            "Each wording is used for all emotions; \u201canger\u201d is only an example.",
            fontsize=8, va="top", linespacing=1.4)
    save(fig, out_dir, "fig_per_fold_war")


def fig_three_step(folds: pd.DataFrame, plain: pd.DataFrame, human, out_dir: Path) -> None:
    by_set = folds.groupby("prompt_set")["WAR"]
    plain_original = float(plain.loc[plain["prompt_set"] == "class_descriptor", "WAR"].iloc[0])
    steps = [
        ("Plain CLIP\ncorrect text", plain_original, 0.0, ORANGE, "///"),
        ("DFER-CLIP\nwrong (shuffled) text", by_set.mean()["shuffled"], by_set.std()["shuffled"], BLUE, None),
        ("DFER-CLIP\ncorrect text", by_set.mean()["class_descriptor"], by_set.std()["class_descriptor"], BLUE, None),
    ]
    if human is not None:
        steps.append(("Humans\nsilent video", human, 0.0, HUMAN_GRAY, None))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for x, (label, value, std, color, hatch) in enumerate(steps):
        ax.bar(x, value, width=0.6, color=color, hatch=hatch, edgecolor="white", linewidth=0)
        if std:
            ax.errorbar(x, value, yerr=std, color=INK, capsize=4, linewidth=1)
        ax.text(x, value + std + 1.5, f"{value:.1f}%", ha="center", fontsize=10, color=INK,
                fontweight="bold")
    # what each step adds
    for x_from, x_to in ((0, 1), (1, 2)):
        gain = steps[x_to][1] - steps[x_from][1]
        top = max(steps[x_from][1] + steps[x_from][2], steps[x_to][1] + steps[x_to][2]) + 7
        ax.annotate("", xy=(x_to - 0.1, top), xytext=(x_from + 0.1, top),
                    arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=1))
        ax.text((x_from + x_to) / 2, top + 1, f"{gain:+.1f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=9)
    ax.set_ylabel("WAR on CREMA-D (%)", color=INK)
    ax.set_ylim(0, 85)
    ax.set_title("Where the accuracy comes from: training vs words", fontsize=11, color=INK, loc="left")
    chance_line(ax, len(steps) - 0.5)
    ax.text(0.0, -0.2, "DFER-CLIP: average of the 5 models trained on DFEW (line = spread between models). "
            "Humans: CREMA-D raters' majority vote\non the same clips without sound.",
            transform=ax.transAxes, fontsize=8, color=INK_MUTED, va="top")
    style_axes(ax)
    save(fig, out_dir, "fig_three_step")


def fig_plain_vs_dfer(folds: pd.DataFrame, plain: pd.DataFrame, out_dir: Path) -> None:
    by_set = folds.groupby("prompt_set")["WAR"]
    plain_war = plain.set_index("prompt_set")["WAR"]
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for x, prompt_set in enumerate(TEXT_SETS):
        p, d, s = plain_war[prompt_set], by_set.mean()[prompt_set], by_set.std()[prompt_set]
        ax.bar(x - width / 2 - 0.01, p, width, color=ORANGE, hatch="///", edgecolor="white", linewidth=0,
               label="Plain CLIP (no emotion training)" if x == 0 else None)
        ax.bar(x + width / 2 + 0.01, d, width, color=BLUE, linewidth=0,
               label="DFER-CLIP (average of 5 trained models; line = spread)" if x == 0 else None)
        ax.errorbar(x + width / 2 + 0.01, d, yerr=s, color=INK, capsize=3, linewidth=1)
        label_box = dict(facecolor="white", edgecolor="none", pad=0.6)  # lines pass behind labels
        ax.text(x - width / 2 - 0.01, p + 1, f"{p:.1f}", ha="center", fontsize=8, color=INK,
                bbox=label_box, zorder=4)
        ax.text(x + width / 2 + 0.01, d + s + 1, f"{d:.1f}", ha="center", fontsize=8, color=INK,
                bbox=label_box, zorder=4)
    ax.set_xticks(range(len(TEXT_SETS)), TEXT_SETS.values(), fontsize=9)
    ax.set_ylabel("WAR on CREMA-D (%)", color=INK)
    ax.set_ylim(0, 60)
    ax.set_title("Plain CLIP vs DFER-CLIP for each text set", fontsize=11, color=INK, loc="left")
    chance_line(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncols=3, bbox_to_anchor=(1.0, 1.0))
    style_axes(ax)
    save(fig, out_dir, "fig_plain_vs_dfer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=Path, default=REPO_ROOT / "results",
                        help="folder with evaluation/, analysis/ and plain_clip/")
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = load_folds(args.results_dir)
    plain = pd.read_csv(args.results_dir / "plain_clip" / "evaluation" / "summary.csv")
    plain = plain[plain["mode"] == "6-class"]
    human = human_accuracy(args.results_dir)

    print(f"{folds['fold'].nunique()} folds, {folds['prompt_set'].nunique()} text sets; "
          f"plain CLIP: {len(plain)} text sets; humans: {human}")
    fig_per_fold(folds, args.results_dir, out_dir)
    fig_three_step(folds, plain, human, out_dir)
    fig_plain_vs_dfer(folds, plain, out_dir)


if __name__ == "__main__":
    main()
