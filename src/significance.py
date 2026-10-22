"""Paired significance tests: is a prompt set really better or worse than the baseline?

Two prompt sets are run on the SAME clips with the SAME model (one DFEW fold),
so we compare them clip by clip (McNemar's test):
* b = clips the baseline gets right and the other prompt set gets wrong
* c = clips the other prompt set gets right and the baseline gets wrong
Clips both get right or both get wrong say nothing about which is better.
If the two were equally good, b and c would be about equal (like coin flips);
the p-value is the probability of an imbalance at least this large by chance
(exact binomial test on b vs c). p < 0.05 is the usual threshold.

Across folds we also count in how many folds each prompt set beats the baseline.
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from common import data_dir


def mcnemar(base_correct: pd.Series, other_correct: pd.Series) -> tuple:
    b = int((base_correct & ~other_correct).sum())
    c = int((~base_correct & other_correct).sum())
    p_value = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p_value


def stars(p_value: float) -> str:
    return "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=Path, default=data_dir() / "results")
    parser.add_argument("--baseline", default="class_descriptor")
    parser.add_argument("--mode", choices=["6", "7"], default="6",
                        help="6-class (surprise removed) or 7-class predictions")
    args = parser.parse_args()

    pred_column = f"pred_{args.mode}"
    runs = {}
    for path in sorted(args.results_dir.glob("*__*.csv")):
        checkpoint, prompt_set = path.stem.split("__", 1)
        table = pd.read_csv(path, usecols=["clip_id", "label", pred_column]).set_index("clip_id")
        runs.setdefault(checkpoint, {})[prompt_set] = table[pred_column] == table["label"]
    if not runs:
        raise SystemExit(f"no result CSVs in {args.results_dir}")

    rows = []
    for checkpoint, prompt_sets in runs.items():
        if args.baseline not in prompt_sets:
            print(f"skipping {checkpoint}: no {args.baseline} run")
            continue
        base = prompt_sets[args.baseline]
        for prompt_set, correct in prompt_sets.items():
            if prompt_set == args.baseline:
                continue
            common_clips = base.index.intersection(correct.index)
            b, c, p_value = mcnemar(base[common_clips], correct[common_clips])
            rows.append({
                "checkpoint": checkpoint,
                "prompt_set": prompt_set,
                "clips": len(common_clips),
                "baseline_WAR": base[common_clips].mean() * 100,
                "WAR": correct[common_clips].mean() * 100,
                "difference": (correct[common_clips].mean() - base[common_clips].mean()) * 100,
                "baseline_only_right": b,
                "other_only_right": c,
                "p_value": p_value,
                "significance": stars(p_value),
            })

    table = pd.DataFrame(rows)
    out_dir = args.results_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"significance_vs_{args.baseline}_{args.mode}class.csv"
    table.to_csv(out_csv, index=False)

    table["cell"] = table.apply(lambda r: f"{r['difference']:+.1f} {r['significance']}", axis=1)
    print(f"WAR difference vs {args.baseline} ({args.mode}-class), McNemar test per fold\n"
          "(* p<0.05, ** p<0.01, *** p<0.001, n.s. = not significant)\n")
    print(table.pivot(index="prompt_set", columns="checkpoint", values="cell").to_string())

    summary = table.groupby("prompt_set").agg(
        folds=("difference", "size"),
        better=("difference", lambda d: int((d > 0).sum())),
        significantly_better=("difference", lambda d: 0),
        significantly_worse=("difference", lambda d: 0),
        mean_difference=("difference", "mean"),
    )
    significant = table["p_value"] < 0.05
    summary["significantly_better"] = table[significant & (table["difference"] > 0)].groupby("prompt_set").size()
    summary["significantly_worse"] = table[significant & (table["difference"] < 0)].groupby("prompt_set").size()
    summary = summary.fillna(0).astype({"significantly_better": int, "significantly_worse": int})
    print("\nper prompt set, across folds:\n", summary.round(2).to_string())
    print(f"\nfull table: {out_csv}")


if __name__ == "__main__":
    main()
