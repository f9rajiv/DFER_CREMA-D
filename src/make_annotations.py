""" turn CREMA-D (or our webcam clips) into one annotation table.

Each video filename encodes its labels, e.g. 1058_IEO_SAD_HI.flv:
actor 1058, sentence IEO, emotion SAD, intensity HI. We parse that into a CSV
with one row per clip, add the DFEW class index the model uses, and, if
CREMA-D's summaryTable.csv is available, the emotion that human raters
perceived from the silent video (FaceVote). Every later step reads this CSV.
"""

import argparse
from pathlib import Path

import pandas as pd

from common import CREMA_EMOTION_TO_INDEX, VOTE_LETTER_TO_EMOTION, data_dir

VIDEO_EXTENSIONS = {".flv", ".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_clip_name(stem: str) -> dict:
    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"unexpected clip name {stem!r}, expected ACTOR_SENTENCE_EMOTION_LEVEL")
    actor, sentence, emotion, intensity = parts
    if emotion not in CREMA_EMOTION_TO_INDEX:
        raise ValueError(f"unknown emotion code {emotion!r} in {stem!r}")
    return {
        "clip_id": stem,
        "actor": int(actor),
        "sentence": sentence,
        "emotion": emotion,
        "intensity": intensity,
        "label": CREMA_EMOTION_TO_INDEX[emotion],
    }


def load_human_votes(summary_csv: Path) -> pd.DataFrame:
    """FaceVote = majority emotion chosen by raters who saw the video without sound.

    Ties are stored like "A:N"; we keep all tied letters.
    """
    votes = pd.read_csv(summary_csv)
    votes = votes.rename(columns={"FileName": "clip_id"})

    def to_codes(value) -> str:
        if pd.isna(value):
            return ""
        return ":".join(VOTE_LETTER_TO_EMOTION.get(v, v) for v in str(value).split(":"))

    votes["human_face_vote"] = votes["FaceVote"].map(to_codes)
    votes["human_face_agreement"] = pd.to_numeric(votes["FaceLevel"], errors="coerce")
    return votes[["clip_id", "human_face_vote", "human_face_agreement"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, required=True,
                        help="folder with the videos, e.g. .../CREMA-D/VideoFlash")
    parser.add_argument("--summary_csv", type=Path, default=None,
                        help="CREMA-D processedResults/summaryTable.csv (optional)")
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--dataset_name", default="crema_d")
    args = parser.parse_args()

    out_csv = args.out_csv or data_dir() / "annotations" / f"{args.dataset_name}.csv"

    rows, skipped = [], []
    for video in sorted(args.video_dir.iterdir()):
        if video.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        # Git LFS placeholders are ~130-byte text files, not videos.
        if video.stat().st_size < 1024:
            skipped.append(video.name)
            continue
        row = parse_clip_name(video.stem)
        row["video_path"] = str(video.resolve())
        rows.append(row)

    if not rows:
        raise SystemExit(f"no videos found in {args.video_dir}")
    table = pd.DataFrame(rows)


    # CREMA-D ships one clip twice (1015_DFA_ANG_XX as .flv and .mp4).
    # Keep one file per clip, preferring the .flv version like all other clips.
    duplicated = sorted(table.loc[table["clip_id"].duplicated(), "clip_id"])
    if duplicated:
        table["is_flv"] = table["video_path"].str.endswith(".flv")
        table = (table.sort_values("is_flv", ascending=False)
                 .drop_duplicates("clip_id", keep="first")
                 .drop(columns="is_flv")
                 .sort_values("clip_id")
                 .reset_index(drop=True))
        print(f"note: {duplicated} existed in two formats; kept the .flv version")

    if args.summary_csv and args.summary_csv.exists():
        table = table.merge(load_human_votes(args.summary_csv), on="clip_id", how="left")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    print(f"wrote {len(table)} clips to {out_csv}")
    if skipped:
        print(f"skipped {len(skipped)} files that are Git LFS pointers (not downloaded yet)")
    print(table.groupby("emotion").size().rename("clips").to_string())
    print(f"actors: {table['actor'].nunique()}")


if __name__ == "__main__":
    main()