""" video -> 16 face-cropped frames per clip, in the format DFER-CLIP reads.

The DFEW checkpoints were trained on folders of face-cropped JPEG frames, not on
raw video. To test the model fairly we give it CREMA-D in the same shape:

1. read every frame of the clip;
2. choose 16 frames with the same rule DFER-CLIP uses at test time
   (the middle frame of 16 equal segments);
3. detect the face on those frames and take the median box, so the crop does not
   jitter between frames (CREMA-D actors sit still in front of a fixed camera);
4. enlarge the box a little (DFEW crops include some hair and chin), make it
   square, crop all 16 frames with it and save them as 224x224 JPEGs.

Clips are split into shards so a Slurm array job can process them in parallel.
A log CSV per shard records how many frames had a detected face, so we can
report and inspect failures.
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import data_dir

IMAGE_SIZE = 224


def test_frame_indices(num_frames: int, num_segments: int = 16) -> np.ndarray:
    """Same as VideoDataset._get_test_indices in DFER-CLIP (duration = 1)."""
    if num_frames > num_segments:
        tick = num_frames / float(num_segments)
        return np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
    return np.pad(np.arange(num_frames), (0, num_segments - num_frames), mode="edge")


def read_all_frames(video_path: str) -> list:
    capture = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


class FaceBoxFinder:
    """Haar cascade face detector bundled with OpenCV (no extra downloads).

    Good enough for CREMA-D's frontal, evenly lit studio faces; the median over
    16 frames removes the occasional false detection.
    """

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"could not load {cascade_path}")

    def largest_face(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_side = max(30, min(gray.shape) // 10)  # ignore tiny false detections
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side)
        )
        if len(faces) == 0:
            return None
        return max(faces, key=lambda box: box[2] * box[3])

    def clip_box(self, frames: list):
        boxes = [b for b in (self.largest_face(f) for f in frames) if b is not None]
        if not boxes:
            return None, 0
        return np.median(np.array(boxes), axis=0), len(boxes)


def square_crop(frame: np.ndarray, box, box_scale: float) -> np.ndarray:
    x, y, w, h = box
    center_x, center_y = x + w / 2.0, y + h / 2.0
    side = int(round(max(w, h) * box_scale))
    left, top = int(round(center_x - side / 2.0)), int(round(center_y - side / 2.0))
    # Pad with edge pixels if the enlarged box leaves the image.
    pad = side
    padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = padded[top + pad: top + pad + side, left + pad: left + pad + side]
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def center_square(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return cv2.resize(frame[top:top + side, left:left + side], (IMAGE_SIZE, IMAGE_SIZE),
                      interpolation=cv2.INTER_AREA)


def process_clip(row, frames_root: Path, finder: FaceBoxFinder, num_frames: int,
                 box_scale: float, overwrite: bool) -> dict:
    out_dir = frames_root / row.clip_id
    if not overwrite and out_dir.exists() and len(list(out_dir.glob("*.jpg"))) == num_frames:
        return {"clip_id": row.clip_id, "status": "exists"}

    frames = read_all_frames(row.video_path)
    if not frames:
        return {"clip_id": row.clip_id, "status": "unreadable", "video_frames": 0}

    selected = [frames[i] for i in test_frame_indices(len(frames), num_frames)]
    box, faces_found = finder.clip_box(selected)

    out_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(selected):
        crop = square_crop(frame, box, box_scale) if box is not None else center_square(frame)
        cv2.imwrite(str(out_dir / f"{index:03d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {
        "clip_id": row.clip_id,
        "status": "ok" if box is not None else "no_face_center_crop",
        "video_frames": len(frames),
        "frames_with_face": faces_found,
        "frame_height": frames[0].shape[0],
        "frame_width": frames[0].shape[1],
    }


def process_rows(rows: pd.DataFrame, frames_root: Path, num_frames: int, box_scale: float,
                 overwrite: bool, report_every: int = 100) -> list:
    finder = FaceBoxFinder()
    log_rows, start = [], time.time()
    for done, row in enumerate(rows.itertuples(index=False), start=1):
        log_rows.append(process_clip(row, frames_root, finder, num_frames, box_scale, overwrite))
        if report_every and (done % report_every == 0 or done == len(rows)):
            rate = done / (time.time() - start)
            print(f"  {done}/{len(rows)} clips ({rate:.1f} clips/s)", flush=True)
    return log_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path,
                        default=data_dir() / "annotations" / "crema_d.csv")
    parser.add_argument("--frames_dir", type=Path, default=data_dir() / "frames" / "crema_d")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--box_scale", type=float, default=1.25,
                        help="enlarge the detected face box by this factor")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--processes", type=int, default=1,
                        help="CPU processes inside this job (e.g. 8 on a laptop)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N clips (debug)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    table = pd.read_csv(args.annotations)
    if args.limit:
        table = table.head(args.limit)
    shard = table.iloc[args.shard_index::args.num_shards]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(shard)} clips -> {args.frames_dir}")

    if args.processes <= 1:
        log_rows = process_rows(shard, args.frames_dir, args.num_frames, args.box_scale,
                                args.overwrite)
    else:
        parts = [shard.iloc[i::args.processes] for i in range(args.processes)]
        start, log_rows = time.time(), []
        with ProcessPoolExecutor(max_workers=args.processes) as pool:
            futures = [pool.submit(process_rows, part, args.frames_dir, args.num_frames,
                                   args.box_scale, args.overwrite, 0) for part in parts]
            for future in futures:
                log_rows.extend(future.result())
        print(f"  {len(log_rows)} clips in {time.time() - start:.0f}s "
              f"with {args.processes} processes", flush=True)

    log = pd.DataFrame(log_rows)
    print(log["status"].value_counts().to_string())
    log_dir = args.frames_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.frames_dir.name}_shard{args.shard_index:03d}.csv"
    # Keep the detection stats from earlier runs; "exists" rows carry no information.
    if log_path.exists():
        log = pd.concat([pd.read_csv(log_path), log])
    log = log[log["status"] != "exists"].drop_duplicates("clip_id", keep="last")
    log.to_csv(log_path, index=False)
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()