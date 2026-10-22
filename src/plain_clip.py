"""Baseline: plain OpenAI CLIP (ViT-B/32), with none of DFER-CLIP's training.

DFER-CLIP starts from this exact CLIP model and adds three trained parts: a
fine-tuned image encoder, a temporal model over the 16 frames, and 8 learned
context tokens per emotion. Here we use CLIP as it was released, on the same
16 face frames per clip and the same text sets, so comparing the two shows
what DFER-CLIP's training adds.

Per clip: encode each frame with CLIP's image encoder, average the 16 frame
vectors into one clip vector, and pick the emotion whose text vector is most
similar (cosine similarity), the standard way CLIP is used zero-shot.

Text sets: every set in configs/prompts.json, used exactly as written (without
the learned tokens), plus "clip_template": the classic CLIP prompt
"a photo of a face showing {emotion}.".

Frames are normalised with CLIP's own mean/std, as CLIP expects. (DFER-CLIP's
loader feeds 0-1 pixel values without this normalisation; its fine-tuned image
encoder learned to work with that.)

Output: one CSV per text set in results/plain_clip/, same columns as
zero_shot.py, named clip_vitb32__<set>.csv, so evaluate.py and
significance.py work on them unchanged.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from common import CLASS_NAMES_7, REPO_ROOT, SURPRISE_INDEX, data_dir, pick_device
from zero_shot import write_list_file  # also puts the authors' code on the import path

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1)
MODEL_NAME = "clip_vitb32"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts_file", type=Path, default=REPO_ROOT / "configs" / "prompts.json")
    parser.add_argument("--annotations", type=Path,
                        default=data_dir() / "annotations" / "crema_d.csv")
    parser.add_argument("--frames_dir", type=Path, default=data_dir() / "frames" / "crema_d")
    parser.add_argument("--out_dir", type=Path, default=data_dir() / "results" / "plain_clip")
    parser.add_argument("--clip_root", type=Path, default=data_dir() / "clip_weights")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="only the first N clips (debug)")
    args = parser.parse_args()

    device = pick_device(args.device)
    table = pd.read_csv(args.annotations)
    if args.limit:
        table = table.head(args.limit)
    table, list_file, missing = write_list_file(table, args.frames_dir, num_frames=16)
    if missing:
        print(f"warning: {len(missing)} clips have no complete frame folder and are skipped")
    if table.empty:
        raise SystemExit("no clips with frames; run extract_frames.py first")

    from clip import clip
    from dataloader.video_dataloader import test_data_loader

    model, _ = clip.load("ViT-B/32", device="cpu", download_root=str(args.clip_root))
    model = model.float().to(device).eval()

    dataset = test_data_loader(list_file=str(list_file), num_segments=16, duration=1, image_size=224)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers,
                                         pin_memory=device.type == "cuda")

    # 1. One vector per clip: encode the 16 frames, average, normalise.
    print(f"encoding {len(dataset)} clips with plain CLIP on {device}")
    start, video_features = time.time(), []
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(loader, start=1):
            images = ((images - CLIP_MEAN) / CLIP_STD).to(device)  # (clips, 16, 3, 224, 224)
            clips, frames = images.shape[:2]
            features = model.encode_image(images.flatten(0, 1)).float()
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.view(clips, frames, -1).mean(dim=1)
            video_features.append((features / features.norm(dim=-1, keepdim=True)).cpu())
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(f"  batch {batch_index}/{len(loader)}", flush=True)
    video_features = torch.cat(video_features)
    encode_seconds = time.time() - start

    # 2. Every text set: encode the 7 sentences, compare with every clip.
    prompts = {k: v for k, v in json.loads(args.prompts_file.read_text()).items()
               if not k.startswith("_")}
    prompts["clip_template"] = [f"a photo of a face showing {name}." for name in CLASS_NAMES_7]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = table.drop(columns=["video_path"], errors="ignore").reset_index(drop=True)
    for prompt_set, texts in prompts.items():
        with torch.no_grad():
            text_features = model.encode_text(clip.tokenize(texts).to(device)).float().cpu()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * video_features @ text_features.T  # 100 = CLIP's usual logit scale

        six_class = logits.clone()
        six_class[:, SURPRISE_INDEX] = float("-inf")
        results = base.copy()
        results["prompt_set"] = prompt_set
        results["pred_7"] = logits.argmax(dim=1).numpy()
        results["pred_6"] = six_class.argmax(dim=1).numpy()
        for index, name in enumerate(CLASS_NAMES_7):
            results[f"logit_{name}"] = logits[:, index].numpy()

        out_csv = args.out_dir / f"{MODEL_NAME}__{prompt_set}.csv"
        results.to_csv(out_csv, index=False)
        out_csv.with_suffix(".json").write_text(json.dumps({
            "model": "OpenAI CLIP ViT-B/32, no DFER-CLIP training",
            "prompt_set": prompt_set,
            "prompts": texts,
            "clips": len(results),
            "device": str(device),
            "encode_seconds": round(encode_seconds, 1),
            "torch": torch.__version__,
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2))
        accuracy_7 = (results["pred_7"] == results["label"]).mean() * 100
        accuracy_6 = (results["pred_6"] == results["label"]).mean() * 100
        print(f"{prompt_set:>26}: 7-class {accuracy_7:5.1f}%, 6-class {accuracy_6:5.1f}%")

    print(f"done in {time.time() - start:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
