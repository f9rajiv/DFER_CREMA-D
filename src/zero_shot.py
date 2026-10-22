"""run the DFEW-trained DFER-CLIP model on CREMA-D without any training.

We build the model with the authors' own classes, load their checkpoint, feed the
frames through the authors' own test data loader, and save the raw scores
(logits) for every clip. All metrics are computed later from that CSV, so the GPU
only has to run once per prompt set.

Three details of the released checkpoints that this script handles:
* the keys carry a "module." prefix (they were saved from nn.DataParallel);
* they contain token_prefix / token_suffix buffers holding the embeddings of the
  ORIGINAL class descriptions. Loading them would silently mix the original
  text into every other prompt set (no error, just wrong scores), so we drop
  them and keep the ones computed from the prompt set we pass in (the authors'
  own comment in Prompt_Learner.py says to ignore them);
* the pickled checkpoint references a RecorderMeter class from their main.py,
  so a stand-in class has to exist for torch.load to unpickle it.

Prompt sets: the learned context tokens (8 per class) stay as trained; only the
description text after them changes.
"""

import argparse
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from common import (CLASS_NAMES_7, REPO_ROOT, SURPRISE_INDEX, add_dfer_clip_to_path,
                    data_dir, pick_device)

# At import time, so DataLoader workers started with "spawn" (macOS) can find
# the authors' dataloader module too.
add_dfer_clip_to_path()

DFEW_MODEL_ARGS = SimpleNamespace(  # train_DFEW.sh settings the checkpoints were trained with
    contexts_number=8,
    class_token_position="end",
    class_specific_contexts="True",
    temporal_layers=1,
    load_and_tune_prompt_learner="False",  # read by PromptLearner, missing from their main.py
)


class RecorderMeter:  # stand-in so torch.load can unpickle the training-curve object
    pass


def load_prompt_set(prompts_file: Path, name: str) -> list:
    prompts = json.loads(prompts_file.read_text())
    if name not in prompts or name.startswith("_"):
        options = [k for k in prompts if not k.startswith("_")]
        raise SystemExit(f"prompt set {name!r} not in {prompts_file}; options: {options}")
    texts = prompts[name]
    if len(texts) != len(CLASS_NAMES_7):
        raise SystemExit(f"prompt set {name!r} needs {len(CLASS_NAMES_7)} entries, has {len(texts)}")
    return texts


def build_model(input_text: list, checkpoint: Path, clip_root: Path):
    add_dfer_clip_to_path()
    from clip import clip
    from models.Generate_Model import GenerateModel

    clip_model, _ = clip.load("ViT-B/32", device="cpu", download_root=str(clip_root))
    model = GenerateModel(input_text=input_text, clip_model=clip_model, args=DFEW_MODEL_ARGS)

    import __main__
    __main__.RecorderMeter = RecorderMeter
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved.get("state_dict", saved)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    text_buffers = [k for k in state if k.endswith(("token_prefix", "token_suffix"))]
    for key in text_buffers:
        del state[key]

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith(("token_prefix", "token_suffix"))]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch. missing: {missing[:5]} unexpected: {unexpected[:5]}")
    print(f"loaded {checkpoint.name} (dropped {len(text_buffers)} original-text buffers)")
    return model.eval()


def write_list_file(table: pd.DataFrame, frames_dir: Path, num_frames: int) -> tuple:
    """DFER-CLIP's loader wants lines of '<frames_folder> <num_frames> <label>'."""
    keep, missing = [], []
    for row in table.itertuples(index=False):
        folder = frames_dir / row.clip_id
        if folder.is_dir() and len(list(folder.glob("*.jpg"))) == num_frames:
            keep.append(row)
        else:
            missing.append(row.clip_id)
    kept = pd.DataFrame(keep)
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    for row in kept.itertuples(index=False):
        handle.write(f"{frames_dir / row.clip_id} {num_frames} {row.label}\n")
    handle.close()
    return kept, Path(handle.name), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt_set", default="class_descriptor")
    parser.add_argument("--prompts_file", type=Path, default=REPO_ROOT / "configs" / "prompts.json")
    parser.add_argument("--annotations", type=Path,
                        default=data_dir() / "annotations" / "crema_d.csv")
    parser.add_argument("--frames_dir", type=Path, default=data_dir() / "frames" / "crema_d")
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--clip_root", type=Path, default=data_dir() / "clip_weights",
                        help="where OpenAI's ViT-B-32.pt is cached")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="only the first N clips (debug)")
    args = parser.parse_args()

    device = pick_device(args.device)
    input_text = load_prompt_set(args.prompts_file, args.prompt_set)
    run_name = f"{args.checkpoint.stem}__{args.prompt_set}"
    out_csv = args.out_csv or data_dir() / "results" / f"{run_name}.csv"

    table = pd.read_csv(args.annotations)
    if args.limit:
        table = table.head(args.limit)
    table, list_file, missing = write_list_file(table, args.frames_dir, num_frames=16)
    if missing:
        print(f"warning: {len(missing)} clips have no complete frame folder and are skipped "
              f"(first: {missing[:3]})")
    if table.empty:
        raise SystemExit("no clips with frames; run extract_frames.py first")

    model = build_model(input_text, args.checkpoint, args.clip_root).to(device)

    from dataloader.video_dataloader import test_data_loader
    dataset = test_data_loader(list_file=str(list_file), num_segments=16, duration=1, image_size=224)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers,
                                         pin_memory=device.type == "cuda")

    print(f"running {len(dataset)} clips on {device} with prompt set {args.prompt_set!r}")
    all_logits, start = [], time.time()
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(loader, start=1):
            all_logits.append(model(images.to(device)).float().cpu())
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(f"  batch {batch_index}/{len(loader)}", flush=True)
    logits = torch.cat(all_logits)
    seconds = time.time() - start

    six_class = logits.clone()
    six_class[:, SURPRISE_INDEX] = float("-inf")
    results = table.drop(columns=["video_path"], errors="ignore").reset_index(drop=True)
    results["prompt_set"] = args.prompt_set
    results["pred_7"] = logits.argmax(dim=1).numpy()
    results["pred_6"] = six_class.argmax(dim=1).numpy()
    for index, name in enumerate(CLASS_NAMES_7):
        results[f"logit_{name}"] = logits[:, index].numpy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    metadata = {
        "run_name": run_name,
        "checkpoint": str(args.checkpoint),
        "prompt_set": args.prompt_set,
        "prompts": input_text,
        "clips": len(results),
        "skipped_clips": len(missing),
        "device": str(device),
        "seconds": round(seconds, 1),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "host": platform.node(),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_csv.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    accuracy_7 = (results["pred_7"] == results["label"]).mean() * 100
    accuracy_6 = (results["pred_6"] == results["label"]).mean() * 100
    print(f"done in {seconds:.0f}s. quick accuracy: 7-class {accuracy_7:.1f}%, "
          f"6-class {accuracy_6:.1f}%  ->  {out_csv}")


if __name__ == "__main__":
    main()