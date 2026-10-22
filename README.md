# Prompt Design and Domain Transfer in Vision-Language Dynamic Facial Expression Recognition

Zero-shot evaluation of DFER-CLIP (Zhao & Patras, BMVC 2023; trained on DFEW movie clips)
on CREMA-D (acted studio clips), with varied class descriptions.

## Pipeline

```
CREMA-D videos ──make_annotations.py──> annotations CSV (labels from filenames + human FaceVote)
               ──extract_frames.py───> 16 face-cropped 224x224 frames per clip   [CPU]
               ──zero_shot.py────────> per-clip scores, one CSV per prompt set   [GPU]
               ──evaluate.py─────────> WAR / UAR table + confusion matrices
               ──analysis.py─────────> intensity, actor, sentence, human-vs-model
               ──overlay_predictions.py> demo figures (webcam clips)
```

| Path | What |
|---|---|
| `src/` | our code (snake_case, one script per step) |
| `configs/prompts.json` | the 6 prompt sets (class names, original descriptions, FACS, other LLM, shuffled, ...) |
| `third_party/DFER-CLIP` | authors' code as a git submodule, never edited |
| `roihu/` | Roihu environment, one-time setup, downloads |
| `slurm/` | batch jobs (frame extraction on CPU, inference on GPU) |
| `results/` | small result files we commit (CSVs, figures) |

Data, frames, checkpoints and venvs are not in Git. On Roihu they live in
`/scratch/<project>/<user>/dfer_crema` (set by `roihu/env.sh` via `DFER_DATA_DIR`);
locally they default to `./data`.

## Quick start

Local (Mac, uv):
```bash
git clone --recurse-submodules <repo-url> && cd dfer_crema
uv sync
uv run python src/make_annotations.py --video_dir <CREMA-D>/VideoFlash
uv run python src/extract_frames.py --processes 8
uv run python src/zero_shot.py --checkpoint checkpoints/dfew_fold1.pth --prompt_set class_descriptor
uv run python src/evaluate.py
```

Roihu: see the step-by-step guide in the course project docs. In short:
```bash
export CSC_PROJECT=project_XXXXXXX
bash roihu/setup.sh                    # once on roihu-gpu and once on roihu-cpu
source roihu/env.sh && bash roihu/download.sh
python src/make_annotations.py --video_dir $DFER_DATA_DIR/raw/CREMA-D/VideoFlash \
    --summary_csv $DFER_DATA_DIR/raw/CREMA-D/processedResults/summaryTable.csv
sbatch slurm/extract_frames.sbatch     # on roihu-cpu
sbatch slurm/zero_shot.sbatch          # on roihu-gpu
```

## Notes on loading the DFEW checkpoints

`src/zero_shot.py` builds the model with the authors' classes and test loader, and:
strips the `module.` prefix (saved from DataParallel); drops the `token_prefix`/`token_suffix`
buffers, which hold the original description embeddings and would otherwise silently mix the
original text into every other prompt set; provides a stand-in `RecorderMeter` so the pickled
checkpoint loads; and sets `load_and_tune_prompt_learner`, which the authors' PromptLearner
reads but their `main.py` never defines. The learned context tokens stay as trained; only the
description text changes between prompt sets.