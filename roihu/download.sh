#!/bin/bash
# Downloads everything into scratch. Run on a LOGIN node (compute nodes may have
# no internet), inside tmux so a dropped connection does not stop it:
#   tmux new -s download
#   source roihu/env.sh && bash roihu/download.sh
# FOLDS="1 2 3 4 5" bash roihu/download.sh   also fetches the other DFEW folds.
set -euo pipefail
source roihu/env.sh

# 1. OpenAI CLIP ViT-B/32. DFER-CLIP builds its model from it (weights are then
#    replaced by the checkpoint). clip.load() checks this file's SHA256.
CLIP_URL="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
if [ ! -f "${DFER_DATA_DIR}/clip_weights/ViT-B-32.pt" ]; then
    wget -O "${DFER_DATA_DIR}/clip_weights/ViT-B-32.pt" "${CLIP_URL}"
fi

# 2. DFER-CLIP checkpoints trained on DFEW (Google Drive IDs from the authors' README).
declare -A FOLD_IDS=(
    [1]=1tH1d2zZv2HtcQFPrGuVWJHXmD9sTeddQ
    [2]=1w6yESY3cNwcOCwQa-dTxIHNWBv57PRUR
    [3]=1kmI1XlZdQIr6R33tBdteyrmYl0UgyXv8
    [4]=10pxe7fCtASuKaxpZSGb7nR6EAjN7YSR6
    [5]=161pNBUgR_uPSvptzN4ZI4_L7-tvF9SD8
)
for fold in ${FOLDS:-1}; do
    target="${DFER_DATA_DIR}/checkpoints/dfew_fold${fold}.pth"
    if [ ! -f "${target}" ]; then
        gdown "${FOLD_IDS[$fold]}" -O "${target}" || {
            echo "gdown failed for fold ${fold}. Download it in a browser from the DFER-CLIP README and scp it to ${target}" >&2; }
    fi
done

# 3. CREMA-D videos only (VideoFlash, ~7,442 .flv files). The repo README
#    recommends the GitLab mirror; audio is skipped to save space.
if ! git lfs version >/dev/null 2>&1; then
    echo "git-lfs not found. Try: module spider git-lfs   (or ask CSC servicedesk)" >&2
    exit 1
fi
CREMA_DIR="${DFER_DATA_DIR}/raw/CREMA-D"
if [ ! -d "${CREMA_DIR}" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://gitlab.com/cs-cooper-lab/crema-d-mirror.git "${CREMA_DIR}"
fi

cd "${CREMA_DIR}"
# Skip if the videos are already here: after the cleanup (deleting .git/lfs/objects)
# git lfs would otherwise download all of them again.
if [ "$(find VideoFlash -name '*.flv' -size +1k | wc -l)" -lt 7442 ]; then
    git lfs pull --include="VideoFlash/*"
fi
echo "videos: $(find VideoFlash -name '*.flv' -size +1k | wc -l) of 7442"


# 4. Human ratings (FaceVote etc.). Not in the GitLab mirror, so take them from
#    the official GitHub repository.
mkdir -p processedResults
if [ ! -s processedResults/summaryTable.csv ]; then
    wget -O processedResults/summaryTable.csv \
        https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/processedResults/summaryTable.csv
fi
echo "human ratings: $(($(wc -l < processedResults/summaryTable.csv) - 1)) rows"