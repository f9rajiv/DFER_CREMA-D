# Source this from the repo root on Roihu:  source roihu/env.sh
# Sets paths, loads CSC's PyTorch module and activates our venv for this CPU type
# (Roihu-GPU is ARM/aarch64, Roihu-CPU is x86_64, so each side has its own venv).

if [ -z "${CSC_PROJECT:-}" ]; then
    echo "Set your project first, e.g.  export CSC_PROJECT=project_2001234  (MyCSC > Projects)" >&2
    return 1 2>/dev/null || exit 1
fi

export DFER_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DFER_DATA_DIR="/scratch/${CSC_PROJECT}/${USER}/dfer_crema"
export PIP_CACHE_DIR="${DFER_DATA_DIR}/.pip_cache"
export SBATCH_ACCOUNT="${CSC_PROJECT}"     # so sbatch and srun do not need --account
export SLURM_ACCOUNT="${CSC_PROJECT}"

module load python-pytorch/2.13

DFER_VENV="${DFER_REPO}/.venvs/$(uname -m)"
if [ -f "${DFER_VENV}/bin/activate" ]; then
    source "${DFER_VENV}/bin/activate"
fi