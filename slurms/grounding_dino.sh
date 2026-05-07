#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=grounding_dino_logo
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/general/vast/u1475870/logo_detection/logs/%j/%j_gdino.out
#SBATCH --error=/scratch/general/vast/u1475870/logo_detection/logs/%j/%j_gdino.err
#SBATCH --mail-user=pravallikaslurm@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue
#SBATCH --open-mode=append

SCRATCH_DIR="/scratch/general/vast/u1475870/logo_detection"
LOG_DIR="$SCRATCH_DIR/logs/$SLURM_JOB_ID"
mkdir -p $LOG_DIR

echo "Job started on $(date)"
echo "Running on node: $SLURMD_NODENAME"

cd $SCRATCH_DIR

# Cache HF models on scratch
export HF_HOME="$SCRATCH_DIR/hf_cache"
mkdir -p "$HF_HOME"

# Deactivate conda if active
conda deactivate 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | grep -v anaconda | tr '\n' ':')

# Load modules
module purge
module load python/3.11
module load cuda/12.5.0
module load cudnn

# Create venv on scratch if it doesn't exist
VENV_DIR="$SCRATCH_DIR/venv_gdino"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv on scratch..."
    python3 -m venv $VENV_DIR
fi
source $VENV_DIR/bin/activate

echo "Python: $(which python)"

# Install deps — HuggingFace pipeline for GroundingDINO + CLIP + SigLIP
pip install -q torch torchvision
pip install -q "transformers>=4.46.0" pillow timm sentencepiece protobuf

# Verify
python -c "from transformers import AutoModelForZeroShotObjectDetection; print('HF GroundingDINO OK')" \
    || { echo "FATAL: transformers doesn't support GroundingDINO"; exit 1; }

nvidia-smi > $LOG_DIR/gpu_info.txt 2>&1

# Run DETR detector with all three embedders
echo "Starting DETR + all embedders..."
python grounding_dino_logo.py \
    --test-dir clean_test/test_images \
    --ref-dir clean_test \
    --detector detr \
    --embedder all \
    --box-threshold 0.30 \
    --output-dir detr_output \
    2>&1 | tee $LOG_DIR/detr_output.txt

if [ $? -eq 0 ]; then
    echo "Done successfully"
else
    echo "Failed — check $LOG_DIR/detr_output.txt"
fi

deactivate
echo "Job ended on $(date)"
