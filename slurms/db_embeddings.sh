#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=db_embeddings
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/general/vast/u1475870/logo_detection/logs/%j/%j_db_embeddings.out
#SBATCH --error=/scratch/general/vast/u1475870/logo_detection/logs/%j/%j_db_embeddings.err
#SBATCH --mail-user=pravallikaslurm@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue
#SBATCH --open-mode=append

# Create directories
SCRATCH_DIR="/scratch/general/vast/u1475870/logo_detection/"
LOG_DIR="$SCRATCH_DIR/logs/$SLURM_JOB_ID"
mkdir -p $LOG_DIR

echo "Job started/resumed on $(date)"
echo "Running on node: $SLURMD_NODENAME"

# Set up scratch directory
cd $SCRATCH_DIR

# Copy the latest version of files
echo "Copying latest version of db_embeddings.py..."
cp -f /uufs/chpc.utah.edu/common/home/$USER/logo_detection/db_embeddings.py .

# Copy logos directory
echo "Copying logos directory..."
cp -rf /uufs/chpc.utah.edu/common/home/$USER/logo_detection/logos .

# Print current directory contents
echo "Contents of current directory:"
ls -l

echo "Contents of logos directory:"
ls -lR logos/

# Set Hugging Face cache on scratch for faster repeated downloads
export HF_HOME="$SCRATCH_DIR/hf_cache"
mkdir -p "$HF_HOME"

# Load required modules
module purge
module load cuda/12.5.0
module load cudnn

# Activate virtual environment
source /uufs/chpc.utah.edu/common/home/$USER/logo_detection/venv/bin/activate

# Print environment info
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Check GPU and save info
nvidia-smi > $LOG_DIR/gpu_info.txt 2>&1
echo "GPU Info saved to: $LOG_DIR/gpu_info.txt"

# Run embedding database builder with real-time output
# Change --model to "clip" or "siglip" if desired
echo "Starting embedding database build..."
python db_embeddings.py \
    --logo_dir "$SCRATCH_DIR/logos" \
    --output "$SCRATCH_DIR/logo_embeddings.pkl" \
    --model dinov2 \
    2>&1 | tee $LOG_DIR/db_embeddings_output.txt

# Check if completed successfully
if [ $? -eq 0 ]; then
    echo "Embedding database build completed successfully"
    
    if [ -f "$SCRATCH_DIR/logo_embeddings.pkl" ]; then
        echo "Embeddings saved to: $SCRATCH_DIR/logo_embeddings.pkl"
        ls -lh "$SCRATCH_DIR/logo_embeddings.pkl"
    else
        echo "Warning: logo_embeddings.pkl not found"
    fi
else
    echo "Embedding database build failed"
    echo "Check error logs at: $LOG_DIR/db_embeddings_output.txt"
fi

# Deactivate virtual environment
deactivate

echo "Job ended on $(date)"
