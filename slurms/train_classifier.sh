#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=train_logo_classifier
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/general/vast/u1475870/logo_detection/logs/%j/%j_train_classifier.out
#SBATCH --error=/scratch/general/vast/u1475870/logo_detection/logs/%j/%j_train_classifier.err
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

# Copy latest version of files
echo "Copying latest version of train_logo_classifier.py..."
cp -f /uufs/chpc.utah.edu/common/home/$USER/logo_detection/train_logo_classifier.py .

# Copy logos directory (with subfolders)
echo "Copying logos directory..."
cp -rf /uufs/chpc.utah.edu/common/home/$USER/logo_detection/logos .

# Create models directory
mkdir -p models

# Print current directory contents
echo "Contents of current directory:"
ls -l

# Set Hugging Face cache
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

# Check GPU
nvidia-smi > $LOG_DIR/gpu_info.txt 2>&1
echo "GPU Info saved to: $LOG_DIR/gpu_info.txt"

# Run training
echo "Starting classifier training..."
python train_logo_classifier.py 2>&1 | tee $LOG_DIR/train_classifier_output.txt

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
    
    # Copy trained model back to home
    MODEL_DIR="/uufs/chpc.utah.edu/common/home/$USER/logo_detection/models"
    mkdir -p $MODEL_DIR
    
    if [ -d "models" ]; then
        cp -r models/* $MODEL_DIR/
        echo "Model copied to: $MODEL_DIR"
    fi
    
    # Copy log files
    OUTPUT_DIR="/uufs/chpc.utah.edu/common/home/$USER/logo_detection/outputs"
    mkdir -p $OUTPUT_DIR
    cp $LOG_DIR/train_classifier_output.txt $OUTPUT_DIR/
else
    echo "Training failed"
fi

# Deactivate virtual environment
deactivate

echo "Job ended on $(date)"
