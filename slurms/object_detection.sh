#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=object_detection
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/general/vast/u1475870/logo_detection/logs/%j/%j_training.out
#SBATCH --error=/scratch/general/vast/u1475870/logo_detection/logs/%j/%j_training.err
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

# Always copy the latest version of the files
echo "Copying latest version of files..."
cp -f /uufs/chpc.utah.edu/common/home/$USER/logo_detection/object_detection.py .
cp -f /uufs/chpc.utah.edu/common/home/$USER/logo_detection/requirements.txt .

# Load required modules
module purge
module load cuda/12.5.0
module load cudnn

# Activate virtual environment
source /uufs/chpc.utah.edu/common/home/$USER/logo_detection/venv/bin/activate

# Print environment info
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "Contents of current directory:"
ls -l

# Check GPU and save info
nvidia-smi > $LOG_DIR/gpu_info.txt 2>&1
echo "GPU Info saved to: $LOG_DIR/gpu_info.txt"

# Run training with real-time output
echo "Starting training..."
echo "Using object_detection.py version:"
head -n 20 object_detection.py

python object_detection.py 2>&1 | tee $LOG_DIR/training_output.txt

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
    
    # Copy results back
    OUTPUT_DIR="/uufs/chpc.utah.edu/common/home/$USER/logo_detection/outputs/"
    mkdir -p $OUTPUT_DIR
    
    if [ -d "model_output" ]; then
        cp -r model_output $OUTPUT_DIR
    else
        echo "Warning: Output directory 'model_output' not found"
    fi
    
    # Copy log files
    cp $LOG_DIR/training_output.txt $OUTPUT_DIR
    cp $LOG_DIR/gpu_info.txt $OUTPUT_DIR
else
    echo "Training failed"
    echo "Check error logs at: $LOG_DIR/training_output.txt"
fi

# Deactivate virtual environment
deactivate

echo "Job ended on $(date)"
