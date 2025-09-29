#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=object_detection_inference
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/general/vast/u1475870/photonode/logs/%j/%j_single_inference.out
#SBATCH --error=/scratch/general/vast/u1475870/photonode/logs/%j/%j_inference.err
#SBATCH --mail-user=pravallikaslurm@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue
#SBATCH --open-mode=append

# Create directories
SCRATCH_DIR="/scratch/general/vast/u1475870/photonode/"
LOG_DIR="$SCRATCH_DIR/logs/$SLURM_JOB_ID"
mkdir -p $LOG_DIR

echo "Job started/resumed on $(date)"
echo "Running on node: $SLURMD_NODENAME"

# Set up scratch directory
cd $SCRATCH_DIR

# Always copy the latest version of files
echo "Copying latest version of inference.py..."
cp -f /uufs/chpc.utah.edu/common/home/$USER/photonode/single_inference.py .

# Print current directory contents
echo "Contents of current directory:"
ls -l

# Print model directory contents
echo "Contents of model directory:"
ls -l model_output/

# Load required modules
module purge
module load cuda/12.5.0
module load cudnn

# Activate virtual environment
source /uufs/chpc.utah.edu/common/home/$USER/photonode/venv/bin/activate

# Print environment info
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Pip packages:"
pip list

# Check if model exists
MODEL_DIR="$SCRATCH_DIR/model_output"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found at $MODEL_DIR"
    exit 1
fi

# Check GPU and save info
nvidia-smi > $LOG_DIR/gpu_info.txt 2>&1
echo "GPU Info saved to: $LOG_DIR/gpu_info.txt"

# Run inference with real-time output
echo "Starting inference..."
python single_inference.py 2>&1 | tee $LOG_DIR/single_inference_output.txt

# Check if inference completed successfully
if [ $? -eq 0 ]; then
    echo "Inference completed successfully"
    
    # Copy results back
    OUTPUT_DIR="/uufs/chpc.utah.edu/common/home/$USER/photonode/outputs"
    mkdir -p $OUTPUT_DIR
    
    # Copy inference results
    if [ -d "outputs" ]; then
        cp -r outputs/* $OUTPUT_DIR/
        echo "Results copied to: $OUTPUT_DIR"
    else
        echo "Warning: Inference results directory not found"
    fi
    
    # Copy log files
    cp $LOG_DIR/single_inference_output.txt $OUTPUT_DIR/
    cp $LOG_DIR/gpu_info.txt $OUTPUT_DIR/
else
    echo "Inference failed"
    echo "Check error logs at: $LOG_DIR/single_inference_output.txt"
fi

# Deactivate virtual environment
deactivate

echo "Job ended on $(date)"
