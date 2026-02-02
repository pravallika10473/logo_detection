#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=logo_inference
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/general/vast/u1475870/logo_detection/logs/%j/%j_logo_inference.out
#SBATCH --error=/scratch/general/vast/u1475870/logo_detection/logs/%j/%j_logo_inference.err
#SBATCH --mail-user=pravallikaslurm@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue
#SBATCH --open-mode=append

# ============================================================
# USAGE:
#   sbatch logo_inference.sh <image_name>
#   sbatch logo_inference.sh utah_first.png
#   sbatch logo_inference.sh pictureline.png
#
# If no image specified, defaults to processing all images in test_images/
# ============================================================

# Get input image from command line argument (or default to all)
INPUT_IMAGE="${1:-}"

# Create directories
SCRATCH_DIR="/scratch/general/vast/u1475870/logo_detection"
LOG_DIR="$SCRATCH_DIR/logs/$SLURM_JOB_ID"
OUTPUT_DIR="/uufs/chpc.utah.edu/common/home/$USER/logo_detection/output_final"
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

echo "Job started on $(date)"
echo "Running on node: $SLURMD_NODENAME"

# Set up scratch directory
cd $SCRATCH_DIR

# Copy the latest version of files
echo "Copying latest scripts..."
cp -f /uufs/chpc.utah.edu/common/home/$USER/logo_detection/logo_inference.py .
cp -f /uufs/chpc.utah.edu/common/home/$USER/logo_detection/db_embeddings.py .

# Copy test images
echo "Copying test images..."
cp -rf /uufs/chpc.utah.edu/common/home/$USER/logo_detection/test_images .

# Print current directory contents
echo "Contents of test_images:"
ls -l test_images/

# Set Hugging Face cache on scratch
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

# Check if embeddings database exists
EMBEDDINGS_PATH="$SCRATCH_DIR/logo_embeddings.pkl"
if [ ! -f "$EMBEDDINGS_PATH" ]; then
    echo "ERROR: Embedding database not found at $EMBEDDINGS_PATH"
    echo "Please run db_embeddings.sh first to build the database."
    exit 1
fi
echo "Found embedding database: $EMBEDDINGS_PATH"

# Function to run inference on a single image
run_inference() {
    local IMAGE_PATH="$1"
    local IMAGE_NAME=$(basename "$IMAGE_PATH" | sed 's/\.[^.]*$//')
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local OUTPUT_NAME="${IMAGE_NAME}_${SLURM_JOB_ID}_${TIMESTAMP}"
    
    echo ""
    echo "============================================================"
    echo "Processing: $IMAGE_PATH"
    echo "Output: $OUTPUT_DIR/${OUTPUT_NAME}.png"
    echo "============================================================"
    
    python logo_inference.py \
        --image "$IMAGE_PATH" \
        --output "$OUTPUT_DIR/${OUTPUT_NAME}.png" \
        --embeddings "$EMBEDDINGS_PATH" \
        --model dinov2 \
        --threshold 0.2 \
        --save_crops
}

# Run inference
if [ -n "$INPUT_IMAGE" ]; then
    # Single image specified
    IMAGE_PATH="$SCRATCH_DIR/test_images/$INPUT_IMAGE"
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "ERROR: Image not found: $IMAGE_PATH"
        echo "Available images:"
        ls -1 $SCRATCH_DIR/test_images/
        exit 1
    fi
    run_inference "$IMAGE_PATH" 2>&1 | tee $LOG_DIR/logo_inference_output.txt
else
    # Process all images in test_images/
    echo "No image specified, processing all images in test_images/"
    for IMAGE_PATH in $SCRATCH_DIR/test_images/*.{png,jpg,jpeg}; do
        [ -f "$IMAGE_PATH" ] || continue
        run_inference "$IMAGE_PATH"
    done 2>&1 | tee $LOG_DIR/logo_inference_output.txt
fi

# Check if completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Logo inference completed successfully"
    echo "Results saved to: $OUTPUT_DIR/"
    ls -lh $OUTPUT_DIR/ | tail -20
else
    echo "Logo inference failed"
    echo "Check error logs at: $LOG_DIR/logo_inference_output.txt"
fi

# Deactivate virtual environment
deactivate

echo "Job ended on $(date)"
