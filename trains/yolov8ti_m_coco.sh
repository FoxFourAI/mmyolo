#!/bin/bash
cd $(dirname $(dirname $0))

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0  # Set to 0 for async CUDA operations

# Performance optimization for large batch training
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_yolov8ti_m_coco_${TIMESTAMP}.log"

# Echo command being run
echo "Starting training with YOLOv8-TI model on COCO dataset..."
echo "Log file: $LOG_FILE"

# Run training with output logging to file
{
  echo "=== Training started at $(date) ==="
  echo "Command: python tools/train.py configs/yolov8ti/yolov8ti_m_coco.py --model-surgery 2 --amp"
  echo "=== Environment ==="
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "PYTHONPATH: $PYTHONPATH"
  echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
  echo "MKL_NUM_THREADS: $MKL_NUM_THREADS"
  echo "=== Training output ==="

  # Run single GPU training with optimizations
  python tools/train.py configs/yolov8ti/yolov8ti_m_coco.py \
    --model-surgery 2 \
    --amp

  echo "=== Training completed at $(date) ==="
} 2>&1 | tee "$LOG_FILE"

echo "Training completed. Log saved to: $LOG_FILE" 