#!/bin/bash
cd $(dirname $(dirname $0))

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0  # Set to 0 for async CUDA operations

# Performance optimization for large batch training
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Pre-cache COCO dataset to RAM for faster training (if enough system RAM available)
# echo "Pre-caching dataset for faster access..."
# find /path/to/coco/dataset -type f -name "*.jpg" | xargs cat > /dev/null

# Run single GPU training with optimizations
echo "Starting optimized training with large batch size..."
python tools/train.py configs/yolov8ti/yolov8ti_m_coco.py \
  --model-surgery 2 \
  --amp \
  --cfg-options train_dataloader.persistent_workers=True \
  default_hooks.checkpoint.max_keep_ckpts=3

# Note: For distributed training on multiple T4s (if available in future), uncomment below:
# NUM_GPUS=1
# bash tools/dist_train.sh configs/yolov8ti/yolov8ti_m_coco.py $NUM_GPUS --model-surgery 2 --amp