#!/bin/bash
# Check if WEIGHTS_PATH is defined
if [ -z "$WEIGHTS_PATH" ]; then
    echo "Error: WEIGHTS_PATH environment variable is not defined."
    echo "Please set it with: export WEIGHTS_PATH=<path_to_checkpoint>"
    echo "Example: export WEIGHTS_PATH=checkpoints/yolov8ti-m-736x1280-coco-fast/epoch_3.pth"
    exit 1
fi

# Verify the weights file exists
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Error: Weights file does not exist at path: $WEIGHTS_PATH"
    exit 1
fi

echo "Using weights from: $WEIGHTS_PATH"

python projects/easydeploy/tools/export_onnx.py \
    configs/yolov8ti/yolov8ti_m_coco.py \
    $WEIGHTS_PATH \
    --work-dir $(dirname $WEIGHTS_PATH) \
    --img-size 736 1280 \
    --batch 1 \
    --device cpu \
    --simplify \
    --opset 11 \
    --pre-topk 1000 \
    --keep-topk 100 \
    --iou-threshold 0.65 \
    --score-threshold 0.25 \
    --export-type YOLOv5 \
    --model-surgery 2

# bash exports/yolov8ti_m_coco.sh