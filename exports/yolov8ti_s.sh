#!/bin/bash
cd $(dirname $(dirname $0))
python projects/easydeploy/tools/export_onnx.py \
    configs/yolov8ti/yolov8ti_s.py \
    work_dirs/yolov8ti-s-exp1/epoch_30.pth \
    --work-dir work_dirs/yolov8ti-s-exp1 \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
    --opset 19 \
    --pre-topk 1000 \
    --keep-topk 100 \
    --iou-threshold 0.65 \
    --score-threshold 0.25 \
    --export-type YOLOv8 \
    --model-surgery 2 