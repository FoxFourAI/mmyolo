#!/bin/bash
python projects/easydeploy/tools/export_onnx.py \
    configs/yolov8ti/yolov8ti_m.py \
    /datasets/romanv/projects/yolov8ti/yolov8ti-m-768x1280-fast/best_coco_bbox_mAP_epoch_127.pth \
    --work-dir /datasets/romanv/projects/yolov8ti/yolov8ti-m-768x1280-fast \
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

# (cd /datasets/romanv/repos/mmyolo && bash ./exports/yolov8ti_m_coco.sh)