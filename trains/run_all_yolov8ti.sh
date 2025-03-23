#!/bin/bash

# Run all YOLOv8 experiments
echo "Starting YOLOv8 TI experiments..."

echo "Running YOLOv8-N experiment..."
bash ./yolov8ti_n.sh

echo "Running YOLOv8-S experiment..."
bash ./yolov8ti_s.sh

echo "Running YOLOv8-M experiment..."
bash ./yolov8ti_m.sh

echo "Running YOLOv8-L experiment..."
bash ./yolov8ti_l.sh

echo "Running YOLOv8-X experiment..."
bash ./yolov8ti_x.sh

echo "All YOLOv8 TI experiments completed!" 