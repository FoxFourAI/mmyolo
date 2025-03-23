#!/bin/bash

# Run all YOLOv8 experiments
echo "Starting YOLOv8 experiments..."

echo "Running YOLOv8-N experiment..."
bash ./yolov8_n.sh

echo "Running YOLOv8-S experiment..."
bash ./yolov8_s.sh

echo "Running YOLOv8-M experiment..."
bash ./yolov8_m.sh

echo "Running YOLOv8-L experiment..."
bash ./yolov8_l.sh

echo "Running YOLOv8-X experiment..."
bash ./yolov8_x.sh

echo "All YOLOv8 experiments completed!" 