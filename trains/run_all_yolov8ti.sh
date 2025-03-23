#!/bin/bash

# Run all YOLO8TI experiments
echo "Starting YOLO8TI experiments..."

echo "Running YOLO8TI-N experiment..."
bash ./trains/yolov8ti_n.sh

echo "Running YOLO8TI-S experiment..."
bash ./trains/yolov8ti_s.sh

echo "Running YOLO8TI-M experiment..."
bash ./trains/yolov8ti_m.sh

echo "Running YOLO8TI-L experiment..."
bash ./trains/yolov8ti_l.sh

echo "Running YOLO8TI-X experiment..."
bash ./trains/yolov8ti_x.sh

echo "All YOLO8TI experiments completed!"