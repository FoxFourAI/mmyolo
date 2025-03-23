#!/bin/bash

# Run all YOLO8TI exports
echo "Starting YOLO8TI model exports..."

echo "Exporting YOLO8TI-N model..."
bash ./exports/yolov8ti_n.sh

echo "Exporting YOLO8TI-S model..."
bash ./exports/yolov8ti_s.sh

echo "Exporting YOLO8TI-M model..."
bash ./exports/yolov8ti_m.sh

echo "Exporting YOLO8TI-L model..."
bash ./exports/yolov8ti_l.sh

echo "Exporting YOLO8TI-X model..."
bash ./exports/yolov8ti_x.sh

echo "All YOLO8TI model exports completed!" 