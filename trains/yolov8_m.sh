#!/bin/bash
cd $(dirname $(dirname $0))
python tools/train.py configs/yolov8ti/yolov8_m.py --model-surgery 2 