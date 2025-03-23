#!/bin/bash
cd $(dirname $(dirname $0))
python tools/train.py configs/yolov8ti/yolov8ti_x.py --model-surgery 2