#!/bin/bash
cd $(dirname $(dirname $0))
python tools/train.py configs/yolov8ti/yolov8ti_n.py --model-surgery 2 --amp 