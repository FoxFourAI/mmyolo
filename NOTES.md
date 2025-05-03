## Installation
```bash
conda create -n edge-mmyolo python=3.10
conda activate edge-mmyolo
conda install -y nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install openmim
pip install numpy==1.26.4
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0,<4.0.0"
mim install -v -e .
pip install albumentations==1.3.1
pip install --no-input protobuf==3.20.2 onnx==1.14.0
pip install onnx-simplifier
pip install git+https://github.com/TexasInstruments/edgeai-modeloptimization.git@r9.1#subdirectory=torchmodelopt
pip install onnxruntime==1.15.1
```

## Setup Wandb
```bash
pip install wandb
wandb init
```

## Download Dataset
```bash
python tools/misc/download_dataset.py --dataset-name coco2017 --unzip --delete
```
## Training
Look to `configs/yolov8ti/yolov8ti_m_coco.py` for training settings.
```bash
bash trains/yolov8ti_m_coco.sh
```
## Export ONNX
```bash
# EXAMPLE: 
# export WEIGHTS_PATH=checkpoints/yolov8ti-m-736x1280-coco-fast/epoch_3.pth

export WEIGHTS_PATH=<PATH_TO_CHECKPOINT>
bash exports/yolov8ti_m_coco.sh
```

Example of created files:
```bash
checkpoints/yolov8ti-m-736x1280-coco-fast/epoch_3.prototxt
checkpoints/yolov8ti-m-736x1280-coco-fast/epoch_3.onnx
```

## Prerequisites
* System dependencies
```bash
sudo apt update
sudo apt install build-essential curl libbz2-dev libffi-dev liblzma-dev libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev libxml2-dev libxmlsec1-dev llvm make tk-dev wget xz-utils zlib1g-dev
```
* Conda