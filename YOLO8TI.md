# YOLO8TI Experiments

This directory contains configurations for running YOLO8TI model variants with model surgery applied.

## Model Variants

- **YOLO8TI-N**: Nano version (smallest)
  - `deepen_factor = 0.33`
  - `widen_factor = 0.25`

- **YOLO8TI-S**: Small version
  - `deepen_factor = 0.33`
  - `widen_factor = 0.5`

- **YOLO8TI-M**: Medium version
  - `deepen_factor = 0.67`
  - `widen_factor = 0.75`

- **YOLO8TI-L**: Large version
  - `deepen_factor = 1.00`
  - `widen_factor = 1.00`

- **YOLO8TI-X**: Extra Large version (largest)
  - `deepen_factor = 1.00`
  - `widen_factor = 1.25`

## Configuration Settings

The model configurations use these default settings:

- **Image Size**: 640×640
- **Batch Size**: 1 per GPU
- **Training**: Full COCO dataset
- **Model Surgery**: Applied during training (level 2)
- **Mixed Precision**: Enabled for faster training

## Running Training

### Individual Model Training

To train a specific model variant:

```bash
# From project root
cd /datasets/romanv/repos/mmyolo

# Train Nano version
./trains/yolov8ti_n.sh

# Train Small version 
./trains/yolov8ti_s.sh

# Train Medium version
./trains/yolov8ti_m.sh

# Train Large version
./trains/yolov8ti_l.sh

# Train Extra Large version
./trains/yolov8ti_x.sh
```

### Run All Model Variants

To train all model variants in sequence:

```bash
# From project root
cd /datasets/romanv/repos/mmyolo
./trains/run_all_yolov8ti.sh
```

## Exporting Models

### Individual Model Export

To export a specific model to ONNX format:

```bash
# From project root
cd /datasets/romanv/repos/mmyolo

# Export Nano version
./exports/yolov8ti_n.sh

# Export Small version 
./exports/yolov8ti_s.sh

# Export Medium version
./exports/yolov8ti_m.sh

# Export Large version
./exports/yolov8ti_l.sh

# Export Extra Large version
./exports/yolov8ti_x.sh
```

### Export All Model Variants

To export all model variants in sequence:

```bash
# From project root
cd /datasets/romanv/repos/mmyolo
./exports/run_all_yolov8ti_exports.sh
```

### Export Settings

All models are exported with the following settings:
- **Format**: ONNX
- **Image Size**: 640×640
- **Batch Size**: 1
- **ONNX Opset**: 11
- **Pre-topk**: 1000
- **Keep-topk**: 100
- **IoU Threshold**: 0.65
- **Score Threshold**: 0.25
- **Export Type**: YOLOv8
- **Model Surgery**: Level 2

## Output Checkpoints

Checkpoints and logs will be saved to these directories:

- Nano: `/datasets/romanv/projects/yolov8ti/yolov8ti-n-exp1`
- Small: `/datasets/romanv/projects/yolov8ti/yolov8ti-s-exp1`
- Medium: `/datasets/romanv/projects/yolov8ti/yolov8ti-m-exp1`
- Large: `/datasets/romanv/projects/yolov8ti/yolov8ti-l-exp1`
- Extra Large: `/datasets/romanv/projects/yolov8ti/yolov8ti-x-exp1`

## Customization

To modify these experiments:

- Adjust image size: Change `img_scale` parameter
- Adjust batch size: Change `train_batch_size_per_gpu` parameter
- Change epochs: Modify `max_epochs` parameter (default is 500)

## Model Surgery Details

These configurations use model surgery level 2, which applies the optimized FX-based transformations to all model components. This makes the generated weights suitable for deployment. 