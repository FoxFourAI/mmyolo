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
- **Training**: Fast experiment with limited dataset
- **Model Surgery**: Applied during training (level 2)
- **Mixed Precision**: Enabled for faster training

## Running Training

### Training Settings

All training scripts use the following settings:
- **Model Surgery**: Level 2 (optimized FX-based transformations)
- **Base Config**: YOLOv8 with SyncBN
- **Fast Training Mode**:
  - Limited dataset size (10 samples per batch)
  - 30 epochs
  - Validation every 10 epochs
  - Checkpoint saving every epoch
  - Disabled mosaic augmentation
  - Minimal data prefetch for speed

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

Each training script:
- Uses model surgery level 2 (`--model-surgery 2`)
- Loads the appropriate configuration from `configs/yolov8ti/`
- Saves outputs to the corresponding experiment directory

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
- **Device**: CPU
- **ONNX Simplification**: Enabled
- **ONNX Opset**: 11
- **Pre-topk**: 1000
- **Keep-topk**: 100
- **IoU Threshold**: 0.65
- **Score Threshold**: 0.25
- **Export Type**: YOLOv5
- **Model Surgery**: Level 2

## Output Checkpoints

Checkpoints and logs will be saved to these directories:

- Nano: `/datasets/romanv/projects/yolov8ti/yolov8ti-n-exp1`
- Small: `/datasets/romanv/projects/yolov8ti/yolov8ti-s-exp1`
- Medium: `/datasets/romanv/projects/yolov8ti/yolov8ti-m-exp1`
- Large: `/datasets/romanv/projects/yolov8ti/yolov8ti-l-exp1`
- Extra Large: `/datasets/romanv/projects/yolov8ti/yolov8ti-x-exp1`

## Training Configuration Details

The training configuration includes:

- **Dataset Settings**:
  ```python
  train_dataloader = dict(
      batch_size=1,
      num_workers=0,  # Disabled for speed
      persistent_workers=False,
      dataset=dict(
          indices=range(10)  # Process only 10 images total
      )
  )

  val_dataloader = dict(
      batch_size=1,
      num_workers=0,
      persistent_workers=False,
      dataset=dict(
          indices=range(5)  # Process only 5 images for validation
      )
  )
  ```

- **Training Parameters**:
  ```python
  max_epochs = 30
  val_interval = 10
  save_epoch_intervals = 1
  close_mosaic_epochs = 0  # Disabled for speed
  ```

## Model Surgery Details

These configurations use model surgery level 2, which applies the optimized FX-based transformations to all model components. This makes the generated weights suitable for deployment on TI devices. 