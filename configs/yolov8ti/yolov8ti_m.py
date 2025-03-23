_base_ = '../yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco.py'

deepen_factor = 0.67
widen_factor = 0.75

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

# Fast experiment settings
img_scale = (640, 640)  # smaller image size for faster training
train_batch_size_per_gpu = 1
max_epochs = 30  # just 1 epoch
num_samples = 10 * train_batch_size_per_gpu
val_interval = 10  # validate after each epoch
save_epoch_intervals = 1  # save checkpoint after each epoch
close_mosaic_epochs = 0  # disable mosaic augmentation completely for speed

# Dataset settings
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=0,  # disable data prefetch for speed
    persistent_workers=False,  # disable persistent workers for speed
    # Limit dataset to only 10 iterations worth of data
    dataset=dict(
        indices=range(num_samples)  # Process only 10 images in total
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,  # disable data prefetch for speed
    persistent_workers=False,  # disable persistent workers for speed
    # Use only 5 samples for validation
    dataset=dict(
        indices=range(5)  # Process only 5 images for validation
    )
)

work_dir = '/datasets/romanv/projects/yolov8ti/yolov8ti-m-exp1'
train_cfg = dict(max_epochs=max_epochs, val_interval=val_interval)