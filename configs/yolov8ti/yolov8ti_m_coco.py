_base_ = '../yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco.py'

deepen_factor = 0.67
widen_factor = 0.75

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)),
    # Enable gradient checkpointing to save memory and allow larger image size
    data_preprocessor=dict(batch_augments=None))


# Training settings - optimized for 16GB Tesla T4 with faster epochs
img_scale = (736, 1280)  # Increased back to 736x1280
train_batch_size_per_gpu = 12  # Increased batch size to 20
val_batch_size_per_gpu = 12
max_epochs = 200  # Total training epochs
save_epoch_intervals = 5  # Save checkpoint every 5 epochs
val_interval = 5  # Validate every 1 epoch
close_mosaic_epochs = 15  # Disable mosaic augmentation for last 15 epochs

# Limit number of images per epoch to make epochs complete faster
# This substantially reduces iterations per epoch (e.g., from 39429 to 1000)
samples_per_epoch = 20000  # Increased to 20000 images per epoch # 20000 means 20000/16 = 1250 iterations per epoch (2 minutes per 100 iterations)
test_visualization_interval = 20 # 20 images

work_dir = 'checkpoints/yolov8ti-m-736x1280-coco'

# COMMENTED FOR QUICK TRAINING
# 800 for quick testing
work_dir = 'checkpoints/yolov8ti-m-736x1280-coco-fast'
samples_per_epoch = 800 # / 16 = 50 iterations per epoch
# val_interval = 1
save_epoch_intervals = 1
max_epochs = 5


# Optimizer settings - using SGD which is compatible with YOLO framework
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', 
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005),
    paramwise_cfg=dict(
        norm_decay_mult=0, 
        bias_decay_mult=0,
        base_total_batch_size=64),  # Required by YOLOv5 optimizer constructor
    # Enable mixed precision training to save memory
    dtype='float16')

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001, 
        by_epoch=False, 
        begin=0, 
        end=500),  # Reduced warmup for faster epochs
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0001,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True)
]

# Use efficient pipeline for faster epochs
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),  # Faster loading
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        max_cached_images=20,  # Cache more images for faster mosaic
        pre_transform=[
            dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=10.0,
        max_shear_degree=2.0,
        scaling_ratio_range=(0.5, 1.5),
        # Ensure border calculations work with 768x1280 resolution
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

# Update train_dataloader to include the train_pipeline
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=4,  # Increased for faster data loading
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False),  # Keep background images to suppress false positives
        indices=range(samples_per_epoch),  # Only process a limited number of images per epoch
        pipeline=train_pipeline  # Explicitly set the pipeline
    )
)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,  # For T4 GPU
    num_workers=4,  # Increased for faster validation
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
)

test_dataloader = val_dataloader

# Model surgery for optimization
model_surgery = 2  # Use v2 surgery for better performance

# Setup for test-time behavior - used by visualization hooks
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.3,  # Score threshold for visualization and test
    nms=dict(type='nms', iou_threshold=0.65),
    max_per_img=300)

# Logging configuration with wandb
visualizer = dict(
    type='mmdet.DetLocalVisualizer',  # Use detection-specific visualizer
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend',
             init_kwargs={
                'project': 'yolov8-training',
                'name': 'yolov8ti_m_coco_768x1280_t4_fast',
                'config': {
                    'epochs': max_epochs,
                    'img_size': img_scale,
                    'batch_size': train_batch_size_per_gpu
                }
             })
    ],
    name='visualizer'
)

# Environment settings
env_cfg = dict(cudnn_benchmark=True)  # Enable for fixed input size

# Reduce evaluation time
val_evaluator = dict(
    proposal_nums=(100, 1, 10),  # Reduced proposal numbers for faster evaluation
)

# Update visualization configuration for more frequent early visualizations
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=save_epoch_intervals,
        max_keep_ckpts=3,
        save_best='auto',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # Increase frequency of validation visualizations
    visualization=dict(
        type='mmdet.DetVisualizationHook',
        draw=True,
        interval=test_visualization_interval,  # Every 20 validation iterations (about 12-13 images per epoch)
    ),
)

# Logging configuration
log_level = 'INFO'
log_file = dict(
    filename_tmpl='logs/{}.log',
    timestamp=True,
    mode='w',
    level=log_level,
)

# Configure custom hooks to capture stdout/stderr
custom_hooks = [
    dict(
        type='mmdet.EmptyCacheHook',
        after_epoch=True
    ),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49
    )
]

# Enable DDP backend for faster training
launcher = 'none'  # Set to 'none' for single GPU training

# Only log every 50 iterations to reduce logging overhead
log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True)

train_cfg = dict(
    max_epochs=max_epochs, 
    val_interval=val_interval,
    dynamic_intervals=[(max_epochs - close_mosaic_epochs, 1)],  # More frequent validation in final epochs
    )
