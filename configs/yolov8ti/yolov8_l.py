_base_ = './yolov8_l_syncbn_fast_8xb16-500e_coco.py'

deepen_factor = 1.00
widen_factor = 1.00

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))


work_dir = '/datasets/romanv/projects/yolov8ti/yolov8l-exp1' 