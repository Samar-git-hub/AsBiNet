# Built for the MMEngine Runner class: https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py

# Block 1: Runtime

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

# Block 2: Model

norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
)

# DeepLabV3 with a ResNet50 backbone, similar to the Torchvision implementation
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='torchvision://resnet50',

    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),

        strides=(1, 2, 1, 1),
        dilations=(1, 1, 2, 4),

        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True
    ),

    decode_head=dict(
        type='ASPPHead',
        in_channels=2048,
        in_index=3,
        channels=256,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.5,
        num_classes=1,
        threshold=0.5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        )
    ),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs = 1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=1,
        threshold=0.5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=0.5
        )
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)


# Block 3: Dataset

dataset_type = 'FracAtlasDataset'
data_root = 'data/FracAtlas'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,

    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='processed/original/train.csv',
        data_prefix=dict(img_path='processed/original/train/images', seg_map_path='processed/original/train/masks'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='processed/original/valid.csv',
        data_prefix=dict(img_path='processed/original/valid/images', seg_map_path='processed/original/valid/masks'),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# Block 4: Schedule and Optimizer

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0001)
)

param_scheduler = [
    dict(
        type='ReduceOnPlateau',
        monitor='mIoU',
        rule='greater',
        factor=0.9,
        patience=5,
        min_lr=1e-6,
        by_epoch=True
    )
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Block 5: Hooks

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),

    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        save_best='mIoU',
        rule='greater'
    ),

    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

randomness = dict(seed=42, deterministic=False)