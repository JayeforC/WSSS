import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']


_C.DATA = CN()
_C.DATA.TYPE = 'ACDC'
_C.DATA.CROP_SIZE = 224
_C.DATA.IGNORE_INDEX = 255
_C.DATA.BATCH_SIZE = 2
_C.DATA.NUM_WORKERS = 4



_C.CAM = CN()
_C.CAM.SCALES = [1, 0.5, 1.5]
_C.CAM.RADIUS = 8
_C.CAM.BKG_SCORE = 0.45
_C.CAM.LOW_THRE = 0.35
_C.CAM.HIGH_THRE = 0.55
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'fusion'
# Model name
_C.MODEL.NAME = 'FusionFormer_w6'
# Pretrained weight from checkpoint
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 2
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Model parameters
_C.MODEL.FUSION = CN()
_C.MODEL.FUSION.PATCH_SIZE = 4
_C.MODEL.FUSION.IN_CHANS = 3
_C.MODEL.FUSION.EMBED_DIM = 512
_C.MODEL.FUSION.DEPTHS = [2, 6, 2]
_C.MODEL.FUSION.NUM_HEADS = [4, 8, 16]
_C.MODEL.FUSION.WINDOW_SIZE = 6
_C.MODEL.FUSION.MLP_RATIO = 4.
_C.MODEL.FUSION.QKV_BIAS = True
_C.MODEL.FUSION.QK_SCALE = False
_C.MODEL.FUSION.APE = False
_C.MODEL.FUSION.POOLING_SIZE=[6,6,6]
_C.MODEL.FUSION.PATCH_NORM = True

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 30
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.SAMPLES_PER_GPU = 2
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'step'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# step size, 
_C.TRAIN.LR_SCHEDULER.STEP_SIZE = 312
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.998

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'sgd'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------# Enable Pytorch automatic mixed precision (amp).
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
_C.SAVE = True
# Frequency to save checkpoint
_C.SAVE_FREQ = 30
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 1
# auc_save threshold
_C.AUC_THRESHOLD = 0.98
# eva mode
_C.EVAL_MODE = False
_C.REPEAT = True

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.pooling_size:
        config.MODEL.FUSION.POOLING_SIZE = args.pooling_size
    if args.tag:
        config.TAG = args.tag

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)
    return config