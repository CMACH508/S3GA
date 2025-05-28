import numpy as np
from easydict import EasyDict as edict
import yaml
import os

# 创建dict
__C = edict()
cfg = __C

# Minibatch size
__C.BATCH_SIZE = 4

# path to load pretrained model weights
__C.PRETRAINED_PATH = ''


#
# Problem settings. Set these parameters the same for fair comparison.
#
__C.PROBLEM = edict()

# Rescaled image size
__C.PROBLEM.RESCALE = (256, 256)

#
# Training options
#

__C.TRAIN = edict()

# Iterations per epochs
__C.TRAIN.EPOCH_ITERS = 7000

# Training start epoch. If not 0, will be resumed from checkpoint.
__C.TRAIN.START_EPOCH = 0

# Total epochs
__C.TRAIN.NUM_EPOCHS = 30

# Optimizer type
__C.TRAIN.OPTIMIZER = 'SGD'

# Start learning rate
__C.TRAIN.LR = 0.01

# Use separate learning rate for the CNN backbone
__C.TRAIN.SEPARATE_BACKBONE_LR = False

# Start learning rate for backbone
__C.TRAIN.BACKBONE_LR = __C.TRAIN.LR

# Learning rate decay
__C.TRAIN.LR_DECAY = 0.1

# Learning rate decay step (in epochs)
__C.TRAIN.LR_STEP = [10, 20]

# SGD momentum
__C.TRAIN.MOMENTUM = 0.9

# RobustLoss normalization
__C.TRAIN.RLOSS_NORM = max(__C.PROBLEM.RESCALE)

# Specify a class for training
__C.TRAIN.CLASS = 'none'

# Loss function. Should be 'offset' or 'perm'
__C.TRAIN.LOSS_FUNC = 'perm'

# SSL Loss function mode. Should be 'G2G' or 'L2L'
__C.TRAIN.LOSS_FUNC_MODE = 'G2G'

# Train Process. Should be 'pre_train' or 'fine_tune
__C.TRAIN.PROCESS = 'pre_train'

__C.TRAIN.LOSS_GCL = 1.0

__C.TRAIN.LOSS_PERM = 1.0

__C.TRAIN.LOSS_CONSISTENCY = 1.0

__C.TRAIN.LR_REDUCE_FACTOR = 0.0
__C.TRAIN.LR_SCHEDULE_PATIENCE = 0.0
__C.TRAIN.MAX_STEPS = 2
__C.TRAIN.NUM_ITER = 10

#
# Evaluation options
#

__C.EVAL = edict()

# Evaluation epoch number
__C.EVAL.EPOCH = 30
__C.EVAL.EPOCH_ITERS = 100

# PCK metric
__C.EVAL.PCK_ALPHAS = []
__C.EVAL.PCK_L = float(max(__C.PROBLEM.RESCALE))  # PCK reference.

# Number of samples for testing. Stands for number of image pairs in each classes (VOC)
__C.EVAL.SAMPLES = 1000

# Evaluated classes
__C.EVAL.CLASS = 'all'

# Parallel GPU indices ([0] for single GPU)
__C.GPUS = [0]

# num of dataloader processes
__C.DATALOADER_NUM = __C.BATCH_SIZE


# Data cache path
__C.CACHE_PATH = 'data/cache'

# Model name and dataset name
__C.MODEL_NAME = ''
__C.DATASET_NAME = ''
__C.DATASET_FULL_NAME = 'PascalVOC' # 'PascalVOC' or 'WillowObject'
__C.DATASET_PATH = './data/PascalVOC_SSL'
__C.DATASET_PARTITION = 'metis'

# Module path of module
__C.MODULE = ''

# Output path (for checkpoints, running logs and visualization results)
__C.OUTPUT_PATH = ''

# The step of iteration to print running statistics.
# The real step value will be the least common multiple of this value and batch_size
__C.STATISTIC_STEP = 100

# random seed used for data loading

__C.RANDOM_SEED = 123
# clustering setting.
__C.NUM_CENTROIDS = 1

# model setting.
__C.MODEL = edict()
__C.MODEL.IN_CHANNEL = 256
__C.MODEL.HIDDEN_CHANNEL = 256
__C.MODEL.OUT_CHANNEL = 256
__C.MODEL.NUM_LAYER = 1
# add self-loops or not.
__C.MODEL.LOOP = False
__C.MODEL.NORM = False

# subgraph setting.
__C.SUBGRAPH = edict()
__C.SUBGRAPH.BATCH_SIZE = 1
__C.SUBGRAPH.ARCHITECTURE_COMPENSATE = True
__C.SUBGRAPH.MERGE_CLUSTER = False
__C.SUBGRAPH.SCORE_FUNC_NAME = "linear"
__C.SUBGRAPH.SHUFFLE = True
__C.SUBGRAPH.NUM_WORKS = 0


# SLOT feature disturbantion ratio.
__C.PERMUTATION_RATIO = 0.0
__C.TRUNCATION_RATIO = 0.0
__C.COMPRESS_RATIO = 0.0
__C.EDGE_NOISE = 0.0

def lcm(x, y):
    """
    Compute the least common multiple of x and y. This function is used for running statistics.
    """
    greater = max(x, y)
    while True:
        if (greater % x == 0) and (greater % y == 0):
            lcm = greater
            break
        greater += 1
    return lcm


def get_output_dir(model, dataset):
    """
    Return the directory where experimental artifacts are placed.
    :param model: model name
    :param dataset: dataset name
    :return: output path (checkpoint and log), visual path (visualization images)
    """
    outp_path = os.path.join('output', '{}_{}'.format(model, dataset))
    return outp_path

# 内部方法，实现yaml配置文件到dict的合并
def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v
# 自动加载yaml文件
def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r', encoding='utf-8') as f:
        yaml_cfg = edict(yaml.load(f,Loader= yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
