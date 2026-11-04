from easydict import EasyDict as edict
import yaml

"""Default configuration for TGTrack."""

cfg = edict()

# MODEL
cfg.MODEL = edict()

# TAKS_INDEX
cfg.MODEL.TASK_NUM=5 #should be the largest index number + 1
cfg.MODEL.TASK_INDEX = edict() # index for tasks
cfg.MODEL.TASK_INDEX.VASTTRACK = 0
cfg.MODEL.TASK_INDEX.LASOT = 0
cfg.MODEL.TASK_INDEX.TRACKINGNET = 0
cfg.MODEL.TASK_INDEX.GOT10K = 0
cfg.MODEL.TASK_INDEX.COCO = 0
cfg.MODEL.TASK_INDEX.TNL2K = 1
cfg.MODEL.TASK_INDEX.DEPTHTRACK = 2
cfg.MODEL.TASK_INDEX.LASHER = 3
cfg.MODEL.TASK_INDEX.VISEVENT = 4


# MODEL.LANGUAGE
cfg.MODEL.TEXT_ENCODER = edict()
cfg.MODEL.TEXT_ENCODER.TYPE = 'ViT-L/14' # clip: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px

# MODEL.ENCODER
cfg.MODEL.ENCODER = edict()
cfg.MODEL.ENCODER.TYPE = "fastitpnb" # encoder model
cfg.MODEL.ENCODER.DROP_PATH = 0
cfg.MODEL.ENCODER.PRETRAIN_TYPE = "pretrained/itpn/fast_itpn_base_clipl_e1600.pt"
cfg.MODEL.ENCODER.PATCHEMBED_INIT = "halfcopy" # copy, halfcopy, random
cfg.MODEL.ENCODER.USE_CHECKPOINT = False # to save the memory.
cfg.MODEL.ENCODER.STRIDE = 16
cfg.MODEL.ENCODER.POS_TYPE = 'index' # type of loading the positional encoding. "interpolate" or "index".
cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE = False # add a token_type_embedding to indicate the search, template_foreground, template_background
cfg.MODEL.ENCODER.TASK_INDICATE = False # add a token_type_embedding to indicate the search, template_foreground, template_background
cfg.MODEL.ENCODER.TEXT_INPUT_MODE = 'concat' # how to input the text_src to the encoder, choosen from concat, add_template, add, mul, mul_template
cfg.MODEL.ENCODER.CLASS_TOKEN = False # class token
cfg.MODEL.ENCODER.USE_TIME_TOKEN = False
cfg.MODEL.ENCODER.LEARNABLE_THETA = False

# MODEL.DECODER
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.TYPE = "CENTER" # MLP, CORNER, CENTER
cfg.MODEL.DECODER.NUM_CHANNELS = 256
cfg.MODEL.DECODER.CONV_TYPE = "normal" # normal: 3*3 conv, small: 1*1 conv, only for the center head for now.
cfg.MODEL.DECODER.XAVIER_INIT = True

# MODEL.TASK_DECODER
cfg.MODEL.TASK_DECODER = edict()
cfg.MODEL.TASK_DECODER.NUM_CHANNELS = 256
cfg.MODEL.TASK_DECODER.FEATURE_TYPE = "average" # class: using class token, average: average the feature, text: using the text token

# MODEL.GENERATIVE_DECODER
cfg.MODEL.GENERATIVE_DECODER = edict()
cfg.MODEL.GENERATIVE_DECODER.USE_GENERATIVE_DECODER = True
cfg.MODEL.GENERATIVE_DECODER.DIM = 512
cfg.MODEL.GENERATIVE_DECODER.DEPTH = 8
cfg.MODEL.GENERATIVE_DECODER.NUM_HEADS = 16
# Existing local checkpoints used 16 heads; preserve their attention semantics.

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.ENCODER_MULTIPLIER = 0.1  # encoder's LR = this factor * LR
cfg.TRAIN.FREEZE_ENCODER = False # for freezing the parameters of encoder
cfg.TRAIN.ENCODER_OPEN = [] # only for debug, open some layers of encoder when FREEZE_ENCODER is True
cfg.TRAIN.CE_WEIGHT = 1.0 # weight for cross-entropy loss
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.GEN_WEIGHT = 1.0
cfg.TRAIN.TASK_CE_WEIGHT = 1.0
cfg.TRAIN.MSE_WEIGHT = 1.0
cfg.TRAIN.PRINT_INTERVAL = 50 # interval to print the training log
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.FIX_BN = False
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.TYPE = "normal" # normal, peft, fft, text_frozen
cfg.TRAIN.PRETRAINED_PATH = None

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.SAMPLER_MODE = "order"  # sample data in order, but sample (num_search + num_template + 1) times
cfg.DATA.LOADER = "tracking"
cfg.DATA.MULTI_MODAL_VISION = True # vision multi-modal
cfg.DATA.MULTI_MODAL_LANGUAGE = True # language multi-modal
cfg.DATA.USE_NLP = edict() # using the text of the dataset
cfg.DATA.USE_NLP.LASOT = False
cfg.DATA.USE_NLP.GOT10K = False
cfg.DATA.USE_NLP.COCO = False
cfg.DATA.USE_NLP.TRACKINGNET = False
cfg.DATA.USE_NLP.VASTTRACK = False
cfg.DATA.USE_NLP.REFCOCOG = False
cfg.DATA.USE_NLP.TNL2K = False
cfg.DATA.USE_NLP.OTB99 = False
cfg.DATA.USE_NLP.DEPTHTRACK = False
cfg.DATA.USE_NLP.LASHER = False
cfg.DATA.USE_NLP.VISEVENT = False
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1  #number of search region, only support 1 for now.
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SAMPLE_NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0


# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 4.0
cfg.TEST.TEMPLATE_SIZE = 256
cfg.TEST.SEARCH_FACTOR = 2.0
cfg.TEST.SEARCH_SIZE = 128
cfg.TEST.EPOCH = 180
cfg.TEST.CHECKPOINT = ""
cfg.TEST.WINDOW = False # window penalty
cfg.TEST.NUM_TEMPLATES = 1

cfg.TEST.UPDATE_INTERVALS = edict()
cfg.TEST.UPDATE_INTERVALS.DEFAULT = 999999
cfg.TEST.UPDATE_INTERVALS.LASOT = 999999
cfg.TEST.UPDATE_INTERVALS.LASOT_EXTENSION_SUBSET = 999999
cfg.TEST.UPDATE_INTERVALS.GOT10K_TEST = 999999
cfg.TEST.UPDATE_INTERVALS.TRACKINGNET = 999999
cfg.TEST.UPDATE_INTERVALS.TNL2K = 999999
cfg.TEST.UPDATE_INTERVALS.OTB99_LANG = 999999
cfg.TEST.UPDATE_INTERVALS.LASHER = 999999
cfg.TEST.UPDATE_INTERVALS.RGBT234 = 999999
cfg.TEST.UPDATE_INTERVALS.VISEVENT = 999999
#
cfg.TEST.UPDATE_THRESHOLD = edict()
cfg.TEST.UPDATE_THRESHOLD.DEFAULT = 1.0
cfg.TEST.UPDATE_THRESHOLD.LASOT = 1.0
cfg.TEST.UPDATE_THRESHOLD.LASOT_EXTENSION_SUBSET = 1.0
cfg.TEST.UPDATE_THRESHOLD.GOT10K_TEST = 1.0
cfg.TEST.UPDATE_THRESHOLD.TRACKINGNET = 1.0
cfg.TEST.UPDATE_THRESHOLD.TNL2K = 1.0
cfg.TEST.UPDATE_THRESHOLD.OTB99_LANG = 1.0
cfg.TEST.UPDATE_THRESHOLD.LASHER = 1.0
cfg.TEST.UPDATE_THRESHOLD.RGBT234 = 1.0
cfg.TEST.UPDATE_THRESHOLD.VISEVENT = 1.0

cfg.TEST.MB = edict()
cfg.TEST.MB.DEFAULT = 500
cfg.TEST.MB.TNL2K = 500
cfg.TEST.MB.TRACKINGNET = 200
cfg.TEST.MB.OTB99_LANG = 500
cfg.TEST.MB.LASOT_EXTENSION_SUBSET = 500

# Explicit RGB-D keys let all benchmarks use the same model YAML.
for _dataset in ("DEPTHTRACK", "VOTRGBD22"):
    cfg.TEST.UPDATE_INTERVALS[_dataset] = 999999
    cfg.TEST.UPDATE_THRESHOLD[_dataset] = 1.0
for _dataset in ("LASOT", "GOT10K_TEST", "VISEVENT", "LASHER", "RGBT234",
                 "DEPTHTRACK", "VOTRGBD22"):
    cfg.TEST.MB[_dataset] = 500
#
cfg.TEST.MULTI_MODAL_VISION = edict()
cfg.TEST.MULTI_MODAL_VISION.DEFAULT = True
#
cfg.TEST.MULTI_MODAL_LANGUAGE = edict()
cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT = False
#
cfg.TEST.USE_NLP = edict()
cfg.TEST.USE_NLP.DEFAULT = False
cfg.TEST.USE_NLP.TNL2K = True




def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)
    # DEFAULT in a supplied mapping is its actual fallback. Unspecified
    # per-dataset defaults must not mask it after recursive configuration merge.
    for name in ("UPDATE_INTERVALS", "UPDATE_THRESHOLD", "MB"):
        supplied = exp_config.get("TEST", {}).get(name)
        if supplied is not None and "DEFAULT" in supplied:
            cfg.TEST[name] = edict(supplied)


