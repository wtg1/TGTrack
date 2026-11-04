"""Resolve TGTrack configurations and independently copied checkpoints."""
from copy import deepcopy
from pathlib import Path
from lib.test.utils import TrackerParams
from lib.test.evaluation.environment import env_settings
from lib.config.tgtrack.config import cfg, update_config_from_file

_DEFAULT_CONFIG = deepcopy(cfg)


def parameters(yaml_name):
    cfg.clear()
    cfg.update(deepcopy(_DEFAULT_CONFIG))
    env = env_settings()
    config_file = Path(env.prj_dir) / "experiments" / "tgtrack" / f"{yaml_name}.yaml"
    update_config_from_file(config_file)
    params = TrackerParams()
    params.cfg = deepcopy(cfg)
    params.yaml_name = yaml_name
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    checkpoint_config = cfg.TEST.CHECKPOINT or yaml_name
    params.checkpoint = str(Path(env.save_dir) / "checkpoints" / "train" / "tgtrack" /
                            checkpoint_config / f"TGTrack_ep{cfg.TEST.EPOCH:04d}.pth.tar")
    params.weights_only = False
    params.save_all_boxes = False
    params.debug = 0
    return params
