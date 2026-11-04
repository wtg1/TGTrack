"""Training entry point for TGTrack."""
import os
import importlib
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn.functional import l1_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.models.tgtrack import build_tgtrack
from lib.train.actors import TGTrackActor
from lib.train.trainers import LTRTrainer
from lib.train.base_functions import update_settings, build_dataloaders, get_optimizer_scheduler
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss


def run(settings):
    if settings.script_name != "tgtrack":
        raise ValueError("This repository supports --script tgtrack")
    settings.description = "Temporal Generative Learning for Unified Single Object Tracking"
    config_module = importlib.import_module("lib.config.tgtrack.config")
    config_module.update_config_from_file(settings.cfg_file)
    cfg = config_module.cfg
    update_settings(settings, cfg)
    log_dir = os.path.join(settings.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    settings.log_file = os.path.join(log_dir, f"tgtrack-{settings.config_name}.log")
    loader_train = build_dataloaders(cfg, settings)
    net = build_tgtrack(cfg, training=True)
    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        if settings.script_name == 'tgtrack':
            find_unused_parameters=True
        else:
            find_unused_parameters = False
        net = DDP(net, broadcast_buffers=False, device_ids=[settings.local_rank], find_unused_parameters=find_unused_parameters) # modify the find_unused_parameters to False to skip a runtime error of twice variable ready
        for buffer in net.buffers(): # if we set broadcast_buffers=True, it occurs a in-place error, thus we manully broadcast buffer
            torch.distributed.broadcast(buffer, src=0)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    objective = {"giou": giou_loss, "l1": l1_loss, "focal": FocalLoss(),
                 "cls": BCEWithLogitsLoss(), "task_cls": CrossEntropyLoss()}
    loss_weight = {"giou": cfg.TRAIN.GIOU_WEIGHT, "l1": cfg.TRAIN.L1_WEIGHT,
                   "focal": 1., "cls": cfg.TRAIN.CE_WEIGHT,
                   "gen": cfg.TRAIN.GEN_WEIGHT, "task_cls": cfg.TRAIN.TASK_CE_WEIGHT}
    actor = TGTrackActor(net, objective, loss_weight, settings, cfg)
    optimizer, scheduler = get_optimizer_scheduler(net, cfg)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, scheduler,
                         use_amp=getattr(cfg.TRAIN, "AMP", False))
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=False)
