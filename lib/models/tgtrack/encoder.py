"""Fast-iTPN encoders used by TGTrack."""
from pathlib import Path
from torch import nn
from lib.models.tgtrack import fastitpn
from lib.utils.misc import is_main_process


class Encoder(nn.Module):
    def __init__(self, cfg, training=True):
        super().__init__()
        options = cfg.MODEL.ENCODER
        channels = {"fastitpnt": 384, "fastitpns": 384,
                    "fastitpnb": 512, "fastitpnl": 768}
        if options.TYPE not in channels:
            raise ValueError(f"Unsupported TGTrack backbone: {options.TYPE}")
        pretrain_path = Path(options.PRETRAIN_TYPE).expanduser()
        if not pretrain_path.is_absolute():
            pretrain_path = Path(__file__).resolve().parents[3] / pretrain_path
        pretrained = training and is_main_process()
        if pretrained and not pretrain_path.is_file():
            raise FileNotFoundError(f"Missing backbone initialization: {pretrain_path}")
        self.body = getattr(fastitpn, options.TYPE)(
            pretrained=pretrained, pretrain_type=str(pretrain_path),
            search_size=cfg.DATA.SEARCH.SIZE, template_size=cfg.DATA.TEMPLATE.SIZE,
            template_number=cfg.DATA.TEMPLATE.NUMBER,
            drop_rate=0., drop_path_rate=options.DROP_PATH, attn_drop_rate=0.,
            init_values=0.1, drop_block_rate=None, use_mean_pooling=True,
            grad_ckpt=options.USE_CHECKPOINT, cls_token=options.CLASS_TOKEN,
            pos_type=options.POS_TYPE, token_type_indicate=options.TOKEN_TYPE_INDICATE,
            task_indicate=options.TASK_INDICATE, task_num=cfg.MODEL.TASK_NUM,
            patchembed_init=options.PATCHEMBED_INIT,
            text_input_mode=options.TEXT_INPUT_MODE,
            use_time_token=options.USE_TIME_TOKEN,
            learnable_theta=options.LEARNABLE_THETA)
        self.num_channels = channels[options.TYPE]
        if options.CLASS_TOKEN:
            raise ValueError("TGTrack uses CLASS_TOKEN: false")
        if cfg.TRAIN.FREEZE_ENCODER or cfg.TRAIN.ENCODER_MULTIPLIER <= 0:
            for name, parameter in self.body.named_parameters():
                if not any(pattern in name for pattern in cfg.TRAIN.ENCODER_OPEN):
                    parameter.requires_grad_(False)

    def forward(self, templates, searches, annotations, frame_ids, text, task):
        return self.body(templates, searches, annotations, frame_ids, text, task)


def build_encoder(cfg, training=True):
    return Encoder(cfg, training=training)
