"""
tgtrack Model
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from .encoder import build_encoder
from .clip import build_textencoder
from .decoder import build_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
from .generative_decoder import build_generative_decoder


class TGTrack(nn.Module):
    """ This is the base class for TGTrack """
    def __init__(self, text_encoder, encoder, decoder,generative_decoder,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER", task_feature_type="average"):
        """ Initializes the model.
        """
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder_type = decoder_type

        self.class_token = False if (encoder.body.cls_token is None) else True
        self.task_feature_type = task_feature_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.decoder = decoder
        self.generative_decoder = generative_decoder

        self.num_frames = num_frames
        self.num_template = num_template


    def forward(self, text_data=None,
                template_list=None, search_list=None, template_anno_list=None,template_frame_ids=None,
                text_src=None, task_index=None,
                feature=None, mode="encoder"):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        """
        if mode == "text":
            return self.forward_textencoder(text_data)
        elif mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list, template_frame_ids, text_src, task_index)
        elif mode == "decoder_generative_decoder":
            return self.forward_decoder(feature), self.forward_generative_decoder(feature, template_list)
        elif mode == "decoder":
            return self.forward_decoder(feature)
        else:
            raise ValueError

    def forward_textencoder(self, text_data):
        text_src = self.text_encoder(text_data)
        return text_src

    def forward_encoder(self, template_list, search_list, template_anno_list,template_frame_ids, text_src, task_index):
        template_list = template_list[0:self.num_template]
        template_anno_list = template_anno_list[0:self.num_template]
        xz = self.encoder(template_list, search_list, template_anno_list,template_frame_ids, text_src, task_index)
        return xz

    def forward_decoder(self, feature, gt_score_map=None):

        feature = feature[0]
        if self.class_token:
            feature = feature[:,1:self.num_patch_x+1]
        else:
            feature = feature[:,0:self.num_patch_x] # (B, HW, C)

        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def forward_generative_decoder(self, feature, template_list):
        template_reconstruct_list = template_list[1:]
        feature = feature[0]
        feature = feature[:,self.num_patch_x:-1]
        feature = self.generative_decoder(feature, template_reconstruct_list)
        return feature


def build_tgtrack(cfg, training=True, clip_state=None):
    encoder = build_encoder(cfg, training=training)
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        text_encoder = build_textencoder(cfg, encoder, state_dict=clip_state)
    else:
        text_encoder = None
    decoder = build_decoder(cfg, encoder)
    generative_decoder = build_generative_decoder(cfg, encoder)
    model = TGTrack(
        text_encoder,
        encoder,
        decoder,
        generative_decoder,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE
    )

    return model
