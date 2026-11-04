import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from lib.models.tgtrack.fastitpn import Block
import torch.utils.checkpoint as checkpoint

class GENERATIVE_DECODER(nn.Module):
    """ The generative decoder to generate intermediate templates from the input templates
    """
    def __init__(self, inplanes=512, dim=512, depth=8,
                 num_heads=16, mlp_ratio=4,
                 qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm,stride=16,num_template=6):
        super().__init__()
        self.norm_pix_loss = True
        self.decoder_embed = nn.Linear(inplanes, dim, bias=True)
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                naiveswiglu=True,
                subln=True
            )
            for _ in range(depth)])

        self.decoder_patch_size = stride
        self.decoder_norm = norm_layer(dim)
        self.decoder_pred = nn.Linear(dim, self.decoder_patch_size**2 * 6, bias=True) # decoder to patch

        self.gate = nn.Linear(dim * 2, dim)  # 输入当前时间步特征 + 历史特征，输出融合特征
        self.num_template = num_template
        self.grad_ckpt = False


    def patchify(self, imgs,in_chan):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.decoder_patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], in_chan, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * in_chan))
        return x

    def forward(self, feature, imgs):
        pred = self.forward_gen_decoder(feature)
        loss = self.forward_loss(pred, imgs)
        return loss

    def forward_gen_decoder(self, feature):
        feature = self.decoder_embed(feature)
        feature = feature.view(feature.size(0), self.num_template, -1, feature.size(2))

        outputs = []
        historical_memory = torch.zeros_like(feature[:, 0])  # (B, 49, 512), 初始化历史信息

        for t in range(1, self.num_template+1):  # 预测第2, 3, 4, 5, 6个模板
            current_input = feature[:, t - 1]  # (B, 49, 512), 当前时间步的输入

            combined_input = torch.cat([historical_memory, current_input], dim=-1)  # (B, 49, 1024)
            fused_feature = torch.sigmoid(self.gate(combined_input)) * historical_memory + (1 - torch.sigmoid(self.gate(combined_input))) * current_input
            for i, blk in enumerate(self.decoder_blocks):
                fused_feature = checkpoint.checkpoint(blk, fused_feature,None, use_reentrant=False) if self.grad_ckpt else blk(fused_feature)
            fused_feature = self.decoder_norm(fused_feature)

            output = self.decoder_pred(fused_feature)
            outputs.append(output)  # 存储结果
            historical_memory = fused_feature  # 更新历史信息

        outputs = torch.stack(outputs, dim=1)  # (B, 5, 49, decoder_output_dim)
        outputs = outputs.view(feature.size(0), -1, outputs.shape[-1])  # (B, 245, decoder_output_dim)

        return outputs

    def forward_loss(self, pred, imgs):
        """
        imgs: list of [B, 3, H, W]
        pred: [B, NHW, p*p*3]
        """
        imgs = torch.stack(imgs, dim=1)  # (b,n,c,h,w)
        batch, imgs_num, c, h, w = imgs.size()
        imgs = imgs.view(-1, *imgs.size()[2:])  # (bn,c,h,w)
        target = self.patchify(imgs,in_chan=6) # (bn, hw, c)
        target = target.reshape(-1, imgs_num, target.size(-2), target.size(-1)) #(b,n,hw,c)
        target = target.reshape(target.size(0), -1, target.size(-1)) #(b,nhw,c)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean()
        return loss

def build_generative_decoder(cfg, encoder):
    num_channels_enc = encoder.num_channels
    stride = cfg.MODEL.ENCODER.STRIDE
    in_channel = num_channels_enc
    generative_decoder = GENERATIVE_DECODER(inplanes=in_channel,
                                            dim=cfg.MODEL.GENERATIVE_DECODER.DIM,
                                            depth=cfg.MODEL.GENERATIVE_DECODER.DEPTH,
                                            num_heads=cfg.MODEL.GENERATIVE_DECODER.NUM_HEADS,
                                            norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                            stride=stride,
                                            num_template=cfg.DATA.TEMPLATE.NUMBER)
    return generative_decoder

