# This file is part of IntrinsicDiffusion.
# Licensed under the Apache License, Version 2.0.
# See https://github.com/JundanLuo/IntrinsicDiffusion for details.
# If you use this code, please cite the paper listed on the above website.
# ------------------------------------------------------------------------


import torch
from torch import nn

from modeling.crefnet.crefnet_swin_v2 import TransformerModule
from modeling.crefnet.crefnet_swin_v2 import Encoder as CRefNetEncoder
from modeling.swin_v2.swin_transformer_v2 import BasicLayer as SwinBasicLayer


checkpoint_module = SwinBasicLayer

class IntrinsicDiffusionConditionEmbedding(nn.Module):
    def __init__(self,
                 conditioning_embedding_channels: int = 320,
                 conditioning_channels: int = 3,
                 swin_features_resolution: int = 96,  # default to 96 for 768x768 images
                 target_domains: tuple = ("albedo", "shading", "surface normal")):
        super().__init__()

        # Validate the input parameter
        assert conditioning_embedding_channels % 8 == 0, \
            "conditioning_embedding_channels must be divisible by 8"

        # Configuration for the encoder
        initial_channels = conditioning_channels
        feature_channels = conditioning_embedding_channels // 8
        encoder_channel_multiplier = (1, 2, 4, 8)
        encoder_residual_blocks = 2

        # Encoder setup
        self.encoder = CRefNetEncoder(
            in_channels=initial_channels,
            ch=feature_channels,
            ch_mult=encoder_channel_multiplier,
            num_res_blocks=encoder_residual_blocks
        )

        # Configuration for the transformer module
        swin_group_depth = 4
        num_swin_groups = 2
        swin_window_size = 8

        # Transformer setup
        self.domain_transformers = nn.ParameterDict()
        for d in target_domains:
            self.domain_transformers[d] = TransformerModule(
                img_size=swin_features_resolution,
                patch_size=1,
                in_chans=feature_channels * encoder_channel_multiplier[-1],
                embed_dim=feature_channels * encoder_channel_multiplier[-1],
                depths=[swin_group_depth] * num_swin_groups,
                num_heads=[8] * num_swin_groups,
                window_size=swin_window_size,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                use_checkpoint=False,
                pretrained_window_sizes=[0] * num_swin_groups
            )
        self.swin_features_resolution = swin_features_resolution

    def forward(self, x, domain_labels):
        """
        Forward pass for the ControlNet conditioning embedding.
        :param x: Input tensor of shape BCHW
        :param domain_labels: str or list/tuple of strings indicating the domain for each sample in the batch.
        :return:
        """
        if type(domain_labels) == str:
            domain_labels = [domain_labels] * x.shape[0]
        assert type(domain_labels) == list or type(domain_labels) == tuple, \
            "domain_labels must be a list or tuple of strings"
        assert len(domain_labels) == x.shape[0], \
            "The number of domain labels must be the same as the batch size"

        # Pass input through the encoder and transformer
        x = self.encoder(x)
        out = []
        for i, label in enumerate(domain_labels):
            out.append(self.domain_transformers[label](x[i:i + 1]))
        x = torch.cat(out, dim=0)
        return x