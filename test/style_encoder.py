import torch
import torch.nn as nn
import numpy as np
from swin_transformer import SwinTransformer  # 假設Swin Transformer模塊已經存在
from swin import SwinTransformer


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=512, num_domains=2):
        super().__init__()

        # self.patch_embed = PatchEmbed(img_size=img_size, patch_size=4, in_chans=3, embed_dim=512)
        
        # Initialize Swin Transformer
        self.swin_transformer = SwinTransformer(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=3,
                                embed_dim=96,
                                depths=[2, 2, 2],
                                num_heads=[3, 6, 6],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False,
                                mode=False
                                )
        self.fc1 = nn.Linear(196, 49)  # 用於降維
        self.fc2 = nn.Linear(9408, style_dim)
        self.conv = nn.Conv1d(384, 192, 3,1,1)
        self.act_layer = nn.GELU()
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(style_dim, style_dim)]

    def forward(self, x, y):
        h = self.swin_transformer(x)
        h = h.permute(0,2,1)
        h = self.fc1(h)
        h = self.act_layer(h)
        # print("fc1",h.shape)
        h = self.conv(h)
        h = self.act_layer(h)
        # print("conv",h.shape)
        
        h = h.view(h.size(0), -1)
        h = self.fc2(h)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.arange(y.size(0)).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s
