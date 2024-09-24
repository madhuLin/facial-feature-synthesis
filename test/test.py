# from swin_transformer import SwinTransformer
import torch
from style_encoder import StyleEncoder
import torch.nn as nn
# model = SwinTransformer(img_size=224,
#                                 patch_size=4,
#                                 in_chans=3,
#                                 num_classes=3,
#                                 embed_dim=96,
#                                 depths=[2, 2, 6, 2],
#                                 num_heads=[3, 6, 12, 24],
#                                 window_size=7,
#                                 mlp_ratio=4.,
#                                 qkv_bias=True,
#                                 qk_scale=None,
#                                 drop_rate=0.0,
#                                 drop_path_rate=0.1,
#                                 ape=False,
#                                 patch_norm=True,
#                                 use_checkpoint=False,
#                                 )

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, content_feat, style_feat):
        size = content_feat.size()
        content_mean, content_std = self.calc_mean_std(content_feat)
        style_mean, style_std = self.calc_mean_std(style_feat)
        print("content_mean",content_mean.shape)
        print("content_std",content_std.shape)
        print("style_mean",style_mean.shape)
        print("style_std", style_std.shape)
        normalized_feat = (content_feat - content_mean) / content_std
        return normalized_feat * style_std + style_mean
    
    def calc_mean_std(self, feat, eps=1e-5):
        N, C = feat.size()[:2]
        feat_var = feat.view(N, -1, C).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1)
        feat_mean = feat.view(N, -1, C).mean(dim=2).view(N, 1, C)
        return feat_mean, feat_std
    

styleEncoder = StyleEncoder(256, 128, 1)
adin = AdaptiveInstanceNorm(256)

input_tensor = torch.randn(1, 3, 224, 224)
# input_tensor = torch.randn(1, 3, 256, 256)
y = torch.tensor([0])  # 假設輸入的域索引
x = torch.randn(1, 196, 384)

# output_tensor = model(input_tensor)

style_tensor = styleEncoder(input_tensor, y)
print("styleEncoder",style_tensor.shape )
output_tensor = adin(x,style_tensor)
print(output_tensor.shape)