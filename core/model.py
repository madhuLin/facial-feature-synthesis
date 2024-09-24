"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
from torchvision.models import swin_t, Swin_T_Weights, swin_b, Swin_B_Weights
# from torchvision.models import Swin_T_Weights

from core.wing import FAN
from core.swin_transformer_unet import SwinTransformerSys
from core.swin_transformerEncoder import SwinTransformer 

# """
# 實現 Swin Transformer，參考論文 "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"。

# 參數:
#     patch_size (List[int]): Patch 的大小。
#     embed_dim (int): Patch 嵌入的維度。
#     depths (List[int]): 每個 Swin Transformer 層的深度。
#     num_heads (List[int]): 各層的注意力頭數量。
#     window_size (List[int]): 窗口大小。
#     mlp_ratio (float): MLP 隱藏層與嵌入層的維度比率，默認為 4.0。
#     dropout (float): Dropout 率，默認為 0.0。
#     attention_dropout (float): 注意力層的 Dropout 率，默認為 0.0。
#     stochastic_depth_prob (float): 隨機深度率，默認為 0.1。
#     num_classes (int): 分類頭的類別數，默認為 1000。
#     norm_layer (nn.Module, optional): 正規化層，默認為 None。
#     block (nn.Module, optional): Swin Transformer 的 Block，默認為 None。
#     downsample_layer (nn.Module): 下採樣層（Patch 合併），默認為 PatchMerging。
# """

# 殘差塊
class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    # 建立權重
    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    # 快捷路徑
    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    # 殘差路徑
    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # 單位方差

# 自適應Instance正規化
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        # print("gamma", gamma, "beta", beta)
        # print("self.norm(x)", self.norm(x))
        return (1 + gamma) * self.norm(x) + beta

# 自適應Instance正規化殘差塊
class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    # 建立權重
    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    # 快捷路徑
    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    # 殘差路徑
    def _residual(self, x, s):
        # print("Befer _residual:", x.shape, "Befer _residual:", s.shape)
        x = self.norm1(x, s)
        # print("After _residual:", x.shape, "After _residual:", s.shape)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

# 高通濾波器
class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

# 生成器
class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))

        # 下/上採樣塊
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # 瓶頸塊
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None):
        # print("Generator", x.shape)
        x = self.from_rgb(x)
        # print("Generator from_rgb:", x.shape)
        cache = {}
        for block in self.encode:
            # print(f"Before block: {block.__class__.__name__}, shape: {x.shape}")
            # print(f"Before block: {block}")
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
                
            x = block(x)
        # print(f"After block: {block.__class__.__name__}, shape: {x.shape}")
        for block in self.decode:
            x = block(x, s)
            # print("dercoder:", x.shape)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])
        # print("return decode",x.shape)
        x = self.to_rgb(x)
        # print("return decode to_rgb",x.shape)     
        return x
    



class AdjustOutput(nn.Module):
    def __init__(self, input_dim, output_dim, scale):
        super(AdjustOutput, self).__init__()
        # self.conv_transpose = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1,output_padding=1)
        # self.bn = nn.BatchNorm2d(output_dim)
        # self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)
        # 用於調整輸出大小的轉置卷積層
        #swin_T
        if scale == "t":
            self.adjust_output = nn.Sequential(
                nn.ConvTranspose2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14x384 -> 28x28x512
                nn.ReLU(),
                nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1),  # 28x28x512 -> 28x28x512
                nn.ReLU(),
                nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1),  # 28x28x512 -> 16x16x512
                nn.ReLU(),
                nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1),  # 28x28x512 -> 16x16x512
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((8, 8)),  # 確保輸出尺寸為 16x16
                nn.LayerNorm([output_dim, 8, 8])  # 根據需要調整
            )
        elif scale == "b":
            self.adjust_output = nn.Sequential(
                nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1),  # 28x28x512 -> 28x28x512
                nn.ReLU(),
                nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1),  # 28x28x512 -> 16x16x512
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((8, 8)),  # 確保輸出尺寸為 16x16
                nn.LayerNorm([output_dim, 8, 8])  # 根據需要調整
            )
    
    def forward(self, x):
        x = self.adjust_output(x)
        return x

#swin_b
class GeneratorSwin_b(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))
        
        
        # 使用 Swin Transformer 的預訓練模型
        # swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        swin = swin_b(weights='IMAGENET1K_V1')
        
        all_layers = list(swin.features.children())
        # print(all_layers[:6])
        # 提取前 6 層作為編碼器
        self.encoderSwin = nn.Sequential(*all_layers[:6])
        
        # 調整輸出層
        self.adjust_output = AdjustOutput(384, 512, "b")
        

        # 下/上採樣塊
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # 瓶頸塊
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None):
        # print("Before x:", x.shape)
        # x = self.from_rgb(x)
        cache = {}
        # print("self.encoderSwin", self.encoderSwin)
        x = self.encoderSwin(x)
        # print("self.encoderSwin x", x.shape)
        x = x.permute(0, 3, 1, 2)
        x = self.adjust_output(x)
        # x = x.permute(0, 2, 3, 1)
        # print("adjust_output x", x.shape)
        
        for block in self.encode:
            
            # print(f"Before block: {block.__class__.__name__}, shape: {x.shape}")
            # print(f"Before block: {block}")
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
            # print(f"After block: {block.__class__.__name__}, shape: {x.shape}")
        # print("encode Resnet", x.shape)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                if(x.size(2) in cache.keys()):
                    x = x + self.hpf(mask * cache[x.size(2)])
        # print("return decode",x.shape)
        x = self.to_rgb(x)
        # print("return decode to_rgb",x.shape)   
        return x


#swin_t
class GeneratorSwin_t(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))
        
        # 使用 Swin Transformer 的預訓練模型
        swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        # swin = swin_b(weights='IMAGENET1K_V1')
        
        all_layers = list(swin.features.children())
        # print(all_layers[:6])
        # 提取前 6 層作為編碼器
        self.encoderSwin = nn.Sequential(*all_layers[:6])
        
        # 調整輸出層
        self.adjust_output = AdjustOutput(384, 512, "t")

        # 下/上採樣塊
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # 瓶頸塊
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None):
        # print("GeneratorSwin x:", x.shape)
        # x = self.from_rgb(x)
        cache = {} # skip conention
        # print("self.encoderSwin", self.encoderSwin)
        x = self.encoderSwin(x)
        # print("self.encoderSwin x", x.shape)
        x = x.permute(0, 3, 1, 2)
        x = self.adjust_output(x)
        # x = x.permute(0, 2, 3, 1)
        # print("adjust_output x", x.shape)
        
        for block in self.encode:
            
            # print(f"Before block: {block.__class__.__name__}, shape: {x.shape}")
            # print(f"Before block: {block}")
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
            # print(f"After block: {block.__class__.__name__}, shape: {x.shape}")
        # print("encode Resnet", x.shape)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                if(x.size(2) in cache.keys()):
                    x = x + self.hpf(mask * cache[x.size(2)])
        # print("return decode",x.shape)
        x = self.to_rgb(x)
        # print("return decode to_rgb",x.shape)   
        return x
    
    
class GeneratorSwinUnet(nn.Module):
    def __init__(self, img_size=224, style_dim=64, num_classes=3, zero_head=False, vis=False, w_hpf=1):
        super(GeneratorSwinUnet, self).__init__()
        
        self.num_classes = num_classes  # 類別數目
        self.zero_head = zero_head  # 是否使用zero head初始化
        self.style_dim = style_dim  # 風格維度
        
        # 初始化 Swin Transformer U-Net 結構
        self.swin_unet = SwinTransformerSys(
            img_size=img_size,  # 輸入圖像大小
            patch_size=4,  # 每個 patch 的大小
            in_chans=3,  # 輸入通道數
            num_classes=num_classes,  # 分類類別數
            embed_dim=96,  # 嵌入維度
            style_dim=style_dim,  # 風格維度
            depths=[2, 2, 6, 2],  # 每個階段的層數
            num_heads=[3, 6, 12, 24],  # 每個階段的頭數
            window_size=7,  # 注意力窗口大小
            mlp_ratio=4.,  # MLP的寬度倍率
            qkv_bias=True,  # 是否添加 QKV 的偏差
            qk_scale=None,  # QK 縮放
            drop_rate=0.0,  # Dropout 比率
            drop_path_rate=0.1,  # DropPath 比率
            ape=False,  # 絕對位置編碼
            patch_norm=True,  # 是否使用 Patch 正規化
            use_checkpoint=False,  # 是否使用檢查點減少記憶體使用
            w_hpf=w_hpf,  # 高通濾波器權重
            noise=True  # 是否添加噪聲
        )
        
    def forward(self, x, s, masks=None):
        # 將輸入圖像 x 調整到指定大小 (224, 224)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 如果輸入圖像是單通道，則將其複製三次以形成三通道圖像
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # 使用 Swin Transformer U-Net 生成 logits
        logits = self.swin_unet(x, s)
        
        # 將 logits 調整到指定大小 (256, 256)
        logits = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print(f"pretrained_path: {pretrained_path}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            
            if "model" not in pretrained_dict:
                print("---開始載入預訓練模型---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print(f"刪除鍵：{k}")
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                return
            
            pretrained_dict = pretrained_dict['model']
            print("---開始載入 swin encoder 的預訓練模型---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_upload) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print(f"刪除：{k};預訓練形狀：{v.shape};模型形狀：{model_dict[k].shape}")
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            print("無預訓練模型")

#將SWIN transfomrer style 轉為適合 Generator大小
class ConvTransform(nn.Module):
    def __init__(self):
        super(ConvTransform, self).__init__()
        # 定義一系列的卷積層和激活函數，用於對輸入張量進行重塑
        self.reshape = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=1, padding=0),
            nn.GELU()
        )
        # 定義第二組卷積層和反卷積層，並包含上採樣層
        self.reshape1 = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose2d(384, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.GELU()
        )
        # 定義第三組卷積層和反卷積層，並包含上採樣層
        self.reshape2 = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose2d(192, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU()
        )
        # 定義第四組卷積層和反卷積層，並包含上採樣層
        self.reshape3 = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose2d(96, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU()
        )
        # 定義一系列全連接層，用於將輸入張量轉換為指定大小
        self.fc = nn.Linear(49, 64)
        self.fc1 = nn.Linear(196, 256)
        self.fc2 = nn.Linear(784, 1024)
        self.fc3 = nn.Linear(3136, 4096)

    def forward(self, x, i):
        # 將輸入張量進行維度變換，適應卷積層輸入格式
        x = x.permute(0, 2, 1)
        if i == 0:
            # 將輸入張量通過全連接層並重塑為 (batch_size, 512, 8, 8)
            x = self.fc(x).view(x.size(0), x.size(1), 8, 8)
            # 通過第一組卷積層進行處理
            x = self.reshape(x)
        elif i == 1:
            # 將輸入張量通過全連接層並重塑為 (batch_size, 512, 16, 16)
            x = self.fc1(x).view(x.size(0), x.size(1), 16, 16)
            # 通過第二組卷積層和反卷積層進行處理
            x = self.reshape1(x)
        elif i == 2:
            # 將輸入張量通過全連接層並重塑為 (batch_size, 256, 32, 32)
            x = self.fc2(x).view(x.size(0), x.size(1), 32, 32)
            # 通過第三組卷積層和反卷積層進行處理
            x = self.reshape2(x)
        elif i == 3:
            # 將輸入張量通過全連接層並重塑為 (batch_size, 128, 64, 64)
            x = self.fc3(x).view(x.size(0), x.size(1), 64, 64)
            # 通過第四組卷積層和反卷積層進行處理
            x = self.reshape3(x)
        return x

#swin_encoder
class GeneratorSwinEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))
        
        
        # 使用 Swin Transformer 的預訓練模型
        self.encoderSwin = SwinTransformer(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=3,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False,
                                )
        
        
        # 調整輸出層
        self.convTransform = ConvTransform()
        

        # 下/上採樣塊
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # 瓶頸塊
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    # def forward(self, x, s, masks=None):
    #     # print("GeneratorSwin x:", x.shape)
    #     # x = self.from_rgb(x)
    #     x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    #     x, x_downsample = self.encoderSwin(x)

    #     cache = {}
    #     # for block in self.encode:
    #         # if (masks is not None) and (x.size(2) in [32, 64, 128]):
    #         #     cache[x.size(2)] = x
    #         # x = block(x)
    #         # print(f"After block: {block.__class__.__name__}, shape: {x.shape}")
    #     # print("encode Resnet", x.shape)
    #     x = self.convTransform(x,0) # b,49,768  --> b,512,8,8
    #     # print("encoder reshape x.shape",x.shape)
    #     value = 16
    #     for i in range(1,4):
    #         value *= 2
    #         # print("x_downsample{}:".format(i),x_downsample[3-i].shape)
    #         cache[value] = self.convTransform(x_downsample[3-i],i)
    #         # print("value_{}:".format(value), cache[value])
    #     for block in self.decode:
    #         x = block(x, s)
    #         # print("decoder x:",x.shape)
    #         if (masks is not None) and (x.size(2) in [32, 64, 128]):
    #             mask = masks[0] if x.size(2) in [32] else masks[1]
    #             mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
    #             if(x.size(2) in cache.keys()):
    #                 x = x + self.hpf(mask * cache[x.size(2)])
    #     # print("return decode",x.shape)
    #     x = self.to_rgb(x)
    #     # print("return decode to_rgb",x.shape)   
    #     return x
    
    def forward(self, x, s, masks=None):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x, x_downsample = self.encoderSwin(x)

        cache = {}
        x = self.convTransform(x,0) # b,49,768  --> b,512,8,8
        value = 16
        for i in range(1,4):
            value *= 2
            cache[value] = self.convTransform(x_downsample[3-i],i)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                if(x.size(2) in cache.keys()):
                    x = x + self.hpf(mask * cache[x.size(2)])
        x = self.to_rgb(x)  
        return x
   


# 映射網絡
class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512,

 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s

# 風格編碼器
class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        # print("style dim:", s.shape)
        return s
    
# 風格編碼器
class StyleSwinEncoder(nn.Module):
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
        # self.fc2 = nn.Linear(9408, 2048)
        # self.fc3 = nn.Linear(2048, style_dim)
        self.fc2 = nn.Linear(9408, style_dim)

        self.conv = nn.Conv1d(384, 192, 3,1,1)
        self.act_layer = nn.ReLU()
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(style_dim, style_dim)]
        # print(self.swin_transformer)

    def forward(self, x, y):
        # print("x:", x.shape, "y:", y.shape)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        h = self.swin_transformer(x)
        # print("h:", h.shape)
        h = h.permute(0,2,1)
        h = self.fc1(h)
        h = self.act_layer(h)
        # print("fc1",h.shape)
        h = self.conv(h)
        h = self.act_layer(h)
        # print("conv",h.shape)
        
        h = h.view(h.size(0), -1)
        h = self.fc2(h)
        # h = self.act_layer(h)
        # h = self.fc3(h)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.arange(y.size(0)).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


# 判別器
class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out

# 建立模型
def build_model(args):
    if args.generator_backbone == "ResNet" or args.generator_backbone == "SwinStyle":
        generator = nn.DataParallel(Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf))
    elif args.generator_backbone == "SwinUnet":
        # generator = nn.DataParallel(GeneratorSwin(args.img_size, args.style_dim, w_hpf=args.w_hpf))
        generator = nn.DataParallel(GeneratorSwinUnet(img_size=args.img_size, style_dim= args.style_dim, w_hpf=args.w_hpf))
    elif args.generator_backbone == "SwinEncoder":
        generator = generator = nn.DataParallel(GeneratorSwinEncoder(args.img_size, args.style_dim, w_hpf=args.w_hpf))
    elif args.generator_backbone == "SwinT":
        generator = generator = nn.DataParallel(GeneratorSwin_t(args.img_size, args.style_dim, w_hpf=args.w_hpf))

        
        
    mapping_network = nn.DataParallel(MappingNetwork(args.latent_dim, args.style_dim, args.num_domains))
    
    if args.generator_backbone == "SwinUnet" or args.generator_backbone == "SwinStyle":
        # style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains))
        style_encoder = nn.DataParallel(StyleSwinEncoder(args.img_size, args.style_dim, args.num_domains))
    else:
        style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains))
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains))
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)
    if args.w_hpf > 0:
        fan = nn.DataParallel(FAN(fname_pretrained=args.wing_path).eval())
        fan.get_heatmap = fan.module.get_heatmap
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema