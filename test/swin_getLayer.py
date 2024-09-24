import torch
from torch import nn
# from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
from torchvision.models import swin_t, Swin_T_Weights

from PIL import Image
import ast
from torchvision import transforms



img_path = '/project/g/pradipta/madhu/stargan-v2/test/pig.jpg'
img = Image.open(img_path)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 读取和预处理图像
img_tensor = preprocess(img)
print(img_tensor.shape)
img_tensor = img_tensor.unsqueeze(0)  # 增加 batch 維度

# 创建 Swin Transformer 模型
model = swin_t(weights="IMAGENET1K_V1")
# 将模型设置为评估模式
model.eval()





# 进行推理
with torch.no_grad():
#     # 获取 Swin Transformer 模型的所有层
    swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        
    all_layers = list(swin.features.children())
    print("all_layers",all_layers[:6])
    # all_layers = list(model.children())
#     print(len(all_layers))
#     # # 寻找 PatchMerging 模块的索引
#     patch_merging_index = None
#     for i, layer in enumerate(all_layers):
#         print("i = layer:" ,i, layer)
#         if isinstance(layer, torch.nn.modules.container.Sequential):
#             for sub_layer in layer:
#                 if isinstance(sub_layer, torch.nn.modules.container.Sequential) and \
#                         isinstance(sub_layer[-1], torch.nn.modules.normalization.LayerNorm):
#                     patch_merging_index = i
                    # break
        # if patch_merging_index is not None:
        #     break

    # if patch_merging_index is None:
    #     raise ValueError("PatchMerging module not found in the model.")

    # 获取目标块之前的所有块
    # pre_blocks = all_layers[0][:6]
    pre_blocks = all_layers[:6]
    # print(pre_blocks)
    pre_model = torch.nn.Sequential(*pre_blocks)
    print(pre_blocks)
    # 获取目标块之前的输出
    pre_output = pre_model(img_tensor)
         # 用於調整輸出大小的轉置卷積層
    
        # 用於調整輸出大小的轉置卷積層
    adjust_output = nn.Sequential(
        nn.ConvTranspose2d(384, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14x384 -> 28x28x512
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 28x28x512 -> 28x28x512
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 28x28x512 -> 16x16x512
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((16, 16)),  # 確保輸出尺寸為 16x16
        nn.LayerNorm([512, 16, 16])  # 根據需要調整
    )
    # pre_output = pre_output.permute(0, 3, 1, 2)
    # pre_output = adjust_output(pre_output)
    # pre_output = pre_output.permute(0, 2, 3, 1)

    # 输出中间层的形状
    print("Shape of the intermediate output:", pre_output.shape)


# import torch
# from torch import nn
# from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
# from PIL import Image
# from torchvision import transforms

# # 创建 Swin Transformer 模型并加载预训练权重
# model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

# # 定義圖像路徑並讀取圖像
# img_path = '/project/g/pradipta/madhu/stargan-v2/test/pig.jpg'
# img = Image.open(img_path)

# # 定義圖像預處理步驟
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # 读取和预处理图像
# img_tensor = preprocess(img)
# img_tensor = img_tensor.unsqueeze(0)  # 增加 batch 維度
# print(img_tensor.shape)
# # 将模型设置为评估模式
# model.eval()

# # 定義一個函數來添加 Swin Transformer 的早期層


# def add_swin_transformer_layers():
#     all_layers = list(model.children())
    
#     # 尋找 PatchMerging 模塊的索引
#     patch_merging_index = None
#     for i, layer in enumerate(all_layers):
#         if isinstance(layer, nn.Sequential):
#             for sub_layer in layer:
#                 if isinstance(sub_layer, nn.Sequential) and \
#                     isinstance(sub_layer[-1], nn.LayerNorm):
#                     patch_merging_index = i
#                     break
#         if patch_merging_index is not None:
#             break

#     if patch_merging_index is None:
#         raise ValueError("PatchMerging module not found in the model.")

#     # 取得目標區塊之前的所有區塊
#     pre_blocks = all_layers[:patch_merging_index]
#     print("pre_blocks",pre_blocks)
#     # 將 pre_blocks 加入到 pre_model 中
#     # for block in pre_blocks:
#     #     pre_model.append(block)
#     pre_model = nn.Sequential(*pre_blocks)
#     # 確保 pre_model 也是評估模式
#     pre_model.eval()

#     # 進行推理
#     with torch.no_grad():
#         # 获取目标块之前的输出
#         pre_output = pre_model(img_tensor)

#     # 输出中间层的形状
#     print("Shape of the intermediate output:", pre_output.shape)

# add_swin_transformer_layers()


