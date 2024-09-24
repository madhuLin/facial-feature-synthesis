import torch
import torch.nn as nn


class ConvolutionalTransformation(nn.Module):
    def __init__(self):
        super(ConvolutionalTransformation, self).__init__()
        self.reshape = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=1, padding=0),
            nn.GELU()
        )
        self.fc1 = nn.Linear(49, 64)
        
    def forward(self, x):
        # 將輸入特徵通道維度轉換為第二個維度，以符合轉置卷積層的預期輸入格式
        x = x.permute(0, 2, 1)
        x = self.fc1(x).view(x.size(0),-1,8,8)
        x = self.reshape(x)
        return x

# 初始化模型
model = ConvolutionalTransformation()


# 假設輸入是形狀為 [8, 49, 768] 的張量 --> [8, 512, 8, 8]
input_tensor = torch.randn(8, 49, 768)


output_tensor = model(input_tensor)

print(output_tensor.shape)
