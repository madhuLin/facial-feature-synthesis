import torch
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.enc_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.enc_3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.enc_4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.enc_5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

# 使用範例
encoder = Encoder()
input_image = torch.randn(1, 3, 256, 256)  # 假設輸入是一個隨機圖像張量
features = encoder.encode_with_intermediate(input_image)

# features 將包含五層的編碼結果
for i, feature in enumerate(features):
    print(f"Layer {i + 1} output shape: {feature.shape}")
