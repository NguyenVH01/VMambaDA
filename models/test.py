import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckMLP(nn.Module):
    def __init__(self):
        super(BottleneckMLP, self).__init__()
        
        # Giảm chiều của từng feature map với bottleneck
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1),  # Reduce channel from 96 -> 48
            nn.ReLU()
        )
        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1),  # Reduce channel from 192 -> 96
            nn.ReLU()
        )
        self.bottleneck3 = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1),  # Reduce channel from 384 -> 192
            nn.ReLU()
        )
        self.bottleneck4 = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=1),  # Reduce channel from 768 -> 384
            nn.ReLU()
        )
        
        # Lớp MLP để tổng hợp đặc trưng
        self.mlp = nn.Sequential(
            nn.Linear(48*56*56 + 96*28*28 + 192*14*14 + 384*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # Output lớp cuối cùng với kích thước mong muốn
        )
    
    def forward(self, features):
        f1, f2, f3, f4 = features
        
        # Apply bottleneck
        f1 = self.bottleneck1(f1).view(f1.size(0), -1)  # Flatten to [batch_size, -1]
        f2 = self.bottleneck2(f2).view(f2.size(0), -1)
        f3 = self.bottleneck3(f3).view(f3.size(0), -1)
        f4 = self.bottleneck4(f4).view(f4.size(0), -1)
        
        # Kết hợp các đặc trưng
        combined_features = torch.cat([f1, f2, f3, f4], dim=1)
        
        # Đưa qua MLP
        output = self.mlp(combined_features)
        
        return output

# Test với đầu vào giả lập
if __name__ == "__main__":
    # Giả sử ta có các đặc trưng sau từ mô hình Swin Transformer
    features = [
        torch.randn(1, 96, 56, 56),
        torch.randn(1, 192, 28, 28),
        torch.randn(1, 384, 14, 14),
        torch.randn(1, 768, 7, 7)
    ]
    
    model = BottleneckMLP()
    output = model(features)
    print("Output size:", output.size())  # Expected: torch.Size([1, 10])
