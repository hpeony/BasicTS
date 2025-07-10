import torch
from torch import nn
import torch.nn.functional as F


class GatedMLP(nn.Module):
    """
    模块功能: 使用门控线性单元(GLU)变体SwiGLU的MLP。
    设计方法: 该结构用SwiGLU替换了标准的ReLU激活层。输入首先被投影到两倍的隐藏维度，
              然后分裂成两部分。一部分通过Swish激活函数，另一部分作为门，
              两者相乘后形成一个被动态门控的表示。这增强了模型的表达能力。
    文献来源: "GLU Variants Improve Transformer" (Noam Shazeer, 2020)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        # 第一个FC层，输出维度是隐藏维度的两倍，用于门控
        self.fc1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim * 2, kernel_size=(1, 1), bias=True)
        # 第二个FC层，将门控后的结果投影回输出维度
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        # 激活函数，使用SiLU (PyTorch中的Swish实现)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p=dropout)

        # 如果输入和输出维度不同，需要一个线性层来匹配残差连接的维度
        self.adjust_dim = nn.Conv2d(input_dim, hidden_dim, (1, 1)) if input_dim != hidden_dim else nn.Identity()

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        参数:
            input_data (torch.Tensor): 输入张量，形状 [B, D_in, N, 1]。
        返回:
            torch.Tensor: 输出张量，形状 [B, D_out, N, 1]。
        """
        # 调整残差连接的维度
        residual = self.adjust_dim(input_data)

        # 门控机制
        hidden = self.fc1(input_data)
        # 沿通道维度 (dim=1) 分裂成两半
        gate, value = hidden.chunk(2, dim=1)
        # 应用SwiGLU
        gated_value = self.act(gate) * value

        # 应用Dropout和第二个FC层
        output = self.fc2(self.drop(gated_value))

        # 应用残差连接
        output = output + residual
        return output
