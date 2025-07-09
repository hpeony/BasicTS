import torch
from torch import nn

class MultiLayerPerceptron(nn.Module):
    """
    模块功能: 带残差连接的多层感知机(MLP)。
    设计方法: 该模块使用经典的MLP结构作为非线性特征提取器。
              为了构建更深的网络并缓解梯度消失问题，引入了来自ResNet的残差连接思想。
    """

    def __init__(self, input_dim, hidden_dim) -> None:
        """
        功能: 初始化MLP层。
        参数:
            input_dim (int): 输入特征的维度。
            hidden_dim (int): 隐藏层的维度 (也是输出维度，因为有残差连接)。
        """
        super().__init__()
        # 第一个全连接层 (通过1x1卷积实现，高效处理[B, C, N, 1]形状的数据)
        self.fc1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        # 第二个全连接层
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        # 激活函数
        self.act = nn.ReLU()
        # Dropout层，防止过拟合
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        功能: MLP的前向传播。
        参数:
            input_data (torch.Tensor): 输入张量，形状 [B, D, N, 1]。
        返回:
            torch.Tensor: 经过MLP和残差连接后的输出张量，形状与输入相同。
        """
        # 保存输入用于残差连接
        residual = input_data
        # 通过第一个FC层、激活函数、Dropout层
        hidden = self.drop(self.act(self.fc1(input_data)))
        # 通过第二个FC层
        hidden = self.fc2(hidden)
        # 应用残差连接
        hidden = hidden + residual
        return hidden
