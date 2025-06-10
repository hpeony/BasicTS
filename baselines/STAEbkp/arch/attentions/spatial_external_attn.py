# STAEbkp/arch/attentions/spatial_external_attn.py
import torch
import torch.nn as nn

class SpatialExternalAttn(nn.Module):
    """
    空间外部注意力 (Spatial External Attention, SEA)。
    它直接在特征维度 (ModelDim) 上操作。

    输入张量的形状应为: [Batch, SeqLen, NumNodes, ModelDim]
    """
    def __init__(self, model_dim: int, s_dim: int = 64):
        """
        初始化空间外部注意力模块。

        Args:
            model_dim (int): 输入特征的维度 (C)。
            s_dim (int): 外部记忆单元的维度 (S)。
        """
        super().__init__()
        self.mk = nn.Linear(model_dim, s_dim, bias=False)
        self.mv = nn.Linear(s_dim, model_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        空间外部注意力的前向传播。
        """
        attn = self.mk(x)
        attn = self.softmax(attn)
        out = self.mv(attn)
        return out
