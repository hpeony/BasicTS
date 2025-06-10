# STAEbkp/arch/attentions/temporal_external_attn.py
import torch
import torch.nn as nn


class TemporalExternalAttn(nn.Module):
    """
    时间外部注意力 (Temporal External Attention, TEA)。
    该模块以线性复杂度高效捕捉时间依赖。
    设计上，它作用于序列长度维度，因此输入应为 [*, SeqLen, FeatureDim]。
    """

    def __init__(self, d_model: int, s_dim: int = 64):
        """
        初始化时间外部注意力。
        这里的 d_model 实际上是序列长度 SeqLen。

        Args:
            d_model (int): 序列的长度 (L)。
            s_dim (int): 外部记忆单元的维度 (S)。
        """
        super().__init__()
        self.mk = nn.Linear(d_model, s_dim, bias=False)
        self.mv = nn.Linear(s_dim, d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        TEA 的前向传播。

        Args:
            queries (torch.Tensor): 输入张量, 形状应为 [Batch_Size * Num_Nodes, Feature_Dim, Seq_Len]

        Returns:
            torch.Tensor: 输出张量, 形状与输入相同 [Batch_Size * Num_Nodes, Feature_Dim, Seq_Len]
        """
        attn = self.mk(queries)
        attn = self.softmax(attn)
        out = self.mv(attn)
        return out
