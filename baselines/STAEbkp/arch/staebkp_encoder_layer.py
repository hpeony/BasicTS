# STAEbkp/arch/staebkp_encoder_layer.py
import torch
import torch.nn as nn
# 已修复: 注意，这里我们不再需要从 attentions 包导入，因为模块在 encoder layer 中被直接实例化。
from .attentions.temporal_external_attn import TemporalExternalAttn
from .attentions.spatial_external_attn import SpatialExternalAttn

class STAEbkpEncoderLayer(nn.Module):
    """
    STAEbkp 模型的编码器层。
    这个层集成了时间和空间外部注意力，用于替代原版Transformer的自注意力。
    它包含残差连接、层归一化和前馈网络，构成了模型的核心处理单元。
    """

    def __init__(self, model_dim: int, seq_len: int, num_nodes: int, s_dim: int, feed_forward_dim: int, dropout: float):
        """
        初始化 STAEbkp 编码器层。

        Args:
            model_dim (int): 模型特征维度。
            seq_len (int): 时间序列长度。
            num_nodes (int): 空间节点数量。
            s_dim (int): 外部注意力记忆单元的维度。
            feed_forward_dim (int): 前馈网络的隐藏层维度。
            dropout (float): Dropout 比率。
        """
        super().__init__()

        # 已修改：实例化时间和空间外部注意力
        # 已修复: 正确实例化TEA和SEA，传入正确的维度参数
        self.temporal_attn = TemporalExternalAttn(seq_len, s_dim)  # TEA的d_model是序列长度
        self.spatial_attn = SpatialExternalAttn(model_dim, s_dim)  # SEA的d_model是特征维度

        self.feed_forward_t = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim), nn.ReLU(inplace=True), nn.Linear(feed_forward_dim, model_dim)
        )
        self.feed_forward_s = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim), nn.ReLU(inplace=True), nn.Linear(feed_forward_dim, model_dim)
        )

        # 层归一化
        self.ln1_t = nn.LayerNorm(model_dim)
        self.ln2_t = nn.LayerNorm(model_dim)
        self.ln1_s = nn.LayerNorm(model_dim)
        self.ln2_s = nn.LayerNorm(model_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_type: str) -> torch.Tensor:
        """
        编码器层的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 [B, L, N, C] 或其转置。
            attn_type (str): 注意力类型，'temporal' 或 'spatial'。

        Returns:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        # 保存残差连接的输入
        batch_size, seq_len, num_nodes, model_dim = x.shape
        residual = x

        # --- 第一部分：外部注意力 ---
        if attn_type == 'temporal':
            # --- 时间注意力处理流程 ---
            # 1. 重塑和转置，为 TEA 准备输入
            # x: [B, L, N, C] -> x_tea_in: [B*N, C, L]
            x_tea_in = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, model_dim, seq_len)

            # 2. 通过 TEA 模块
            # out_tea: [B*N, C, L]
            out_tea = self.temporal_attn(x_tea_in)

            # 3. 恢复原始形状
            # out_tea: [B*N, C, L] -> out: [B, N, C, L] -> [B, L, N, C]
            out = out_tea.reshape(batch_size, num_nodes, model_dim, seq_len).permute(0, 3, 1, 2)

            # 4. 残差、Dropout、LayerNorm
            out = self.dropout(out)
            x = self.ln1_t(residual + out)

            # 5. 前馈网络
            residual = x
            out = self.feed_forward_t(x)
            out = self.dropout(out)
            x = self.ln2_t(residual + out)
        elif attn_type == 'spatial':
            # --- 空间注意力处理流程 ---
            # 1. SEA可以直接处理 [B, L, N, C] 形状的输入
            out = self.spatial_attn(x)

            # 2. 残差、Dropout、LayerNorm
            out = self.dropout(out)
            x = self.ln1_s(residual + out)

            # 3. 前馈网络
            residual = x
            out = self.feed_forward_s(x)
            out = self.dropout(out)
            x = self.ln2_s(residual + out)
        else:
            raise ValueError("Unsupported attention type. Choose 'temporal' or 'spatial'.")

        return x
