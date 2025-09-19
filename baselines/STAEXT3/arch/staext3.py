import torch.nn as nn
import torch
# ==================== 导入重构后的注意力层 ====================
from .layers import TemporalExternalAttnLayer, SpatialExternalAttnLayer, SelfAttentionLayer
# ==========================================================


class FeatureProj(nn.Module):
    def __init__(self, input_dim, output_dim, expansion_factor=4):
        """
        基于 (1x1) 卷积的特征 MLP 模块 (先升维再降维)
        :param input_dim: 输入特征维度
        :param output_dim: 输出特征维度
        :param expansion_factor: 隐藏层维度相对于输入维度的扩展因子
        """
        super().__init__()
        hidden_dim = int(input_dim * expansion_factor)
        self.mlp = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        )

    def forward(self, x):
        # x shape: (B, D, T, N)
        return self.mlp(x)

class STAEXT3(nn.Module):
    """
    Task: Spatial-Temporal Forecasting
    重构说明：
    此版本的模型实现了跨层共享的外部注意力记忆单元 (Mk, Mv)。
    - 在模型初始化时，会创建一对用于所有时间注意力层的共享 Mk/Mv，
      以及另一对用于所有空间注意力层的共享 Mk/Mv。
    - 这些共享的记忆单元作为参数传递给每一层的注意力模块，从而避免了
      “层间信息孤岛”，允许模型在所有层级上学习统一的全局模式。
    """

    def __init__(
            self,
            num_nodes,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=80,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            use_mixed_proj=True,
            # ==================== 外部注意力相关参数 ====================
            interleave_layers=False,
            use_temporal_external_attn=False,
            use_spatial_external_attn=False,
            # external_memory_S=64,
            # ==========================================================
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + spatial_embedding_dim
                + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.interleave_layers = interleave_layers
        self.use_temporal_external_attn = use_temporal_external_attn
        self.use_spatial_external_attn = use_spatial_external_attn

        self.input_proj = FeatureProj(
            input_dim * in_steps,
            input_embedding_dim,
            2
        )

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        # ==================== 共享外部记忆单元的初始化 ====================
        # 在这里静态初始化Mk和Mv，确保它们在所有层之间共享。
        # PyTorch会自动将这些nn.Module注册为模型的可训练参数。

        # --- 时间注意力共享记忆 ---
        if self.use_temporal_external_attn:
            # d_model是时间步长 T, S是记忆单元大小
            self.mk_temporal = nn.Linear(self.model_dim, self.in_steps, bias=False)
            self.mv_temporal = nn.Linear(self.in_steps, self.model_dim, bias=False)

        # --- 空间注意力共享记忆 ---
        if self.use_spatial_external_attn:
            # d_model是节点数 N, S是记忆单元大小
            self.mk_spatial = nn.Linear(self.model_dim, self.num_nodes, bias=False)
            self.mv_spatial = nn.Linear(self.num_nodes, self.model_dim, bias=False)

        # ==================== 可配置的注意力层初始化 ====================
        # 根据配置选择时间注意力层
        if self.use_temporal_external_attn:
            self.attn_layers_t = nn.ModuleList(
                [
                    TemporalExternalAttnLayer(
                        model_dim=self.model_dim,
                        in_steps=self.in_steps,
                        feed_forward_dim=feed_forward_dim,
                        dropout=dropout,
                        # 将共享的记忆单元传递给每一层
                        shared_mk=self.mk_temporal,
                        shared_mv=self.mv_temporal
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            # 保持原有的自注意力实现
            self.attn_layers_t = nn.ModuleList(
                [
                    SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                    for _ in range(num_layers)
                ]
            )

        # 根据配置选择空间注意力层
        if self.use_spatial_external_attn:
            self.attn_layers_s = nn.ModuleList(
                [
                    SpatialExternalAttnLayer(
                        model_dim=self.model_dim,
                        num_nodes=self.num_nodes,
                        feed_forward_dim=feed_forward_dim,
                        dropout=dropout,
                        # 将共享的记忆单元传递给每一层
                        shared_mk=self.mk_spatial,
                        shared_mv=self.mv_spatial
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            # 保持原有的自注意力实现
            self.attn_layers_s = nn.ModuleList(
                [
                    SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                    for _ in range(num_layers)
                ]
            )
        # =================================================================

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        x = history_data
        batch_size, seq_len, num_nodes, _ = x.shape

        # 1. 准备数据
        #    a. 数值型时序数据，用于展平和生成序列嵌入
        input_data = x[..., :self.input_dim]

        #    b. 时间特征，用于生成时间嵌入
        if self.tod_embedding_dim > 0:
            tod_ids = (x[..., 1] * self.steps_per_day).long()
            tod_emb = self.tod_embedding(tod_ids)  # Shape: [B, T, N, D_tod]
        else:
            tod_emb = None

        if self.dow_embedding_dim > 0:
            dow_ids = (x[..., 2] * 7).long()
            dow_emb = self.dow_embedding(dow_ids)  # Shape: [B, T, N, D_dow]
        else:
            dow_emb = None

        # 2. 构造正确的输入张量给 self.input_proj
        #    [B, T, N, C] -> [B, N, T, C]
        input_data_transposed = input_data.transpose(1, 2).contiguous()
        #    [B, N, T, C] -> [B, N, T*C] -> [B, T*C, N] -> [B, T*C, N, 1]
        input_data_flattened = input_data_transposed.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)

        #    现在 input_data_flattened 的形状是 [B, L*C, N, 1]，例如 [32, 36, 170, 1]。
        #    它的通道数(dim=1)是36，与 self.input_proj 期望的输入维度完全匹配。
        #    输出 time_series_emb 的形状是 [B, D_emb, N, 1]
        time_series_emb = self.input_proj(input_data_flattened)

        # 3. 准备并对齐其他嵌入以进行拼接
        #    将时序嵌入的形状从 [B, D_emb, N, 1] 调整并扩展到 [B, T, N, D_emb]
        time_series_emb_expanded = time_series_emb.squeeze(-1).transpose(1, 2).unsqueeze(1).expand(-1, seq_len, -1, -1)

        features = [time_series_emb_expanded]

        if tod_emb is not None:
            features.append(tod_emb)
        if dow_emb is not None:
            features.append(dow_emb)

        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(batch_size, seq_len, *self.node_emb.shape)
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(batch_size, *self.adaptive_embedding.shape)
            features.append(adp_emb)

        # 4. 在特征维度上拼接所有嵌入
        x = torch.cat(features, dim=-1)

        # ==================== 可配置的注意力层调用 ====================
        if self.interleave_layers:
            # 交替运行模式
            for i in range(self.num_layers):
                x = self.attn_layers_t[i](x)
                x = self.attn_layers_s[i](x)
        else:
            # 原有的分离运行模式
            # 时间注意力
            for attn in self.attn_layers_t:
                x = attn(x)
            # 空间注意力
            for attn in self.attn_layers_s:
                x = attn(x)
        # =============================================================

        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out
