# STAEbkp/arch/staebkp.py
import torch.nn as nn
import torch
from .staebkp_encoder_layer import STAEbkpEncoderLayer


class STAEbkp(nn.Module):
    """
    STAEbkp 模型主架构。
    该模型融合了 STAEformer 的丰富输入嵌入和 MEAformer 的高效外部注意力机制。
    它保留了 STAEformer 的时空自适应嵌入（STAE），但将时间/空间自注意力
    替换为了线性的时间/空间外部注意力（TEA/SEA）。
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
            # 已修改：新增 s_dim 用于外部注意力
            s_dim=64,
            feed_forward_dim=256,
            num_heads=4,  # num_heads 保留，但已不被外部注意力使用，仅为兼容旧配置
            num_layers=3,
            dropout=0.1,
            use_mixed_proj=True,  # use_mixed_proj 保留，但推荐的MLP头已不使用它
    ):
        super().__init__()
        # 已修改-begin: 将所有重要的初始化参数保存为类的属性，解决 AttributeError
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
        self.num_layers = num_layers
        # 已修改-end

        # --- 嵌入层 (保留自 STAEformer) ---
        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)  # 已修改：使用 self.input_dim
        if self.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        if self.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, self.num_nodes, self.adaptive_embedding_dim))  # 已修改：使用 self 属性
            )

        self.encoder_layers = nn.ModuleList(
            [
                STAEbkpEncoderLayer(self.model_dim, self.in_steps, self.num_nodes, s_dim, feed_forward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # 已修改：仿照MEAformer，使用MLP头作为输出层，增强非线性能力并直接预测
        # 我们将模型展平后的特征维度作为MLP的输入
        mlp_input_dim = self.in_steps * self.model_dim
        self.output_proj = nn.Sequential(
            nn.Linear(mlp_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.out_steps * self.output_dim)
        )

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        STAEbkp 模型的前向传播函数。

        Args:
            history_data (torch.Tensor): 符合BasicTS规范的历史数据，形状为 [B, L, N, C_in]。
            其他参数为BasicTS框架传入，此处不直接使用。

        Returns:
            torch.Tensor: 模型的预测结果，形状为 [B, L_out, N, C_out]。
        """
        # x 形状: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        x = history_data
        batch_size = x.shape[0]

        # --- 1. 输入嵌入 ---
        features_to_cat = []
        # 已修改-begin: 确保所有if判断都使用 self.xxx 属性
        if self.input_embedding_dim > 0:
            features_to_cat.append(self.input_proj(x[..., :self.input_dim]))
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding((x[..., 1] * self.steps_per_day).long())
            features_to_cat.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding((x[..., 2] * 7).long())
            features_to_cat.append(dow_emb)
        if self.adaptive_embedding_dim > 0:  # 已修改：修正了属性名称
            adp_emb = self.adaptive_embedding.expand(size=(batch_size, *self.adaptive_embedding.shape))
            features_to_cat.append(adp_emb)
        # 已修改-end

        x = torch.cat(features_to_cat, dim=-1)  # Shape: [B, L, N, model_dim]

        # --- 2. 时空编码器 ---
        for layer in self.encoder_layers:
            x = layer(x, attn_type='temporal')
            x = layer(x, attn_type='spatial')

        # --- 3. 输出层 ---
        out = x.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, -1)
        out = self.output_proj(out)
        out = out.view(batch_size, self.num_nodes, self.out_steps, self.output_dim).permute(0, 2, 1, 3)

        return out
