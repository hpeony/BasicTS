这是在staext18上将进入模型的输入X 进行升维扩展，降维精华

```python
class FeatureMLP(nn.Module):
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

    
# 将输入X展平，使用数据与时间对齐的模式嵌入
self.input_proj = FeatureMLP(input_dim*in_steps, input_embedding_dim, 4)


# 对于转向数据时间展平并对齐的思想，还需要进行额外的嵌入维度对齐：
# 数据时间：[B, N, T, C] -> [B, N, T*C] -> [B, T*C, N] -> [B, T*C, N, 1] -> 
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

```
