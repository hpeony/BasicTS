# File: basicts/arch/optimized_regression.py

from torch import nn
import torch.nn.functional as F


class OptimizedRegressionHead(nn.Module):
    # 初始化函数增加了 dropout 参数，以便可以从外部配置
    def __init__(self, hidden_dim, output_len, num_nodes, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_len = output_len
        self.hidden_dim = hidden_dim

        self.conv_kernel_size = (1, 1)

        # LayerNorm for the feature dimension (hidden_dim // 2)
        self.layer_norm_temp = nn.LayerNorm(hidden_dim // 2)
        self.layer_norm_spatial = nn.LayerNorm(hidden_dim // 2)

        # --- Define branches with explicit in/out channels for clarity ---

        # Temporal Branch
        self.conv1d_temp_in = nn.Conv2d(hidden_dim, hidden_dim // 2, self.conv_kernel_size)
        self.relu_temp = nn.ReLU()
        self.dropout_temp = nn.Dropout(dropout)
        # Explicitly define the final convolution layer with correct channels
        self.final_conv_temp = nn.Conv2d(hidden_dim // 2, output_len, self.conv_kernel_size)
        self.temporal_branch = nn.Sequential(
            self.conv1d_temp_in,
            self.relu_temp,
            self.dropout_temp,
            self.final_conv_temp
        )

        # Spatial Branch
        self.conv1d_spatial_in = nn.Conv2d(hidden_dim, hidden_dim // 2, self.conv_kernel_size)
        self.relu_spatial = nn.ReLU()
        self.dropout_spatial = nn.Dropout(dropout)
        # Explicitly define the final convolution layer with correct channels
        self.final_conv_spatial = nn.Conv2d(hidden_dim // 2, output_len, self.conv_kernel_size)
        self.spatial_branch = nn.Sequential(
            self.conv1d_spatial_in,
            self.relu_spatial,
            self.dropout_spatial,
            self.final_conv_spatial
        )

        # Fusion Attention branch
        self.fusion_attention = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, self.conv_kernel_size),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, 2, self.conv_kernel_size),  # Produces weights for 2 branches
            nn.Softmax(dim=1)  # Softmax over channels (branch dimension)
        )

    def forward(self, hidden):
        # hidden 形状: (B, hidden_dim, N, 1)
        B, C, N, W = hidden.shape  # C is hidden_dim

        # --- Temporal Branch Processing ---
        x_temp = self.conv1d_temp_in(hidden)  # Shape: (B, C//2, N, 1)

        # Apply LayerNorm: permute feature dim to last
        x_temp_for_ln = x_temp.permute(0, 2, 3, 1) # (B, N, 1, C//2)
        x_temp_ln = self.layer_norm_temp(x_temp_for_ln)
        x_temp = x_temp_ln.permute(0, 3, 1, 2) # Back to (B, C//2, N, 1)

        # Pass through ReLU, Dropout, and the final Conv2d
        x_temp = self.relu_temp(x_temp)
        x_temp = self.dropout_temp(x_temp)
        temporal_pred = self.final_conv_temp(x_temp)  # Final Conv2d: (B, output_len, N, 1)

        # --- Spatial Branch Processing ---
        x_spatial = self.conv1d_spatial_in(hidden)  # Shape: (B, C//2, N, 1)

        # Apply LayerNorm (same logic)
        x_spatial_for_ln = x_spatial.permute(0, 2, 3, 1) # (B, N, 1, C//2)
        x_spatial_ln = self.layer_norm_spatial(x_spatial_for_ln)
        x_spatial = x_spatial_ln.permute(0, 3, 1, 2) # Back to (B, C//2, N, 1)

        # Pass through ReLU, Dropout, and the final Conv2d
        x_spatial = self.relu_spatial(x_spatial)
        x_spatial = self.dropout_spatial(x_spatial)
        spatial_pred = self.final_conv_spatial(x_spatial)  # Final Conv2d: (B, output_len, N, 1)

        # --- Fusion Attention ---
        branch_weights_raw = self.fusion_attention[0](hidden)
        branch_weights_raw = self.fusion_attention[1](branch_weights_raw)
        branch_weights_raw = self.fusion_attention[2](branch_weights_raw)
        branch_weights = self.fusion_attention[3](branch_weights_raw)

        final_prediction = branch_weights[:, 0:1, :, :] * temporal_pred + \
                           branch_weights[:, 1:2, :, :] * spatial_pred

        return final_prediction
