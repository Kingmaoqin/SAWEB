import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个Dense Layer
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout=0.1):
        super(DenseLayer, self).__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.ReLU()
        self.fc = nn.Linear(in_channels, growth_rate)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, in_channels)
        out = self.norm(x)
        out = self.act(out)
        out = self.fc(out)
        out = self.act(out)
        out = self.drop(out)
        # 输出特征与输入拼接 (DenseNet风格)
        return torch.cat([x, out], dim=1)  # (batch_size, in_channels + growth_rate)

# Dense Block由多个Dense Layer堆叠组成
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers=2, dropout=0.1):
        super(DenseBlock, self).__init__()
        layers = []
        current_channels = in_channels
        for i in range(num_layers):
            layers.append(DenseLayer(current_channels, growth_rate, dropout=dropout))
            current_channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = current_channels

    def forward(self, x):
        return self.block(x) # (batch_size, out_channels)

# 通道注意力模块 (Channel Attention)
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(num_channels, num_channels // reduction, bias=False)
        self.fc2 = nn.Linear(num_channels // reduction, num_channels, bias=False)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, num_channels)
        y = self.fc1(x)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        out = x * y
        return out

# ResidualBlock定义与之前相同
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.act(out)
        out = self.norm1(out)
        out = self.drop(out)

        out = self.fc2(out)
        out = self.act(out)
        out = self.norm2(out)
        out = self.drop(out)

        # 残差连接
        out = out + residual
        return out

class SimplifiedSharedLayer(nn.Module):
    def __init__(self, in_features, hidden_dim, num_res_blocks=2):
        super(SimplifiedSharedLayer, self).__init__()
        self.input_projection = nn.Linear(in_features, hidden_dim)
        self.norm_input = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.1)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.act(x)
        x = self.norm_input(x)
        x = self.drop(x)
        
        x = self.res_blocks(x)
        return x

class MultiTaskModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=17,
                 num_res_blocks=2, dense_growth_rate=4, dense_num_layers=2,
                 use_channel_attention=False, ca_reduction=16):
        super(MultiTaskModel, self).__init__()
        self.num_tasks = out_features

        # Dense Block
        self.dense_block = DenseBlock(in_features, growth_rate=dense_growth_rate, num_layers=dense_num_layers, dropout=0.3)
        dense_out_channels = self.dense_block.out_channels

        # attention (optional)
        self.use_channel_attention = use_channel_attention
        if self.use_channel_attention:
            self.channel_attention = ChannelAttention(dense_out_channels, reduction=ca_reduction)

        # share layer
        self.shared_layer = SimplifiedSharedLayer(dense_out_channels, hidden_dim, num_res_blocks=num_res_blocks)

        # output
        self.final_hidden_dim = hidden_dim + dense_out_channels
        self.hazard_layer = nn.Linear(self.final_hidden_dim, out_features)

    def forward(self, x):
        dense_output = self.dense_block(x)
        if self.use_channel_attention:
            dense_output = self.channel_attention(dense_output)
        shared_output = self.shared_layer(dense_output)
        concat_features = torch.cat([shared_output, dense_output], dim=1)
        hazard_logits = self.hazard_layer(concat_features)
        hazard_probs = torch.sigmoid(hazard_logits)
        log_hazard = torch.log(1 - hazard_probs + 1e-8)
        log_survival = torch.cumsum(log_hazard, dim=1)
        survival_probs = torch.exp(log_survival)
        task_outputs = [survival_probs[:, i:i+1] for i in range(self.num_tasks)]
        return task_outputs

    def custom_loss(self, task_outputs, targets, masks):
        loss = 0
        prev_output = None
        for i, task_output in enumerate(task_outputs):
            task_target = targets[i]
            task_mask = masks[i]
            task_loss = F.binary_cross_entropy(task_output, task_target.float(), reduction='none')
            task_loss = task_loss * task_mask.float()
            loss += task_loss.sum() / task_mask.sum()

            # markover
            if prev_output is not None:
                markov_loss = F.mse_loss(task_output, prev_output.detach(), reduction='mean')
                loss += 0.1 * markov_loss
            prev_output = task_output

        return loss
