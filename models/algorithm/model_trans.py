import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithm.model_imp import MultiTaskModel
class SimpleTabTransformer(nn.Module):
    def __init__(
        self, 
        num_categories_list,  # 每个categorical特征的类别数列表，如 [10, 5, 20]
        embed_dim=8, 
        num_transformer_layers=2, 
        num_heads=2, 
        dropout=0.1
    ):
        super(SimpleTabTransformer, self).__init__()
        
        # 1) 针对每个类别特征，定义 embedding 层
        #    比如有3个类别特征: embedding1 -> (vocab1, embed_dim), embedding2 -> (vocab2, embed_dim), ...
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embed_dim) 
            for num_categories in num_categories_list
        ])
        
        # 2) 一个可选的 learnable [CLS] token 向量，用于在序列最前面聚合信息
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3) Transformer Encoder
        #    - 我们使用 PyTorch 自带的 nn.TransformerEncoder
        #    - EncoderLayer 里面包含多头注意力和feedforward等模块
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # 使输入维度为 (batch_size, seq_len, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_transformer_layers
        )

        # 4) LayerNorm, 用于 Transformer 输出后的规范化
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, categorical_inputs):
        """
        categorical_inputs: List/Tensor of shape [(batch_size), (batch_size), ...],
                           每个元素是某个类别特征的下标，长度与 self.embeddings 相同。
                           或者事先把它们拼成一个 (batch_size, num_categorical) 的张量，再切开。
        """
        # 1) 对每个特征做 embedding, 并把它们拼成序列
        #    假设 categorical_inputs 是一个列表，里面每个元素是 (batch_size,)
        embed_seq = []
        for i, emb in enumerate(self.embeddings):
            # shape: (batch_size, embed_dim)
            emb_i = emb(categorical_inputs[:, i])  
            embed_seq.append(emb_i)
        # 拼成 (batch_size, num_categorical, embed_dim)
        embed_seq = torch.stack(embed_seq, dim=1)
        
        # 2) 在序列最前面拼接 [CLS] 向量，(batch_size, 1, embed_dim)
        batch_size = embed_seq.size(0)
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        embed_seq = torch.cat([cls_token, embed_seq], dim=1)  # (batch_size, 1 + num_categorical, embed_dim)
        
        # 3) 过 Transformer Encoder
        transformer_out = self.transformer_encoder(embed_seq)  # (batch_size, seq_len, embed_dim)
        
        # 4) 取第0个位置 [CLS] 向量，作为聚合后的类别特征表示
        cls_output = transformer_out[:, 0, :]  # (batch_size, embed_dim)
        
        # 5) 做一个 norm
        cls_output = self.norm(cls_output)
        return cls_output  # (batch_size, embed_dim)

class TabTransformerFusion(nn.Module):
    def __init__(
        self,
        num_categories_list,
        num_numerical_features,
        embed_dim=8,
        num_transformer_layers=2,
        num_heads=2,
        dropout=0.1
    ):
        super(TabTransformerFusion, self).__init__()
        
        # 1) 用前面定义的简易 TabTransformer，获取类别特征的表示
        self.tab_transformer = SimpleTabTransformer(
            num_categories_list=num_categories_list,
            embed_dim=embed_dim,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 2) 对数值特征做一个线性映射到和 embed_dim 相同
        self.numerical_proj = nn.Linear(num_numerical_features, embed_dim)
        
        # 3) 再把 (embed_dim + embed_dim) 投影到 MultiTaskModel 需要的 in_features 尺寸
        #    假设你想要的 in_features = 16(只是举例，需要和 MultiTaskModel 保持一致)
        self.in_features = 16
        self.fusion_proj = nn.Linear(embed_dim * 2, self.in_features)
        
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(self.in_features)

    def forward(self, categorical_inputs, numerical_inputs):
        """
        categorical_inputs: (batch_size, num_categorical)
        numerical_inputs: (batch_size, num_numerical)
        """
        # 1) 得到类别特征的表示 (batch_size, embed_dim)
        cat_embed = self.tab_transformer(categorical_inputs)
        # 2) 得到数值特征的表示 (batch_size, embed_dim)
        num_embed = self.numerical_proj(numerical_inputs)  # (batch_size, embed_dim)
        
        # 3) 拼接并投影到 in_features
        fusion = torch.cat([cat_embed, num_embed], dim=-1)  # (batch_size, 2*embed_dim)
        fusion = self.fusion_proj(fusion)                  # (batch_size, in_features)
        fusion = self.act(fusion)
        fusion = self.norm(fusion)
        return fusion  # (batch_size, in_features)

class TabTransformerMultiTaskModel(nn.Module):
    def __init__(self,
                 num_categories_list,
                 num_numerical_features,
                 # 与 TabTransformerFusion 相关
                 tt_embed_dim=8,
                 tt_num_transformer_layers=2,
                 tt_num_heads=2,
                 tt_dropout=0.1,
                 # 与 MultiTaskModel 相关
                 out_features=5, 
                 hidden_dim=16,
                 num_res_blocks=2, 
                 dense_growth_rate=4, 
                 dense_num_layers=2,
                 use_channel_attention=False, 
                 ca_reduction=16):
        super(TabTransformerMultiTaskModel, self).__init__()
        
        # 1) TabTransformer融合模块
        self.fusion = TabTransformerFusion(
            num_categories_list=num_categories_list,
            num_numerical_features=num_numerical_features,
            embed_dim=tt_embed_dim,
            num_transformer_layers=tt_num_transformer_layers,
            num_heads=tt_num_heads,
            dropout=tt_dropout
        )
        
        # 2) MultiTaskModel
        #    注意： fusion 会输出 (batch_size, in_features)，
        #          这里需要和 MultiTaskModel.__init__(..., in_features=??) 对应。
        self.multi_task_model = MultiTaskModel(
            in_features=self.fusion.in_features,  # 16
            out_features=out_features,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            dense_growth_rate=dense_growth_rate,
            dense_num_layers=dense_num_layers,
            use_channel_attention=use_channel_attention,
            ca_reduction=ca_reduction
        )

    def forward(self, categorical_inputs, numerical_inputs):
        """
        categorical_inputs: (batch_size, num_categorical)
        numerical_inputs: (batch_size, num_numerical)
        """
        # 1) 先过 TabTransformerFusion 得到 (batch_size, in_features)
        fused_input = self.fusion(categorical_inputs, numerical_inputs)
        # 2) 再过 multi_task_model
        task_outputs = self.multi_task_model(fused_input)
        return task_outputs

    def custom_loss(self, task_outputs, targets, masks):
        # 直接调用 multi_task_model 内的 custom_loss
        return self.multi_task_model.custom_loss(task_outputs, targets, masks)
