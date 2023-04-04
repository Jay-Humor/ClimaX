# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# 导入必要的库
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_

# 导入自定义函数
from climax.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)

# 定义 ClimaX 模型
class ClimaX(nn.Module):
    """实现 ClimaX 模型，如论文所述，
    https://arxiv.org/abs/2301.10343

    Args:
        default_vars (list): 默认用于训练的变量列表
        img_size (list): 输入数据的图像大小
        patch_size (int): 输入数据的块大小
        embed_dim (int): 嵌入维度
        depth (int): transformer 层数
        decoder_depth (int): 解码器层数
        num_heads (int): 注意力头数
        mlp_ratio (float): mlp 隐藏维度与嵌入维度的比例
        drop_path (float): 随机深度率
        drop_rate (float): dropout 率
    """

    def __init__(
        self,
        default_vars,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
    ):
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size # 输入图像的大小
        self.patch_size = patch_size # 输入块的大小
        self.default_vars = default_vars # 默认用于训练的变量列表

        # 变量的 tokenization：为每个输入变量单独创建一个嵌入层
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, 1, embed_dim) for i in range(len(default_vars))]
        )
        self.num_patches = self.token_embeds[0].num_patches # 每个输入图像块的数量相同

        # 变量嵌入以指示每个标记属于哪个变量，有助于聚合变量
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim) # 变量嵌入及其变量映射

        # 变量聚合：一个可学习的查询和单层交叉注意力
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True) # 学习可调的查询
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) # 单层交叉注意力

        # 位置编码和前导时间编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True) # 位置编码
        self.lead_time_embed = nn.Linear(1, embed_dim) # 时间编码

        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate) # 位置编码矩阵的 dropout 操作
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # 随机深度学习（stochastic depth）的dropout规则
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim, # 输入嵌入维度大小
                    num_heads, # 多头注意力头数目
                    mlp_ratio, # MLP隐藏层大小和输入大小的比率
                    qkv_bias=True, # 是否包含 QKV 偏置参数
                    drop_path=dpr[i], # 对于每个Block的dropout比率，从dpr列表中取值
                    norm_layer=nn.LayerNorm, # 采用的归一化方式，这里采用LayerNorm
                    drop=drop_rate, # Block中各子层的dropout比率
                )
                for i in range(depth) # 构建depth个Block，每个Block都一样
            ]
        )
        self.norm = nn.LayerNorm(embed_dim) # 对输出做LayerNorm，归一化

        # --------------------------------------------------------------------------

        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))  # 用全连接层来降低特征维度
            self.head.append(nn.GELU())  # GELU激活函数
        self.head.append(nn.Linear(embed_dim, len(self.default_vars) * patch_size**2))  # 全连接层输出最终结果
        self.head = nn.Sequential(*self.head)

        # --------------------------------------------------------------------------

        self.initialize_weights()  # 初始化模型参数

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],  # 位置嵌入维度
            int(self.img_size[0] / self.patch_size),  # 图像高度
            int(self.img_size[1] / self.patch_size),  # 图像宽度
            cls_token=False,  # 是否使用分类嵌入
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))  # 得到变量嵌入
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        # token embedding layer
        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)  # 随机初始化

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)  # 随机初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 常数初始化
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # 常数初始化
            nn.init.constant_(m.weight, 1.0)  # 常数初始化

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)  # 声明变量嵌入
        # TODO: create a mapping from var --> idx
        var_map = {}  # 声明变量索引映射
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx  # 将变量和索引映射起来
            idx += 1
        return var_embed, var_map  # 返回变量嵌入和索引映射

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])  # 获取变量索引
        return torch.from_numpy(ids).long().to(device)  # 转为tensor并移到设备上

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)  # 获取变量在默认变量列表中的索引
        return var_emb[:, ids, :]  # 返回变量嵌入的子集

    def unpatchify(self, x, h=None, w=None):
        """
        Args:
            x (torch.Tensor): (B, L, V * patch_size**2) 待解压的天气/气候变量
            h (int): 图像高度
            w (int): 图像宽度
        Returns:
            imgs (torch.Tensor): (B, V, H, W)
        """
        p = self.patch_size  # 块大小
        c = len(self.default_vars)  # 变量数量
        h = self.img_size[0] // p if h is None else h // p  # 图像高度
        w = self.img_size[1] // p if w is None else w // p  # 图像宽度
        assert h * w == x.shape[1]  # 检查块数和图像尺寸是否匹配

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))  # 重新调整形状
        x = torch.einsum("nhwpqc->nchpwq", x)  # 交换一些维度
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))  # 将块拼接成完整图像
        return imgs

    def aggregate_variables(self, x):
        """
        聚合输入变量
        Args:
            x (torch.Tensor): (B, V, L, D) 输入的天气/气候变量
        Returns:
            x (torch.Tensor): (B, V, H, W)
        """
        b, _, l, _ = x.shape  # 获取批量大小、变量数量、序列长度和嵌入维度
        x = torch.einsum("bvld->blvd", x)  # 交换一些维度
        x = x.flatten(0, 1)  # 将变量和序列维度合并为第一维度，形状为（BxL，V，D）

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)  # 将变量查询重复
        x, _ = self.var_agg(var_query, x, x)  # 聚合嵌入
        x = x.squeeze()  # 移除单维度

        x = x.unflatten(dim=0, sizes=(b, l))  # 将前面拆开的维度重新分离，形状为（B，L，D）
        return x

    def forward_encoder(self, x, lead_times, variables):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): [B, Vi, H, W] 输入的天气/气候变量
            y (torch.Tensor): [B, Vo, H, W] 目标天气/气候变量
            lead_times (torch.Tensor): [B] 每个批次元素的预报时段
            variables (list): 字符串列表，表示输入的天气/气候变量

        Returns:
            x (torch.Tensor): [B, L, D] 输出的编码张量, 其中L是序列长度, D是嵌入维度
        """

        if isinstance(variables, list):
            variables = tuple(variables)  # 如果变量是列表，转换为元组

        embeds = []  # 存储嵌入
        var_ids = self.get_var_ids(variables, x.device)  # 获取变量在默认变量列表中的索引
        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1]))  # 将每个变量单独嵌入
        x = torch.stack(embeds, dim=1)  # 组合

        # 添加变量嵌入
        var_embed = self.get_var_emb(self.var_embed, variables)  # 获取变量嵌入
        x = x + var_embed.unsqueeze(2)  # 将变量嵌入与嵌入的张量相加 B, V, L, D

        # 变量聚合
        x = self.aggregate_variables(x)  # B, L, D

        # 添加位置嵌入
        x = x + self.pos_embed

        # 添加提前时间嵌入
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # 应用 Transformer 模块
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, x, y, lead_times, variables, out_variables, metric, lat):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): [B, Vi, H, W] 输入的天气/气候变量
            y (torch.Tensor): [B, Vo, H, W] 目标天气/气候变量
            lead_times (torch.Tensor): [B] 每个批次元素的预报时段
            variables (list): 字符串列表，表示输入的天气/气候变量
            out_variables (list): 字符串列表，表示输出的天气/气候变量
            metric (list): 损失函数列表
            lat (int): H

        Returns:
            loss (list): 不同度量的损失值列表
            preds (torch.Tensor): [B, Vo, H, W] 预测的天气/气候变量
        """
        out_transformers = self.forward_encoder(x, lead_times, variables)  # 使用forward_encoder函数进行前向传递，输出为B, L, D
        preds = self.head(out_transformers)  # 将out_transformers输入至self.head中，得到B, L, V*p*p的输出

        preds = self.unpatchify(preds)  # 将B, L, V*p*p的输出reshape为B, V, H, W
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)  # 获取输出变量的id
        preds = preds[:, out_var_ids]  # 从输出中选取out_variables对应的部分

        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]  # 根据传入的metric计算损失

        return loss, preds

    def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, clim, log_postfix):
        """
        模型推理和评估

        Args:
            x (torch.Tensor): [B, Vi, H, W] 输入的天气/气候变量
            y (torch.Tensor): [B, Vo, H, W] 目标天气/气候变量
            lead_times (torch.Tensor): [B] 每个批次元素的预报时段
            variables (list): 字符串列表，表示输入的天气/气候变量
            out_variables (list): 字符串列表，表示输出的天气/气候变量
            transform (function): 输出变量的转换函数
            metric (list): 损失函数列表
            lat (int): H
            clim (torch.Tensor): 气候
            log_postfix (string): 日志后缀

        Returns:
            results (list): 评估结果
        """
        _, preds = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat)  # 获取预测结果
        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]  # 根据传入的metric计算评估结果
