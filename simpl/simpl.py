from typing import Any, Dict, List, Tuple, Union, Optional
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from fractions import gcd
#
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention

from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.nn import MessagePassing, GCN2Conv, NNConv
from torch_scatter import scatter_softmax, scatter_add, scatter_max
from utils.utils import gpu, init_weights, to_long, pad_feats_and_create_mask
from .transformer_blocks import Block as transformerblock
from torch.nn.utils.rnn import pad_sequence



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # learnable scaling factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D]
        norm = x.norm(dim=-1, keepdim=True)  # [N, 1]
        rms = norm / (x.shape[-1] ** 0.5)    # Root Mean Square
        return self.scale * (x / (rms + self.eps))  # [N, D]


class DynamicDenseConnection(nn.Module):
    def __init__(self, dim: int, num_layers: int, hidden_dim: int = 128, use_softmax: bool = True):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.W1 = nn.Linear(dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, num_layers)
        self.bias = nn.Parameter(torch.zeros(num_layers))
        self.use_softmax = use_softmax

    def forward(self, x_cur: torch.Tensor, x_prev_list: list) -> torch.Tensor:
        N, D = x_cur.shape
        num_layers = len(x_prev_list)
        assert num_layers == self.W2.out_features

        x_stack = torch.stack(x_prev_list, dim=1)  # [N, num_layers, D]

        # Dynamic weights
        h = self.norm(x_cur)         # [N, D]
        a = F.gelu(self.W1(h))       # [N, H]
        a = self.W2(a) + self.bias   # [N, num_layers]
        if self.use_softmax:
            a = F.softmax(a, dim=-1)
        a = a.unsqueeze(-1)          # [N, num_layers, 1]

        # Aggregate
        x_input = (a * x_stack).sum(dim=1)  # [N, D]
        return x_input


#* 1. Actor Encoder

class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Conv1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=(
            int(kernel_size) - 1) // 2, stride=stride, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Res1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """

    def __init__(self, n_in=3, hidden_size=128, n_fpn_scale=4, num_layers=2, dropout=0.1, d_rpe_in = 8, n_a2a_head = 8):
        super(ActorNet, self).__init__()
        norm = "GN"
        ng = 1

        n_out = [2**(5 + s) for s in range(n_fpn_scale)]  # [32, 64, 128， 256]
        blocks = [Res1d] * n_fpn_scale
        num_blocks = [2] * n_fpn_scale
        self.num_layers = num_layers
        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], hidden_size, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)
        self.output = Res1d(hidden_size, hidden_size, norm=norm, ng=ng)

        #* a2a
        if num_layers != 0:
            self.a2a = nn.ModuleList(
                EdgeAwareGATLayer(d_model = hidden_size, d_edge=hidden_size,
                                heads=n_a2a_head, dropout=dropout, d_ffn=2*hidden_size)
                                for _ in range(num_layers)
            )
            self.proj_a2a_rpes = nn.Sequential(
                nn.Linear(d_rpe_in, 128),
                nn.LayerNorm(128),
                nn.ReLU(inplace=True)
            )
        

    def forward(self, actors: Tensor, rpes: Dict[str, Tensor]) -> Tensor:
        out = actors
        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        actors = self.output(out)[:, :, -1]


        a2a_attr = self.proj_a2a_rpes(rpes['a2a_fused_rpes'])

        
        for layer in self.a2a:
            actors, a2a_attr = layer(actors, rpes['a2a_edges'], a2a_attr)

        return actors, a2a_attr


#* 2. Map Encoder


class MyPointAggregateBlock(nn.Module):
    def __init__(self, hidden_size: int, aggre_out: bool, dropout: float = 0.1) -> None:
        super(MyPointAggregateBlock, self).__init__()
        self.aggre_out = aggre_out
        self.hidden_size = hidden_size

        # 点级别特征变换
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )

        # 聚合特征处理
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x_inp, lane_ids):
        """
        x_inp: [batch_num_nodes, hidden_size]
        lane_ids: [batch_num_nodes] 车道分组标识
        """
        # 点级别特征变换
        x = self.fc1(x_inp)  # [batch_num_nodes, hidden_size]

        # 车道级别聚合（最大池化）
        max_per_lane, _ = scatter_max(
            x,
            lane_ids,
            dim=0,
            dim_size=torch.max(lane_ids).item() + 1 if lane_ids.numel() > 0 else 1
        )  # [num_lanes, hidden_size]

        # 将车道级特征扩展回点级别
        expanded_max = max_per_lane[lane_ids]  # [batch_num_nodes, hidden_size]

        # 拼接点特征和车道聚合特征
        x_aggre = torch.cat([x, expanded_max], dim=-1)  # [batch_num_nodes, hidden_size * 2]

        # 残差连接
        out = self.norm(x_inp + self.fc2(x_aggre))  # [batch_num_nodes, hidden_size]

        # 是否返回聚合输出
        if self.aggre_out:
            # 再次聚合到车道级别
            lane_out, _ = scatter_max(
                out,
                lane_ids,
                dim=0,
                dim_size=torch.max(lane_ids).item() + 1 if lane_ids.numel() > 0 else 1
            )  # [num_lanes, hidden_size]
            return lane_out
        else:
            return out

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_model, d_ffn)
        self.fc3 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.fc1(x)) * self.fc2(x)   # 门控结构
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        return x



class EdgeAwareGATLayer(MessagePassing):
    def __init__(self,
                 d_model: int = 128,
                 d_edge: int = 128,
                 d_ffn: int = 512,
                 heads: int = 8,
                 dropout: float = 0.1,
                 update_edge: bool = True):
        super(EdgeAwareGATLayer, self).__init__(aggr='add',node_dim=0)  # aggregate messages by sum
        self.d_model = d_model
        self.d_edge = d_edge
        self.heads = heads
        self.d_head = d_model // heads
        self.update_edge = update_edge
        self.dropout = nn.Dropout(dropout)
        # self.head_mixer = nn.Conv1d(self.heads, self.heads, kernel_size=1, bias=False)
        # Project to memory (attention input): f(x_i, x_j, e_ij) -> d_model
        self.memory_proj = nn.Sequential(
            nn.Linear(2 * d_model + d_edge, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

        # Attention scoring
        self.att_q = nn.Linear(d_model, d_model, bias=False)
        self.att_k = nn.Linear(d_model, d_model, bias=False)
        self.att_v = nn.Linear(d_model, d_model, bias=False)

        self.att_o = nn.Linear(d_model, d_model, bias=False)
        self.edge_bias_proj = nn.Linear(d_model,self.heads)
        # Edge update block
        if update_edge:
            self.edge_update_block = nn.Sequential(
                nn.Linear(d_model, d_edge),
                nn.LayerNorm(d_edge),
                nn.ReLU(inplace=True)
            )
            self.edge_norm = nn.LayerNorm(d_edge)

        # Feed-forward network

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )
        
        # Normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def build_mem(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        src_node = x[edge_index[0]]
        tgt_node = x[edge_index[1]]
        # print("src_node:", src_node.shape)
        # print("tgt_node:", tgt_node.shape)
        # print("edge_attr:", edge_attr.shape)
        mem = self.memory_proj(torch.cat([tgt_node,src_node,edge_attr],dim=-1))
        if self.update_edge:
            delta_edge = self.edge_update_block(mem)  # [E, d_edge]
            edge_attr = self.edge_norm(edge_attr + delta_edge)
        return mem, edge_attr
    
    def message(self, x_i: Tensor, mem: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        '''
        x_i: [E, d_model] receiver node features, edge index[1]
        x_j: [E, d_model] sender node features, edge_index[0]
        edge_attr: [E, d_edge] edge features
        '''

        # attention heads
        q = self.att_q(x_i).view(-1, self.heads, self.d_head)  # [E, H, d_h]
        k = self.att_k(mem).view(-1, self.heads, self.d_head)
        v = self.att_v(mem).view(-1, self.heads, self.d_head)

        # attention scores
        attn_logits = (q * k).sum(dim=-1) / (self.d_head ** 0.5)  # [E, H]
        # bias = self.edge_bias_proj(edge_attr)
        # attn_logits = attn_logits + bias

        # local softmax per target node
        attn = pyg_softmax(attn_logits, index=edge_index[1])  # [E, H]

        # weighted message
        out = attn.unsqueeze(-1) * v  # [E, H, d_h]
        # out = self.head_mixer(out)
        out = out.view(-1, self.d_model)  # [E, d_model]

        return self.att_o(out)
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        '''
        x:          [N, d_model] node features
        edge_index: [2, E]       edges (source, target)
        edge_attr:  [E, d_edge]  edge features
        '''
        mem, updated_edge_attr = self.build_mem(x, edge_index, edge_attr)
        out = self.propagate(x=x, mem=mem, edge_index=edge_index, edge_attr=updated_edge_attr)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x, updated_edge_attr

    def update(self, aggr_out: Tensor) -> Tensor:
        return self.dropout(aggr_out)  # [N, d_model]


class Point_RPE_MAP_Encoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 10,
            d_model: int = 128,
            dropout: int = 0.1,
            num_layers: int = 2,
            d_rpe_in: int = 8,
            n_l2l_head: int = 8
    ):
        super(Point_RPE_MAP_Encoder, self).__init__()

        self.num_layers = num_layers
        # 特征投影层
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

        # 聚合模块
        self.point_aggr = MyPointAggregateBlock(
            hidden_size=d_model,
            aggre_out=False,  # 点级别输出
            dropout=dropout
        )
        self.lane_aggr = MyPointAggregateBlock(
            hidden_size=d_model,
            aggre_out=True,  # 车道级别输出
            dropout=dropout
        )

        # Lane Fusion
        self.rpe_proj = nn.Sequential(
            nn.Linear(d_rpe_in, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True))

        self.l2l = nn.ModuleList(
            EdgeAwareGATLayer(d_model = d_model, d_edge=d_model,
                              heads=n_l2l_head, dropout=dropout, d_ffn=2*d_model)
                              for _ in range(num_layers)
        )

    def forward(self, node_feats,  nodes_of_lanes, rpes):
        # out: {N_lane, d_model}
        # [N_{batch_node},input_dim]
        # edge_index: dict{} key:pre,suc,left_pair,right_pair,pre_pairs,suc_pairs
        # nodes_of_lanes shape [num_nodes], 每个节点属于哪一条车道

        # 特征投影
        l2l_edge = rpes['l2l_edges']
        l2l_attr = rpes['l2l_fused_rpes']
        x = self.proj(node_feats)  # [batch_num_nodes, hidden_size]

        # 第一级聚合（点级别增强）
        x = self.point_aggr(x, nodes_of_lanes)  # [batch_num_nodes, hidden_size]

        # 第二级聚合（车道级别聚合）
        lane_features = self.lane_aggr(x, nodes_of_lanes)  # [num_lanes, hidden_size]

        # 车道之间交互建模
        l2l_attr = self.rpe_proj(l2l_attr)

        for layer in self.l2l:
            lane_features, l2l_attr = layer(lane_features, l2l_edge, l2l_attr)
        
        return lane_features, l2l_attr


#* 3. Fusion
class EdgeAwareGATFusion(nn.Module):
    def __init__(self, device, cfg):
        super(EdgeAwareGATFusion, self).__init__()

        d_model = cfg['d_embed']
        d_rpe_in = cfg['rpe_type_d']
        num_heads = cfg['n_scene_head']
        num_layers = cfg['n_scene_layer']
        dropout = cfg['dropout']
        self.num_layers = cfg['n_scene_layer']

        self.l2a2l = nn.ModuleList(
            EdgeAwareGATLayer(d_model=d_model,
                                d_edge=d_model,
                                d_ffn=2 * d_model,
                                heads=num_heads,
                                dropout=dropout)
            for _ in range(num_layers)
        )
        self.proj_a2l_l2a_rpe = nn.Sequential(
            nn.Linear(d_rpe_in, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

        self.res_fused = nn.ModuleList(
            DynamicDenseConnection(d_model, i)
            for i in range(1, num_layers)
        )
        
        self.l2l = nn.ModuleList(
                GCN2Conv(
                    channels=d_model,
                    alpha=0.1,
                    theta=0.5,
                    shared_weights=False,
                    layer=i + 1
                )
                for i in range(num_layers)
            )
            
    def forward(self, actors, actor_idcs, lanes, lane_idcs, rpes, l2l_attr, a2a_attr):
        a2a_fusion_edges = rpes['a2a_fusion_edges']
        l2l_fusion_edges = rpes['l2l_fusion_edges']
        a2l_l2a_fusion_edges = rpes['a2l_l2a_fusion_edges']
        a2l_l2a_attr = rpes['a2l_l2a_fused_rpes']

        device = actors[0].device

        token_list = []
        lane_global_ids = []
        actor_global_ids = []
        offset = 0  # 当前拼接 token 的起始位置
        for lane_ids, actor_ids in zip(lane_idcs, actor_idcs):
            # 拼接每个场景的 actor + lane 特征
            token_piece = torch.cat([actors[actor_ids], lanes[lane_ids]], dim=0)
            token_list.append(token_piece)

            lane_count = len(lane_ids)
            actor_count = len(actor_ids)
            # actor 在前，lane 的起始位置是 offset + len(actor_ids)
            lane_start = offset + actor_count

            actor_global_ids.append(torch.tensor(list(range(offset, offset + actor_count)), 
                                                 device=device, dtype=torch.long))
            lane_global_ids.append(torch.tensor(list(range(lane_start, lane_start + lane_count)), 
                                                device=device, dtype=torch.long))
            # 更新 offset，跳过当前场景的 token（actor + lane）
            offset += actor_count + lane_count

        token = torch.cat(token_list, dim=0)
        a2l_l2a_attr = self.proj_a2l_l2a_rpe(a2l_l2a_attr)
        edge_index = torch.cat([a2a_fusion_edges, l2l_fusion_edges, a2l_l2a_fusion_edges],dim=-1)
        edge_attr = torch.cat([a2a_attr, l2l_attr, a2l_l2a_attr],dim=0)


        token_list = []
        token_list.append(token)
        for i in range(self.num_layers):
            token, edge_attr = self.l2a2l[i](token, edge_index, edge_attr)
            if i < len(self.res_fused):
                token = self.res_fused[i](token, token_list)
            token_list.append(token)
        
        token_fused = token
        actors_list = [token_fused[idx] for idx in actor_global_ids]
        lanes_list  = [token_fused[idx] for idx in lane_global_ids]

        return actors_list, lanes_list

#* 4. Decoder

class GlobalQueryTokenExtractor(nn.Module):
    def __init__(self, d_model=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 多头注意力
        self.norm1 = nn.LayerNorm(d_model) 
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True, dropout=dropout)


    def forward(self, agent_feat_global, lane_feat_global,
                agent_padding_mask, lane_padding_mask):
        """
        Inputs:
          agent_feat_global:     [B, N_a, D]
          lane_feat_global:      [B, N_l, D]
          agent_padding_mask:    [B, N_a]  (1: valid, 0: pad)
          lane_padding_mask:     [B, N_l]
          query:                 [B, N_a * K, D]  (可选) 使用 agent_feat_global 作为 query

        Returns:
          global_token:          [B, N_a*K, D]  每个 agent 感知的全局语义
        """
        B, N_a, D = agent_feat_global.shape

        # 环境特征与 mask
        env_feat = torch.cat([agent_feat_global, lane_feat_global], dim=1)     # [B, N_all, D]
        env_mask = torch.cat([agent_padding_mask, lane_padding_mask], dim=1)   # [B, N_all]
        key_padding_mask = env_mask                                            # [B, N_all]
        # 多头注意力
        attn_out, _ = self.attn(
            query=agent_feat_global,
            key=env_feat,
            value=env_feat,
            key_padding_mask=key_padding_mask
        )  # [B, N_a * K, D]
        return attn_out  # 每个 agent 的全局感知特征

class ModeQueryRefineDecoder(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.num_modes = cfg['g_num_modes']
        self.T = cfg['g_pred_len']
        self.d_model = cfg['d_embed']
        num_heads = cfg['n_decoder_head']
        d_pos = cfg['d_pos']
        # dropout = 0.1
        self.out_prob = cfg['out_prob']

        # 位置映射层
        self.agent_pos_g = nn.Sequential(nn.Linear(self.d_model + d_pos, self.d_model),
                                         nn.LayerNorm(self.d_model),
                                         nn.ReLU(inplace=True))
        
        self.lane_pos_g = nn.Sequential(nn.Linear(self.d_model + d_pos, self.d_model),
                                        nn.LayerNorm(self.d_model),
                                        nn.ReLU(inplace=True))

        # 生成全局 query 的模块
        self.global_query_generator = GlobalQueryTokenExtractor(self.d_model, num_heads, dropout=0.0)
        # 
        dim_mm = self.d_model * self.num_modes
        dim_inter = dim_mm // 2
        self.multihead_proj = nn.Sequential(
            nn.Linear(self.d_model, dim_inter),
            nn.LayerNorm(dim_inter),
            nn.ReLU(inplace=True),
            nn.Linear(dim_inter, dim_mm),
            nn.LayerNorm(dim_mm),
            nn.ReLU(inplace=True)
        )

        self.reg = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU(inplace=True),
                nn.Linear(self.d_model, self.T * 2)
            )
        
        self.cls = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model , 1)
        )

        # self.offset_gate = nn.Sequential(
        #     nn.Linear(2 * self.d_model, self.d_model),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model, 1),
        #     nn.Sigmoid()  # 输出范围在 [0,1]
        # )

        # refine MLP
        self.refine_mlp = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            nn.Linear(self.d_model, self.T * 2)
        )
        self.offset_scale = 1.0  # 可调的偏移缩放系数

        nn.init.zeros_(self.refine_mlp[-1].weight)
        nn.init.zeros_(self.refine_mlp[-1].bias)
    def forward(self, agent_feats, lane_feats, pose):
        """
            agent_feats: List([N_a, D]) 
            actor_idcs: 每个 batch 场景中的 agent id 索引
            lane_feats: List([N_l,D])
            pose: dict{agent_pose:([Na,4]), lane_pose:([Nl,4])}
            return: list of [N_i, K, T, 2], list of [N_i, K]
        """
        padded_agent_pose = pad_sequence(pose['agent_pose'], batch_first=True)
        padded_lane_pose = pad_sequence(pose['lane_pose'], batch_first=True)
        # agent_feats_padded: [B,N_max_a,D], mask: [B, N_max_a]
        agent_feats_padded, agent_padding_mask, lane_feats_padded, lane_padding_mask = pad_feats_and_create_mask(agent_feats,lane_feats)
        B, N_a, D = agent_feats_padded.shape
        K = self.num_modes

        # 1. 位置映射
        # [B, N_a, D+2] -> [B, N_a, D]
        agent_feat_global = self.agent_pos_g(torch.cat([agent_feats_padded, padded_agent_pose],dim=2))  
        lane_feat_global = self.lane_pos_g(torch.cat([lane_feats_padded, padded_lane_pose],dim=2)) 

        # 2. 生成多模态查询,同时初步预测轨迹
        # todo:尝试采用query与embed cat得到多模态特征
        embed = self.multihead_proj(agent_feats_padded).view(B, N_a ,K, D)  # [B, N_a * K, D]
        traj = self.reg(embed).view(B,N_a,K,-1,2)  # [B, N_a, K, T*2]

        # 3. 提取全局特征
        offset_embed = self.global_query_generator(agent_feat_global, lane_feat_global,
                                                   agent_padding_mask, lane_padding_mask) # [B, N_a, D]
        offset_embed = offset_embed.unsqueeze(2).repeat(1, 1, K, 1)  # [B, N_a, K, D]

        # 4. 微调轨迹
        fused_embed = torch.cat([offset_embed, embed], dim=-1)  # [B, N_a, K, D*2]
        # fused_embed = torch.cat([offset_embed, embed.detach()], dim=-1)  # [B, N_a, K, D*2]
        offset = self.refine_mlp(fused_embed).view(B, N_a, K, -1, 2)   # [B, N_a, K, T, 2]
        # gate = self.offset_gate(fused_embed).view(B, N_a, K, 1, 1)  # [B, N_a, K, 1]
        traj = traj + (offset * self.offset_scale)                        # [B, N_a, K, T, 2]
        vel = torch.gradient(traj, dim=-2)[0] / 0.1
        # 5. 计算模态置信度
        # conf = self.cls(torch.cat([offset_embed.detach(), embed], dim=-1)).view(B, N_a, K)
        conf = self.cls_refine(fused_embed).view(B, N_a, K)
        if self.out_prob:
            conf = F.softmax(conf,dim=-1)
        res_traj, res_conf, res_aux = [], [], []
        # 遍历每个 batch
        for i in range(traj.shape[0]):  # B
            # 取出第 i 个 batch 的 mask 和 query
            mask = ~agent_padding_mask[i]  # [N_a], True 表示有效
            res_traj.append(traj[i][mask])
            res_conf.append(conf[i][mask])
            res_aux.append((vel[i][mask], None))

        return res_conf, res_traj, res_aux





class ModeSeqDecoderTwoStage(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.device = device
        self.two_stage = cfg["two_stage"]
        hidden_dim = cfg['d_embed']
        self.num_modes = cfg['g_num_modes']
        pred_len = cfg['g_pred_len']
        num_layers = cfg['n_decoder_layer']
        num_heads = cfg['n_decoder_head']
        dropout = cfg['dropout']
        cross_first = cfg['cross_first']

        # learnable 模态 token
        self.mode_queries = nn.Embedding(self.num_modes, embedding_dim=hidden_dim)  

        self.agent_pose_proj = nn.Linear(hidden_dim + 4, hidden_dim)
        self.lane_pose_proj = nn.Linear(hidden_dim + 4, hidden_dim)

        # transformer 层对 mode_query 与 agent_feat 做交互
        self.layers = nn.ModuleList([
            ModeSeqLayer(hidden_dim, num_heads, dropout, cross_first) for _ in range(num_layers)
        ])
        self.query_proj = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        if self.two_stage:
            # 阶段一：goal 解码器
            self.goal_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2)  # 输出 goal [x, y]
            )

            self.goal_offset = nn.Sequential(
                nn.Linear(hidden_dim+2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2)  # 输出 goal [x, y]
            )

            # 阶段二：trajectory 解码器（输入是 goal + agent_feat）
            self.traj_mlp = nn.Sequential(
                nn.Linear(hidden_dim+2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, (pred_len-1) * 2)
            )
            # 模态置信度预测
            self.conf_head = nn.Sequential(
                nn.Linear(hidden_dim + 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
                
            )

        else:
            self.traj_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2 * pred_len)
            )

            # 模态置信度预测
            self.conf_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )



    def forward(self, agent_feats, lane_feats, pose):
        """
        agent_feats: List([N_a, D]) 
        actor_idcs: 每个 batch 场景中的 agent id 索引
        lane_feats: List([N_l,D])
        pose: dict{agent_pose:([Na,4]), lane_pose:([Nl,4])}
        return: list of [N_i, K, T, 2], list of [N_i, K]
        """
        padded_agent_pose = pad_sequence(pose['agent_pose'], batch_first=True)
        padded_lane_pose = pad_sequence(pose['lane_pose'], batch_first=True)
        # agent_feats_padded: [B,N_max_a,D], mask: [B, N_max_a]
        agent_feats_padded, agent_padding_mask, lane_feats_padded, lane_padding_mask = pad_feats_and_create_mask(agent_feats,lane_feats)
        agent_feats_padded = self.agent_pose_proj(torch.cat([agent_feats_padded,padded_agent_pose],dim=-1))
        lane_feats_padded = self.lane_pose_proj(torch.cat([lane_feats_padded, padded_lane_pose],dim=-1))


        B = agent_feats_padded.shape[0]
        N_a = agent_feats_padded.shape[1]
        # 跨模态 attention 融合 agent features
        # -> [B, N_a * K, D]
        mode_queries = self.mode_queries.weight.view(1, 1, 
                                                     self.num_modes, -1).expand(B, N_a, self.num_modes, -1).reshape(B, N_a * self.num_modes, -1)
        mode_queries = torch.cat([agent_feats_padded.unsqueeze(2).expand(-1, -1, self.num_modes, -1).reshape(B, N_a * self.num_modes, -1), mode_queries],dim=-1)
        mode_queries = self.query_proj(mode_queries)
        for layer in self.layers:
            mode_queries = layer(mode_queries, agent_feats_padded, 
                                 lane_feats_padded, agent_padding_mask, lane_padding_mask)  # 每个模态 query 都 attend agent_feats
        mode_queries = mode_queries.view(B,N_a,self.num_modes,-1) # [B,N_a,K,D]
        if self.two_stage:
            # 预测每个模态的终点 goal
            goal = self.goal_head(mode_queries)  # [B, N_a, K, 2]
            offset = self.goal_offset(torch.cat([mode_queries, goal.detach()], dim=-1))
            goal = goal + offset

            # 拼接 goal 和 mode embedding 做轨迹预测
            goal_detached = goal.detach()  
            mode_cat = torch.cat([mode_queries, goal_detached], dim=-1)  # [B, N_a, K, D+2]
            traj = self.traj_mlp(mode_cat).view(mode_cat.size(0), mode_cat.size(1), mode_cat.size(2), -1, 2)  # [B, N_a, K, (T-1), 2]
            traj = torch.cat([traj, goal.unsqueeze(3)],dim=3) # [B, N_a, K, 30, 2]
            # 模态置信度
            conf = self.conf_head(mode_cat).squeeze(-1)  # [B, N_a, K]

        else:
            traj = self.traj_mlp(mode_queries).view(mode_queries.shape[0],mode_queries.shape[1],mode_queries.shape[2],-1,2) # [B, N_a, K, T, 2]
            conf = self.conf_head(mode_queries).squeeze(-1)  # [B, N_a, K]

        # conf = F.softmax(conf,dim=-1)
        res_traj, res_conf = [], []
        # 遍历每个 batch
        for i in range(traj.shape[0]):  # B
            # 取出第 i 个 batch 的 mask 和 query
            mask = ~agent_padding_mask[i]  # [N_a], True 表示有效
            res_traj.append(traj[i][mask])
            res_conf.append(conf[i][mask])

        return res_conf, res_traj


class ModeSeqLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, cross_first=False):
        """
        cross_first: 若为 True，则先做 cross-attn，再 self-attn
                     若为 False（默认），则先做 self-attn，再 cross-attn（与你原始的一致）
        """
        super().__init__()
        self.cross_first = cross_first

        self.mode_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.agent_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.lane_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )


        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)

    def forward(self,  mode_query, agent_feats, lane_feats, agent_padding_mask, lane_padding_mask):
        """
        mode_query: [B, K*N_a, D] - mode queries
        agent_feats: [B, N_a, D] - agent-level features
        lane_feats: [B, N_l, D]
        agent_padding_mask: [B,N_a]
        lane_padding_mask: [B,N_l]
        """
        
        B = mode_query.shape[0]
        if self.cross_first:
            # ==== 先做 cross-agent-attn ====
            # [B,N_a*K,D]
            agent_out, _ = self.agent_attn(query=mode_query, key=agent_feats, 
                                         value=agent_feats, key_padding_mask=agent_padding_mask)
            mode_query = self.norm1(mode_query + self.dropout(agent_out))

            # lane-agent-attn
            # [B,N_a*K,D]
            lane_out, _ = self.lane_attn(query=mode_query, key=lane_feats, 
                                         value=lane_feats, key_padding_mask=lane_padding_mask)
            mode_query = self.norm2(mode_query + self.dropout(lane_out))

            # mode-attn -> [B*N_a,K,D]
            mode_query = mode_query.view(-1, 6, mode_query.shape[-1])
            mode_out, _ = self.mode_attn(query=mode_query, key=mode_query, value=mode_query)
            x = self.norm3(mode_query + self.dropout(mode_out)).view(B, -1, mode_query.shape[-1])

        else:
            # ==== 先做 self-attn（原始方式）====
            attn_out, _ = self.mem_attn(query=x, key=x, value=x)
            x = self.norm1(x + self.dropout(attn_out))

            # 再做 cross-attn
            ctx_out, _ = self.ctx_attn(query=x, key=agent_feats, value=agent_feats)
            x = self.norm2(x + self.dropout(ctx_out))
        # FFN 层
        x = self.norm4(x + self.ffn(x))
        return x


class MLPDecoder(nn.Module):
    def __init__(self,
                 device,
                 config) -> None:
        super(MLPDecoder, self).__init__()
        self.device = device
        self.config = config
        self.hidden_size = config['d_embed']
        self.future_steps = config['g_pred_len']
        self.num_modes = config['g_num_modes']
        self.param_out = config['param_out']  # parametric output: bezier/monomial/none
        self.N_ORDER = config['param_order']
        self.out_prob = config['out_prob']
        dim_mm = self.hidden_size * self.num_modes
        dim_inter = dim_mm // 2
        self.multihead_proj = nn.Sequential(
            nn.Linear(self.hidden_size, dim_inter),
            nn.LayerNorm(dim_inter),
            nn.ReLU(inplace=True),
            nn.Linear(dim_inter, dim_mm),
            nn.LayerNorm(dim_mm),
            nn.ReLU(inplace=True)
        )

        self.cls = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1)
        )

        if self.param_out == 'bezier':
            self.mat_T = self._get_T_matrix_bezier(n_order=self.N_ORDER, n_step=self.future_steps).to(self.device)
            self.mat_Tp = self._get_Tp_matrix_bezier(n_order=self.N_ORDER, n_step=self.future_steps).to(self.device)

            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, (self.N_ORDER + 1) * 2)
            )
        elif self.param_out == 'monomial':
            self.mat_T = self._get_T_matrix_monomial(n_order=self.N_ORDER, n_step=self.future_steps).to(self.device)
            self.mat_Tp = self._get_Tp_matrix_monomial(n_order=self.N_ORDER, n_step=self.future_steps).to(self.device)

            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, (self.N_ORDER + 1) * 2)
            )
        elif self.param_out == 'none':
            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 2)
            )
        else:
            raise NotImplementedError

    def _get_T_matrix_bezier(self, n_order, n_step):
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        T = []
        for i in range(n_order + 1):
            coeff = math.comb(n_order, i) * (1.0 - ts)**(n_order - i) * ts**i
            T.append(coeff)
        return torch.Tensor(np.array(T).T)

    def _get_Tp_matrix_bezier(self, n_order, n_step):
        # ~ 1st derivatives
        # ! NOTICE: we multiply n_order inside of the Tp matrix
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        Tp = []
        for i in range(n_order):
            coeff = n_order * math.comb(n_order - 1, i) * (1.0 - ts)**(n_order - 1 - i) * ts**i
            Tp.append(coeff)
        return torch.Tensor(np.array(Tp).T)

    def _get_T_matrix_monomial(self, n_order, n_step):
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        T = []
        for i in range(n_order + 1):
            coeff = ts ** i
            T.append(coeff)
        return torch.Tensor(np.array(T).T)

    def _get_Tp_matrix_monomial(self, n_order, n_step):
        # ~ 1st derivatives
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        Tp = []
        for i in range(n_order):
            coeff = (i + 1) * (ts ** i)
            Tp.append(coeff)
        return torch.Tensor(np.array(Tp).T)

    def forward(self,
                embed: torch.Tensor,
                actor_idcs: List[Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # input embed: [159, 128]
        embed = self.multihead_proj(embed).view(-1, self.num_modes, self.hidden_size).permute(1, 0, 2)
        # print('embed: ', embed.shape)  # e.g., [6, 159, 128]

        cls = self.cls(embed).view(self.num_modes, -1).permute(1, 0)  # e.g., [159, 6]
        if self.out_prob:
            cls = F.softmax(cls * 1.0, dim=1)  # e.g., [159, 6]

        if self.param_out == 'bezier':
            param = self.reg(embed).view(self.num_modes, -1, self.N_ORDER + 1, 2)  # e.g., [6, 159, N_ORDER + 1, 2]
            param = param.permute(1, 0, 2, 3)  # e.g., [159, 6, N_ORDER + 1, 2]
            reg = torch.matmul(self.mat_T, param)  # e.g., [159, 6, 30, 2]
            vel = torch.matmul(self.mat_Tp, torch.diff(param, dim=2)) / (self.future_steps * 0.1)
        elif self.param_out == 'monomial':
            param = self.reg(embed).view(self.num_modes, -1, self.N_ORDER + 1, 2)  # e.g., [6, 159, N_ORDER + 1, 2]
            param = param.permute(1, 0, 2, 3)  # e.g., [159, 6, N_ORDER + 1, 2]
            reg = torch.matmul(self.mat_T, param)  # e.g., [159, 6, 30, 2]
            vel = torch.matmul(self.mat_Tp, param[:, :, 1:, :]) / (self.future_steps * 0.1)
        elif self.param_out == 'none':
            reg = self.reg(embed).view(self.num_modes, -1, self.future_steps, 2)  # e.g., [6, 159, 30, 2]
            reg = reg.permute(1, 0, 2, 3)  # e.g., [159, 6, 30, 2]
            vel = torch.gradient(reg, dim=-2)[0] / 0.1  # vel is calculated from pos

        # print('reg: ', reg.shape, 'cls: ', cls.shape)
        # de-batchify
        res_cls, res_reg, res_aux = [], [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            res_cls.append(cls[idcs])
            res_reg.append(reg[idcs])

            if self.param_out == 'none':
                res_aux.append((vel[idcs], None))  # ! None is a placeholder
            else:
                res_aux.append((vel[idcs], param[idcs]))  # List[Tuple[Tensor,...]]

        return res_cls, res_reg, res_aux


class Simpl(nn.Module):
    # Initialization
    def __init__(self, cfg, device):
        super(Simpl, self).__init__()
        self.device = device
        self.decoder_type = cfg["decoder_type"]

        self.actor_net = ActorNet(n_in=cfg['in_actor'],
                                  hidden_size=cfg['d_actor'],
                                  n_fpn_scale=cfg['n_fpn_scale'],
                                  num_layers=cfg['num_a2a_layer'],
                                  dropout=cfg['dropout'],
                                  d_rpe_in=cfg['rpe_type_d'],
                                  n_a2a_head=cfg['n_a2a_head'])

        # self.lane_net = LaneNet(device=self.device,
        #                         in_size=cfg['in_lane'],
        #                         hidden_size=cfg['d_lane'],
        #                         dropout=cfg['dropout'])
        self.lane_net = Point_RPE_MAP_Encoder(input_dim=cfg['in_lane'], d_model=cfg['d_lane'],
                                                  dropout=cfg['dropout'], num_layers=cfg['num_l2l_layer'],d_rpe_in=cfg['rpe_type_d'],n_l2l_head=cfg['n_l2l_head'])

        # self.fusion_net = FusionNet(device=self.device,
        #                             config=cfg)
        self.fusion_net = EdgeAwareGATFusion(device, cfg)

        if self.decoder_type == 'MLP':
            self.pred_net = MLPDecoder(device=self.device,
                                   config=cfg)
        elif self.decoder_type == 'QueryBased':
            self.pred_net = ModeSeqDecoderTwoStage(device, cfg)
        elif self.decoder_type == 'GlobalQueryRefine':
            self.pred_net = ModeQueryRefineDecoder(device, cfg)

        self.edge_type_embedding = nn.Embedding(num_embeddings=7, embedding_dim=cfg['edge_type_d'])
        self.out_prob = cfg['out_prob']  # 是否输出概率
        if cfg["init_weights"]:
            self.apply(init_weights)

    def forward(self, data):
        actors, actor_idcs, lanes, lane_idcs, rpes, nodes_of_lane, node_edge_index, pose = data

        # * 对边类型编码
        rpes = self.fuse_rpe_with_edge_type_embed_and_edges(rpes)
        # * actors/lanes encoding
        actors,a2a_attr = self.actor_net(actors, rpes)  # output: [N_{actor}, 128]

        lanes, l2l_attr = self.lane_net(lanes, nodes_of_lane, rpes)  # output: [N_{lane}, 128]
        # * fusion
        # actors,_ = self.fusion_net(actors, actor_idcs, lanes, lane_idcs)
        actors_list, lanes_list = self.fusion_net(actors, actor_idcs, lanes, lane_idcs, rpes, a2a_attr, l2l_attr)
        # * decoding
        # out = self.pred_net(actors_list, lanes_list, pose)
        if self.decoder_type == 'MLP':
            actors = torch.cat(actors_list,dim=0)
            out = self.pred_net(actors, actor_idcs)
        elif self.decoder_type == 'QueryBased':
            out = self.pred_net(actors_list, lanes_list, pose)
        elif self.decoder_type == 'GlobalQueryRefine':
            out = self.pred_net(actors_list, lanes_list, pose)

        return out
    
    def fuse_rpe_with_edge_type_embed_and_edges(self, rpes_dict):
        '''
            rpes_dict['a2a_edges']          # [2, E_a2a]
            rpes_dict['a2a_rpes']           # [E_a2a, 5]
            rpes_dict['a2a_onehot']         # [E_a2a, 7] 

            rpes_dict['l2l_edges']        
            rpes_dict['l2l_rpes'] 
            rpes_dict['l2l_onehot'] 

            rpes_dict['a2a_fusion_edges']
            rpes_dict['a2l_l2a_fusion_edges]
            rpes_dict['l2l_fusion_edges']
            rpes_dict['a2l_l2a_rpes'] 
            rpes_dict['a2l_l2a_onehot'] 
        '''
        for key in list(rpes_dict.keys()):
            if key.endswith('_rpes'):
                prefix = key[:-5]  # 去掉 "_rpes"，比如 a2a_rpes -> a2a
                onehot_key = f"{prefix}_onehot"
                fused_key = f"{prefix}_fused_rpes"

                if onehot_key not in rpes_dict:
                    continue  # 没有 onehot，就跳过

                rpe = rpes_dict[key]                # [E, rpe_dim]
                onehot = rpes_dict[onehot_key]      # [E, num_edge_types]
                edge_type_idx = onehot.argmax(dim=-1)  # [E]
                edge_embed = self.edge_type_embedding(edge_type_idx)  # [E, embed_dim]

                fused_rpe = torch.cat([rpe, edge_embed], dim=-1)  # [E, rpe_dim + embed_dim]
                rpes_dict[fused_key] = fused_rpe  # 新增到原字典
        return rpes_dict

    def pre_process(self, data):
        '''
            Send to device
            'BATCH_SIZE', 'SEQ_ID', 'CITY_NAME',
            'ORIG', 'ROT',
            'TRAJS_OBS', 'TRAJS_FUT', 'PAD_OBS', 'PAD_FUT', 'TRAJS_CTRS', 'TRAJS_VECS',
            'LANE_GRAPH',
            'RPE',
            'ACTORS', 'ACTOR_IDCS', 'LANES', 'LANE_IDCS'
        '''
        actors = gpu(data['ACTORS'], self.device)
        actor_idcs = gpu(data['ACTOR_IDCS'], self.device)
        lanes = gpu(data['LANES'], self.device)
        lane_idcs = gpu(data['LANE_IDCS'], self.device)
        rpe = to_long(gpu(data['RPE'], self.device))
        nodes_of_lane = to_long(gpu(data['NODES_OF_LANES'],self.device))
        node_edge_index = to_long(gpu(['NODE_EDGES_INDEX'], self.device))
        pose = gpu(data['POSE'], self.device)

        return (actors, actor_idcs, lanes, lane_idcs, rpe, nodes_of_lane, node_edge_index, pose)

    def post_process(self, out):
        post_out = dict()
        res_cls = out[0]
        res_reg = out[1]

        # get prediction results for target vehicles only
        reg = torch.stack([trajs[0] for trajs in res_reg], dim=0)
        cls = torch.stack([probs[0] for probs in res_cls], dim=0)
        if not self.out_prob:   # 解码器输出不是概率
            cls = F.softmax(cls, dim=1)
        post_out['out_raw'] = out
        post_out['traj_pred'] = reg  # batch x n_mod x pred_len x 2
        post_out['prob_pred'] = cls  # batch x n_mod

        return post_out
