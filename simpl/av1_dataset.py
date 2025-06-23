import os
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
#
import numpy as np
import pandas as pd
#
import torch
from torch.utils.data import Dataset
#
from utils.utils import from_numpy
import torch.nn.functional as F



class ArgoDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 mode: str,
                 obs_len: int = 20,
                 pred_len: int = 30,
                 aug: bool = False,
                 verbose: bool = False):
        self.mode = mode
        self.aug = aug
        self.verbose = verbose

        self.dataset_files = []
        self.dataset_len = -1
        self.prepare_dataset(dataset_dir)

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

        self.use_gnn_fusion = True
        self.l2a_dist_th = 50

        if self.verbose:
            print('[Dataset] Dataset Info:')
            print('-- mode: ', self.mode)
            print('-- total frames: ', self.dataset_len)
            print('-- obs_len: ', self.obs_len)
            print('-- pred_len: ', self.pred_len)
            print('-- seq_len: ', self.seq_len)
            print('-- aug: ', self.aug)

    def prepare_dataset(self, feat_path):
        if self.verbose:
            print("[Dataset] preparing {}".format(feat_path))

        if isinstance(feat_path, list):
            for path in feat_path:
                sequences = os.listdir(path)
                sequences = sorted(sequences)
                for seq in sequences:
                    file_path = f"{path}/{seq}"
                    self.dataset_files.append(file_path)
        else:
            sequences = os.listdir(feat_path)
            sequences = sorted(sequences)
            for seq in sequences:
                file_path = f"{feat_path}/{seq}"
                self.dataset_files.append(file_path)

        self.dataset_len = len(self.dataset_files)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        df = pd.read_pickle(self.dataset_files[idx])
        '''
            "SEQ_ID", "CITY_NAME",
            "ORIG", "ROT",
            "TIMESTAMP", "TRAJS", "TRAJS_CTRS", "TRAJS_VECS", "PAD_FLAGS",
            "LANE_GRAPH"
        '''

        data = self.data_augmentation(df)

        seq_id = data['SEQ_ID']
        city_name = data['CITY_NAME']
        orig = data['ORIG']
        rot = data['ROT']

        # timestamp = data['TIMESTAMP']
        trajs = data['TRAJS']
        trajs_obs = trajs[:, :self.obs_len]
        trajs_fut = trajs[:, self.obs_len:]

        pad_flags = data['PAD_FLAGS']
        pad_obs = pad_flags[:, :self.obs_len]
        pad_fut = pad_flags[:, self.obs_len:]

        trajs_ctrs = data['TRAJS_CTRS']
        trajs_vecs = data['TRAJS_VECS']

        graph = data['LANE_GRAPH']
        # for k, v in graph.items():
        #     print(k, type(v), v.shape if type(v) == np.ndarray else [])
        '''
            'node_ctrs'         (164, 10, 2)
            'node_vecs'         (164, 10, 2)
            'turn'              (164, 10, 2)
            'control'           (164, 10)
            'intersect'         (164, 10)
            'left'              (164, 10)
            'right'             (164, 10)
            'lane_ctrs'         (164, 2)
            'lane_vecs'         (164, 2)
            'num_nodes'         1640
            'num_lanes'         164
            'rel_lane_flags'    (164,)
        '''

        lane_ctrs = graph['lane_ctrs']
        lane_vecs = graph['lane_vecs']

        # ~ calc rpe
        scene_ctrs = torch.cat([torch.from_numpy(trajs_ctrs), torch.from_numpy(lane_ctrs)], dim=0)
        scene_vecs = torch.cat([torch.from_numpy(trajs_vecs), torch.from_numpy(lane_vecs)], dim=0)
        if self.use_gnn_fusion:
            rpes = self._get_rpe(scene_ctrs, scene_vecs, trajs_ctrs.shape[0], graph)
        else:
            rpes = self._get_rpe(scene_ctrs, scene_vecs)

        data = {}
        data['SEQ_ID'] = seq_id
        data['CITY_NAME'] = city_name
        data['ORIG'] = orig
        data['ROT'] = rot
        data['TRAJS_OBS'] = trajs_obs
        data['TRAJS_FUT'] = trajs_fut
        data['PAD_OBS'] = pad_obs
        data['PAD_FUT'] = pad_fut
        data['TRAJS_CTRS'] = trajs_ctrs
        data['TRAJS_VECS'] = trajs_vecs
        data['LANE_GRAPH'] = graph
        data['RPE'] = rpes

        return data

    def _get_cos(self, v1, v2):
        ''' input: [M, N, 2], [M, N, 2]
            output: [M, N]
            cos(<a,b>) = (a dot b) / |a||b|
        '''
        v1_norm = v1.norm(dim=-1)
        v2_norm = v2.norm(dim=-1)
        v1_x, v1_y = v1[..., 0], v1[..., 1]
        v2_x, v2_y = v2[..., 0], v2[..., 1]
        cos_dang = (v1_x * v2_x + v1_y * v2_y) / (v1_norm * v2_norm + 1e-10)
        return cos_dang

    def _get_sin(self, v1, v2):
        ''' input: [M, N, 2], [M, N, 2]
            output: [M, N]
            sin(<a,b>) = (a x b) / |a||b|
        '''
        v1_norm = v1.norm(dim=-1)
        v2_norm = v2.norm(dim=-1)
        v1_x, v1_y = v1[..., 0], v1[..., 1]
        v2_x, v2_y = v2[..., 0], v2[..., 1]
        sin_dang = (v1_x * v2_y - v1_y * v2_x) / (v1_norm * v2_norm + 1e-10)
        return sin_dang

    def _get_rpe(self, ctrs, vecs, num_actor = None, lane_graph = None, radius=100.0):
        if self.use_gnn_fusion:
            diff = ctrs.unsqueeze(0) - ctrs.unsqueeze(1)
            d_pos = diff.norm(dim=-1)
            pos_rpe = d_pos * 2 / radius  # scale [0, radius] to [0, 2]

            # angle diff
            cos_a1 = self._get_cos(vecs.unsqueeze(0), vecs.unsqueeze(1))
            sin_a1 = self._get_sin(vecs.unsqueeze(0), vecs.unsqueeze(1))
            cos_a2 = self._get_cos(vecs.unsqueeze(0), diff)
            sin_a2 = self._get_sin(vecs.unsqueeze(0), diff)

            rpes = torch.stack([cos_a1, sin_a1, cos_a2, sin_a2, pos_rpe], dim=-1) # [N,N,5]

            #* l2a 提取连接边
            dist_mask = pos_rpe < (2 * self.l2a_dist_th / radius)
            src, tgt = dist_mask.nonzero(as_tuple=True)
            is_lane_src = src >= num_actor
            is_lane_tgt = tgt >= num_actor
            lane_lane_mask = ~(is_lane_src & is_lane_tgt)  # 只保留非 lane↔lane 的边

            l2a_edges = torch.stack([src[lane_lane_mask], tgt[lane_lane_mask]], dim=0).to(torch.int64) # [2,num_edges]
            l2a_rpes = rpes[l2a_edges[0],l2a_edges[1]]

            #* l2l_encoder 提取车道建模时的连接边

            pre_pairs = torch.as_tensor(lane_graph['pre_pairs'], dtype= torch.int64)
            suc_pairs = torch.as_tensor(lane_graph['suc_pairs'], dtype= torch.int64)
            left_pairs = torch.as_tensor(lane_graph['left_pairs'], dtype= torch.int64)
            right_pairs = torch.as_tensor(lane_graph['right_pairs'], dtype= torch.int64)  # [2, num_edges]

            self_loop_nodes = torch.arange(0, (ctrs.shape[0] - num_actor))
            self_paris = torch.stack([self_loop_nodes, self_loop_nodes], dim=0)  # shape: [2, num_self_loops]

            l2l_encoder_edges = torch.cat([pre_pairs, suc_pairs, left_pairs, right_pairs, self_paris], dim=1)
            edge_type = torch.cat([
                torch.full((pre_pairs.shape[1],), 0, dtype=torch.long),
                torch.full((suc_pairs.shape[1],), 1, dtype=torch.long),
                torch.full((left_pairs.shape[1],), 2, dtype=torch.long),
                torch.full((right_pairs.shape[1],), 3, dtype=torch.long),
                torch.full((self_paris.shape[1],), 4, dtype=torch.long),
            ], dim=0)

            l2l_encoder_rpes = self.concat_rpe_with_type_encoding_torch(rpes, l2l_encoder_edges + num_actor, edge_type) # [2, num_l2l_edges]

            #* l2l fusion
            l2l_edges = torch.cat([pre_pairs, suc_pairs, left_pairs, right_pairs], dim=1) + num_actor
            edge_type = torch.cat([
                torch.full((pre_pairs.shape[1],), 0, dtype=torch.long),
                torch.full((suc_pairs.shape[1],), 1, dtype=torch.long),
                torch.full((left_pairs.shape[1],), 2, dtype=torch.long),
                torch.full((right_pairs.shape[1],), 3, dtype=torch.long)
            ], dim=0)
            l2l_rpes = self.concat_rpe_with_type_encoding_torch(rpes, l2l_edges, edge_type)


            rpes_dict = dict()
            rpes_dict['l2a_edges'] = l2a_edges  # [2,num_edges]
            rpes_dict['l2a_rpes'] = l2a_rpes    # [num_edges, 5
            rpes_dict['l2l_encoder_edges'] = l2l_encoder_edges
            rpes_dict['l2l_encoder_rpes'] = l2l_encoder_rpes
            rpes_dict['l2l_edges'] = l2l_edges
            rpes_dict['l2l_rpes'] = l2l_rpes

            return  rpes_dict

        else:
            # distance encoding
            d_pos = (ctrs.unsqueeze(0) - ctrs.unsqueeze(1)).norm(dim=-1)
            # if False:
            #     mask = d_pos >= radius
            #     d_pos = d_pos * 2 / radius  # scale [0, radius] to [0, 2]
            #     pos_rpe = []
            #     for l_pos in range(10):
            #         pos_rpe.append(torch.sin(2**l_pos * math.pi * d_pos))
            #         pos_rpe.append(torch.cos(2**l_pos * math.pi * d_pos))
            #     # print('pos rpe: ', [x.shape for x in pos_rpe])
            #     pos_rpe = torch.stack(pos_rpe)
            #     # print('pos_rpe: ', pos_rpe.shape)
            # else:
            mask = None
            d_pos = d_pos * 2 / radius  # scale [0, radius] to [0, 2]
            pos_rpe = d_pos.unsqueeze(0)

            # angle diff
            cos_a1 = self._get_cos(vecs.unsqueeze(0), vecs.unsqueeze(1))
            sin_a1 = self._get_sin(vecs.unsqueeze(0), vecs.unsqueeze(1))
            # print('cos_a1: ', cos_a1.shape, 'sin_a1: ', sin_a1.shape)

            v_pos = ctrs.unsqueeze(0) - ctrs.unsqueeze(1)
            cos_a2 = self._get_cos(vecs.unsqueeze(0), v_pos)
            sin_a2 = self._get_sin(vecs.unsqueeze(0), v_pos)
            # print('cos_a2: ', cos_a2.shape, 'sin_a2: ', sin_a2.shape)

            ang_rpe = torch.stack([cos_a1, sin_a1, cos_a2, sin_a2])
            rpe = torch.cat([ang_rpe, pos_rpe], dim=0)
            rpes_dict = dict()
            rpes_dict['scene'] = rpe    # [5, N, N]
            rpes_dict['scene_mask'] = mask
            return rpes_dict

    def concat_rpe_with_type_encoding_torch(self, rpe_matrix, edge_index, edge_type=None, num_edge_types=4):
        """
        将原始 RPE 特征与边类型的 One-Hot 编码拼接（PyTorch 版本）。

        参数:
            rpe_matrix: torch.Tensor, shape [num_nodes, num_nodes, rpe_dim]
            edge_index: torch.Tensor, shape [2, num_edges]
            edge_type: torch.Tensor or None, shape [num_edges], 类型编号为 0~num_edge_types-1

        返回:
            rpe_with_type: torch.Tensor, shape [num_edges, rpe_dim + num_edge_types]
        """
        assert edge_index.shape[0] == 2, "edge_index 应该是 shape [2, num_edges]"

        row_idx, col_idx = edge_index[0], edge_index[1]  # 起点、终点索引

        # 从 RPE 矩阵中取出对应边的特征: [num_edges, rpe_dim]
        rpe = rpe_matrix[row_idx, col_idx]

        # one-hot 编码
        if edge_type is None:
            one_hot = torch.zeros(rpe.size(0), num_edge_types, device=rpe.device, dtype=rpe.dtype)
        else:
            valid_mask = (edge_type >= 0) & (edge_type < num_edge_types)
            one_hot = torch.zeros(rpe.size(0), num_edge_types, device=rpe.device, dtype=rpe.dtype)
            one_hot[valid_mask] = F.one_hot(edge_type[valid_mask], num_classes=num_edge_types).to(rpe.dtype)

        # 拼接
        rpe_with_type = torch.cat([rpe, one_hot], dim=-1)  # shape: [num_edges, rpe_dim + num_edge_types]

        return rpe_with_type

    def collate_fn(self, batch: List[Any]) -> Dict[str, Any]:
        batch = from_numpy(batch)
        data = dict()
        data['BATCH_SIZE'] = len(batch)
        # Batching by use a list for non-fixed size
        for key in batch[0].keys():
            data[key] = [x[key] for x in batch]
        '''
            Keys:
            'BATCH_SIZE', 'SEQ_ID', 'CITY_NAME',
            'ORIG', 'ROT',
            'TRAJS_OBS', 'PAD_OBS', 'TRAJS_FUT', 'PAD_FUT', 'TRAJS_CTRS', 'TRAJS_VECS',
            'LANE_GRAPH',
            'RPE'
        '''

        actors, actor_idcs = self.actor_gather(data['BATCH_SIZE'], data['TRAJS_OBS'], data['PAD_OBS'])
        lanes, lane_idcs, nodes_of_lane, node_edge_index = self.graph_gather(data['BATCH_SIZE'], data["LANE_GRAPH"])
        if self.use_gnn_fusion:
            rpes = self.rpe_gather(data['RPE'], lane_idcs, actor_idcs)
            data['RPE'] = rpes

        data['ACTORS'] = actors
        data['ACTOR_IDCS'] = actor_idcs
        data['LANES'] = lanes
        data['LANE_IDCS'] = lane_idcs
        data['NODES_OF_LANES'] = nodes_of_lane
        data['NODE_EDGES_INDEX'] = node_edge_index
        return data

    def rpe_gather(self, rpes_dicts, lane_idcs, actor_idcs):
        '''
            rpes_dict['l2a_edges'] = l2a_edges  # [2,num_edges]
            rpes_dict['l2a_rpes'] = l2a_rpes    # [num_edges, 5
            rpes_dict['l2l_encoder_edges'] = l2l_encoder_edges
            rpes_dict['l2l_encoder_rpes'] = l2l_encoder_rpes
            rpes_dict['l2l_edges'] = l2l_edges
            rpes_dict['l2l_rpes'] = l2l_rpes
        '''
        token_count = 0
        lane_count = 0
        for i in range(len(rpes_dicts)):
            rpes_dicts[i]['l2a_edges'] = rpes_dicts[i]['l2a_edges'] + token_count
            rpes_dicts[i]['l2l_edges'] = rpes_dicts[i]['l2l_edges'] + token_count
            rpes_dicts[i]['l2l_encoder_edges'] = rpes_dicts[i]['l2l_encoder_edges'] + lane_count

            token_count += (len(lane_idcs[i]) + len(actor_idcs[i]))
            lane_count += len(lane_idcs[i])

        rpes_dict = dict()
        for key in rpes_dicts[0].keys():
            vals = [x[key] for x in rpes_dicts if x[key].numel() > 0]
            if len(vals) == 0:
                continue
            if 'rpes' in key:
                rpes_dict[key] = torch.cat(vals, dim=0)
            else:
                rpes_dict[key] = torch.cat(vals, dim=1)

        return rpes_dict



    def actor_gather(self, batch_size, actors, pad_flags):
        num_actors = [len(x) for x in actors]

        act_feats = []
        for i in range(batch_size):
            # actors[i] (N_a, 20, 2)
            # pad_flags[i] (N_a, 20)
            vel = torch.zeros_like(actors[i])
            vel[:, 1:, :] = actors[i][:, 1:, :] - actors[i][:, :-1, :]
            act_feats.append(torch.cat([vel, pad_flags[i].unsqueeze(2)], dim=2))
        act_feats = [x.transpose(1, 2) for x in act_feats]
        actors = torch.cat(act_feats, 0)  # [N_a, feat_len, 20], N_a is agent number in a batch
        actor_idcs = []  # e.g. [tensor([0, 1, 2, 3]), tensor([ 4,  5,  6,  7,  8,  9, 10])]
        count = 0
        for i in range(batch_size):
            idcs = torch.arange(count, count + num_actors[i])
            actor_idcs.append(idcs)
            count += num_actors[i]
        return actors, actor_idcs

    def graph_gather(self, batch_size, graphs):
        '''
            graphs[i]
                node_ctrs           torch.Size([116, 10, 2])
                node_vecs           torch.Size([116, 10, 2])
                turn                torch.Size([116, 10, 2])
                control             torch.Size([116, 10])
                intersect           torch.Size([116, 10])
                left                torch.Size([116, 10])
                right               torch.Size([116, 10])
                lane_ctrs           torch.Size([116, 2])
                lane_vecs           torch.Size([116, 2])
                num_nodes           1160
                num_lanes           116
        '''
        lane_idcs = list()
        lane_count = 0
        node_count = 0
        for i in range(batch_size):
            l_idcs = torch.arange(lane_count, lane_count + graphs[i]["num_lanes"])
            lane_idcs.append(l_idcs)

            for key in ["pre", "suc"]:
                graphs[i][key] = graphs[i][key].type(torch.int32) + torch.tensor(node_count, dtype=torch.int32)
            graphs[i]["nodes_of_lane"] = graphs[i]["nodes_of_lane"] + lane_count

            lane_count = lane_count + graphs[i]["num_lanes"]
            node_count = node_count + graphs[i]["num_nodes"]
        # print('lane_idcs: ', lane_idcs)

        graph = dict()
        for key in ["node_ctrs", "node_vecs", "turn", "control", "intersect", "left", "right", "nodes_of_lane"]:
            graph[key] = torch.cat([x[key] for x in graphs], 0)
            # print(key, graph[key].shape)
        # for key in ["lane_ctrs", "lane_vecs"]:
        #     graph[key] = [x[key] for x in graphs]
            # print(key, [x.shape for x in graph[key]])
        graph['node_edge_index'] = dict()
        for key in ["pre", "suc"]:
            graph['node_edge_index'][key] = torch.cat([x[key] for x in graphs], 1)

        lanes = torch.cat([graph['node_ctrs'],
                           graph['node_vecs'],
                           graph['turn'],
                           graph['control'].unsqueeze(1),
                           graph['intersect'].unsqueeze(1),
                           graph['left'].unsqueeze(1),
                           graph['right'].unsqueeze(1)], dim=-1) # [N_{nodes}, F]

        return lanes, lane_idcs, graph['nodes_of_lane'], graph['node_edge_index']

    # def rpe_gather(self, rpes):
    #     rpe = dict()
    #     for key in list(rpes[0].keys()):
    #         rpe[key] = [x[key] for x in rpes]
    #     return rpe

    def data_augmentation(self, df):
        '''
            "SEQ_ID", "CITY_NAME", "ORIG", "ROT",
            "TIMESTAMP", "TRAJS", "TRAJS_CTRS", "TRAJS_VECS", "PAD_FLAGS", "LANE_GRAPH"

            "node_ctrs", "node_vecs",
            "turn", "control", "intersect", "left", "right"
            "lane_ctrs", "lane_vecs"
            "num_nodes", "num_lanes", "node_idcs", "lane_idcs"
        '''

        data = {}
        for key in list(df.keys()):
            data[key] = df[key].values[0]

        is_aug = random.choices([True, False], weights=[0.3, 0.7])[0]
        if not (self.aug and is_aug):
            return data

        # ~ random vertical flip
        data['TRAJS_CTRS'][..., 1] *= -1
        data['TRAJS_VECS'][..., 1] *= -1
        data['TRAJS'][..., 1] *= -1

        data['LANE_GRAPH']['lane_ctrs'][..., 1] *= -1
        data['LANE_GRAPH']['lane_vecs'][..., 1] *= -1
        data['LANE_GRAPH']['node_ctrs'][..., 1] *= -1
        data['LANE_GRAPH']['node_vecs'][..., 1] *= -1
        data['LANE_GRAPH']['left'], data['LANE_GRAPH']['right'] = data['LANE_GRAPH']['right'], data['LANE_GRAPH']['left']

        return data
