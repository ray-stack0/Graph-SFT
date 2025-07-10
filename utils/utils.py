import imp
import os
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
#
import numpy as np
#
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from thop import profile
from thop import clever_format
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def remove_thop_attributes(model):
    for name, module in model.named_modules():
        for attr in ["total_ops", "total_params"]:
            if hasattr(module, attr):
                delattr(module, attr)

def get_pf(model, input):
    print("calculate flops and params:\n")
    model.eval()  # 切换到 eval 模式，防止 Dropout / BN 出错
    with torch.no_grad():  # 禁用 autograd
        macs, params = profile(model, inputs=(input,))
        macs, params = clever_format([macs, params], "%.3f")
        wrapped_model = WrappedModel(model)
        flops = FlopCountAnalysis(wrapped_model, input)
        remove_thop_attributes(model)
    return macs, flops.total(), params, parameter_count_table(model)

class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):  # 解包 tuple
        return self.model(inputs)  # 保持原模型习惯接收一个 tuple

def pad_feats_and_create_mask(agent_feats, lane_feats):
    """
    Args:
        agent_feats: List[Tensor[N_a_i, D]]
        lane_feats:  List[Tensor[N_l_i, D]]
    Returns:
        agent_feats_padded: [B, N_max_a, D]
        agent_padding_mask: [B, N_max_a]  # True 表示被mask掉
        lane_feats_padded:  [B, N_max_l, D]
        lane_padding_mask:  [B, N_max_l]
    """
    B = len(agent_feats)
    device = agent_feats[0].device
    D = agent_feats[0].size(1)

    # [N_a_i, D] -> pad to [N_max, D], then stack -> [B, N_max, D]
    agent_feats_padded = pad_sequence(agent_feats, batch_first=True)  # [B, N_max_a, D]
    lane_feats_padded  = pad_sequence(lane_feats, batch_first=True)   # [B, N_max_l, D]

    # 创建 mask，True 表示被 mask（即 padding）
    agent_padding_mask = torch.ones(agent_feats_padded.shape[:2], dtype=torch.bool, device=device)
    lane_padding_mask  = torch.ones(lane_feats_padded.shape[:2], dtype=torch.bool, device=device)

    for i, (a, l) in enumerate(zip(agent_feats, lane_feats)):
        agent_padding_mask[i, :a.size(0)] = False
        lane_padding_mask[i, :l.size(0)]  = False

    return agent_feats_padded, agent_padding_mask, lane_feats_padded, lane_padding_mask

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data


def gpu(data, device):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x, device=device) for x in data]
    elif isinstance(data, dict):
        data = {key: gpu(_data, device=device) for key, _data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().to(device, non_blocking=True)
    return data


def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data


def str2bool(v):
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_tensor_memory(data):
    return data.element_size() * data.nelement() / 1024 / 1024


def save_ckpt(net, opt, epoch, save_dir, filename, best_metric=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict(), 
         "scheduler_state": opt.scheduler.state_dict(), "best_metric": best_metric},
        os.path.join(save_dir, filename),
    )


def load_pretrain(net, pretrain_dict):
    state_dict = net.state_dict()
    for key in pretrain_dict.keys():
        if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
            value = pretrain_dict[key]
            if not isinstance(value, torch.Tensor):
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)


def distributed_mean(tensor):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.stack(output_tensors, dim=0)
    return concat.mean(0)


def distributed_concat(tensor):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, (nn.GRU, nn.GRUCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().item()
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterForDict(object):
    def __init__(self):
        self.reset()       # __init__():reset parameters

    def reset(self):
        self.metrics = {}

    def update(self, elem, n=1):
        for key, val in elem.items():
            if not key in self.metrics:
                self.metrics[key] = AverageMeter()

            self.metrics[key].update(val, n)

    def get_info(self):
        info = ''
        for key, elem in self.metrics.items():
            info += "{}: {:.3f} ".format(key, elem.avg)
        return info

    def print(self):
        info = self.get_info()
        print('-- ' + info)
    
    def get_avg_dict(self):
        avg_dict = {}
        for key, elem in self.metrics.items():
            avg_dict[key] = elem.avg
        return avg_dict


class ScalarMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = []

    def push(self, val):
        self.vals.append(val)

    def mean(self):
        return np.mean(self.vals)

    def max(self):
        return np.max(self.vals)

    def min(self):
        return np.min(self.vals)


class ScalarMeterForDict(object):
    def __init__(self):
        self.reset()       # __init__():reset parameters

    def reset(self):
        self.metrics = {}

    def push(self, elem):
        for key, val in elem.items():
            if not key in self.metrics:
                self.metrics[key] = ScalarMeter()

            self.metrics[key].push(val)

    def get_info(self):
        info = ''
        for key, elem in self.metrics.items():
            info += "{}: [{:.3f} {:.3f} {:.3f}] ".format(key, elem.min(), elem.mean(), elem.max())
        return info

    def print(self):
        info = self.get_info()
        print('-- ' + info)
