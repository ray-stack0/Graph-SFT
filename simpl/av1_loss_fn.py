from typing import Any, Dict, List, Tuple, Union
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import gpu, to_long

def compute_ade(reg, gt_preds, has_preds):
    """
    Args:
        reg: Tensor [N, K, T, 2] - predicted trajectories
        gt_preds: Tensor [N, T, 2] - ground truth
        has_preds: Bool Tensor [N, T] - mask of valid GT

    Returns:
        ade: Tensor [N, K] - ADE per agent per mode
    """
    N, K, T, _ = reg.shape

    # Expand gt and mask to [N, K, T, 2] 和 [N, K, T]
    gt_exp = gt_preds.unsqueeze(1).expand(N, K, T, 2)          # [N, K, T, 2]
    mask_exp = has_preds.unsqueeze(1).expand(N, K, T)          # [N, K, T]

    # L2 distance
    l2_dist = torch.norm(reg - gt_exp, dim=-1)                 # [N, K, T]

    # Mask invalid steps
    l2_dist = l2_dist * mask_exp                               # [N, K, T]

    # Sum and normalize
    valid_counts = mask_exp.sum(dim=-1).clamp(min=1)           # [N, K]
    ade = l2_dist.sum(dim=-1) / valid_counts                   # [N, K]

    return ade



class LossFunc(nn.Module):
    def __init__(self, config, device):
        super(LossFunc, self).__init__()
        self.config = config
        self.device = device
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")
        self.pred_len = config["g_pred_len"]
        self.cur_epoch = None
        self.use_aWTA = config['use_aWTA']
        self.use_aWTA_cls = config['use_aWTA_cls']
        if self.use_aWTA:
            self.init_temperature_reg = config["init_temperature_reg"]
            self.exp_base_reg = config["exp_base_reg"]
        if self.use_aWTA_cls:
            self.init_temperature_cls = config["init_temperature_cls"]
            self.exp_base_cls = config["exp_base_cls"]
            self.cls_mode = config["cls_mode"] 


    def forward(self, out, data, cur_epoch = None):
        # print('TRAJS_FUT: ', len(data["TRAJS_FUT"]), data["TRAJS_FUT"][0].shape)
        # print('PAD_FUT: ', len(data["PAD_FUT"]), data["PAD_FUT"][0].shape)
        # print('out: ', out[1][0].shape, out[0][0].shape)
        self.cur_epoch = cur_epoch
        loss_out = self.pred_loss(out,
                                  gpu(data["TRAJS_FUT"], self.device),
                                  to_long(gpu(data["PAD_FUT"], self.device)))
        loss_out["loss"] = sum([x for x in loss_out.values()])
        return loss_out

    def compute_diversity_loss_euclidean(
    reg: torch.Tensor,           # [N, K, T, 2]
    min_dist: torch.Tensor,      # [N]
    threshold: float = 2.0,
    temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        多样性损失函数,基于欧几里得距离。reg 是轨迹序列 (x, y)，单位为 meter。
        """
        device = reg.device
        N, K, T, _ = reg.shape
        div_weights = torch.clamp((threshold - min_dist) / threshold, min=0.0, max=1.0)  # [N]
        valid_mask = div_weights > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        reg_valid = reg[valid_mask]          # [N', K, T, 2]
        weights_valid = div_weights[valid_mask]  # [N']
        N_valid = reg_valid.shape[0]
        reg_flat = reg_valid.view(N_valid, K, -1)  # [N', K, T*2]

        # pairwise distances
        diff = reg_flat.unsqueeze(2) - reg_flat.unsqueeze(1)     # [N', K, K, T*2]
        dist_matrix = torch.norm(diff, dim=-1)                   # [N', K, K]
        dist_matrix = dist_matrix / temperature                  # 调整梯度 scale

        # 去除对角线
        eye = torch.eye(K, device=device).unsqueeze(0)
        dist_matrix = dist_matrix * (1 - eye)

        diversity = dist_matrix.sum(dim=(1, 2)) / (K * (K - 1))  # [N']
        loss = (-weights_valid * diversity).sum() / (weights_valid.sum() + 1e-6)
        return loss

    def compute_cls_loss(self, logits: torch.Tensor, dist: torch.Tensor, 
                         last_idcs: torch.Tensor, mode: str = "fde_kl") -> torch.Tensor:
        device = logits.device
        cls_th = self.config.get("cls_th", 2.0)
        temperature = max(self.temperature_scheduler(scheduler_type='cls'), 1.0)  # 温度不要小于 1

        if mode == "fde_kl":
            min_dist = dist.min(dim=1).values
            min_dist_mask = (min_dist < cls_th)
            if min_dist_mask.sum() == 0:
                return torch.tensor(0.0, device=device)
            dist_sel = dist[min_dist_mask]
            logits_sel = logits[min_dist_mask]
        elif mode == "ade_kl":
            dist_sel = dist
            logits_sel = logits
        else:
            raise ValueError(f"Unsupported mode '{mode}'")

        with torch.no_grad():
            soft_label = F.softmax(-dist_sel / temperature, dim=1)  # 不再手动加 log，避免数值不稳定
            traj_len = (last_idcs + 1).float()
            len_weight = traj_len/traj_len.mean()  # [N]

        # 注意 F.kl_div 要求 logits_sel 已经是 log_softmax
        log_probs = F.log_softmax(logits_sel, dim=1)
        kl_per_sample = F.kl_div(log_probs, soft_label, reduction='none').sum(dim=1)  # N
        cls_loss = (kl_per_sample * len_weight).mean()
        return cls_loss



    def pred_loss(self, out: Dict[str, List[torch.Tensor]], gt_preds: List[torch.Tensor], pad_flags: List[torch.Tensor]):
        cls = out[0]
        reg = out[1]
        # cls = torch.cat([x[0:2] for x in cls], 0)
        # reg = torch.cat([x[0:2] for x in reg], 0)
        # gt_preds = torch.cat([x[0:2] for x in gt_preds], 0)
        # has_preds = torch.cat([x[0:2] for x in pad_flags], 0).bool()
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in pad_flags], 0).bool()

        loss_out = dict()
        num_modes = self.config["g_num_modes"]
        num_preds = self.pred_len
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(self.device) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]             # [N,K]
        reg = reg[mask]             # [N,K,T,2]
        gt_preds = gt_preds[mask]   # [N,T,2]
        has_preds = has_preds[mask] # [N,T]
        last_idcs = last_idcs[mask] # [N]

        _reg = reg[..., 0:2].clone()  # for WTA strategy, in case of (5-dim) prob output

        row_idcs = torch.arange(len(last_idcs)).long().to(self.device)
        fde = []
        for j in range(num_modes):
            fde.append(
                torch.sqrt(
                    (
                        (_reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        #* fde
        fde = torch.cat([x.unsqueeze(1) for x in fde], 1) #fde, fde
        #* ade
        ade = compute_ade(reg, gt_preds, has_preds) # [N,K], ade
        #* cls
        if not self.use_aWTA_cls:
            min_dist, min_idcs = fde.min(1)
            cls = F.softmax(cls,dim=1) # [N,K]
            mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
            mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
            mask1 = fde - min_dist.view(-1, 1) > self.config["cls_ignore"]
            mgn = mgn[mask0 * mask1]
            mask = mgn < self.config["mgn"]
            num_cls = mask.sum().item()
            cls_loss = (self.config["mgn"] * mask.sum() - mgn[mask].sum()) / (num_cls + 1e-10)
        else:
            if self.cls_mode == 'ade_kl':
                cls_loss = self.compute_cls_loss(cls, ade, last_idcs, self.cls_mode)
            else:
                cls_loss = self.compute_cls_loss(cls, fde, last_idcs, self.cls_mode)

        loss_out["cls_loss"] = self.config["cls_coef"] * cls_loss

        #* reg
        if self.use_aWTA:
            loss_out["reg_loss"] = self.config["reg_coef"] * self.awta_loss(ade)
        else:
            reg = reg[row_idcs, min_idcs]
            num_reg = has_preds.sum().item()
            reg_loss = self.reg_loss(reg[has_preds], gt_preds[has_preds]) / (num_reg + 1e-10)
            loss_out["reg_loss"] = self.config["reg_coef"] * reg_loss

        return loss_out
    
    def awta_loss(self, distance):
        '''
        prediction: predicted forecasts, of shape [batch, hypotheses, timesteps, 2]
        gt: ground-truth forecasting trajectory, of shape [batch, timesteps, 2]
        gt_valid_mask: ground-truth forecasting mask indicating the valid future steps, of shape [batch, timesteps]
        '''
        cur_temperature = self.temperature_scheduler(scheduler_type="reg")

        # calculate the weights q(t): softmin of the distance, controlled by the current temperature
        awta_weights = torch.softmax(-1.0*distance/cur_temperature, dim=-1).detach() # [N, K], 
        
        # weight the distance by awta weights
        loss_reg = distance * awta_weights # [N, K]
        return loss_reg.sum(-1).mean() # sum over weighted hypotheses and average over the batch
    
    def temperature_scheduler(self,scheduler_type = "reg"):
        '''
            init_temperature: initial temperature
            cur_epoch: current number of epochs
            exp_base: exponential scheduler base    
        '''
        if scheduler_type == "reg":
            return self.init_temperature_reg*self.exp_base_reg**self.cur_epoch
        elif scheduler_type == "cls":
            return self.init_temperature_cls*self.exp_base_cls**self.cur_epoch
        else:
            raise ValueError(f"Unsupported scheduler_type: {scheduler_type}. Use 'reg' or 'cls'.")
    
    def print(self):
        print('\nloss_fn config:', self.config)
        print('loss_fn device:', self.device)
        
