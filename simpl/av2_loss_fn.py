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

        self.yaw_loss = config['yaw_loss']
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out, data, cur_epoch=None):
        self.cur_epoch = cur_epoch
        traj_fut = [x['TRAJS_POS_FUT'] for x in data["TRAJS"]]
        traj_fut = gpu(traj_fut, device=self.device)

        pad_fut = [x['PAD_FUT'] for x in data["TRAJS"]]
        pad_fut = to_long(gpu(pad_fut, device=self.device))

        cls, reg, aux = out

        train_mask = [x["TRAIN_MASK"] for x in data["TRAJS"]]
        train_mask = gpu(train_mask, device=self.device)
        # print('train_mask:', [x.shape for x in train_mask])
        # print('whitelist num: ', [x.sum().item() for x in train_mask])

        cls = [x[train_mask[i]] for i, x in enumerate(cls)]
        reg = [x[train_mask[i]] for i, x in enumerate(reg)]
        traj_fut = [x[train_mask[i]] for i, x in enumerate(traj_fut)]
        pad_fut = [x[train_mask[i]] for i, x in enumerate(pad_fut)]


        if self.yaw_loss:
            # yaw angle GT
            ang_fut = [x['TRAJS_ANG_FUT'] for x in data["TRAJS"]]
            ang_fut = gpu(ang_fut, device=self.device)
            # for yaw loss
            yaw_loss_mask = gpu([x["YAW_LOSS_MASK"] for x in data["TRAJS"]], device=self.device)
            # collect aux info
            vel = [x[0] for x in aux]
            # apply train mask
            vel = [x[train_mask[i]] for i, x in enumerate(vel)]
            ang_fut = [x[train_mask[i]] for i, x in enumerate(ang_fut)]
            yaw_loss_mask = [x[train_mask[i]] for i, x in enumerate(yaw_loss_mask)]

            loss_out = self.pred_loss_with_yaw(cls, reg, vel, traj_fut, ang_fut, pad_fut, yaw_loss_mask)
            loss_out["loss"] = loss_out["cls_loss"] + loss_out["reg_loss"] + loss_out["yaw_loss"]
        else:
            loss_out = self.pred_loss(cls, reg, traj_fut, pad_fut)
            loss_out["loss"] = loss_out["cls_loss"] + loss_out["reg_loss"]

        return loss_out

    def pred_loss_with_yaw(self,
                           cls: List[torch.Tensor],
                           reg: List[torch.Tensor],
                           vel: List[torch.Tensor],
                           gt_preds: List[torch.Tensor],
                           gt_ang: List[torch.Tensor],
                           pad_flags: List[torch.Tensor],
                           yaw_flags: List[torch.Tensor]):
        cls = torch.cat([x for x in cls], dim=0)                     # [98, 6]
        reg = torch.cat([x for x in reg], dim=0)                     # [98, 6, 60, 2]
        vel = torch.cat([x for x in vel], dim=0)                     # [98, 6, 60, 2]
        gt_preds = torch.cat([x for x in gt_preds], dim=0)           # [98, 60, 2]
        gt_ang = torch.cat([x for x in gt_ang], dim=0)               # [98, 60, 2]
        has_preds = torch.cat([x for x in pad_flags], dim=0).bool()  # [98, 60]
        has_yaw = torch.cat([x for x in yaw_flags], dim=0).bool()    # [98]

        loss_out = dict()
        num_modes = self.config["g_num_modes"]
        num_preds = self.config["g_pred_len"]
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(self.device) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        vel = vel[mask]
        gt_preds = gt_preds[mask]
        gt_ang = gt_ang[mask]
        has_preds = has_preds[mask]
        has_yaw = has_yaw[mask]
        last_idcs = last_idcs[mask]

        _reg = reg[..., 0:2].clone()  # for WTA strategy

        row_idcs = torch.arange(len(last_idcs)).long().to(self.device)
        dist = []
        for j in range(num_modes):
            dist.append(
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
            # cls = F.softmax(cls,dim=1) # [N,K]
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

        # ~ yaw loss
        vel = vel[row_idcs, min_idcs]  # select the best mode, keep identical to reg

        _has_preds = has_preds[has_yaw].view(-1)
        _v1 = vel[has_yaw].view(-1, 2)[_has_preds]
        _v2 = gt_ang[has_yaw].view(-1, 2)[_has_preds]
        # print('_has_preds: ', _has_preds.shape)
        # print('_v1: ', _v1.shape)
        # print('_v2: ', _v2.shape)
        # ang diff loss use cosine similarity
        cos_sim = torch.cosine_similarity(_v1, _v2)  # [-1, 1]
        # print('cos_sim: ', cos_sim.shape, cos_sim[:100])
        loss_out["yaw_loss"] = ((1 - cos_sim) / 2).mean()  # [0, 1]

        return loss_out

    def pred_loss(self,
                  cls: List[torch.Tensor],
                  reg: List[torch.Tensor],
                  gt_preds: List[torch.Tensor],
                  pad_flags: List[torch.Tensor]):
        cls = torch.cat([x for x in cls], 0)                        # [98, 6]
        reg = torch.cat([x for x in reg], 0)                        # [98, 6, 60, 2]
        gt_preds = torch.cat([x for x in gt_preds], 0)              # [98, 60, 2]
        has_preds = torch.cat([x for x in pad_flags], 0).bool()     # [98, 60]

        loss_out = dict()
        num_modes = self.config["g_num_modes"]
        num_preds = self.config["g_pred_len"]
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(self.device) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        _reg = reg[..., 0:2].clone()  # for WTA strategy

        row_idcs = torch.arange(len(last_idcs)).long().to(self.device)
        dist = []
        for j in range(num_modes):
            dist.append(
                torch.sqrt(
                    (
                        (_reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        #* fde
        fde = torch.cat([x.unsqueeze(1) for x in dist], 1)
        #* ade
        ade = compute_ade(reg, gt_preds, has_preds) # [N,K], ade
        #* cls
        if not self.use_aWTA_cls:
            min_dist, min_idcs = dist.min(1)
            row_idcs = torch.arange(len(min_idcs)).long().to(self.device)

            mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
            mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
            mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
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

    def print(self):
        print('\nloss_fn config:', self.config)
        print('loss_fn device:', self.device)

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
