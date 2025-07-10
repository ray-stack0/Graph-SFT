import os
import sys


class AdvCfg():
    def __init__(self, is_ddp=False):
        self.g_cfg = dict()
        self.g_cfg['g_num_modes'] = 6
        self.g_cfg['g_obs_len'] = 50
        self.g_cfg['g_pred_len'] = 60

        #* dataset config
        self.data_cfg = dict()
        self.data_cfg['dataset'] = "simpl.av2_dataset:AV2Dataset"

        #* network config
        self.net_cfg = dict()
        self.net_cfg["network"] = "simpl.simpl:Simpl"
        self.net_cfg["init_weights"] = False
        self.net_cfg["in_actor"] = 14
        self.net_cfg["d_actor"] = 128
        self.net_cfg["num_a2a_layer"] = 3 # 编码的层数
        self.net_cfg["n_fpn_scale"] = 4
        self.net_cfg["in_lane"] = 16
        self.net_cfg["d_lane"] = 128
        self.net_cfg["num_l2l_layer"] = 3 # 编码的层数
        self.net_cfg["edge_type_d"] = 5
        self.net_cfg["rpe_type_d"] = 5 + self.net_cfg["edge_type_d"]

        self.net_cfg["token_fuse_mode"] = 'res'
        self.net_cfg["d_rpe_in"] = 5
        self.net_cfg["d_rpe"] = 128
        self.net_cfg["d_embed"] = 128
        self.net_cfg["n_scene_layer"] = 4
        self.net_cfg["n_scene_head"] = 8
        self.net_cfg['use_diff_mha'] = False
        self.net_cfg["dropout"] = 0.1
        self.net_cfg["update_edge"] = True
        self.net_cfg["use_nnconv"] = False  # 交互建模阶段是否采用NNConv
        self.net_cfg["use_fusion_gate"] = False
        self.net_cfg["use_SwiGLU_fnn"] = False
        
        self.net_cfg["decoder_type"] = "GlobalQueryRefine"
        self.net_cfg["d_pos"] = 4  # 位置编码的维度
        if self.net_cfg["decoder_type"] == "MLP":
            self.net_cfg["param_out"] = 'none'  # bezier/monomial/none
            self.net_cfg["param_order"] = 5     # 5-th order polynomials
        elif self.net_cfg["decoder_type"] == "QueryBased":
            self.net_cfg["cross_first"] = True
            self.net_cfg["two_stage"] = False
            self.net_cfg["n_decoder_layer"] = 3
            self.net_cfg["n_decoder_head"] = 8
        elif self.net_cfg["decoder_type"] == "GlobalQueryRefine":
            self.net_cfg["n_decoder_head"] = 8

       
        #* loss config
        self.loss_cfg = dict()
        self.loss_cfg["loss_fn"] = "simpl.av2_loss_fn:LossFunc"
        self.loss_cfg["cls_coef"] = 0.1
        self.loss_cfg["reg_coef"] = 0.9
        self.loss_cfg["mgn"] = 0.2
        self.loss_cfg["cls_th"] = 2.0
        self.loss_cfg["cls_ignore"] = 0.2
        self.loss_cfg["yaw_loss"] = False
        self.loss_cfg["use_aWTA"] = True
        self.loss_cfg["use_aWTA_cls"] = True
        if self.loss_cfg["use_aWTA_cls"]:
            self.loss_cfg["exp_base_cls"] = 0.834
            self.loss_cfg["init_temperature_cls"] = 8
            self.loss_cfg["cls_mode"] = "ade_kl" # ade_kl/fde_kl
        if self.loss_cfg['use_aWTA']:
            self.loss_cfg["exp_base_reg"] = 0.834
            self.loss_cfg["init_temperature_reg"] = 8

        #* optimizer config
        opt_cfg = dict()
        opt_cfg['opt'] = 'adam'
        opt_cfg['weight_decay'] = 0.0
        opt_cfg['lr_scale_func'] = 'none'  # none/sqrt/linear

        # scheduler
        self.opt_cfg = dict()
        self.opt_cfg['scheduler'] = 'polyline'

        if self.opt_cfg['scheduler'] == 'cosine':
            self.opt_cfg['init_lr'] = 6e-4
            self.opt_cfg['T_max'] = 50
            self.opt_cfg['eta_min'] = 1e-5
        elif self.opt_cfg['scheduler'] == 'cosine_warmup':
            self.opt_cfg['init_lr'] = 1e-3
            self.opt_cfg['T_max'] = 50
            self.opt_cfg['eta_min'] = 1e-4
            self.opt_cfg['T_warmup'] = 5
        elif self.opt_cfg['scheduler'] == 'step':
            self.opt_cfg['init_lr'] = 1e-3
            self.opt_cfg['step_size'] = 40
            self.opt_cfg['gamma'] = 0.1
        elif self.opt_cfg['scheduler'] == 'polyline':
            self.opt_cfg['init_lr'] = 1e-4
            self.opt_cfg['milestones'] = [0, 5, 35, 40]
            self.opt_cfg['values'] = [1e-4, 1e-3, 1e-3, 1e-4]

        # * eval config
        self.eval_cfg = dict()
        self.eval_cfg['evaluator'] = 'utils.evaluator:TrajPredictionEvaluator'
        self.eval_cfg['data_ver'] = 'av2'
        self.eval_cfg['miss_thres'] = 2.0


        

    def get_dataset_cfg(self):
        self.data_cfg.update(self.g_cfg)  # append global config
        return self.data_cfg

    def get_net_cfg(self):
        self.net_cfg.update(self.g_cfg)  # append global config
        return self.net_cfg

    def get_loss_cfg(self):
        self.loss_cfg.update(self.g_cfg)  # append global config
        return self.loss_cfg

    def get_opt_cfg(self):
        self.opt_cfg.update(self.g_cfg)  # append global config
        return self.opt_cfg

    def get_eval_cfg(self):
        self.eval_cfg.update(self.g_cfg)  # append global config
        return self.eval_cfg
