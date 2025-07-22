import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import numpy as np
import faulthandler
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from loader import Loader
from utils.utils import AverageMeter, AverageMeterForDict

import pandas as pd


from av2.datasets.motion_forecasting import scenario_serialization

from pathlib import Path

def restore_trajectories(trajs,trajs_ctrs, trajs_vecs, orig, rot):
    # trajs: [batch,num_mode,num_step,2]
    # trajs_ctrs: List([num_agent,2])   trajs_vecs: List([num_agent,2])
    # orig: List[2]     rot: List([2,2])
    trajs_restored = []
    
    # 1.还原局部坐标系
    for traj, ctr, vec , seq_rot, seq_orig in zip(trajs.unsqueeze(1), trajs_ctrs, trajs_vecs, rot, orig):
        # 得到每一个seq的数据
        agents_trajs = []
        for agent_traj,agent_ctr,agent_vec in zip(traj,ctr,vec):
            agent_traj = agent_traj.clone().cpu().numpy()
            theta = np.arctan2(agent_vec[1], agent_vec[0])
            act_rot_inv = np.array([[np.cos(theta), np.sin(theta)],  # 顺时针旋转矩阵
                                    [-np.sin(theta), np.cos(theta)]])
            # 还原局部坐标系下的轨迹
            agent_trajs = np.matmul(agent_traj,act_rot_inv) + agent_ctr.numpy()
            agents_trajs.append(agent_trajs)
        seq_trajs = np.stack(agents_trajs,axis=0)
        # 第二次旋转平移
        rot_inv = np.linalg.inv(seq_rot)
        seq_trajs = np.matmul(seq_trajs,rot_inv) + seq_orig.numpy()
        trajs_restored.append(seq_trajs)
    #     print(seq_trajs.shape)
    # print(len(trajs_restored))
    return trajs_restored

def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="val", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", default="data_argo/features/", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=1, help="Val batch size")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--adv_cfg_path", default="config.simpl_cfg", type=str)
    parser.add_argument("--model_path",default="saved_models/20250306-151346_gat_ckpt_epoch33.tar" ,type=str, help="path to the saved model")
    return parser.parse_args()


def main():
    args = parse_arguments()
    print('Args: {}\n'.format(args))

    faulthandler.enable()
    # for av2
    # data_dir = os.path.join("/home/nvidia/ltp/Dataset/AV2", args.mode)
    data_dir = os.path.join("/host_home/AV2", args.mode)
    date_set = 'av1'
    if "av2" in args.features_dir:
        date_set = 'av2'

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    if not args.model_path.endswith(".tar"):
        assert False, "Model path error - '{}'".format(args.model_path)

    loader = Loader(args, device, is_ddp=False)
    print('[Resume] Loading state_dict from {}'.format(args.model_path))
    loader.set_resmue(args.model_path)
    test_set, net, loss_fn, _, evaluator = loader.load()

    dl_val = DataLoader(test_set,
                        batch_size=args.val_batch_size,
                        shuffle=False,
                        num_workers=32,
                        collate_fn=test_set.collate_fn,
                        drop_last=False,
                        pin_memory=True)

    net.eval()
    preds = {}
    probabilities = {}
    with torch.no_grad():
        # * Validation
        test_start = time.time()

        for i, data in enumerate(tqdm(dl_val)):
            # obs_trajectory = data['TRAJS_OBS'] List([num_agent,num_obs_steps,2]),相对坐标
            data_in = net.pre_process(data)
            out = net(data_in)
            post_out = net.post_process(out)
            # List{[1,6,30,2]}
            

            if date_set == 'av1':
                restored_traj = restore_trajectories(post_out['traj_pred'],data['TRAJS_CTRS'],data['TRAJS_VECS'],data['ORIG'],data['ROT'])
                for argo_idx, pred_traj, pred_prob in zip(data['SEQ_ID'], restored_traj, post_out['prob_pred']):
                    preds[argo_idx] = pred_traj.squeeze()
                    x = pred_prob.squeeze().detach().cpu().numpy()
                    probabilities[argo_idx] = x


            elif date_set == 'av2':
                data['TRAJS_CTRS'] = [x["TRAJS_CTRS"]  for x in data['TRAJS']]
                data['TRAJS_VECS'] = [x["TRAJS_VECS"]  for x in data['TRAJS']]
                restored_traj = restore_trajectories(post_out['traj_pred'],data['TRAJS_CTRS'],data['TRAJS_VECS'],data['ORIG'],data['ROT'])
                for argo_idx, pred_traj, pred_prob in zip(data['SEQ_ID'], restored_traj, post_out['prob_pred']):

                    seq_path = os.path.join(data_dir, argo_idx)
                    scenario_path = Path(seq_path + f"/scenario_{argo_idx}.parquet")
                    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
                    track_id = scenario.focal_track_id

                    pred_traj = pred_traj.squeeze()
                    pred_prob = pred_prob.squeeze().detach().cpu().numpy()


                    preds[argo_idx] = {track_id : (pred_traj, pred_prob)} # 
        
                # print(f"index {argo_idx} trajs shape: {preds[argo_idx].shape}")
                # print(f"index {argo_idx} prob shape: {x.shape}")
                # print(preds[argo_idx])
            # if i == 10:
            #     break
        print('\nTest set finish, cost {:.2f} secs'.format(time.time() - test_start))
    
    if date_set == 'av1':
        from argoverse.evaluation.competition_util import generate_forecasting_h5
        file_path = 'results'
        folder_name = os.path.basename(os.path.dirname(args.model_path))  # '20250618-083330'
        file_stem = os.path.splitext(os.path.basename(args.model_path))[0]  # 'Simpl_ddp_best'
        abs_path = os.path.join(file_path,folder_name, file_stem)
        
        generate_forecasting_h5(preds, abs_path, probabilities=probabilities)
    elif date_set == 'av2':
        from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
                # 1. 构造 ChallengeSubmission 对象
        submission = ChallengeSubmission(preds)

        # 2. 保存为 parquet 文件（可提交）
        submission.to_parquet(Path("submission_av2.parquet"))
    print('\nExit...')


if __name__ == "__main__":
    main()
