from datetime import datetime
import argparse
from torch.utils.data import DataLoader
import torch
from typing import Any
import faulthandler
import time
from tqdm import tqdm
from utils.utils import AverageMeterForDict
import os
from loader import Loader
from utils.logger import Logger
from utils.utils import set_seed, AverageMeterForDict
import logging
import sys
import numpy as np

def parse_arguments() -> Any:

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", required=True, default="data_df/features/", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Val batch size")
    parser.add_argument("--train_epoches", type=int, default=30, help="Number of epoches for training")
    parser.add_argument("--val_interval", type=int, default=2, help="Validation intervals")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation") 
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--logger_writer", action="store_true", help="Enable tensorboard")
    parser.add_argument("--adv_cfg_path", required=True, default="", type=str)
    parser.add_argument("--rank_metric", required=False, type=str, default="minade_k", help="Ranking metric")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--no_pbar", action="store_true", help="Hide progress bar")
    parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")
    return parser.parse_args()


import optuna

def objective(trial, args):

    # 初始化设备和种子
    set_seed(args.seed)

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")


    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "opt/"+ date_str
    logger = Logger(date_str=date_str, log_dir=logdir,
                    enable_flags={'writer': args.logger_writer})
    logger.log_basics(args=args, datetime=date_str)
    loader = Loader(args, device, is_ddp=False, logger=logger )
    #* 设置优化参数
    # loader.adv_cfg.net_cfg['dropout'] = trial.suggest_float("dropout", 0.1, 0.4)
    loader.adv_cfg.net_cfg['num_l2l_layer'] = trial.suggest_int("num_l2l_layer", 2, 6)
    loader.adv_cfg.net_cfg['n_scene_layer'] = trial.suggest_int("n_scene_layer", 2, 6)
    loader.adv_cfg.net_cfg['n_decoder_layer'] = trial.suggest_int("n_decoder_layer", 2, 6)
    # lr = trial.suggest_float("lr", 5e-4, 2e-3, log=True)
    loader.adv_cfg.opt_cfg['values'][1] = 8e-4
    loader.adv_cfg.opt_cfg['values'][2] = 8e-4



    (train_set, val_set), net, loss_fn, optimizer, evaluator = loader.load()


    dl_train = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=4, shuffle=True,
                          collate_fn=train_set.collate_fn, drop_last=True, pin_memory=True)
     

    dl_val = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=4,
                        collate_fn=val_set.collate_fn, drop_last=True, pin_memory=True)

    best_metric = 1e6
    rank_metric = args.rank_metric
    for epoch in range(args.train_epoches):

        torch.cuda.empty_cache()   
        torch.cuda.reset_peak_memory_stats()
        # * Train

        train_loss_meter = AverageMeterForDict()
        net.train()
        for i, data in enumerate(tqdm(dl_train, disable=args.no_pbar, ncols=80)):
            data_in = net.pre_process(data)
            out = net(data_in)
            loss_out = loss_fn(out, data)

            optimizer.zero_grad()
            loss_out['loss'].backward()
            lr = optimizer.step()

            train_loss_meter.update(loss_out)

        logger.print(f"Epoch {epoch+1}/{args.train_epoches}, Train Loss: {train_loss_meter.metrics['loss'].avg:.4f}, LR: {lr:.6f}")
        
        optimizer.step_scheduler()
        # 验证
        with torch.no_grad():
            net.eval()
            val_eval_meter = AverageMeterForDict()
            val_loss_meter = AverageMeterForDict()
            for i, data in enumerate(dl_val):
                data_in = net.pre_process(data)
                out = net(data_in)
                loss_out = loss_fn(out, data)
                
                post_out = net.post_process(out)
                eval_out = evaluator.evaluate(post_out, data)

                # val_loss_meter.update(loss_out)
                val_eval_meter.update(eval_out, len(data))
                val_loss_meter.update(loss_out, len(data))
            logger.print('[Validation] Avg. loss: {:.6}'.format(
                    val_loss_meter.metrics['loss'].avg, ))
            logger.print('-- ' + val_eval_meter.get_info())

            metric = val_eval_meter.metrics[rank_metric].avg
            best_metric = min(best_metric, metric)
            trial.report(best_metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    logger.print(f"[Trial {trial.number}] Metric {rank_metric}: {best_metric:.4f}, Params: {trial.params}")
    trial.set_user_attr(args.rank_metric, best_metric)
    return best_metric  # Optuna默认是最小化目标函数

if __name__ == "__main__":

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    storage_path = "sqlite:///opt/optuna_study.db"
    study_name = "simple_av1_opt"
    
    study = optuna.create_study(
        direction="minimize",
        storage=storage_path,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    args = parse_arguments()
    study.optimize(lambda trial: objective(trial, args), n_trials=20)

    print("Best trial:")
    print("  Value (metric):", study.best_trial.value)
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")