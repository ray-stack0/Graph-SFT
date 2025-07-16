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
import swanlab
from utils.utils import get_pf
import os

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
    parser.add_argument("--is_main_thread", action="store_true")
    return parser.parse_args()


import optuna

def objective(trial, args):

    # base_port = 29500  # 你可以改为 29500 等常见起始端口
    # port = base_port + trial.number % args.train_epoches  # 防止太大
    # os.environ["MASTER_PORT"] = str(port)
    # 初始化设备和种子
    set_seed(args.seed)


    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    patience = 5
    no_improve_count = 0

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "opt/"+ date_str
    logger = Logger(date_str=date_str, log_dir=logdir,
                    enable_flags={'writer': args.logger_writer})
    logger.log_basics(args=args, datetime=date_str)
    loader = Loader(args, device, is_ddp=False, logger=logger )
    #* 设置优化参数
    # loader.adv_cfg.net_cfg['dropout'] = trial.suggest_float("dropout", 0.10, 0.35, step=0.05)
    # lr_peak = trial.suggest_float("lr", 5e-4, 1.5e-3, log=True)
    # loader.adv_cfg.opt_cfg['values'][1] = lr_peak
    # loader.adv_cfg.opt_cfg['values'][2] = lr_peak

    # loader.adv_cfg.net_cfg['num_l2l_layer'] = trial.suggest_int("num_l2l_layer", 1, 3)
    # loader.adv_cfg.net_cfg['num_a2a_layer'] = trial.suggest_int("num_a2a_layer", 1, 3)
    # loader.adv_cfg.net_cfg['n_scene_layer'] = trial.suggest_int("n_scene_layer", 3, 6)
    # loader.adv_cfg.net_cfg['n_decoder_layer'] = trial.suggest_int("n_decoder_layer", 1, 3)
    loader.adv_cfg.loss_cfg['exp_base_reg'] = trial.suggest_float("exp_base_reg", 0.6, 0.9, log = True)
    loader.adv_cfg.loss_cfg['init_temperature_reg'] = trial.suggest_int("init_temperature_reg", 4, 10)
    
    adv_cfg = loader.adv_cfg.get_all_dict()
    args_dict = vars(args)
    merged_dict = {
            "args": args_dict,
            "adv_cfg": adv_cfg,
            "opt_params": trial.params}
    swanlab.init(
        project="SIMPL-Baseline-opt",
        config=merged_dict,
        mode='cloud',
        experiment_name=f"trial_{trial.number}")
    (train_set, val_set), net, loss_fn, optimizer, evaluator = loader.load()


    dl_train = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=8, shuffle=True,
                          collate_fn=train_set.collate_fn, drop_last=True, pin_memory=True)
     

    dl_val = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=8,
                        collate_fn=val_set.collate_fn, drop_last=True, pin_memory=True)
    
    #* 记录FLOPs
    batch = next(iter(dl_val))
    input = net.pre_process(batch)
    macs, flops, params, parameter_count_table= get_pf(net, input)
    logger.print(f"[Model Stats]")
    logger.print(f"  - MACs:      {macs}")
    logger.print(f"  - Parameters:{params}")
    logger.print(f"  - FLOPs:     {flops / 1e9:.2f}G")
    logger.print("  - Parameter breakdown:\n" + parameter_count_table)

    logger.print(f"\n[Trial {trial.number}] Start with parameters:")
    for k, v in trial.params.items():
        logger.print(f"  - {k}: {v}")
    best_metric = 1e6
    rank_metric = args.rank_metric
    try:
        for epoch in range(args.train_epoches):

            torch.cuda.empty_cache()   
            torch.cuda.reset_peak_memory_stats()
            # * Train
            train_loss_meter = AverageMeterForDict()
            net.train()
            for i, data in enumerate(tqdm(dl_train, disable=args.no_pbar, ncols=80)):
                data_in = net.pre_process(data)
                out = net(data_in)
                loss_out = loss_fn(out, data, epoch)

                optimizer.zero_grad()
                loss_out['loss'].backward()
                lr = optimizer.step()

                train_loss_meter.update(loss_out)

            logger.print(f"Epoch {epoch+1}/{args.train_epoches}, Train Loss: {train_loss_meter.metrics['loss'].avg:.4f}, LR: {lr:.6f}")
            swanlab.log({"train/loss/": train_loss_meter.get_avg_dict()}, step=epoch)
            optimizer.step_scheduler()
            # 验证
            with torch.no_grad():
                net.eval()
                val_eval_meter = AverageMeterForDict()
                val_loss_meter = AverageMeterForDict()
                for i, data in enumerate(tqdm(dl_val, disable=args.no_pbar, ncols=80)):
                    data_in = net.pre_process(data)
                    out = net(data_in)
                    loss_out = loss_fn(out, data, epoch)
                    
                    post_out = net.post_process(out)
                    eval_out = evaluator.evaluate(post_out, data)

                    # val_loss_meter.update(loss_out)
                    val_eval_meter.update(eval_out, len(data))
                    val_loss_meter.update(loss_out, len(data))
                logger.print('[Validation] Avg. loss: {:.6}'.format(
                        val_loss_meter.metrics['loss'].avg, ))
                logger.print('-- ' + val_eval_meter.get_info())
                # 记录信息
                swanlab.log({"val/metric/": val_eval_meter.get_avg_dict()}, step=epoch)
                swanlab.log({"val/loss/": val_loss_meter.get_avg_dict()}, step=epoch)

                metric = val_eval_meter.metrics[rank_metric].avg
                if metric < best_metric:
                    best_metric = metric
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                trial.report(best_metric, epoch)
                if trial.should_prune():
                    print(f"Trial {trial.number} is pruned by Optuna scheduler (trial.should_prune()).")
                    raise optuna.TrialPruned()

                if no_improve_count >= patience:
                    print(f"Trial {trial.number} is pruned due to early stopping (no improvement for {patience} epochs).")
                    raise optuna.TrialPruned()

                if metric > 3:
                    print(f"Trial {trial.number} is pruned due to poor metric performance (metric={metric} > 3).")
                    raise optuna.TrialPruned()

        logger.print(f"[Trial {trial.number}] Metric {rank_metric}: {best_metric:.4f}, Params: {trial.params}")
        trial.set_user_attr(args.rank_metric, best_metric)
        return best_metric  # Optuna默认是最小化目标函数
    finally:
        swanlab.finish()

if __name__ == "__main__":

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    os.makedirs("./opt", exist_ok=True)
    storage_path = "sqlite:///opt/init_temperature_exp_reg.db"
    study_name = "init_temperature_exp_reg"
    
    study = optuna.create_study(
        direction="minimize",
        storage=storage_path,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )

    args = parse_arguments()
    if args.is_main_thread:
        study.enqueue_trial({
            "exp_base_reg": 0.834,
            # "n_scene_layer": 4,
            # "num_l2l_layer": 2,
            # "num_a2a_layer": 2,
            # "n_decoder_layer": 3,
            "init_temperature_reg": 8
        })
        

    study.optimize(lambda trial: objective(trial, args), n_trials=20, n_jobs=1)
    
    print("Best trial:")
    print("Value (metric):", study.best_trial.value)
    print("Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")