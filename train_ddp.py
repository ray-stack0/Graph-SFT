import os
import sys
import time
import copy
import subprocess
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import faulthandler
from tqdm import tqdm
import numpy as np
#
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#
from loader import Loader
from utils.logger import Logger
from utils.utils import AverageMeterForDict
from utils.utils import save_ckpt, set_seed, distributed_mean
import yaml
import swanlab

def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Val batch size")
    parser.add_argument("--train_epoches", type=int, default=10, help="Number of epoches for training")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation intervals")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--logger_writer", action="store_true", help="Enable tensorboard")
    parser.add_argument("--adv_cfg_path", required=True, default="", type=str)
    parser.add_argument("--rank_metric", required=False, type=str, default="brier_fde_k", help="Ranking metric")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--no_pbar", action="store_true", help="Hide progress bar")
    parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")
    parser.add_argument("--experiment_name", required=False, type=str, default="test")
    return parser.parse_args()


def main():
    args = parse_arguments()
    faulthandler.enable()
    start_time = time.time()

    local_rank = int(os.environ['LOCAL_RANK'])
    set_seed(args.seed + local_rank)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()

    is_main = True if local_rank == 0 else False

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log/" + date_str
    save_dir = os.path.join("saved_models",date_str)
    logger = Logger(date_str=date_str, enable=is_main, log_dir=log_dir,
                    enable_flags={'writer': args.logger_writer, 'mailbot': False})
    logger.print(args)
    # log basic info
    logger.log_basics(args=args, datetime=date_str)
    
    epoch_now = 0
    loader = Loader(args, device, is_ddp=True, world_size=world_size, local_rank=local_rank, verbose=is_main, logger=logger)
    if args.resume:
        logger.print('[Resume] Loading state_dict from {}'.format(args.model_path))
        loader.set_resmue(args.model_path)
        epoch_now = loader.get_last_epoch() + 1
    (train_set, val_set), net, loss_fn, optimizer, evaluator = loader.load()

    #* 启动swanlab
    if is_main and args.logger_writer:

        adv_cfg = loader.adv_cfg.get_all_dict()
        args_dict = vars(args)
        merged_dict = {
            "args": args_dict,
            "adv_cfg": adv_cfg}
        cfg_path = os.path.join(log_dir, 'config.yaml')

        with open(cfg_path, 'w') as file:
            yaml.dump(merged_dict, file, default_flow_style=False)
        swanlab.init(
        project="SIMPL-Baseline",
        config=merged_dict,
        mode='cloud',
        experiment_name=args.experiment_name)
    
    #* dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    dl_train = DataLoader(train_set,
                          batch_size=args.train_batch_size,
                          num_workers=20,
                          prefetch_factor=2,
                          collate_fn=train_set.collate_fn,
                          drop_last=True,
                          sampler=train_sampler,
                          pin_memory=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        num_workers=16,
                        prefetch_factor=2,
                        collate_fn=val_set.collate_fn,
                        drop_last=True,
                        sampler=val_sampler,
                        pin_memory=True)

    best_metric = loader.get_best_metric()
    rank_metric = args.rank_metric
    net_name = loader.network_name()

    for epoch in range(epoch_now, args.train_epoches):
        dist.barrier()  # sync
        logger.print('\nEpoch {}'.format(epoch))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # * Train
        dl_train.sampler.set_epoch(epoch)
        epoch_start = time.time()
        train_loss_meter = AverageMeterForDict()
        train_eval_meter = AverageMeterForDict()
        net.train()
        for i, data in enumerate(tqdm(dl_train, disable=(not is_main) or args.no_pbar, ncols=80)):
            data_in = net.module.pre_process(data)
            out = net(data_in)
            loss_out = loss_fn(out, data, epoch)

            post_out = net.module.post_process(out)
            eval_out = evaluator.evaluate(post_out, data)

            optimizer.zero_grad()
            loss_out['loss'].backward()
            lr = optimizer.step()

            train_loss_meter.update(loss_out)
            train_eval_meter.update(eval_out)


        # print('epoch: {}, lr: {}'.format(epoch, lr))
        optimizer.step_scheduler()
        max_memory = torch.cuda.max_memory_allocated(device=device) // 2 ** 20

        loss_avg = train_loss_meter.metrics['loss'].avg
        logger.print('[Training] Avg. loss: {:.6}, time cost: {:.3} mins, lr: {:.3}, peak mem: {} MB'.
                     format(loss_avg, (time.time() - epoch_start) / 60.0, lr, max_memory))
        logger.print('-- ' + train_eval_meter.get_info())

        logger.add_scalar('train/lr', lr, it=epoch)
        logger.add_scalar('train/max_mem', max_memory, it=epoch)
        for key, elem in train_eval_meter.metrics.items():
            logger.add_scalar(title='train/{}'.format(key), value=elem.avg, it=epoch)
        for key, elem in train_loss_meter.metrics.items():
            logger.add_scalar(title='train/{}'.format(key), value=elem.avg, it=epoch)

        dist.barrier()  # sync
        if ((epoch + 1) % args.val_interval == 0) or epoch > 25:
            # * Validation
            with torch.no_grad():
                val_start = time.time()
                dl_val.sampler.set_epoch(epoch)
                val_loss_meter = AverageMeterForDict()
                val_eval_meter = AverageMeterForDict()
                net.eval()
                for i, data in enumerate(tqdm(dl_val, disable=(not is_main) or args.no_pbar, ncols=80)):
                    data_in = net.module.pre_process(data)
                    out = net(data_in)
                    loss_out = loss_fn(out, data, epoch)

                    post_out = net.module.post_process(out)
                    eval_out = evaluator.evaluate(post_out, data)

                    val_loss_meter.update(loss_out)
                    val_eval_meter.update(eval_out)

                # make eval results into a Tensor
                eval_res = [elem.avg for k, elem in val_eval_meter.metrics.items()]
                eval_res = torch.from_numpy(np.array(eval_res)).to(device)

                val_eval_mean = distributed_mean(eval_res)
                val_eval = dict()
                for i, key in enumerate(list(val_eval_meter.metrics.keys())):
                    val_eval[key] = val_eval_mean[i].item()
                val_eval_meter.reset()
                val_eval_meter.update(val_eval)

                loss_res = [elem.avg for k, elem in val_loss_meter.metrics.items()]
                loss_res = torch.from_numpy(np.array(loss_res)).to(device)
                val_loss_mean = distributed_mean(loss_res)
                val_loss = dict()
                for i, key in enumerate(list(val_loss_meter.metrics.keys())):
                    val_loss[key] = val_loss_mean[i].item()
                val_loss_meter.reset()
                val_loss_meter.update(val_loss)


                logger.print('[Validation] Avg. loss: {:.6}, time cost: {:.3} mins'.format(
                    val_loss_meter.metrics['loss'].avg, (time.time() - val_start) / 60.0))
                logger.print('-- ' + val_eval_meter.get_info())

                for key, elem in val_loss_meter.metrics.items():
                    logger.add_scalar(title='val/{}'.format(key), value=elem.avg, it=epoch)
                for key, elem in val_eval_meter.metrics.items():
                    logger.add_scalar(title='val/{}'.format(key), value=elem.avg, it=epoch)

                if is_main:
                    if args.logger_writer:
                        swanlab.log({"val/metric/": val_eval_meter.get_avg_dict()}, step=epoch)
                        swanlab.log({"val/loss/": val_loss_meter.get_avg_dict()}, step=epoch)

                    if val_eval_meter.metrics[rank_metric].avg < best_metric:
                        model_name = '{}_ddp_best.tar'.format(net_name)
                        
                        save_ckpt(net.module, optimizer, epoch, save_dir, model_name)
                        best_metric = val_eval_meter.metrics[rank_metric].avg
                        logger.print('Save the model: {}, {}: {:.4}, epoch: {}'.format(
                            model_name, rank_metric, best_metric, epoch))

        if is_main:
            if args.logger_writer:
                swanlab.log({"train/loss/": train_loss_meter.get_avg_dict()}, step=epoch)
                swanlab.log({"train/metric/": train_eval_meter.get_avg_dict()}, step=epoch)
            if int(100 * epoch / args.train_epoches) in [20, 40, 60, 80] or (epoch > int(args.train_epoches * 0.8)):
                model_name = '{}_ddp_ckpt_epoch{}.tar'.format(net_name, epoch)
                save_ckpt(net.module, optimizer, epoch, save_dir, model_name)
                logger.print('Save the model to {}'.format('saved_models/' + model_name))

    logger.print("\nTraining completed in {:.2f} mins".format((time.time() - start_time) / 60.0))

    if is_main:
        # save trained model
        swanlab.finish()
        model_name = '{}_ddp_epoch{}.tar'.format(net_name, args.train_epoches)
        save_ckpt(net.module, optimizer, epoch, save_dir, model_name)
        logger.print('Save the model to {}'.format(save_dir + model_name))

    dist.destroy_process_group()
    logger.print('\nExit...\n')


if __name__ == "__main__":
    main()
