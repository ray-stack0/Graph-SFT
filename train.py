import os
import sys
import time
import subprocess
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import faulthandler
from tqdm import tqdm
#
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
#
from loader import Loader
from utils.logger import Logger
from utils.utils import AverageMeterForDict
from utils.utils import save_ckpt, set_seed, get_pf
import swanlab
import yaml


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
    set_seed(args.seed)

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log/" + date_str
    logger = Logger(date_str=date_str, log_dir=log_dir, enable_flags={'writer': args.logger_writer})
    # log basic info
    logger.log_basics(args=args, datetime=date_str)

    loader = Loader(args, device, is_ddp=False, logger=logger)
    epoch_now = 0
    if args.resume:
        logger.print('[Resume] Loading state_dict from {}'.format(args.model_path))
        loader.set_resmue(args.model_path)
        epoch_now = loader.get_last_epoch() + 1
    (train_set, val_set), net, loss_fn, optimizer, evaluator = loader.load()

        #* 启动swanlab
    if args.logger_writer:

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

    dl_train = DataLoader(train_set,
                          batch_size=args.train_batch_size,
                          shuffle=True,
                          num_workers=8,
                          prefetch_factor=4,
                          collate_fn=train_set.collate_fn,
                          drop_last=True,
                          pin_memory=True)
    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        shuffle=False,
                        num_workers=8,
                        prefetch_factor=4,
                        collate_fn=val_set.collate_fn,
                        drop_last=True,
                        pin_memory=True)
    
    # 记录计算量
    batch = next(iter(dl_val))
    input = net.pre_process(batch)
    macs, flops, params, parameter_count_table= get_pf(net, input)
    logger.print(f"[Model Stats]")
    logger.print(f"  - MACs:      {macs}")
    logger.print(f"  - Parameters:{params}")
    logger.print(f"  - FLOPs:     {flops / 1e9:.2f}G")
    logger.print("  - Parameter breakdown:\n" + parameter_count_table)


    best_metric = loader.get_best_metric()
    rank_metric = args.rank_metric
    net_name = loader.network_name()
    step = 0
    model_dir = os.path.join('saved_models', date_str)
    try:
        for epoch in range(args.train_epoches):
            logger.print('\nEpoch {}'.format(epoch))
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # * Train
            epoch_start = time.time()
            train_loss_meter = AverageMeterForDict()
            train_eval_meter = AverageMeterForDict()
            net.train()
            for i, data in enumerate(tqdm(dl_train, disable=args.no_pbar, ncols=80)):
                data_in = net.pre_process(data)
                out = net(data_in)
                loss_out = loss_fn(out, data, epoch)

                post_out = net.post_process(out)
                eval_out = evaluator.evaluate(post_out, data)

                optimizer.zero_grad()
                loss_out['loss'].backward()
                lr = optimizer.step()

                train_loss_meter.update(loss_out)
                train_eval_meter.update(eval_out)

                # if args.logger_writer:
                    # swanlab.log({"train/metric/step/": eval_out}, step=step)
                #     swanlab.log({"train/loss/step/": loss_out}, step=step)
                # step = step + 1


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

            if args.logger_writer:
                swanlab.log({"train/loss/": train_loss_meter.get_avg_dict()}, step=epoch)
                swanlab.log({"train/metric/": train_eval_meter.get_avg_dict()}, step=epoch)

            if ((epoch + 1) % args.val_interval == 0) or epoch > 20:
                # * Validation
                with torch.no_grad():
                    val_start = time.time()
                    val_loss_meter = AverageMeterForDict()
                    val_eval_meter = AverageMeterForDict()
                    net.eval()
                    for i, data in enumerate(tqdm(dl_val, disable=args.no_pbar, ncols=80)):
                        data_in = net.pre_process(data)
                        out = net(data_in)
                        loss_out = loss_fn(out, data, epoch)

                        post_out = net.post_process(out)
                        eval_out = evaluator.evaluate(post_out, data)

                        val_loss_meter.update(loss_out)
                        val_eval_meter.update(eval_out)

                    logger.print('[Validation] Avg. loss: {:.6}, time cost: {:.3} mins'.format(
                        val_loss_meter.metrics['loss'].avg, (time.time() - val_start) / 60.0))
                    logger.print('-- ' + val_eval_meter.get_info())

                    for key, elem in val_loss_meter.metrics.items():
                        logger.add_scalar(title='val/{}'.format(key), value=elem.avg, it=epoch)
                    for key, elem in val_eval_meter.metrics.items():
                        logger.add_scalar(title='val/{}'.format(key), value=elem.avg, it=epoch)

                    if args.logger_writer:
                            swanlab.log({"val/metric/": val_eval_meter.get_avg_dict()}, step=epoch)
                            swanlab.log({"val/loss/": val_loss_meter.get_avg_dict()}, step=epoch)
                    
                    if (epoch >= args.train_epoches / 2):
                        if val_eval_meter.metrics[rank_metric].avg < best_metric:
                            
                            model_name = '{}_best.tar'.format(net_name)
                            save_ckpt(net, optimizer, epoch, model_dir, model_name)
                            best_metric = val_eval_meter.metrics[rank_metric].avg
                            logger.print('Save the model: {}, {}: {:.4}, epoch: {}'.format(
                                model_name, rank_metric, best_metric, epoch))

            if int(100 * epoch / args.train_epoches) in [20, 40, 60, 80]:
                model_name = '{}_ckpt_epoch{}.tar'.format(net_name, epoch)
                save_ckpt(net, optimizer, epoch, model_dir, model_name)
                logger.print('Save the model to {}'.format(os.path.join(model_dir, model_name)))

        logger.print("\nTraining completed in {:.2f} mins".format((time.time() - start_time) / 60.0))
        # save trained model
        model_name = date_str + '_{}_epoch{}.tar'.format(net_name, args.train_epoches)
        save_ckpt(net, optimizer, epoch, model_dir, model_name)
        print('Save the model to {}'.format(os.path.join(model_dir, model_name)))
        print('\nExit...\n')
    finally:
        swanlab.finish()



if __name__ == "__main__":
    main()
