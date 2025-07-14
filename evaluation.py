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


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="val", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Val batch size")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--adv_cfg_path", required=True, default="", type=str)
    parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")
    parser.add_argument("--warm_up_epoch", default=500, type=int, help="Warm up epochs")
    parser.add_argument("--measure_epoch", default=1000, type=int, help="measure iterations")
    parser.add_argument("--inference_time", action="store_true", help="Enable inference time measurement")
    return parser.parse_args()


def main():
    args = parse_arguments()
    print('Args: {}\n'.format(args))

    faulthandler.enable()

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    if not args.model_path.endswith(".tar"):
        assert False, "Model path error - '{}'".format(args.model_path)

    loader = Loader(args, device, is_ddp=False)
    print('[Resume] Loading state_dict from {}'.format(args.model_path))
    loader.set_resmue(args.model_path)
    (train_set, val_set), net, loss_fn, _, evaluator = loader.load()

    if args.inference_time:
        args.val_batch_size = 1
    
    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=val_set.collate_fn,
                        drop_last=False,
                        pin_memory=True)

    net.eval()
    if args.inference_time:
        print('Inference time measurement enabled')
        with torch.no_grad():
            # * Validation
            for i, data in enumerate(tqdm(dl_val)):
                data_in = net.pre_process(data)
                if i < args.warm_up:
                    _ = net(data_in)
                    continue

                # === 开始计时 ===
                torch.cuda.synchronize()
                start = time.time()

                out = net(data_in)

                torch.cuda.synchronize()
                end = time.time()

                total_time += (end - start)
                count += 1

                if count >= args.measure_epoch:
                    break
            avg_time_per_batch = total_time / count
            print(f"Avg. inference time per batch(batch size = {args.val_batch_size}): {avg_time_per_batch*1000:.2f} ms")

    else:
        with torch.no_grad():
            # * Validation
            val_start = time.time()
            val_eval_meter = AverageMeterForDict()
            for i, data in enumerate(tqdm(dl_val)):
                data_in = net.pre_process(data)
                out = net(data_in)
                _ = loss_fn(out, data)
                post_out = net.post_process(out)

                eval_out = evaluator.evaluate(post_out, data)
                val_eval_meter.update(eval_out, n=data['BATCH_SIZE'])

            print('\nValidation set finish, cost {:.2f} secs'.format(time.time() - val_start))
            print('-- ' + val_eval_meter.get_info())

    print('\nExit...')


if __name__ == "__main__":
    main()
