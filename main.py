import os
import time
import json
import argparse
import tqdm
import numpy as np
import torch
from config import get_config
from data.custom_dataset import CustomDataset
from solver import Solver
from utils.logger import create_logger
from models.domain_adaptation.vmamba_da import Feature, Predictor, init_weights
from loss.label_smoothing import LabelSmoothingCrossEntropy
import torch.optim as optim


if torch.multiprocessing.get_start_method() != "spawn":
    print(f"||{torch.multiprocessing.get_start_method()}||", end="")
    torch.multiprocessing.set_start_method("spawn", force=True)




def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE",
                        default="", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--source-path', type=str,
                        default="/public", help='path to public dataset')
    parser.add_argument('--target-path', type=str,
                        default="/private", help='path to target dataset')
    parser.add_argument('--zip', action='store_true',
                        help='use zipped dataset instead of folder dataset')
    
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default=time.strftime("%Y%m%d%H%M%S",
                        time.localtime()), help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')

    parser.add_argument('--fused_layernorm',
                        action='store_true', help='Use fused layernorm.')
    parser.add_argument(
        '--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float,
                        default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu',
                        type=str2bool, default=False, help='')

    parser.add_argument('--memory_limit_rate', type=float,
                        default=-1, help='limitation of gpu memory use')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, args):
    solver = Solver(config, batch_size=config.DATA.BATCH_SIZE)
    EPOCHS = config.TRAIN.EPOCHS

    max = 0
    for epoch in range(EPOCHS):
        solver.train(epoch=epoch)
        max = solver.test(max)
        print(f'Maxium acc val target: {max:2f}') 



if __name__ == '__main__':
    args, config = parse_option()

    
    os.makedirs(config.OUTPUT, exist_ok=True)
    

    if args.memory_limit_rate > 0 and args.memory_limit_rate < 1:
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_rate)
        usable_memory = torch.cuda.get_device_properties(
            0).total_memory * args.memory_limit_rate / 1e6
        print(f"===========> GPU memory is limited to {usable_memory}MB", flush=True)

    main(config, args)
