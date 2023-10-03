from torch import nn
import torch
import builtins
import os

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.distributed as dist
import torch.optim as optim
from model import ViTTinyAutoencoder
from data_loader import get_data_loader
from train import Trainer
import argparse
from utils import get_model_files


parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning Rate")
parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
parser.add_argument("--epochs", type=int, default=1000, help="# of epochs")
parser.add_argument("--in_channel", type=int, default=1, help="# of input channel")
parser.add_argument("--sequence_length", type=int, default=4, help="# of images in seq")
parser.add_argument("--image_size", default=84, type=int, help="Size of images")
parser.add_argument("--data_type", default="Random", choices=["Random", "Gamer",
                                                            "Agent"], 
                    help="Type of dataset which was collected")
parser.add_argument("--patch_size", default=16, type=int, help="Patch size for ViT")
parser.add_argument("--transformer_layer", type=int, default=2, help="# of transformer")
parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
parser.add_argument("--n_heads", type=int, default=4, help="# of heads for transformer")
parser.add_argument("--checkpoint_interval", type=int, default=10)
parser.add_argument("--resume_checkpoint", action="store_true", help="Load checkpoint")
parser.add_argument("--checkpoint_file", type=str, default="./encoder_model_files/{}/{}_{}_{}_{}_{}_{}_{}.pth",
                    help="url of last checkpoint")

## DDP Config
parser.add_argument("--distributed", action="store_true", help="Using distributed")
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')


def load_train_objects(args):
    train_loader = get_data_loader(args)
    model = ViTTinyAutoencoder(args)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=0.001)

    return model, train_loader, optimizer, scheduler, criterion


def main(args):
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    world_size = torch.cuda.device_count()
    print(f'World Size: {world_size}')
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1:
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    checkpoint_dir, best_model_dir = get_model_files(args)
    model, train_loader, optimizer, scheduler, criterion = load_train_objects(args)
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(model, optimizer, scheduler, train_loader, criterion, 
                      checkpoint_dir, best_model_dir, args)
    trainer.train()
 

if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
