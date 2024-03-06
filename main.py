# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function


import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn


import timm
from models import *
from models.vit_small import ViT
from models.mlpmixer import MLPMixer
from models.convmixer import ConvMixer
from models.simplevit import SimpleViT

from data import get_dataloader
from trainer import train_one_epoch, evaluate

from utils import *
from nc_metrics import analysis
from models.vit import ViT


def main(args):
    # take in args

    import wandb
    os.environ["WANDB_API_KEY"] = "0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee"
    os.environ["WANDB_MODE"] = "online"  # "dryrun"
    os.environ["WANDB_CACHE_DIR"] = "./wandb"
    os.environ["WANDB_CONFIG_DIR"] = "./wandb"
    wandb.login(key='0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee')
    wandb.init(project='nv' + args.dataset,
               name=args.store_name.split('/')[-1]
               )
    wandb.config.update(args)

    global best_acc
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # ==================== data loader ====================
    print('==> Preparing data..')
    train_loader, test_loader = get_dataloader(args)

    # ====================  define model ====================
    # Model factory..
    print('==> Building model..')
    if args.net=='res18':
        net = ResNet18()
    elif args.net=='res50':
        net = ResNet50()
    elif args.net=="convmixer":
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
    elif args.net=="mlpmixer":
        net = MLPMixer(image_size=32, channels=3, patch_size=args.patch, dim=512, depth=6, num_classes = args.num_classes)
    elif args.net=="vit_small":
        net = ViT(
            image_size=args.imsize, patch_size=args.patch,
            num_classes=args.num_classes, dim=int(args.dimhead),
            depth=6, heads=8, mlp_dim=512,
            dropout=0.1, emb_dropout=0.1
        )
    elif args.net=="vit_tiny":
        net = ViT(
            image_size=args.imsize, patch_size=args.patch,
            num_classes=args.num_classes, dim=int(args.dimhead),
            depth=4, heads=6, mlp_dim=256,
            dropout=0.1, emb_dropout=0.1
        )
    elif args.net=="simplevit":
        net = SimpleViT(
            image_size=args.imsize, patch_size=args.patch,
            num_classes=args.num_classes, dim=int(args.dimhead),
            depth=6, heads=8, mlp_dim=512
    )
    elif args.net=="vit":
        # ViT for cifar10
        net = ViT(
            image_size=args.imsize, patch_size=args.patch,
            num_classes=args.num_classes, dim=int(args.dimhead),
            depth=6, heads=8, mlp_dim=512,
            dropout=0.1, emb_dropout=0.1
    )
    elif args.net=="vit_timm":
        net = timm.create_model("vit_base_patch16_384", pretrained=True)
        net.head = nn.Linear(net.head.in_features, 10)
    elif args.net=="swin":
        from models.swin import swin_t
        net = swin_t(window_size=args.patch, num_classes=10, downscaling_factors=(2,2,2,1))
    if torch.cuda.is_available():
        net.cuda()

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # ====================  Training utilities ====================
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(args, optimizer)

    graphs1 = Graph_Vars()  # for training nc
    graphs2 = Graph_Vars()  # for testing nc

    for epoch in range(args.max_epochs):
        train_loss, train_acc = train_one_epoch(net, criterion, optimizer, train_loader, args)
        if args.scheduler in ['step', 'ms', 'multi_step', 'poly']:
            lr_scheduler.step()

        val_loss, val_acc = evaluate(net, criterion, test_loader)

        wandb.watch(net)
        wandb.log({'train/train_loss': train_loss,
                   'train/train_acc': train_acc,
                   'train/lr': optimizer.param_groups[0]["lr"],
                   'val/val_loss': val_loss,
                   'val/val_acc': val_acc},
                  step=epoch
                  )

        if epoch % 10 ==0:
            nc_train = analysis(net, train_loader, args)
            nc_val = analysis(net, test_loader, args)
            graphs1.load_dt(nc_train, epoch=epoch, lr=optimizer.param_groups[0]['lr'])
            graphs2.load_dt(nc_val, epoch=epoch, lr=optimizer.param_groups[0]['lr'])

        # ===== save model
        if args.save_ckpt and val_acc > best_acc:
            state = {"model": net.state_dict(),
                     "optimizer": optimizer.state_dict(), # "scaler": scaler.state_dict()
                     }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/'+args.model+'-{}-ckpt.t7'.format(args.patch))
            best_acc = val_acc

        log('Epoch:{}, lr:{:.6f}, train loss:{:.4f}, train acc:{:.4f}; val loss:{:.4f}, val acc:{:.4f}'.format(
            epoch, optimizer.param_groups[0]["lr"], train_loss, train_acc, val_loss, val_acc
        ))


def set_seed(SEED=666):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # parsers
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument("--seed", type=int, default=2021, help="random seed")
    parser.add_argument('--dset', type=str, default='cifar10')
    parser.add_argument('--model', default='vit')
    parser.add_argument('--num_classes', default=10)
    parser.add_argument('--save_ckpt', default=False, action='store_true', help='save best model')

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')
    parser.add_argument('--use_amp', default=False, action='store_true', help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--scheduler', type=str, default='ms')
    parser.add_argument('--max_epochs', type=int, default='200')

    # args for ViT
    parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
    parser.add_argument('--dimhead', default="512", type=int)
    parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

    parser.add_argument('--exp_name', type=str, default='baseline')
    args = parser.parse_args()

    if args.dset == 'cifar10':
        args.imsize = 32
        args.num_classes = 10
    elif args.dset == 'cifar100':
        args.imsize = 32
        args.num_classes = 100

    args.output_dir = os.path.join('./result/{}/{}/'.format(args.dset, args.model), args.exp_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    set_log_path(args.output_dir)
    log('save log to path {}'.format(args.output_dir))
    log(print_args(args))

    set_seed(SEED=args.seed)
    main(args)

