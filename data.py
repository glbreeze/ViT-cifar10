
import os
import datetime

import numpy

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
data_folder = 'data' # for greene,  '../dataset' for local

def get_dataloader(args):

    # cifar10/cifar100: 32x32, stl10: 96x96, fmnist: 28x28, TinyImageNet 64x64
    if args.dset == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(args.imsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2434, 0.2615])
        ])
        # transform_train.transforms.insert(0, RandAugment(2, 14))
        test_tranform = transform
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_folder, train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_folder, train=False, download=True, transform=test_tranform),
            batch_size=args.batch_size, shuffle=False)

    elif args.dset == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize(args.imsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
        ])
        test_tranform = transform
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(data_folder, train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(data_folder, train=False, download=True, transform=test_tranform),
            batch_size=args.batch_size, shuffle=False)

    if args.dset == 'stl10':  # 96x96
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2434, 0.2615])
        ])
        test_tranform = transform
        train_loader = torch.utils.data.DataLoader(
            datasets.STL10(data_folder, split='train', download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.STL10(data_folder, split='test', download=True, transform=test_tranform),
            batch_size=args.batch_size, shuffle=False)

    elif args.dset == 'tinyi': # image_size:64 x 64
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        test_tranform = transform
        train_dataset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'train'), transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        test_dataset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'val'), test_tranform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader