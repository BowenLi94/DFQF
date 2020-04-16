from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
import os

def mnist(batch_size, train=True, val=True):
    ds = []
    if train:
        trainloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                root="/data/MNIST", train=True, download=False,
                transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(), 
                    transforms.Normalize((0.1307,), (0.3081,))
                ]),
            ),
            batch_size=batch_size, shuffle=True,
        )
        ds.append(trainloader)
    if val:
        testloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                root="/data/MNIST", train=False, download=False,
                transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(), 
                    transforms.Normalize((0.1307,), (0.3081,))
                ]),
            ),
            batch_size=batch_size, shuffle=True,
        )
        ds.append(testloader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def cifar10(batch_size, train=True, val=True):
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ds = []
    if train:
        trainloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root="/data/CIFAR10", train=True, download=False,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(), 
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]),
            ),
            batch_size=batch_size, shuffle=True,
        )
        ds.append(trainloader)
    if val:
        testloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root="/data/CIFAR10", train=False, download=False,
                transform=transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]),
            ),
            batch_size=batch_size, shuffle=True,
        )
        ds.append(testloader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def cifar100(batch_size, train=True, val=True):
    ds = []
    if train:
        trainloader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root="/data/CIFAR100", train=True, download=False,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(), 
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]),
            ),
            batch_size=batch_size, shuffle=True,
        )
        ds.append(trainloader)
    if val:
        testloader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root="/data/CIFAR100", train=False, download=False,
                transform=transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]),
            ),
            batch_size=batch_size, shuffle=True,
        )
        ds.append(testloader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def imagenet(batch_size, train=True, val=True):
    ds = []
    input_size=224
    if train:
        trainloader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root="/data/imagenet/train", 
                transform=transforms.transforms.Compose([
                    transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                ]),
            ),
            batch_size=batch_size, shuffle=True,
        )
        ds.append(trainloader)
    if val:
        testloader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                root="/data/imagenet/valid", 
                transform=transforms.Compose([
                    transforms.Resize(int(input_size/0.875)),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
                ]),
            ),
            batch_size=128, shuffle=True,
        )
        ds.append(testloader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_dataset(args):
    if args.dataset == 'mnist':
        args.channels = 1
        args.img_size = 28
        args.num_classes =10
        return mnist(batch_size=args.batch_size)
    elif args.dataset == 'cifar10':
        args.channels = 3
        args.img_size = 32
        args.num_classes =10
        return cifar10(batch_size=args.batch_size)
    elif args.dataset == 'cifar100':
        args.channels = 3
        args.img_size = 32
        args.num_classes =100
        return cifar100(batch_size=args.batch_size)
    elif args.dataset == 'imagenet':
        args.channels = 3
        args.img_size = 224
        args.num_classes =1000
        return imagenet(batch_size=args.batch_size)
