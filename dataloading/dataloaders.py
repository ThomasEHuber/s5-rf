import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import math
import numpy as np
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset


def bitreversal_po2(n):
    """
    From S4 codebase.

    :param n:
    :return:
    """
    m = int(math.log(n)/math.log(2))
    perm = np.arange(n).reshape(n,1)
    for i in range(m):
        n1 = perm.shape[0]//2
        perm = np.hstack((perm[:n1],perm[n1:]))
    return perm.squeeze(0)

def bitreversal_permutation(n):
    """
    From S4 codebase.

    :param n:
    :return:
    """
    m = int(math.ceil(math.log(n)/math.log(2)))
    N = 1 << m
    perm = bitreversal_po2(N)
    return np.extract(perm < n, perm)


def create_mnist_dataloaders(batch_size:int, num_classes:int, shuffle:bool, permute:bool) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = [
        transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(1, 28*28).t()),
    ]
    if permute:
        permutation = bitreversal_permutation(28*28)
        transform.append(torchvision.transforms.Lambda(lambda x: x[permutation]))

    transform = transforms.Compose(transform)

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='../../data', train=True, download=True, transform=transform, target_transform=lambda y: torch.tensor(F.one_hot(y, num_classes), dtype=torch.long))
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
    test_dataset = datasets.MNIST(root='../../data', train=False, download=True, transform=transform, target_transform=lambda y: torch.tensor(F.one_hot(y, num_classes), dtype=torch.long))


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, val_dataloader, test_dataloader


def create_shd_dataloader(batch_size:int, num_classes:int, shuffle:bool, bins:int=250, downsample_size:int=140) -> tuple[DataLoader, DataLoader, DataLoader]:
    shd_sensor_size = (700, 1, 1)
    pre_cache_transform = transforms.Compose([
        tonic.transforms.Downsample(sensor_size=shd_sensor_size, target_size=(downsample_size, 1)),
    ])
    post_cache_train_transform = transforms.Compose([
        tonic.transforms.TimeJitter(std=100, clip_negative=True),
        tonic.transforms.ToFrame(
            sensor_size=(downsample_size, 1, 1), 
            n_time_bins=bins,
        ),
        torchvision.transforms.Lambda(lambda x: x[:, 0, :]),
        torchvision.transforms.Lambda(lambda x: (x > 0).astype(np.int32)),
    ])
    post_cache_val_transform = transforms.Compose([
        tonic.transforms.ToFrame(
            sensor_size=(downsample_size, 1, 1), 
            n_time_bins=bins,
        ),
        torchvision.transforms.Lambda(lambda x: x[:, 0, :]),
        torchvision.transforms.Lambda(lambda x: (x > 0).astype(np.int32)),
    ])

    train_dataset = tonic.datasets.SHD(save_to="../../data", train=True, transform=pre_cache_transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
    test_dataset = tonic.datasets.SHD(save_to="../../data", train=False, transform=pre_cache_transform)

    train_dataset = DiskCachedDataset(train_dataset, cache_path="../../data/tonic/cache/shd/train", transform=post_cache_train_transform, target_transform=lambda y: torch.tensor(F.one_hot(y, num_classes), dtype=torch.long))
    val_dataset = DiskCachedDataset(val_dataset, cache_path="../../data/tonic/cache/shd/val", transform=post_cache_val_transform, target_transform=lambda y: torch.tensor(F.one_hot(y, num_classes), dtype=torch.long))
    test_dataset = DiskCachedDataset(test_dataset, cache_path="../../data/tonic/cache/shd/test", transform=post_cache_val_transform, target_transform=lambda y: torch.tensor(F.one_hot(y, num_classes), dtype=torch.long))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, val_dataloader, test_dataloader


def create_ssc_dataloader(batch_size:int, num_classes:int, shuffle:bool, bins:int=250, downsample_size:int=140) -> tuple[DataLoader, DataLoader, DataLoader]:
    shd_sensor_size = (700, 1, 1)
    pre_cache_transform = transforms.Compose([
        tonic.transforms.Downsample(sensor_size=shd_sensor_size, target_size=(downsample_size, 1)),
    ])
    post_cache_train_transform = transforms.Compose([
        tonic.transforms.TimeJitter(std=100, clip_negative=True),
        tonic.transforms.ToFrame(
            sensor_size=(downsample_size, 1, 1), 
            n_time_bins=bins,
        ),
        torchvision.transforms.Lambda(lambda x: x[:, 0, :]),
        torchvision.transforms.Lambda(lambda x: (x > 0).astype(np.int32)),
    ])
    post_cache_val_transform = transforms.Compose([
        tonic.transforms.ToFrame(
            sensor_size=(downsample_size, 1, 1), 
            n_time_bins=bins,
        ),
        torchvision.transforms.Lambda(lambda x: x[:, 0, :]),
        torchvision.transforms.Lambda(lambda x: (x > 0).astype(np.int32)),
    ])

    train_dataset = tonic.datasets.SSC(save_to="../../data", split="train", transform=pre_cache_transform)
    val_dataset = tonic.datasets.SSC(save_to="../../data", split="valid", transform=pre_cache_transform)
    test_dataset = tonic.datasets.SSC(save_to="../../data", split="test", transform=pre_cache_transform)

    train_dataset = DiskCachedDataset(train_dataset, cache_path="../../data/tonic/cache/ssc/train", transform=post_cache_train_transform, target_transform=lambda y: torch.tensor(F.one_hot(y, num_classes), dtype=torch.long))
    val_dataset = DiskCachedDataset(val_dataset, cache_path="../../data/tonic/cache/ssc/val", transform=post_cache_val_transform, target_transform=lambda y: torch.tensor(F.one_hot(y, num_classes), dtype=torch.long))
    test_dataset = DiskCachedDataset(test_dataset, cache_path="../../data/tonic/cache/ssc/test", transform=post_cache_val_transform, target_transform=lambda y: torch.tensor(F.one_hot(y, num_classes), dtype=torch.long))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, val_dataloader, test_dataloader