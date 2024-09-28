import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
import math
import numpy as np
import tonic
from tonic import DiskCachedDataset
from tonic.datasets import SHD, SSC
from tqdm import tqdm


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


def create_mnist_dataloaders(path:str, batch_size:int, permute:bool, shuffle:bool=True) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(1, 28*28).t()),
    ]
    if permute:
        permutation = bitreversal_permutation(28*28)
        transform.append(torchvision.transforms.Lambda(lambda x: x[permutation]))

    transform = Compose(transform)
    target_transform = lambda y: F.one_hot(torch.tensor(y,  dtype=torch.long), 10)

    # Load the MNIST dataset
    train_dataset = MNIST(root=path, train=True, download=True, transform=transform, target_transform=target_transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
    test_dataset = MNIST(root=path, train=False, download=True, transform=transform, target_transform=target_transform)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, val_dataloader, test_dataloader


def create_shd_dataloaders(path:str, batch_size:int, shuffle:bool=True, bins:int=250, downsample_size:int=140) -> tuple[DataLoader, DataLoader, DataLoader]:
    shd_sensor_size = (700, 1, 1)
    pre_cache_transform = Compose([
        tonic.transforms.Downsample(sensor_size=shd_sensor_size, target_size=(downsample_size, 1)),
    ])
    post_cache_train_transform = Compose([
        tonic.transforms.TimeJitter(std=100, clip_negative=True),
        tonic.transforms.ToFrame(
            sensor_size=(downsample_size, 1, 1), 
            n_time_bins=bins,
        ),
        torchvision.transforms.Lambda(lambda x: x[:, 0, :]),
        torchvision.transforms.Lambda(lambda x: (x > 0).astype(np.int32)),
    ])
    post_cache_val_transform = Compose([
        tonic.transforms.ToFrame(
            sensor_size=(downsample_size, 1, 1), 
            n_time_bins=bins,
        ),
        torchvision.transforms.Lambda(lambda x: x[:, 0, :]),
        torchvision.transforms.Lambda(lambda x: (x > 0).astype(np.int32)),
    ])

    target_transform = lambda y: F.one_hot(torch.tensor(y, dtype=torch.long), 20)

    train_dataset = SHD(save_to=path, train=True, transform=pre_cache_transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
    test_dataset = SHD(save_to=path, train=False, transform=pre_cache_transform)

    train_dataset = DiskCachedDataset(train_dataset, cache_path=path + "/tonic/cache/shd/train", transform=post_cache_train_transform, target_transform=target_transform)
    # val_dataset = DiskCachedDataset(val_dataset, cache_path=path + "/tonic/cache/shd/val", transform=post_cache_val_transform, target_transform=target_transform)
    test_dataset = DiskCachedDataset(test_dataset, cache_path=path + "/tonic/cache/shd/test", transform=post_cache_val_transform, target_transform=target_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = []
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, val_dataloader, test_dataloader


def create_ssc_dataloaders(path:str, batch_size:int, shuffle:bool=True, bins:int=250, downsample_size:int=140, load_into_ram:bool=True) -> tuple[DataLoader, DataLoader, DataLoader]:
    shd_sensor_size = (700, 1, 1)
    pre_cache_transform = Compose([
        tonic.transforms.Downsample(sensor_size=shd_sensor_size, target_size=(downsample_size, 1)),
    ])
    post_cache_train_transform = Compose([
        tonic.transforms.TimeJitter(std=100, clip_negative=True),
        tonic.transforms.ToFrame(
            sensor_size=(downsample_size, 1, 1), 
            n_time_bins=bins,
        ),
        torchvision.transforms.Lambda(lambda x: x[:, 0, :]),
        torchvision.transforms.Lambda(lambda x: (x > 0).astype(np.int32)),
    ])
    post_cache_val_transform = Compose([
        tonic.transforms.ToFrame(
            sensor_size=(downsample_size, 1, 1), 
            n_time_bins=bins,
        ),
        torchvision.transforms.Lambda(lambda x: x[:, 0, :]),
        torchvision.transforms.Lambda(lambda x: (x > 0).astype(np.int32)),
    ])

    target_transform = lambda y: F.one_hot(torch.tensor(y, dtype=torch.long), 35)

    class CustomDataset(torch.utils.data.Dataset):
        """
        The only use for this dataset is to load the entire dataset into ram to speed up dataloading.
        This is just a workaround solution.
        """
        x: torch.Tensor
        y: torch.Tensor

        def __init__(self, dataset: torch.utils.data.Dataset) -> None:
            xs = []
            ys = []
            print("Loading data into RAM. This may take a few minutes...")
            for x, y in tqdm(dataset):
                xs.append(x)
                ys.append(y)
            self.x = np.array(xs)
            self.y = np.array(ys)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.tensor(self.x[idx,...], dtype=torch.float32), torch.tensor(self.y[idx,...], dtype=torch.long)
        
        def __len__(self) -> int:
            return self.x.shape[0]

    train_dataset = SSC(save_to=path, split="train", transform=pre_cache_transform)
    val_dataset = SSC(save_to=path, split="valid", transform=pre_cache_transform)
    test_dataset = SSC(save_to=path, split="test", transform=pre_cache_transform)

    train_dataset = DiskCachedDataset(train_dataset, cache_path=path + "/tonic/cache/ssc/train", transform=post_cache_train_transform, target_transform=target_transform)
    val_dataset = DiskCachedDataset(val_dataset, cache_path=path + "/tonic/cache/ssc/val", transform=post_cache_val_transform, target_transform=target_transform)
    test_dataset = DiskCachedDataset(test_dataset, cache_path=path + "/tonic/cache/ssc/test", transform=post_cache_val_transform, target_transform=target_transform)

    if load_into_ram:
        print("#######################")
        print(load_into_ram)
        train_dataset = CustomDataset(train_dataset)
        val_dataset = CustomDataset(val_dataset)
        test_dataset = CustomDataset(test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, val_dataloader, test_dataloader