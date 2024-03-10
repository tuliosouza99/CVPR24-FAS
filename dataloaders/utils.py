import random

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .dataset import FaceDataset, Identity, RandomCutout, RoundRobinDataset


def get_datasets(
    data_dir,
    data_list,
    train=True,
    img_size=256,
    map_size=32,
    transform=None,
    balance=False,
):
    datasets = []
    sum_n = 0
    labels = ['live', 'spoof'] if balance else [None]

    for i in range(len(data_list)):
        for label in labels:
            data_tmp = FaceDataset(
                dataset_name=data_list[i],
                root_dir=data_dir,
                is_train=train,
                img_size=img_size,
                map_size=map_size,
                transform=transform,
                UUID=i,
                label=label,
            )
            datasets.append(data_tmp)
            sum_n += len(data_tmp)

    if balance:
        print("Balanced loader for each class and domain")
        data_set_sum = RoundRobinDataset(datasets)
    else:
        data_set_sum = sum(datasets) if len(datasets) > 1 else datasets[0]

    print("{} videos: {}".format('Train' if train else 'Test', sum_n))

    return data_set_sum


def get_train_test_loader(config):
    """
    Return the train and test data loader

    """
    if config.TRAIN.get('auto_augment'):
        print("Apply AUTO AUGMENTATION")
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (config.MODEL.image_size, config.MODEL.image_size),
                scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(
                degrees=(-180, 180), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            RandomCutout(1, 0.5) if config.TRAIN.cutout else Identity(),
            normalization,
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((config.MODEL.image_size, config.MODEL.image_size)),
            transforms.ToTensor(),
            normalization,
        ]
    )
    train_set = get_datasets(
        config.PATH.data_folder,
        train=True,
        data_list=config.train_set,
        img_size=config.MODEL.image_size,
        map_size=32,
        transform=train_transform,
        balance=config.TRAIN.get('balance_loader'),
    )

    test_set = get_datasets(
        config.PATH.data_folder,
        train=False,
        data_list=config.test_set,
        img_size=config.MODEL.image_size,
        map_size=32,
        transform=test_transform,
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(
        train_set,
        batch_size=config.TRAIN.batch_size,
        shuffle=True,
        num_workers=config.SYS.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.TRAIN.batch_size,
        shuffle=False,
        num_workers=config.SYS.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, test_loader
