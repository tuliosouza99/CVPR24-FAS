import io
import os
import random
import re
from glob import glob
from typing import Sequence, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .meta import DEVICE_INFOS


class FaceDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        is_train: bool = True,
        label: Optional[str] = None,
        transform=None,
        map_size: int = 32,
        UUID: int = -1,
        img_size: int = 256,
    ):
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.root_dir = os.path.join(root_dir, self.dataset_name, 'preprocess')

        self.video_list = os.listdir(self.root_dir)
        if is_train:
            self.video_list = list(filter(lambda x: 'train' in x, self.video_list))
        else:
            self.video_list = list(filter(lambda x: 'train' not in x, self.video_list))

        if label is not None and label != 'all':
            self.video_list = list(filter(lambda x: label in x, self.video_list))

        print(
            "({dataset}) Total video: {total} | Live: {live} | Spoof: {spoof}".format(
                dataset=self.dataset_name,
                total=len(self.video_list),
                live=len([u for u in self.video_list if 'live' in u]),
                spoof=len([u for u in self.video_list if 'live' not in u]),
            )
        )

        self.transform = transform
        self.map_size = map_size
        self.UUID = UUID
        self.image_size = img_size

    def __len__(self):
        return len(self.video_list)

    def shuffle(self):
        if self.is_train:
            random.shuffle(self.video_list)

    def get_client_from_video_name(self, video_name: str):
        video_name = video_name.split('/')[-1]

        if 'msu' in self.dataset_name.lower() or 'replay' in self.dataset_name.lower():
            match = re.findall(r'client(\d\d\d)', video_name)
        elif 'oulu' in self.dataset_name.lower():
            match = re.findall(r'(\d+)_\d$', video_name)
        elif 'casia' in self.dataset_name.lower():
            match = re.findall(r'(\d+)_[H|N][R|M]_\d$', video_name)
        else:
            raise RuntimeError("no dataset found")

        if len(match) == 0:
            raise RuntimeError('no client')
        client_id = match[0]

        return client_id

    def __getitem__(self, idx):
        video_name = self.video_list[idx]
        spoofing_label = int('live' in video_name)
        device_tag = 'live' if spoofing_label else 'spoof'

        if self.dataset_name in DEVICE_INFOS:
            if 'live' in video_name:
                patterns = DEVICE_INFOS[self.dataset_name]['live']
            elif 'spoof' in video_name:
                patterns = DEVICE_INFOS[self.dataset_name]['spoof']
            else:
                raise RuntimeError(
                    f"Cannot find the label info from the video: {video_name}"
                )

            device_tag = None
            for pattern in patterns:
                if len(re.findall(pattern, video_name)) > 0:
                    if device_tag is not None:
                        raise RuntimeError("Multiple Match")
                    device_tag = pattern

            if device_tag is None:
                raise RuntimeError("No Match")

        client_id = self.get_client_from_video_name(video_name)
        image_dir = os.path.join(self.root_dir, video_name)
        image_x = self.sample_image(image_dir)

        transformed_image1 = self.transform(image_x)
        if self.is_train:
            transformed_image2 = self.transform(image_x)
        else:
            transformed_image2 = transformed_image1

        sample = {
            "image_x_v1": transformed_image1,
            "image_x_v2": transformed_image2,
            "label": spoofing_label,
            "UUID": self.UUID,
            'device_tag': device_tag,
            'video': video_name,
            'client_id': client_id,
        }

        return sample

    def sample_image(self, image_dir: str):
        frames = glob(os.path.join(image_dir, 'crop_*.jpg'))
        image_path = np.random.choice(frames)
        image = Image.open(image_path)

        return image


class Identity:  # used for skipping transforms
    def __call__(self, im):
        return im


class RandomCutout(object):
    def __init__(self, n_holes, p=0.5):
        """
        Args:
            n_holes (int): Number of patches to cut out of each image.
            p (int): probability to apply cutout
        """
        self.n_holes = n_holes
        self.p = p

    def rand_bbox(self, W, H, lam):
        """
        Return a random box
        """
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if np.random.rand(1) > self.p:
            return img

        h = img.size(1)
        w = img.size(2)
        lam = np.random.beta(1.0, 1.0)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(w, h, lam)
        for n in range(self.n_holes):
            img[:, bby1:bby2, bbx1:bbx2] = img[:, bby1:bby2, bbx1:bbx2].mean(
                dim=[-2, -1], keepdim=True
            )
        return img


class RandomJPEGCompression(object):
    def __init__(self, quality_min=30, quality_max=90, p=0.5):
        assert 0 <= quality_min <= 100 and 0 <= quality_max <= 100
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img):
        if np.random.rand(1) > self.p:
            return img
        # Choose a random quality for JPEG compression
        quality = np.random.randint(self.quality_min, self.quality_max)

        # Save the image to a bytes buffer using JPEG format
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)

        # Reload the image from the buffer
        img = Image.open(buffer)
        return img


class RoundRobinDataset(Dataset):
    def __init__(self, datasets: Sequence[FaceDataset]):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_len = sum(self.lengths)

    def __getitem__(self, index):
        # Determine which dataset to sample from
        dataset_id = index % len(self.datasets)

        # Adjust index to fit within the chosen dataset's length
        inner_index = index // len(self.datasets)
        inner_index = inner_index % self.lengths[dataset_id]
        return self.datasets[dataset_id][inner_index]

    def shuffle(self):
        for dataset in self.datasets:
            dataset.shuffle()

    def __len__(self):
        # Return the length of the largest dataset times the number of datasets
        return max(self.lengths) * len(self.datasets)
