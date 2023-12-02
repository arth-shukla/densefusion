import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

import os
from pathlib import Path
from typing import Union

class SegmentationDataset(Dataset):
    def __init__(
            self, 
            data_dir: Union[str, Path],
            train=True,
            add_noise=False,
            transform=torch.from_numpy,
        ):
        self.data_dir = Path(data_dir)

        self.train = train
        self.transform = transform

        self.len = int(len(os.listdir(data_dir)) / (2 if train else 1))

        self.add_noise = add_noise
        if self.add_noise:
            import torchvision.transforms as transforms
            self.img_noise = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

    def __getitem__(self, index):
        rgb = np.load(self.data_dir / f'{index}_color.npy')

        if not self.train:
            return rgb

        if self.add_noise:
            rgb_pil = Image.fromarray(np.uint8(rgb * 255)).convert('RGB')
            rgb = np.array(self.img_noise(rgb_pil)) / 255

        label = np.load(self.data_dir / f'{index}_label.npy')

        return rgb, label

    def __len__(self):
        return self.len