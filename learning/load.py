import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from learning.utils import OBJ_NAMES, OBJ_NAMES_TO_IDX
from PIL import Image
import numpy as np
import pickle

import os
from pathlib import Path
from typing import Union

class PoseDataset(Dataset):
    def __init__(
            self, 
            data_dir: Union[str, Path],
            train=True,
            cloud=False, cloud_rgb=False, rgb=False, model=False, choose=False, target=False,
            image_base_size=(240, 240),
            add_noise=False,
            transform=torch.from_numpy,
        ):
        self.data_dir = Path(data_dir)

        self.train = train
        self.cloud, self.cloud_rgb, self.rgb, self.model, self.choose, self.target = cloud, cloud_rgb, rgb, model, choose, target
        self.image_base_size = image_base_size
        self.transform = transform

        self.len = int(len(os.listdir(data_dir)) / (8 if train else 6))

        self.add_noise = add_noise
        if self.add_noise:
            import torchvision.transforms as transforms
            self.img_noise = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            self.translate_noise = 0.03

        # self.dps = [None] * len(self.obj_data)

    def __getitem__(self, index):

        rets = []

        if self.add_noise:
            add_t = self.translate_noise * (2 * np.random.rand(3) - 1)

        if self.cloud:
            cloud = np.load(self.data_dir / f'{index}_point_cloud.npy')
            if self.add_noise:
                cloud += add_t
            rets.append(cloud)
        if self.cloud_rgb:
            rets.append(np.load(self.data_dir / f'{index}_point_cloud_rgb.npy'))
        if self.rgb:
            rgb = np.load(self.data_dir / f'{index}_cropped_rgb.npy')
            if self.add_noise:
                rgb_pil = Image.fromarray(np.uint8(rgb * 255)).convert('RGB')
                rgb = np.array(self.img_noise(rgb_pil)) / 255
            # rets.append(rgb)
            base = np.zeros((*self.image_base_size, rgb.shape[-1]))
            base[:rgb.shape[0],:rgb.shape[1]] = rgb
            rets.append(base)
        if self.model:
            rets.append(np.load(self.data_dir / f'{index}_model.npy'))
        if self.choose:
            choose = np.load(self.data_dir / f'{index}_choose.npy')
            # rets.append(choose)
            base = np.zeros(self.image_base_size)
            base[:choose.shape[0],:choose.shape[1]] = choose
            rets.append(base)
        if self.target:
            target = np.load(self.data_dir / f'{index}_target.npy')
            if self.add_noise:
                target += add_t
            rets.append(target)
        # obj_idx required for any model to work
        meta = pickle.load(open(self.data_dir / f'{index}_meta.pkl', 'rb'))
        rets.append(np.array([OBJ_NAMES_TO_IDX[meta['obj_name']]]))
        if self.train:
            pose = np.load(self.data_dir / f'{index}_pose.npy')
            if self.add_noise:
                pose[:3, 3] += add_t
            rets.append(pose)

        if self.transform:
            for i, x in enumerate(rets):
                rets[i] = self.transform(x)

        return tuple(rets)

    def __len__(self):
        return self.len

def pad_test(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)

def pad_train(batch):
    rets = []
    for x in zip(*batch):
        padded = pad_sequence(x, batch_first=True, padding_value=0)
        rets.append(padded)
    return tuple(rets)
    
if __name__ == '__main__':
    print(int(len(os.listdir('processed_data/train')) / 4))
