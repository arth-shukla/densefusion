import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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
            cloud=False, cloud_rgb=False, rgb=False, model=False, choose=False,
            image_base_size=(240, 240),
            transform=torch.from_numpy,
        ):
        self.data_dir = Path(data_dir)

        self.train = train
        self.cloud, self.cloud_rgb, self.rgb, self.model, self.choose = cloud, cloud_rgb, rgb, model, choose
        self.image_base_size = image_base_size
        self.transform = transform

        self.len = int(len(os.listdir(data_dir)) / (7 if train else 6))

        # self.dps = [None] * len(self.obj_data)

    def __getitem__(self, index):

        rets = []

        if self.cloud:
            rets.append(np.load(self.data_dir / f'{index}_point_cloud.npy'))
        if self.cloud_rgb:
            rets.append(np.load(self.data_dir / f'{index}_point_cloud_rgb.npy'))
        if self.rgb:
            rgb = np.load(self.data_dir / f'{index}_cropped_rgb.npy')
            base = np.zeros((*self.image_base_size, rgb.shape[-1]))
            base[:rgb.shape[0],:rgb.shape[1]] = rgb
            rets.append(base)
        if self.model:
            rets.append(np.load(self.data_dir / f'{index}_model.npy'))
        if self.choose:
            choose = np.load(self.data_dir / f'{index}_choose.npy')
            base = np.zeros(self.image_base_size)
            base[:choose.shape[0],:choose.shape[1]] = choose
            rets.append(base)
        # obj_idx required for any model to work
        meta = pickle.load(open(self.data_dir / f'{index}_meta.pkl', 'rb'))
        rets.append(np.array([meta['obj_id']]))
        if self.train:
            rets.append(np.load(self.data_dir / f'{index}_pose.npy'))

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
