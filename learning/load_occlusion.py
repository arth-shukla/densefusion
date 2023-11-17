import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from learning.utils import OBJ_NAMES, OBJ_NAMES_TO_IDX
from PIL import Image
import numpy as np
import pickle
import random

import os
from pathlib import Path
from typing import Union

class PoseOcclusionDataset(Dataset):
    def __init__(
            self, 
            data_dir: Union[str, Path],
            train=True,
            cloud=False, cloud_rgb=False, rgb=False, model=False, choose=False, target=False,
            image_base_size=(240, 240),
            add_noise=False,
            transform=torch.from_numpy,
            max_ptcld_len=1000,
            n_occlusion_masks=2,
        ):
        self.data_dir = Path(data_dir)
        self.scene_mask_dir = self.data_dir / 'scene_masks'
        self.rgb_dir = self.data_dir / 'rgbs'
        self.depth_dir = self.data_dir / 'depths'
        self.instrinsic_dir = self.data_dir / 'intrinsics'
        self.inv_extrinsic_dir = self.data_dir / 'inv_extrinsics'

        self.n_scene_masks = len(os.listdir(self.scene_mask_dir))
        self.n_occlusion_masks = n_occlusion_masks

        self.train = train
        self.cloud, self.cloud_rgb, self.rgb, self.model, self.choose, self.target = cloud, cloud_rgb, rgb, model, choose, target
        self.image_base_size = image_base_size
        self.transform = transform
        self.max_ptcld_len = max_ptcld_len

        self.len = int(len(os.listdir(data_dir)) / (5 if train else 3))

        self.add_noise = add_noise
        if self.add_noise:
            import torchvision.transforms as transforms
            self.img_noise = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
            self.translate_noise = 0.03

    def __getitem__(self, index):

        meta = pickle.load(open(self.data_dir / f'{index}_meta.pkl', 'rb'))
        rmin, rmax, cmin, cmax = meta['rmin'], meta['rmax'], meta['cmin'], meta['cmax']
        scene_num = meta['scene_num']

        rgb = np.load(self.rgb_dir / f'{scene_num}_rgb.npy')
        depth = np.load(self.depth_dir / f'{scene_num}_depth.npy')
        intrinsic = np.load(self.instrinsic_dir / f'{scene_num}_intrinsic.npy')
        inv_extrinsic = np.load(self.inv_extrinsic_dir / f'{scene_num}_inv_extrinsic.npy')

        mask = np.load(self.data_dir / f'{index}_mask.npy').astype(np.int8)
        n_successful_occlusions = 0
        mask_len = np.sum(mask)
        if mask_len > 200:
            while True:
                scene_mask = np.load(self.scene_mask_dir / f'{random.randint(0, self.n_scene_masks-1)}_scene_mask.npy').astype(np.int8)
                new_mask = mask * (-scene_mask + 1)
                new_mask_len = np.sum(new_mask)
                if (mask_len >= 700 and new_mask_len >= 500) or (mask_len < 700 and new_mask_len > 100):
                    mask = new_mask
                    n_successful_occlusions += 1
                if n_successful_occlusions >= self.n_occlusion_masks:
                    break

        keep_ms, keep_ns = np.nonzero(mask > 0)
        if len(keep_ms) > self.max_ptcld_len:
            keep_idxs = random.sample(list(range(len(keep_ms))), self.max_ptcld_len)
            keep_ms = keep_ms[keep_idxs]
            keep_ns = keep_ns[keep_idxs]
            choose = np.zeros_like(mask)
            choose[(keep_ms, keep_ns)] = 1
        else:
            choose = mask.copy()

        cloud, cloud_rgb = self.get_ptcld_from_depth_using_mask(depth, choose, rgb, intrinsic, inv_extrinsic)

        rgb = rgb * mask.reshape(*mask.shape, 1)
        rgb = rgb[rmin:rmax, cmin:cmax]
        choose = choose[rmin:rmax, cmin:cmax]

        rets = []

        if self.add_noise:
            add_t = self.translate_noise * (2 * np.random.rand(3) - 1)

        if self.cloud:
            if self.add_noise:
                cloud += add_t
            rets.append(cloud)
        if self.cloud_rgb:
            rets.append(cloud_rgb)
        if self.rgb:
            if self.add_noise:
                rgb_pil = Image.fromarray(np.uint8(rgb * 255)).convert('RGB')
                rgb = np.array(self.img_noise(rgb_pil)) / 255
            base = np.zeros((*self.image_base_size, rgb.shape[-1]))
            base[:rgb.shape[0],:rgb.shape[1]] = rgb
            rets.append(base)
        if self.model:
            rets.append(np.load(self.data_dir / f'{index}_model.npy'))
        if self.choose:
            base = np.zeros(self.image_base_size)
            base[:choose.shape[0],:choose.shape[1]] = choose
            rets.append(base)
        if self.target:
            target = np.load(self.data_dir / f'{index}_target.npy')
            if self.add_noise:
                target += add_t
            rets.append(target)
        # obj_idx required for any model to work
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
    
    @staticmethod
    def get_ptcld_from_depth_using_mask(depth, choose, rgb, intrinsic, inv_extrinsic):
        def apply_transformation(x, T):
            return x @ T[:3, :3].T + T[:3, 3]

        z = depth * choose
        v, u = np.indices(z.shape)

        uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
        points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]

        pv = points_viewer.reshape([-1, 3])
        keep_pts = ~np.all(pv == 0, axis=1)

        pts_rgb = rgb.reshape([-1, 3])[keep_pts]

        return apply_transformation(pv[keep_pts], inv_extrinsic), pts_rgb

    def __len__(self):
        return self.len
    
if __name__ == '__main__':
    print(int(len(os.listdir('processed_data/train')) / 4))
