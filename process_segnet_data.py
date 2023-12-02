import os
import open3d as o3d
import pickle
from PIL import Image
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm


def process_raw_data(output_dir = 'processed_data'):
    def get_dists(p0, pts):
        return np.linalg.norm(p0 - pts, axis=-1)

    def fp_sampling(pts: np.ndarray, num_samples: int = 4000, seed: int = 0):
        np.random.seed(seed)
        fps = np.zeros((num_samples, pts.shape[-1]))
        fps[0] = pts[np.random.randint(0, len(pts))]
        dists = get_dists(fps[0], pts)
        for i in range(1, num_samples):
            fps[i] = pts[np.argmax(dists)]
            dists = np.minimum(dists, get_dists(fps[i], pts))
        return fps
    
    def get_stripped_lines(fp, levels=[1, 2, 3]):
        return [x.strip() for x in open(fp, 'r').readlines() if int(x[0]) in levels]
    
    def process_raw_obs_data(raw_obj_dir, scene_names, output_dir, processed_models_dir, test=False, min_ptcld_len=1, max_ptcld_len=1000):
        os.makedirs(output_dir, exist_ok=True)

        for scene_num, scene in enumerate(tqdm(scene_names)):
            color_path = raw_obj_dir / f'{scene}_color_kinect.png'
            label_path = raw_obj_dir / f'{scene}_label_kinect.png'

            color = np.array(Image.open(color_path)) / 255
            label = np.array(Image.open(label_path))

            np.save(output_dir / f'{scene_num}_color', color)
            np.save(output_dir / f'{scene_num}_label', label)

    raw_data_dir = Path('raw_data_all')

    raw_train_dir = raw_data_dir / 'training_data'

    raw_train_splits_dir = raw_train_dir / 'splits/v2'
    raw_train_obj_dir = raw_train_dir / 'v2.2'

    processed_data_dir = Path(output_dir)
    processed_train_dir = processed_data_dir / 'train'
    processed_val_dir = processed_data_dir / 'val'
    processed_models_dir = processed_data_dir / 'models'

    train_scene_names = get_stripped_lines(raw_train_splits_dir / 'train.txt')
    val_scene_names = get_stripped_lines(raw_train_splits_dir / 'val.txt')

    print('Processing val data...')
    process_raw_obs_data(raw_train_obj_dir, val_scene_names, processed_val_dir, processed_models_dir)
    print('Processing train data...')
    process_raw_obs_data(raw_train_obj_dir, train_scene_names, processed_train_dir, processed_models_dir)

if __name__ == '__main__':
    process_raw_data(output_dir='processed_segnet_data')