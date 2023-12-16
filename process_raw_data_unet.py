from segnet.unet import get_unet_cls
from learning.utils import mask_to_bbox, get_bbox

import torch
import numpy as np

import open3d as o3d
import trimesh
from PIL import Image
from pathlib import Path
import os
import pickle
import random
from tqdm import tqdm


def process_raw_data(output_dir = 'processed_data'):
    def load_model(model, optimizer, load_path, device=torch.device('cpu')):
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer

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
    
    def get_stripped_lines(fp, levels=[1, 2, 3], test=False):
        return [
            x.strip() for x in open(fp, 'r').readlines()
            if (
                int(x.split('-')[0]) in levels and
                (
                    test or 
                    (int(x.split('-')[1]) <= 100 and int(x.split('-')[2]) <= 20)
                )
            )
        ]

    def get_ptcld_from_depth_using_mask(depth, mask, rgb, intrinsic, inv_extrinsic):
        def apply_transformation(x, T):
            return x @ T[:3, :3].T + T[:3, 3]

        z = depth * mask
        v, u = np.indices(z.shape)

        uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
        points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]

        pv = points_viewer.reshape([-1, 3])
        keep_pts = ~np.all(pv == 0, axis=1)

        pts_rgb = rgb.reshape([-1, 3])[keep_pts]

        return apply_transformation(pv[keep_pts], inv_extrinsic), pts_rgb

    def crop_img_using_mask(img, choose, mask, ret_masked=True):
        if ret_masked:
            img = img * mask[..., None]
        keep = np.ix_(mask.any(1),mask.any(0))
        return img[keep], choose[keep]
    
    def process_raw_obs_data(raw_obj_dir, scene_names, output_dir, processed_models_dir, test=False, min_ptcld_len=1, max_ptcld_len=1000):
        dp_num = 0

        os.makedirs(output_dir, exist_ok=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        unet = get_unet_cls(bilinear=True)().to(device).eval()
        unet = torch.nn.parallel.DataParallel(unet, device_ids=list(range(torch.cuda.device_count())), dim=0)
        optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
        unet, optimizer = load_model(unet, optimizer, 'select-checkpoints/unet.pt', device=device)
        unet = unet.module.to(device).eval()

        for scene in tqdm(scene_names):
            color_path = raw_obj_dir / f'{scene}_color_kinect.png'
            depth_path = raw_obj_dir / f'{scene}_depth_kinect.png'
            meta_path = raw_obj_dir / f'{scene}_meta.pkl'

            color = np.array(Image.open(color_path)) / 255
            depth = np.array(Image.open(depth_path)) / 1000
            meta = pickle.load(open(meta_path, 'rb'))

            rgb_torch = torch.moveaxis(torch.from_numpy(color).float().to(device).unsqueeze(0), -1, 1)
            pred = unet(rgb_torch)
            label = torch.argmax(pred, dim=1).squeeze(0).detach().cpu().numpy()

            inv_extrinsic = np.linalg.inv(meta['extrinsic'])

            for obj_id, obj_name in zip(meta['object_ids'], meta['object_names']):
                mask = (label.copy() == obj_id)

                rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask))
                mask_zeroed = np.zeros_like(mask)
                mask_zeroed[rmin:rmax, cmin:cmax] = mask[rmin:rmax, cmin:cmax]
                mask = mask_zeroed.copy()

                keep_ms, keep_ns = np.nonzero(mask > 0)
                if len(keep_ms) > max_ptcld_len:
                    keep_idxs = random.sample(list(range(len(keep_ms))), max_ptcld_len)
                    keep_ms = keep_ms[keep_idxs]
                    keep_ns = keep_ns[keep_idxs]
                    choose = np.zeros_like(mask)
                    choose[(keep_ms, keep_ns)] = 1
                else:
                    choose = mask.copy()
                
                scale = np.array(meta['scales'][obj_id])

                pts, pts_rgb = get_ptcld_from_depth_using_mask(depth, choose, color, meta['intrinsic'], inv_extrinsic)
                if len(pts) < min_ptcld_len:
                    continue

                color_masked = color.copy() * mask.reshape(*mask.shape, 1)
                cropped_img = color_masked[rmin:rmax, cmin:cmax]
                choose = choose[rmin:rmax, cmin:cmax]
                # mask = mask[rmin:rmax, cmin:cmax]

                model = np.array(o3d.io.read_point_cloud(str(processed_models_dir / f'{obj_name}.pcd')).points) * scale
                new_meta = dict(
                    obj_id=obj_id,
                    obj_name=obj_name,
                )

                np.save(output_dir / f'{dp_num}_point_cloud', pts)
                np.save(output_dir / f'{dp_num}_point_cloud_rgb', pts_rgb)
                np.save(output_dir / f'{dp_num}_cropped_rgb', cropped_img)
                # np.save(output_dir / f'{dp_num}_mask', mask)
                np.save(output_dir / f'{dp_num}_model', model)
                np.save(output_dir / f'{dp_num}_choose', choose)
                with open(output_dir / f'{dp_num}_meta.pkl', 'wb') as handle:
                    pickle.dump(new_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

                if not test:
                    pose = meta['poses_world'][obj_id]
                    np.save(output_dir / f'{dp_num}_pose', pose)
                    target = model @ pose[:3, :3].T + (pose[:3, 3])
                    np.save(output_dir / f'{dp_num}_target', target)

                dp_num += 1
    
    raw_data_dir = Path('raw_data_all')

    raw_train_dir = raw_data_dir / 'training_data'
    raw_test_dir = raw_data_dir / 'testing_data'
    raw_models_dir = raw_data_dir / 'models'

    raw_train_splits_dir = raw_train_dir / 'splits/v2'
    raw_train_obj_dir = raw_train_dir / 'v2.2'
    raw_test_obj_dir = raw_test_dir / 'v2.2'

    processed_data_dir = Path(output_dir)
    processed_train_dir = processed_data_dir / 'train'
    processed_val_dir = processed_data_dir / 'val'
    processed_test_dir = processed_data_dir / 'test'
    processed_models_dir = processed_data_dir / 'models'

    print('Farthest point sampling model pointclouds...')
    os.makedirs(processed_models_dir, exist_ok=True)
    for obj in tqdm(os.listdir(raw_models_dir)):
        if str(obj) == '.gitignore': continue
        model = trimesh.load(raw_models_dir / obj / 'visual_meshes/visual.dae', force='mesh')
        pts, _ = trimesh.sample.sample_surface(model, 10000, seed=0)
        pts = fp_sampling(pts, num_samples=1000)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(str(processed_models_dir / f'{obj}.pcd'), pcd)

    train_scene_names = get_stripped_lines(raw_train_splits_dir / 'train.txt')
    val_scene_names = get_stripped_lines(raw_train_splits_dir / 'val.txt')
    test_scene_names = get_stripped_lines(raw_test_dir / 'test.txt', test=True)

    with torch.no_grad():
        print('Processing test data...')
        process_raw_obs_data(raw_test_obj_dir, test_scene_names, processed_test_dir, processed_models_dir, test=True, min_ptcld_len=-np.inf)
        print('Processing val data...')
        process_raw_obs_data(raw_train_obj_dir, val_scene_names, processed_val_dir, processed_models_dir)
        print('Processing train data...')
        process_raw_obs_data(raw_train_obj_dir, train_scene_names, processed_train_dir, processed_models_dir)

if __name__ == '__main__':
    process_raw_data(output_dir='processed_data_unet')