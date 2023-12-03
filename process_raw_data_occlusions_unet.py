import os
import open3d as o3d
import pickle
from PIL import Image
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import trimesh
import cv2
import torch
from segnet.unet import get_unet_cls

def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


def get_bbox(bbox):
    img_width = 720
    img_length = 1280
    step = 40
    border_list = [-1] + np.arange(step, img_width+step+step, step=step).tolist()

    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= img_width:
        bbx[1] = img_width-1
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= img_length:
        bbx[3] = img_length-1                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


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

    def crop_img_using_mask(img, choose, mask, ret_masked=True):
        if ret_masked:
            img = img * mask[..., None]
        keep = np.ix_(mask.any(1),mask.any(0))
        return img[keep], choose[keep]
    
    def process_raw_obs_data(raw_obj_dir, scene_names, output_dir, processed_models_dir, test=False, min_ptcld_len=1, max_ptcld_len=1000):
        dp_num = 0
        scene_num = 0

        os.makedirs(output_dir, exist_ok=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        unet = get_unet_cls(bilinear=True)().to(device).eval()
        unet = torch.nn.parallel.DataParallel(unet, device_ids=list(range(torch.cuda.device_count())), dim=0)
        optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
        unet, optimizer = load_model(unet, optimizer, 'checkpoints/unet-final.pt', device=device)
        unet = unet.module.to(device).eval()

        scene_dir = output_dir / 'scene_masks'
        rgb_dir = output_dir / 'rgbs'
        depth_dir = output_dir / 'depths'
        intrinsic_dir = output_dir / 'intrinsics'
        inv_extrinsic_dir = output_dir / 'inv_extrinsics'
        for dirname in [scene_dir, rgb_dir, depth_dir, intrinsic_dir, inv_extrinsic_dir]:
            os.makedirs(dirname, exist_ok=True)

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

            scene_mask = np.zeros_like(label)
            intrinsic = meta['intrinsic']
            inv_extrinsic = np.linalg.inv(meta['extrinsic'])

            for obj_id, obj_name in zip(meta['object_ids'], meta['object_names']):
                mask = (label.copy() == obj_id)

                rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask))
                mask_zeroed = np.zeros_like(mask)
                mask_zeroed[rmin:rmax, cmin:cmax] = mask[rmin:rmax, cmin:cmax]
                mask = mask_zeroed.copy()

                scene_mask += mask

                if np.all(mask == False):
                    continue
                
                scale = np.array(meta['scales'][obj_id])

                model = np.array(o3d.io.read_point_cloud(str(processed_models_dir / f'{obj_name}.pcd')).points) * scale
                new_meta = dict(
                    obj_id=obj_id,
                    obj_name=obj_name,
                    rmin=rmin, rmax=rmax,
                    cmin=cmin, cmax=cmax,
                    scene_num=scene_num,
                )

                np.save(output_dir / f'{dp_num}_model', model)
                np.save(output_dir / f'{dp_num}_mask', mask)
                with open(output_dir / f'{dp_num}_meta.pkl', 'wb') as handle:
                    pickle.dump(new_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

                if not test:
                    pose = meta['poses_world'][obj_id]
                    np.save(output_dir / f'{dp_num}_pose', pose)
                    target = model @ pose[:3, :3].T + (pose[:3, 3])
                    np.save(output_dir / f'{dp_num}_target', target)

                dp_num += 1

            scene_mask = scene_mask > 0
            np.save(scene_dir / f'{scene_num}_scene_mask', scene_mask)
            np.save(rgb_dir / f'{scene_num}_rgb', color)
            np.save(depth_dir / f'{scene_num}_depth', depth)
            np.save(intrinsic_dir / f'{scene_num}_intrinsic', intrinsic)
            np.save(inv_extrinsic_dir / f'{scene_num}_inv_extrinsic', inv_extrinsic)
            scene_num += 1
    
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

    # print('Farthest point sampling model pointclouds...')
    # os.makedirs(processed_models_dir, exist_ok=True)
    # for obj in tqdm(os.listdir(raw_models_dir)):
    #     if str(obj) == '.gitignore': continue
    #     model = trimesh.load(raw_models_dir / obj / 'visual_meshes/visual.dae', force='mesh')
    #     pts, _ = trimesh.sample.sample_surface(model, 10000, seed=0)
    #     pts = fp_sampling(pts, num_samples=1000)

    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(pts)
    #     o3d.io.write_point_cloud(str(processed_models_dir / f'{obj}.pcd'), pcd)

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
    process_raw_data(output_dir='processed_occlusions_unet')