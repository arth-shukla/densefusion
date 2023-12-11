# %%
from icp import run_icp, rigid_transform, R_and_T
import numpy as np

# %%
def pred(ds, data_idx, pose_evaluator, max_attempts=100, max_iters=1000):
    from learning.utils import IDX_TO_OBJ_NAMES
    cloud, model, obj_idx, pose = ds[data_idx]
    if len(cloud) < 1:
        R_pred, t_pred = R_and_T(np.eye(4))
    else:
        R_pred_cloud, t_pred_cloud = R_and_T(run_icp(cloud, model, max_attempts=max_attempts, max_iters=max_iters))
        R_pred, t_pred = R_and_T(rigid_transform(model, (model - t_pred_cloud) @ R_pred_cloud))
    return pose_evaluator.evaluate(IDX_TO_OBJ_NAMES[obj_idx[0]], R_pred, pose[:3, :3], t_pred, pose[:3, 3])

# %%
def pred_over_ds(data_dir='processed_for_icp/val', max_attempts=100, max_iters=1000):
    from benchmark_utils.pose_evaluator import PoseEvaluator
    pose_evaluator = PoseEvaluator()

    from learning.load import PoseDataset
    ds = PoseDataset(data_dir=data_dir, cloud=True, model=True, transform=None)
    
    successes, rre_syms, rres, rtes = [], [], [], []
    pbar = tqdm(range(len(ds)))
    for i in pbar:
        eval = pred(ds, i, pose_evaluator, max_attempts=max_attempts, max_iters=max_iters)
        rre_syms.append(eval['rre_symmetry'])
        rres.append(eval['rre'])
        rtes.append(eval['rte'])
        successes.append(int(rre_syms[-1] <= 5 and rtes[-1] <= 0.01))
        pbar.set_description('\t'.join([
            'running rates', f'success={np.mean(successes):.4f}',
            f'rre_sym={np.mean(rre_syms):.4f}', f'rte={np.mean(rtes):.4f}']
        ))
    return np.mean(successes)

# %%
# pred_over_ds(data_dir='processed_for_icp_2/val', max_attempts=10, max_iters=1000)

# %%
# pred_over_ds(data_dir='processed_for_icp_2/val', max_attempts=100, max_iters=1000)

# %%
from learning.load import PoseDataset
from pathlib import Path
from tqdm import tqdm
import pickle

def pred_over_raw_data(output_json_name='icp_pred.json', raw_data_dir='raw_data', processed_data_dir='processed_for_icp_2', max_attempts=100, max_iters=1000, thresh=1e-5):
    def get_stripped_lines(fp, levels=[1, 2, 3]):
        return [x.strip() for x in open(fp, 'r').readlines() if int(x[0]) in levels]

    raw_data_dir = Path(raw_data_dir)
    raw_test_dir = raw_data_dir / 'testing_data'
    raw_test_obj_dir = raw_test_dir / 'v2.2'
    processed_data_dir = Path(processed_data_dir)
    processed_test_dir = processed_data_dir / 'test'

    test_scene_names = get_stripped_lines(raw_test_dir / 'test.txt')
    test_ds = PoseDataset(data_dir=processed_test_dir, train=False, cloud=True, model=True, transform=None)

    data_point_num = 0
    all_data = dict()
    pbar = tqdm(test_scene_names)
    for scene_name in pbar:
        meta_path = raw_test_obj_dir / f'{scene_name}_meta.pkl'
        meta = pickle.load(open(meta_path, 'rb'))

        scene_data = dict(poses_world=[None] * 79)

        for obj_id, obj_name in zip(meta['object_ids'], meta['object_names']):

            pbar.set_description(f'dp_num={data_point_num}')

            cloud, model, _ = test_ds[data_point_num]
            if len(cloud) < 1:
                T = np.eye(4)
            else:
                R_pred_cloud, t_pred_cloud = R_and_T(run_icp(cloud, model, max_attempts=max_attempts, max_iters=max_iters, finish_loop_thresh=thresh, acceptable_thresh=thresh))
                T = rigid_transform(model, (model - t_pred_cloud) @ R_pred_cloud)

            scene_data['poses_world'][obj_id] = T.tolist()

            data_point_num += 1

        all_data[scene_name] = scene_data

    import json
    with open(output_json_name, 'w') as fp:
        json.dump(all_data, fp)

# %% [markdown]
# ## HW2: Levels 1-2

# %%
# pred_over_raw_data(output_json_name='icp_pred.json', raw_data_dir='raw_data_all', processed_data_dir='processed_for_icp_2', max_attempts=100, max_iters=1000, thresh=-np.inf)

# %% [markdown]
# ## HW3: Levels 1-3

# %%
pred_over_raw_data(output_json_name='icp_pred_unet.json', raw_data_dir='raw_data_all', processed_data_dir='processed_data_unet', max_attempts=100, max_iters=1000, thresh=-np.inf)


