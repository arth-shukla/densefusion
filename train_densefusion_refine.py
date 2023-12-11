from learning.loss import DenseFusionLoss, quat_to_rot, global_pred_quat_to_rot
from learning.densefusion import DenseFuseNet
from learning.densefusion_refine import DenseRefinerFuseNet
from learning.utils import compute_rre, compute_rte
from learning.utils import compute_rre, compute_rte, OBJ_NAMES, OBJ_NAMES_TO_IDX, IDX_TO_OBJ_NAMES
from benchmark_utils.pose_evaluator import PoseEvaluator

import torch
import torch.nn as nn
import numpy as np

import wandb
import uuid
import argparse
from pathlib import Path
import os
import time


LR_RATE = 0.3
W_RATE = 0.3


def handle_dirs(checkpoint_dir, refiner, optimizer, model, model_load_checkpoint, load_checkpoint=None, run_name=None):
    c_overall_dir = Path(checkpoint_dir)
    os.makedirs(c_overall_dir, exist_ok=True)

    print('Loading base model and optimizer...')
    model, _ = load_model(model, torch.optim.Adam(model.parameters()), load_path=Path(model_load_checkpoint))

    if load_checkpoint is not None:
        print('Loading refiner model and optimizer...')
        refiner, optimizer = load_model(refiner, optimizer, load_path=c_overall_dir / f'{load_checkpoint}' / 'latest.pt')

    cnum = 0
    for dirname in os.listdir(c_overall_dir):
        try: past_cnum = int(dirname)
        except: continue
        if (past_cnum >= cnum):
            cnum = past_cnum + 1 

    cdir = c_overall_dir / (f'{cnum}_refiner' if run_name is None else run_name)
    os.makedirs(cdir, exist_ok=True)

    return cdir, refiner, optimizer, model

def get_num_successes(R_pred, t_pred, R_gt, t_gt):
    def check_success(R_pred, t_pred, R_gt, t_gt):
        rre = np.rad2deg(compute_rre(R_pred, R_gt))
        rte = compute_rte(t_pred, t_gt)
        return int(rre < 5 and rte < 1), rre, rte
    
    batch_size = R_pred.size(0)
    R_p_cpu, t_p_cpu = R_pred.detach().cpu().numpy(), t_pred.detach().cpu().numpy()
    R_gt_cpu, t_gt_cpu = R_gt.detach().cpu().numpy(), t_gt.detach().cpu().numpy(),
    successes = 0
    rres, rtes = [], []
    for b in range(batch_size):
        s, rre, rte = check_success(
            R_p_cpu[b], t_p_cpu[b],
            R_gt_cpu[b], t_gt_cpu[b],
        )
        successes += s
        rres.append(rre)
        rtes.append(rte)
    print([f'{x:.2f}' for x in rres])
    print([f'{x:.2f}' for x in rtes])
    return successes, np.mean(rres), np.mean(rtes)

def save_model(model, optimizer, save_path):
    torch.save(dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
    ), save_path)

def load_model(model, optimizer, load_path, device=torch.device('cpu')):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def train_step(
        refiner, dfnet, dl, optimizer, loss_fn,
        train=True,
        refine_iters=2,
        device=torch.device('cpu'), epoch=0,
        print_batch_metrics=False
    ):

    print_pref = 'train' if train else 'val'
    num_data_points = 0
    tot_successes = 0
    step_loss = 0
    for batch, (cloud, rgb, model, choose, target, obj_idxs, pose) in enumerate(iter(dl)):
        batch_size = cloud.size(0)
        num_data_points += batch_size

        cloud = cloud.to(device).float()
        rgb = rgb.to(device).float()
        model = model.to(device).float()
        choose = choose.to(device).float()
        target = target.to(device).float()
        obj_idxs = obj_idxs.to(device)
        pose = pose.to(device).float()

        if train: optimizer.zero_grad()

        # get predictions
        cloud = cloud.transpose(2, 1)
        rgb = rgb.transpose(3, 2).transpose(2, 1)
        choose = choose.view(choose.size(0), -1)
        R_quat_pred, t_pred, c_pred, cemb = dfnet(cloud, rgb, choose, obj_idxs, ret_cemb=True)

        cloud = cloud.transpose(2, 1)
        R_pred = quat_to_rot(R_quat_pred)
        t_pred = t_pred.transpose(2, 1)
        c_pred = c_pred.unsqueeze(1)
        best_Rs = torch.stack([R_pred[b][torch.argmax(c_pred[b])] for b in range(batch_size)]).detach()
        best_ts = torch.stack([t_pred[b][torch.argmax(c_pred[b])] for b in range(batch_size)]).detach()
        new_cloud = (torch.bmm(cloud, best_Rs.transpose(2, 1)) + best_ts.unsqueeze(1)).detach()

        running_R, running_T = best_Rs.detach(), best_ts.detach()
        for _ in range(refine_iters):
            R_quat_pred, t_pred = refiner(new_cloud.transpose(2, 1), cemb, obj_idxs)
            R_pred = global_pred_quat_to_rot(R_quat_pred)

            loss = loss_fn(R_pred, t_pred, model, target, reduction='mean')
            if train: loss.backward()

            new_cloud = (torch.bmm(new_cloud, R_pred.transpose(2, 1)) + t_pred.unsqueeze(1)).detach()

            running_R = torch.bmm(running_R, R_pred.detach().transpose(2, 1))
            running_T = running_T + t_pred.detach()

            step_loss += loss

        # descent step
        optimizer.step()

        R_gt, t_gt = pose[:,:3,:3], pose[:,:3,3]
        successes, batch_rre, batch_rte = get_num_successes(running_R, running_T, R_gt, t_gt)
        accuracy = successes / batch_size
        tot_successes += successes

        if print_batch_metrics:
            print(f'\t\tepoch: {epoch}\tbatch: {batch+1}/{len(dl)}\t{print_pref}_acc: {accuracy:.4f}\t{print_pref}_loss: {loss.item() / batch_size:.4f}\t{print_pref}_running_acc: {tot_successes / num_data_points:.4f}\t{print_pref}_rre={batch_rre:.4f}\t{print_pref}_rte={batch_rte:.4f}')

    step_accuracy = tot_successes / num_data_points
    step_loss = step_loss / len(dl)
    print(f'epoch: {epoch}\t{print_pref}_acc: {step_accuracy}\t{print_pref}_loss: {step_loss}')
    
    return step_accuracy, step_loss

def train(
        refiner_model_cls, model_cls, loss_fn,
        train_dl, val_dl,
        model_load_checkpoint,
        epochs=1000, acc_req=0.95, lr=0.001, batch_size=-1,
        run_val_every=10,
        checkpoint_dir='checkpoints', load_checkpoint=None,
        wandb_logs=False, print_batch_metrics=False,
        run_name=None,
        data_dir = 'processed_data_new'
    ):
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('USING DEVICE', device)

    if run_name is not None: 
        run_name = f'{run_name}_refiner_{uuid.uuid4()}'

    # init pointnet
    num_objects = len(os.listdir(Path(data_dir) / 'models')) - 1
    model = model_cls(num_objects)
    model.to(device)
    model = torch.nn.parallel.DataParallel(model, device_ids=list(range(torch.cuda.device_count())), dim=0)

    refiner = refiner_model_cls(num_objects)
    refiner.to(device)
    refiner = torch.nn.parallel.DataParallel(refiner, device_ids=list(range(torch.cuda.device_count())), dim=0)

    optimizer = torch.optim.Adam(refiner.parameters(), lr=lr)

    # make checkpoint dir and load checkpoints
    cdir, refiner, optimizer, model = handle_dirs(checkpoint_dir + '_refiner', refiner, optimizer, model, model_load_checkpoint, load_checkpoint=load_checkpoint, run_name=run_name)

    if wandb_logs:
        run = wandb.init(
            project='PointNet 6D Pose Est',
            name=run_name,
            config=dict(
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                acc_req=acc_req,
                run_val_every=run_val_every,
                n_gpu=torch.cuda.device_count(),
            )
        )

    # run train loop
    for epoch in range(epochs):

        print(f'Starting epoch {epoch}...')


        if epoch >= 30:
            lr = lr * LR_RATE
            loss_fn.module.w = loss_fn.module.w * W_RATE
            optimizer = torch.optim.Adam(refiner.parameters(), lr=lr)


        train_accuracy, train_loss = train_step(
            refiner.train(), model.eval(), train_dl, optimizer, loss_fn,
            train=True,
            device=device, epoch=epoch,
            print_batch_metrics=print_batch_metrics
        )

        wandb_log = dict()

        # validation + wandb val visualizing
        if epoch % run_val_every == 0 and run_val_every > 0:
            with torch.no_grad():
                val_accuracy, val_loss = train_step(
                    refiner.eval(), model.eval(), val_dl, optimizer, loss_fn,
                    train=False,
                    device=device, epoch=epoch,
                    print_batch_metrics=print_batch_metrics
                )

            if wandb_logs:
                # log val metrics to wandb
                wandb_log['val/val_acc'] = val_accuracy
                wandb_log['val/val_loss'] = val_loss


        # logging to wandb
        if wandb_logs:
            wandb_log['train/train_acc'] = train_accuracy
            wandb_log['train/train_loss'] = train_loss
            run.log(wandb_log)

        # break if req accuracy reached
        if (train_accuracy > acc_req):
            break

        save_model(model, optimizer, save_path=cdir / f'epoch_{epoch}.pt')
        save_model(model, optimizer, save_path=cdir / 'latest.pt')

    return model

def run_training(
        refiner_model_cls, model_cls, loss_fn,
        model_load_checkpoint,
        batch_size = 8,
        epochs = 1000,
        lr = 0.0001,
        acc_req = 0.95,
        run_val_every = 3,
        checkpoint_dir = 'checkpoints',
        load_checkpoint = None,
        wandb_logs=True, 
        print_batch_metrics=True,
        run_name = 'pnet',
        data_dir = 'processed_data_new',
    ):

    from learning.load import PoseDataset, pad_train
    from torch.utils.data import DataLoader

    data_dir = Path(data_dir)

    train_ds = PoseDataset(data_dir=data_dir / 'train', cloud=True, rgb=True, model=True, choose=True, target=True)
    val_ds = PoseDataset(data_dir=data_dir / 'val', cloud=True, rgb=True, model=True, choose=True, target=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_train, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_train, num_workers=2)

    trained_dfnet = train(
        refiner_model_cls, model_cls, loss_fn,
        train_dl, val_dl, 
        model_load_checkpoint,
        epochs=epochs,
        acc_req=acc_req,
        lr=lr, 
        batch_size=batch_size,
        run_val_every=run_val_every,
        checkpoint_dir=checkpoint_dir,
        load_checkpoint=load_checkpoint,
        wandb_logs=wandb_logs, 
        print_batch_metrics=print_batch_metrics,
        run_name=run_name,
        data_dir=data_dir,
    )

if __name__ == '__main__':

    pose_evaluator = PoseEvaluator()

    inf_sim, n_sim, no_sim = [], [], []
    for obj_name in OBJ_NAMES:
        obj_data = pose_evaluator.objects_db[obj_name]
        # if obj_data['geometric_symmetry'] != 'no':
        #     sym_list.append(OBJ_NAMES_TO_IDX[obj_name])
        if obj_data['rot_axis'] is not None:
            inf_sim.append(OBJ_NAMES_TO_IDX[obj_name])
        elif len(obj_data['sym_rots']) > 1:
            n_sim.append(OBJ_NAMES_TO_IDX[obj_name])
        else:
            no_sim.append(OBJ_NAMES_TO_IDX[obj_name])

    sym_rots = dict((OBJ_NAMES_TO_IDX[name], pose_evaluator.objects_db[name]['sym_rots']) for name in OBJ_NAMES)
        
    print(inf_sim, n_sim, no_sim, sep='\n')
    print(dict((i, sym_rots[i].shape) for i in range(len(sym_rots))))

    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batches', type=int, default=16)
    parser.add_argument('-e', '--epochs', type=int, default=1000)
    parser.add_argument('-l', '--lr', type=float, default=0.0001)
    parser.add_argument('--model_load_checkpoint', type=str, default='')
    parser.add_argument('--acc_req', type=float, default=0.95)
    parser.add_argument('--val_every', type=int, default=3)
    parser.add_argument('-c', '--checkpoint_dir', type=str, default='checkpoints/densefusion')
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('-w', '--wandb', action='store_true')
    parser.add_argument('--print_batch_metrics', type=bool, default=True)
    parser.add_argument('--run_name', type=str, default='dfnet')
    parser.add_argument('-d', '--data_dir', type=str, default='processed_like_df')

    args = parser.parse_args()

    print(args)

    run_training(
        DenseRefinerFuseNet, DenseFuseNet, torch.nn.DataParallel(DenseFusionLoss(inf_sim=inf_sim, n_sim=n_sim, sym_rots=sym_rots, w=0.015, reduction='mean')),
        args.model_load_checkpoint,
        batch_size = args.batches,
        epochs = args.epochs,
        lr = args.lr,
        acc_req = args.acc_req,
        run_val_every = args.val_every,
        checkpoint_dir = args.checkpoint_dir,
        load_checkpoint = args.load_checkpoint,
        wandb_logs = args.wandb, 
        print_batch_metrics = args.print_batch_metrics,
        run_name = args.run_name,
        data_dir = args.data_dir
    )
