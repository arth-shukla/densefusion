import torch
import torch.nn as nn
import numpy as np
from learning.loss import densefusion_symmetry_aware_loss, densefusion_symmetry_unaware_loss, rre_rte_loss, quat_to_rot
from learning.densefusion import DenseFuseNet
from learning.utils import compute_rre, compute_rte

import wandb
import uuid

from pathlib import Path
import os
import time

def handle_dirs(checkpoint_dir, pnet, optimizer, load_checkpoint=None, run_name=None):
    c_overall_dir = Path(checkpoint_dir)
    os.makedirs(c_overall_dir, exist_ok=True)

    if load_checkpoint is not None:
        print('Loading pnet and optimizer...')
        pnet, optimizer = load_model(pnet, optimizer, load_path=c_overall_dir / f'{load_checkpoint}' / 'latest.pt')

    cnum = 0
    for dirname in os.listdir(c_overall_dir):
        try: past_cnum = int(dirname)
        except: continue
        if (past_cnum >= cnum):
            cnum = past_cnum + 1 

    cdir = c_overall_dir / (f'{cnum}' if run_name is None else run_name)
    os.makedirs(cdir, exist_ok=True)

    return cdir, pnet, optimizer

def get_num_successes(R_pred, t_pred, c_pred, R_gt, t_gt):
    def check_success(R_pred, t_pred, R_gt, t_gt):
        rre = np.rad2deg(compute_rre(R_pred, R_gt))
        rte = compute_rte(t_pred, t_gt)
        return int(rre < 5 and rte < 1), rre, rte
    
    batch_size = R_pred.size(0)
    R_p_cpu, t_p_cpu = R_pred.detach().cpu().numpy(), t_pred.detach().cpu().numpy()
    R_gt_cpu, t_gt_cpu = R_gt.detach().cpu().numpy(), t_gt.detach().cpu().numpy(),
    c_p = c_pred.detach().cpu().numpy()
    successes = 0
    rres, rtes = [], []
    for b in range(batch_size):
        best_ind = np.argmax(c_p[b])
        s, rre, rte = check_success(
            R_p_cpu[b][best_ind], t_p_cpu[b][best_ind],
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
        dfnet, dl, optimizer, loss_fn,
        train=True,
        device=torch.device('cpu'), epoch=0,
        print_batch_metrics=False
    ):

    print_pref = 'train' if train else 'val'
    num_data_points = 0
    tot_successes = 0
    step_loss = 0
    for batch, (cloud, rgb, model, choose, obj_idxs, pose) in enumerate(iter(dl)):
        batch_size = cloud.size(0)
        num_data_points += batch_size

        cloud = cloud.to(device).float()
        rgb = rgb.to(device).float()
        model = model.to(device).float()
        choose = choose.to(device).float()
        obj_idxs = obj_idxs.to(device)
        pose = pose.to(device).float()

        if train: optimizer.zero_grad()

        # get predictions
        cloud = cloud.transpose(2, 1)
        rgb = rgb.transpose(3, 2).transpose(2, 1)
        choose = choose.view(choose.size(0), -1)
        R_quat_pred, t_pred, c_pred = dfnet(cloud, rgb, choose, obj_idxs)

        R_pred = quat_to_rot(R_quat_pred)
        t_pred = t_pred.transpose(2, 1)

        # calc loss
        R_gt, t_gt = pose[:,:3,:3], pose[:,:3,3]
        loss = loss_fn(R_pred, t_pred, c_pred, R_gt, t_gt, model, reduction='mean')
        if train: loss.backward()
        step_loss += loss

        # descent step
        optimizer.step()

        successes, batch_rre, batch_rte = get_num_successes(R_pred, t_pred, c_pred, R_gt, t_gt)
        accuracy = successes / batch_size
        tot_successes += successes

        if print_batch_metrics:
            print(f'\t\tepoch: {epoch}\tbatch: {batch+1}/{len(dl)}\t{print_pref}_acc: {accuracy:.4f}\t{print_pref}_loss: {loss.item() / batch_size:.4f}\t{print_pref}_running_acc: {tot_successes / num_data_points:.4f}\t{print_pref}_rre={batch_rre:.4f}\t{print_pref}_rte={batch_rte:.4f}')

    step_accuracy = tot_successes / num_data_points
    step_loss = step_loss / len(dl)
    print(f'epoch: {epoch}\t{print_pref}_acc: {step_accuracy}\t{print_pref}_loss: {step_loss}')
    
    return step_accuracy, step_loss

def train(
        model_cls, loss_fn,
        train_dl, val_dl, 
        epochs=1000, acc_req=0.95, lr=0.001, batch_size=-1,
        run_val_every=10,
        checkpoint_dir='checkpoints', load_checkpoint=None,
        wandb_logs=False, print_batch_metrics=False,
        run_name=None,
    ):
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if run_name is not None: 
        run_name = f'{run_name}_{uuid.uuid4()}'

    # init pointnet
    num_objects = len(os.listdir('/arshukla-fast-vol-1/densefusion/processed_data_new/models')) - 1
    pnet = model_cls(num_objects)
    pnet.to(device)

    optimizer = torch.optim.Adam(pnet.parameters(), lr=lr)

    # make checkpoint dir and load checkpoints
    cdir, pnet, optimizer = handle_dirs(checkpoint_dir, pnet, optimizer, load_checkpoint=load_checkpoint, run_name=run_name)

    pnet = torch.nn.parallel.DataParallel(pnet, device_ids=list(range(torch.cuda.device_count())), dim=0)

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
            )
        )

    # run train loop
    for epoch in range(epochs):

        print(f'Starting epoch {epoch}...')

        train_accuracy, train_loss = train_step(
            pnet.train(), train_dl, optimizer, loss_fn,
            train=True,
            device=device, epoch=epoch,
            print_batch_metrics=print_batch_metrics
        )

        wandb_log = dict()

        # validation + wandb val visualizing
        if epoch % run_val_every == 0 and run_val_every > 0:
            with torch.no_grad():
                val_accuracy, val_loss = train_step(
                    pnet.eval(), val_dl, optimizer, loss_fn,
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

        save_model(pnet, optimizer, save_path=cdir / f'epoch_{epoch}.pt')
        save_model(pnet, optimizer, save_path=cdir / 'latest.pt')

    return pnet

def run_training(
        model_cls, loss_fn,
        batch_size = 8,
        epochs = 1000,
        lr = 0.0001,
        acc_req = 0.95,
        run_val_every = 3,
        checkpoint_dir = 'checkpoints',
        load_checkpoint = None,
        wandb_logs=True, 
        print_batch_metrics=True,
        run_name = 'pnet'
    ):

    from learning.load import PoseDataset, pad_train
    from torch.utils.data import DataLoader

    train_ds = PoseDataset(data_dir='/arshukla-fast-vol-1/densefusion/processed_data_new/train', cloud=True, rgb=True, model=True, choose=True)
    val_ds = PoseDataset(data_dir='/arshukla-fast-vol-1/densefusion/processed_data_new/val', cloud=True, rgb=True, model=True, choose=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_train, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_train, num_workers=2)

    trained_dfnet = train(
        model_cls, loss_fn,
        train_dl, val_dl, 
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
    )

if __name__ == '__main__':
    run_training(
        DenseFuseNet, densefusion_symmetry_unaware_loss,
        batch_size = 16,
        epochs = 1000,
        lr = 0.0001,
        acc_req = 0.95,
        run_val_every = 3,
        checkpoint_dir = '/arshukla-fast-vol-1/densefusion/checkpoints/densefusion',
        # load_checkpoint = 'dfnet-sym_unaware-resume_0bda885f-7b23-4981-8e4c-c9c4ea49c3af',
        wandb_logs=True, 
        print_batch_metrics=True,
        run_name = 'dfnet-sym_unaware-mgpu'
    )
