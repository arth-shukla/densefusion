import torch
import torch.nn.functional as F
from segnet.segnet import SegNet
import numpy as np

import wandb
import uuid

from pathlib import Path
import argparse
import os


def handle_dirs(checkpoint_dir, segnet, optimizer, load_checkpoint=None, run_name=None):
    c_overall_dir = Path(checkpoint_dir)
    os.makedirs(c_overall_dir, exist_ok=True)

    if load_checkpoint is not None:
        print('Loading segnet and optimizer...')
        if '.pt' in str(load_checkpoint):
            segnet, optimizer = load_model(segnet, optimizer, load_path=c_overall_dir / f'{load_checkpoint}')
        else:
            segnet, optimizer = load_model(segnet, optimizer, load_path=c_overall_dir / f'{load_checkpoint}' / 'latest.pt')

    cnum = 0
    for dirname in os.listdir(c_overall_dir):
        try: past_cnum = int(dirname)
        except: continue
        if (past_cnum >= cnum):
            cnum = past_cnum + 1 

    cdir = c_overall_dir / (f'{cnum}' if run_name is None else run_name)
    os.makedirs(cdir, exist_ok=True)

    return cdir, segnet, optimizer

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
        segnet, dl, optimizer, loss_fn,
        train=True,
        device=torch.device('cpu'), epoch=0,
        print_batch_metrics=False
    ):

    print_pref = 'train' if train else 'val'
    num_correct, num_pix = 0, 0
    step_acc, step_loss = 0, 0
    for iter_num, (rgb, label) in enumerate(iter(dl)):
        batch_size = rgb.size(0)

        rgb = rgb.to(device).float()
        label = label.to(device).long()

        if train: optimizer.zero_grad()

        # get predictions
        rgb = torch.moveaxis(rgb, -1, 1)
        pred = segnet(rgb)

        # calc loss
        loss = loss_fn(pred, label).mean()
        if train: loss.backward()
        step_loss += loss

        # descent step
        if train: optimizer.step()

        # accuracy
        pred, label = pred.detach(), label.detach()
        pred = torch.argmax(pred, dim=1)
        correct = torch.sum(pred == label).item()
        pixels = pred.nelement()
        batch_acc = correct / pixels
        num_correct += correct
        num_pix += pixels
        running_acc = num_correct / num_pix
            
        if print_batch_metrics:
            print(f'\t\tepoch: {epoch}\titer: {iter_num+1}/{len(dl)}\t{print_pref}_running_acc: {running_acc:.4f}\t{print_pref}_acc: {batch_acc:.4f}\t{print_pref}_loss: {loss.item():.4f}')

    step_loss = step_loss / len(dl)
    step_acc = num_correct / num_pix
    print(f'epoch: {epoch}\t{print_pref}_acc: {step_acc}\t{print_pref}_loss: {step_loss}')
    
    return step_loss

def train(
        model_cls, loss_fn,
        train_dl, val_dl, 
        epochs=1000, acc_req=0.95, lr=0.001, batch_size=-1,
        run_val_every=10,
        checkpoint_dir='checkpoints', load_checkpoint=None,
        wandb_logs=False, print_batch_metrics=False,
        run_name=None,
        data_dir = 'processed_data_new',
    ):
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('USING DEVICE', device)

    if run_name is not None: 
        run_name = f'{run_name}_{uuid.uuid4()}'

    # init segnet
    segnet = model_cls().to(device)
    # segnet = torch.nn.parallel.DataParallel(segnet, device_ids=list(range(torch.cuda.device_count())), dim=0)

    optimizer = torch.optim.Adam(segnet.parameters(), lr=lr)

    # make checkpoint dir and load checkpoints
    cdir, segnet, optimizer = handle_dirs(checkpoint_dir, segnet, optimizer, load_checkpoint=load_checkpoint, run_name=run_name)

    for g in optimizer.param_groups:
        g['lr'] = lr

    if wandb_logs:
        run = wandb.init(
            project='SegNet Semantic Segmentation',
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

        train_loss = train_step(
            segnet.train(), train_dl, optimizer, loss_fn,
            train=True,
            device=device, epoch=epoch,
            print_batch_metrics=print_batch_metrics
        )

        if wandb_logs:
            wandb_log = dict()

        if epoch % run_val_every == 0 and run_val_every > 0:
            with torch.no_grad():
                val_loss = train_step(
                    segnet.eval(), val_dl, optimizer, loss_fn,
                    train=False,
                    device=device, epoch=epoch,
                    print_batch_metrics=print_batch_metrics
                )

            if wandb_logs:
                wandb_log['val/val_loss'] = val_loss

        if wandb_logs:
            wandb_log['train/train_loss'] = train_loss
            run.log(wandb_log)

        save_model(segnet, optimizer, save_path=cdir / f'epoch_{epoch}.pt')
        save_model(segnet, optimizer, save_path=cdir / 'latest.pt')

    return segnet

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
        run_name = 'segnet',
        data_dir = 'processed_data_new',
        add_train_noise = False,
        dl_workers = 0,
    ):

    from segnet.load import SegmentationDataset
    from torch.utils.data import DataLoader

    data_dir = Path(data_dir)

    train_ds = SegmentationDataset(data_dir=data_dir / 'train', add_noise=add_train_noise)
    val_ds = SegmentationDataset(data_dir=data_dir / 'val')
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=dl_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    trained_segnet = train(
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
        data_dir=data_dir,
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batches', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--acc_req', type=float, default=0.95)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/segnet')
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--print_batch_metrics', type=bool, default=True)
    parser.add_argument('--run_name', type=str, default='segnet')
    parser.add_argument('--data_dir', type=str, default='processed_segnet_data')
    parser.add_argument('--add_train_noise', action='store_true')
    parser.add_argument('--dl_workers', type=int, default=0)

    args = parser.parse_args()

    print(args)

    run_training(
        SegNet, F.cross_entropy,
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
        data_dir = args.data_dir,
        add_train_noise = args.add_train_noise,
        dl_workers = args.dl_workers,
    )
