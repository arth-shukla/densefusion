from learning.densefusion import DenseFuseNet
from learning.loss_batch import DenseFusionLossBatch, quat_to_rot
from learning.utils import OBJ_NAMES, OBJ_NAMES_TO_IDX, IDX_TO_OBJ_NAMES
from benchmark_utils.pose_evaluator import PoseEvaluator

import torch
import numpy as np

import wandb
import uuid
import argparse
from pathlib import Path
import os


def get_success_metrics(R_pred, t_pred, c_pred, R_gt, t_gt, obj_idxs):
    free = lambda x: x.detach().cpu().numpy()
    R_pred, t_pred, c_pred = free(R_pred), free(t_pred), free(c_pred)
    R_gt, t_gt = free(R_gt), free(t_gt)

    s, rresy, rre, rte = [], [], [], []
    s_rre, s_rte = [], []
    for b in range(R_pred.shape[0]):
        evaluation = pose_evaluator.evaluate(
            IDX_TO_OBJ_NAMES[obj_idxs[b].item()],
            R_pred[b][np.argmax(c_pred[b].squeeze())],
            R_gt[b],
            t_pred[b][np.argmax(c_pred[b].squeeze())],
            t_gt[b],
        )
        s.append(evaluation['rre_symmetry'] <= 5 and evaluation['rte'] <= 0.01)
        s_rre.append(evaluation['rre_symmetry'] <= 5)
        s_rte.append(evaluation['rte'] <= 0.01)
        rresy.append(evaluation['rre_symmetry'])
        rre.append(evaluation['rre'])
        rte.append(evaluation['rte'])
    return np.sum(s), np.sum(s_rre), np.sum(s_rte), np.mean(rresy), np.mean(rre), np.mean(rte)

def handle_dirs(checkpoint_dir, pnet, optimizer, load_checkpoint=None, run_name=None):
    c_overall_dir = Path(checkpoint_dir)
    os.makedirs(c_overall_dir, exist_ok=True)

    if load_checkpoint is not None:
        print('Loading pnet and optimizer...')
        if '.pt' in str(load_checkpoint):
            pnet, optimizer = load_model(pnet, optimizer, load_path=c_overall_dir / f'{load_checkpoint}')
        else:
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

def save_model(model, optimizer, save_path):
    torch.save(dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
    ), save_path)

def load_model(model, optimizer, load_path, device=torch.device('cpu'), compatibility=False):
    checkpoint = torch.load(load_path, map_location=device)
    if compatibility:
        for key, num in [
            ('module.conv4_r.weight', 4), ('module.conv4_r.bias', 4),
            ('module.conv4_t.weight', 3), ('module.conv4_t.bias', 3),
            ('module.conv4_c.weight', 1), ('module.conv4_c.bias', 1)
        ]:
            given = checkpoint['model'][key]
            desired0dim = len(OBJ_NAMES) * num
            needed_dim = desired0dim - given.size(0)
            needed = torch.zeros(needed_dim, *given.shape[1:]).to(device)
            final = torch.cat([given, needed], 0)
            checkpoint['model'][key] = final
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
    num_data_points, tot_successes, tot_rre_sym, tot_rre, tot_rte = 0, 0, 0, 0, 0
    tot_s_rre, tot_s_rte = 0, 0
    step_loss = 0
    for iter_num, (cloud, rgb, model, choose, target, obj_idxs, pose) in enumerate(iter(dl)):
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
        rgb = torch.moveaxis(rgb, -1, 1)
        choose = choose.view(choose.size(0), -1)
        R_quat_pred, t_pred, c_pred = dfnet(cloud, rgb, choose, obj_idxs)

        R_pred = quat_to_rot(R_quat_pred)

        # calc loss
        loss = loss_fn(R_pred, t_pred, c_pred, model, target, obj_idxs).mean()
        if train: loss.backward()
        step_loss += loss

        # descent step
        if train: optimizer.step()

        successes, s_rre, s_rte, rre_sym, rre, rte = get_success_metrics(R_pred, t_pred, c_pred, pose[:, :3, :3], pose[:, :3, 3], obj_idxs)
        tot_successes += successes
        tot_s_rre += s_rre
        tot_s_rte += s_rte
        tot_rre_sym += rre_sym
        tot_rre += rre
        tot_rte += rte
            
        if print_batch_metrics:
            print(f'\t\tepoch: {epoch}\titer: {iter_num+1}/{len(dl)}\t{print_pref}_acc: {successes/batch_size:.4f}\t{print_pref}_loss: {loss.item():.4f}\t{print_pref}_running_acc: {tot_successes / num_data_points:.4f}\t{print_pref}_rre_sym={rre_sym:.4f}\t{print_pref}_rre={rre:.4f}\t{print_pref}_rte={rte:.4f}\t{print_pref}_rre_acc={s_rre/batch_size:.4f}\t{print_pref}_rte_acc={s_rte/batch_size:.4f}\t{print_pref}_running_rre_acc={tot_s_rre/num_data_points:.4f}\t{print_pref}_running_rte_acc={tot_s_rte/num_data_points:.4f}')

    step_accuracy = tot_successes / num_data_points
    rre_acc = tot_s_rre / num_data_points
    rte_acc = tot_s_rte / num_data_points
    tot_rre_sym /= len(dl)
    tot_rre /= len(dl)
    tot_rte /= len(dl)
    step_loss = step_loss / len(dl)
    print(f'epoch: {epoch}\t{print_pref}_acc: {step_accuracy}\t{print_pref}_loss: {step_loss}\t{print_pref}_rre_sym: {tot_rre_sym}\t{print_pref}_rre: {tot_rre}\t{print_pref}_rte: {tot_rte}\t{print_pref}_rre_acc={rre_acc:.4f}\t{print_pref}_rte_acc={rte_acc:.4f}')
    
    return step_accuracy, rre_acc, rte_acc, step_loss, tot_rre_sym, tot_rre, tot_rte

def train(
        model_cls, loss_fn,
        train_dl, val_dl, 
        epochs=1000, acc_req=0.95, lr=0.001, batch_size=-1,
        run_val_every=10,
        checkpoint_dir='checkpoints', load_checkpoint=None,
        wandb_logs=False, print_batch_metrics=False,
        run_name=None,
        data_dir = 'processed_data_new',
        do_decay = False,
        min_over_cham_prob = 0.1,
    ):
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('USING DEVICE', device)

    if run_name is not None: 
        run_name = f'{run_name}_{uuid.uuid4()}'

    # init pointnet
    num_objects = len(OBJ_NAMES)
    pnet = model_cls(num_objects)
    pnet.to(device)
    pnet = torch.nn.parallel.DataParallel(pnet, device_ids=list(range(torch.cuda.device_count())), dim=0)

    optimizer = torch.optim.Adam(pnet.parameters(), lr=lr)

    # make checkpoint dir and load checkpoints
    cdir, pnet, optimizer = handle_dirs(checkpoint_dir, pnet, optimizer, load_checkpoint=load_checkpoint, run_name=run_name)

    for g in optimizer.param_groups:
        g['lr'] = lr

    print('w', loss_fn.module.w)
    print('opt', optimizer.state_dict())

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
                min_over_cham_prob=min_over_cham_prob,
            )
        )

    # run train loop
    for epoch in range(epochs):

        print(f'Starting epoch {epoch}...')

        train_accuracy, train_rre_acc, train_rte_acc, train_loss, train_rre_sym, train_rre, train_rte = train_step(
            pnet.train(), train_dl, optimizer, loss_fn,
            train=True,
            device=device, epoch=epoch,
            print_batch_metrics=print_batch_metrics
        )

        if wandb_logs:
            wandb_log = dict()

        # validation + wandb val visualizing
        if epoch % run_val_every == 0 and run_val_every > 0:
            with torch.no_grad():
                val_accuracy, val_rre_acc, val_rte_acc, val_loss, val_rre_sym, val_rre, val_rte = train_step(
                    pnet.eval(), val_dl, optimizer, loss_fn,
                    train=False,
                    device=device, epoch=epoch,
                    print_batch_metrics=print_batch_metrics
                )

            if wandb_logs:
                # log val metrics to wandb
                wandb_log['val/val_acc'] = val_accuracy
                wandb_log['val/train_rre_acc'] = val_rre_acc
                wandb_log['val/train_rte_acc'] = val_rte_acc
                wandb_log['val/val_loss'] = val_loss
                wandb_log['val/val_rre_sym'] = val_rre_sym
                wandb_log['val/val_rre'] = val_rre
                wandb_log['val/val_rte'] = val_rte


        # logging to wandb
        if wandb_logs:
            wandb_log['train/train_acc'] = train_accuracy
            wandb_log['train/train_rre_acc'] = train_rre_acc
            wandb_log['train/train_rte_acc'] = train_rte_acc
            wandb_log['train/train_loss'] = train_loss
            wandb_log['train/train_rre_sym'] = train_rre_sym
            wandb_log['train/train_rre'] = train_rre
            wandb_log['train/train_rte'] = train_rte
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
        run_name = 'pnet',
        data_dir = 'processed_data_new',
        add_train_noise = False,
        occlusion_data_dir = None,
        dl_workers = 0,
        do_decay = False,
        min_over_cham_prob = 0.1,
    ):

    from learning.load import PoseDataset, pad_train
    from learning.load_occlusion import PoseOcclusionDataset
    from torch.utils.data import DataLoader

    data_dir = Path(data_dir)

    if occlusion_data_dir is not None:
        occlusion_data_dir = Path(occlusion_data_dir)
        train_ds = PoseOcclusionDataset(data_dir=occlusion_data_dir / 'train', cloud=True, rgb=True, model=True, choose=True, target=True, add_noise=add_train_noise)
    else:
        train_ds = PoseDataset(data_dir=data_dir / 'train', cloud=True, rgb=True, model=True, choose=True, target=True, add_noise=add_train_noise)
    val_ds = PoseDataset(data_dir=data_dir / 'val', cloud=True, rgb=True, model=True, choose=True, target=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_train, num_workers=dl_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_train, num_workers=0)

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
        data_dir=data_dir,
        do_decay=do_decay,
        min_over_cham_prob=min_over_cham_prob,
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
    parser.add_argument('--loss_w', type=float, default=0.015)
    parser.add_argument('--acc_req', type=float, default=0.95)
    parser.add_argument('--val_every', type=int, default=3)
    parser.add_argument('-c', '--checkpoint_dir', type=str, default='checkpoints/densefusion')
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--print_batch_metrics', type=bool, default=True)
    parser.add_argument('--run_name', type=str, default='dfnet')
    parser.add_argument('-d', '--data_dir', type=str, default='processed_like_df')
    parser.add_argument('--add_train_noise', action='store_true')
    parser.add_argument('--occlusion_data_dir', default=None)
    parser.add_argument('--dl_workers', type=int, default=0)
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--min_over_cham_prob', type=float, default=0.1)

    args = parser.parse_args()

    print(args)

    run_training(
        DenseFuseNet, torch.nn.DataParallel(DenseFusionLossBatch(
            inf_sim=inf_sim, n_sim=n_sim, sym_rots=sym_rots, w=args.loss_w, reduction='mean',
            min_over_cham_prob=args.min_over_cham_prob,
        )),
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
        occlusion_data_dir = args.occlusion_data_dir,
        dl_workers = args.dl_workers,
        do_decay = args.decay,
        min_over_cham_prob = args.min_over_cham_prob,
    )
