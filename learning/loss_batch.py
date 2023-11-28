import torch
from torch.nn.modules.loss import _Loss

class DenseFusionLossBatch(_Loss):

    def __init__(self, inf_sim=[], n_sim=[], sym_rots=dict(), w=0.015, reduction='mean'):
        super().__init__()
        self.inf_sim = inf_sim
        self.n_sim = n_sim
        self.sym_rots = sym_rots
        self.w = w
        self.reduction = reduction

    def forward(
            self,
            R_pred: torch.Tensor, t_pred: torch.Tensor, c_pred: torch.Tensor,
            model: torch.Tensor, target: torch.Tensor, obj_idx,
        ):

        return densefusion_loss_batch(
            R_pred, t_pred, c_pred,
            model, target, obj_idx,
            inf_sim=self.inf_sim,
            n_sim=self.n_sim,
            sym_rots=self.sym_rots,
            w=self.w, reduction=self.reduction
        )

def densefusion_loss_batch(
        R_pred: torch.Tensor, t_pred: torch.Tensor, c_pred: torch.Tensor,
        model: torch.Tensor, target: torch.Tensor, obj_idx,
        inf_sim=[],
        n_sim=[],
        sym_rots=dict(),
        w=0.015, reduction='mean',
    ):
    inf_sim_bidxs, n_sim_bidxs, no_sim_bidxs = [], [], []

    for i, x in enumerate(obj_idx.squeeze(-1).tolist()):
        if x in inf_sim:
            inf_sim_bidxs.append(i)
        elif x in n_sim:
            n_sim_bidxs.append(i)
        else:
            no_sim_bidxs.append(i)


    inf_sim_loss, n_sim_loss, no_sim_loss = 0, 0, 0
    if len(inf_sim_bidxs) > 0:
        inf_sim_loss = cham_loss(
            R_pred[inf_sim_bidxs],
            t_pred[inf_sim_bidxs],
            c_pred[inf_sim_bidxs],
            model[inf_sim_bidxs],
            target[inf_sim_bidxs],
            w=w, reduction=reduction,
        )
    if len(n_sim_bidxs) > 0:
        n_sim_loss = min_of_n_loss(
            R_pred[n_sim_bidxs],
            t_pred[n_sim_bidxs],
            c_pred[n_sim_bidxs],
            model[n_sim_bidxs],
            target[n_sim_bidxs],
            obj_idx[n_sim_bidxs],
            sym_rots=sym_rots,
            w=w, reduction=reduction,
        )
    if len(no_sim_bidxs) > 0:
        no_sim_loss = per_point_mse_loss(
            R_pred[no_sim_bidxs],
            t_pred[no_sim_bidxs],
            c_pred[no_sim_bidxs],
            model[no_sim_bidxs],
            target[no_sim_bidxs],
            w=w, reduction=reduction,
        )
    
    bs = R_pred.size(0)

    return inf_sim_loss * (len(inf_sim_bidxs) / bs) + \
        n_sim_loss * (len(n_sim_bidxs) / bs) + \
        no_sim_loss * (len(no_sim_bidxs) / bs)

def cham_loss(
    R_pred: torch.Tensor, t_pred: torch.Tensor, c_pred: torch.Tensor,
    model: torch.Tensor, target: torch.Tensor,
    w=0.015, reduction='mean',
):
    def pairwise_dist(xyz1, xyz2):
        r_xyz1 = torch.sum(xyz1 * xyz1, dim=2, keepdim=True)
        r_xyz2 = torch.sum(xyz2 * xyz2, dim=2, keepdim=True)
        mul = torch.matmul(xyz2, xyz1.permute(0,2,1))
        dist = r_xyz2 - 2 * mul + r_xyz1.permute(0,2,1)
        return dist
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    bs, ps = R_pred.size(0), R_pred.size(1)

    Rps = R_pred.view(-1, 3, 3)
    tps = t_pred.view(-1, 3)
    cps = c_pred.view(-1, 1).squeeze(-1)

    ms = model.unsqueeze(1).repeat(1, ps, 1, 1).view(bs*ps, -1, 3)
    gt_transform = target.unsqueeze(1).repeat(1, ps, 1, 1).view(bs*ps, -1, 3)

    pred_transform = torch.bmm(ms, Rps.transpose(2, 1)) + tps.unsqueeze(1)
    # chamfer loss
    closest_to_x_inds = []
    for b_range in chunks(range(bs*ps), bs*3):
        pwise_dist = pairwise_dist(pred_transform[b_range], gt_transform[b_range])
        closest_inds = torch.argmin(pwise_dist, dim=1)
        closest = torch.stack([gt_transform[b][closest_inds[i]] for i, b in enumerate(b_range)])
        closest_to_x_inds.append(closest)
    closest_gt_pts = torch.cat(closest_to_x_inds)
    dists = torch.mean(torch.norm(pred_transform - closest_gt_pts, dim=2), dim=1)
    loss = dists * cps - w * torch.log(cps)
    loss = loss.mean()

    return loss


def min_of_n_loss(
    R_pred: torch.Tensor, t_pred: torch.Tensor, c_pred: torch.Tensor,
    model: torch.Tensor, target: torch.Tensor, obj_idx,
    sym_rots=dict(),
    w=0.015, reduction='mean',
):
    og_ps = R_pred.size(1)

    srots, Rps, tps, cps, ms, targs = [], [], [], [], [], []
    srot_sizes = []
    for b in range(obj_idx.size(0)):
        oidx = obj_idx[b][0].item()
        srot = torch.from_numpy(sym_rots[oidx]).to(R_pred.get_device()).float()
        srots.append(srot)
        srot_sizes.append(srot.size(0))

        rp = R_pred[b].unsqueeze(0).repeat(srot.size(0), 1, 1, 1)
        Rps.append(rp)

        tp = t_pred[b].unsqueeze(0).repeat(srot.size(0), 1, 1)
        tps.append(tp)

        cp = c_pred[b].unsqueeze(0).repeat(srot.size(0), 1, 1)
        cps.append(cp)

        m = model[b].unsqueeze(0).repeat(srot.size(0), 1, 1)
        ms.append(m)

        targ = target[b].unsqueeze(0).repeat(srot.size(0), 1, 1)
        targs.append(targ)

    srots = torch.cat(srots)
    Rps = torch.cat(Rps)
    tps = torch.cat(tps)
    cps = torch.cat(cps)
    ms = torch.cat(ms)
    targs = torch.cat(targs)

    all_sims = ms @ srots
    all_sims = all_sims.unsqueeze(1).repeat(1, og_ps, 1, 1)
    targs = targs.unsqueeze(1).repeat(1, og_ps, 1, 1)

    srot_bs = srots.size(0)

    all_sims = all_sims.view(srot_bs*og_ps, -1, 3)
    targs = targs.view(srot_bs*og_ps, -1, 3)
    Rps = Rps.view(-1, 3, 3)
    tps = tps.view(-1, 3)
    cps = cps.view(-1, 1).squeeze(-1)

    preds = all_sims @ Rps.transpose(-1, -2) + tps.unsqueeze(-2)
    dists = torch.mean(torch.norm(preds - targs, dim=2), dim=1)
    loss = dists * cps - w * torch.log(cps)

    prev = 0
    final_losses = []
    for b in range(obj_idx.size(0)):
        srot_s = srot_sizes[b]

        curr = prev + og_ps * srot_s
        curr_split = loss[prev:curr].view(srot_s, og_ps, 1)

        curr_split_losses = torch.mean(curr_split, dim=1)
        curr_split_loss = torch.min(curr_split_losses)
        final_losses.append(curr_split_loss)

        prev = curr

    final_losses = torch.stack(final_losses)
    return final_losses.mean()


def per_point_mse_loss(
    R_pred: torch.Tensor, t_pred: torch.Tensor, c_pred: torch.Tensor,
    model: torch.Tensor, target: torch.Tensor,
    w=0.015, reduction='mean',
):
    bs, ps = R_pred.size(0), R_pred.size(1)

    Rps = R_pred.view(-1, 3, 3)
    tps = t_pred.view(-1, 3)
    cps = c_pred.view(-1, 1).squeeze(-1)

    ms = model.unsqueeze(1).repeat(1, ps, 1, 1).view(bs*ps, -1, 3)
    gt_transform = target.unsqueeze(1).repeat(1, ps, 1, 1).view(bs*ps, -1, 3)

    pred_transform = torch.bmm(ms, Rps.transpose(2, 1)) + tps.unsqueeze(1)
    # per-point mse loss
    dists = torch.mean(torch.norm(pred_transform - gt_transform, dim=2), dim=1)
    loss = dists * cps - w * torch.log(cps)
    loss = loss.mean()

    return loss