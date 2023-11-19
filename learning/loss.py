import torch
from torch.nn.modules.loss import _Loss

def densefusion_loss(
        R_pred: torch.Tensor, t_pred: torch.Tensor, c_pred: torch.Tensor,
        model: torch.Tensor, target: torch.Tensor, obj_idx,
        inf_sim=[],
        n_sim=[],
        sym_rots=dict(),
        w=0.015, reduction='mean',
    ):
    def pairwise_dist(xyz1, xyz2):
        r_xyz1 = torch.sum(xyz1 * xyz1, dim=2, keepdim=True)
        r_xyz2 = torch.sum(xyz2 * xyz2, dim=2, keepdim=True)
        mul = torch.matmul(xyz2, xyz1.permute(0,2,1))
        dist = r_xyz2 - 2 * mul + r_xyz1.permute(0,2,1)
        return dist

    bs, ps = R_pred.size(0), R_pred.size(1)

    Rps = R_pred.view(-1, 3, 3)
    tps = t_pred.view(-1, 3)
    cps = c_pred.view(-1, 1).squeeze(-1)

    ms = model.unsqueeze(1).repeat(1, ps, 1, 1).view(bs*ps, -1, 3)
    gt_transform = target.unsqueeze(1).repeat(1, ps, 1, 1).view(bs*ps, -1, 3)

    obj_idx = obj_idx[0].item()
    if obj_idx in inf_sim:
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
    elif obj_idx in n_sim:
        srots = torch.from_numpy(sym_rots[obj_idx]).to(Rps.get_device()).float()
        all_sims = ms @ srots.unsqueeze(1)
        preds = all_sims @ Rps.transpose(2, 1) + tps.unsqueeze(1)
        dists = torch.mean(torch.norm(preds - gt_transform, dim=3), dim=2)
        losses = torch.mean(dists * cps - w * torch.log(cps), dim=1)
        loss = torch.min(losses)
    else:
        pred_transform = torch.bmm(ms, Rps.transpose(2, 1)) + tps.unsqueeze(1)
        # per-point mse loss
        dists = torch.mean(torch.norm(pred_transform - gt_transform, dim=2), dim=1)
        loss = dists * cps - w * torch.log(cps)
        loss = loss.mean()

    return loss

class DenseFusionLoss(_Loss):

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

        return densefusion_loss(
            R_pred, t_pred, c_pred,
            model, target, obj_idx,
            inf_sim=self.inf_sim,
            n_sim=self.n_sim,
            sym_rots=self.sym_rots,
            w=self.w, reduction=self.reduction
        )


def quat_to_rot(bquats: torch.Tensor, base=1e-15):
    """
        bquats: Bs x Ps x 4
        base: avoids zero-div error w/ norms when converting to unit quaternion

        output: Bs x Ps x 3 x 3
    """

    bs, ps = bquats.size(0), bquats.size(1)

    bquats += torch.tensor(base)
    bquats = bquats / torch.norm(bquats, dim=2).view(bs, ps, 1)
    pred_r = bquats

    R = torch.cat([
        (1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, ps, 1),
        (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, ps, 1),
        (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, ps, 1),
        (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, ps, 1),
        (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, ps, 1),
        (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, ps, 1),
        (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, ps, 1),
        (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, ps, 1),
        (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, ps, 1)
    ], dim=2).contiguous().view(bs, ps, 3, 3).contiguous()

    return R


def global_pred_like_df_loss(
        R_pred: torch.Tensor, t_pred: torch.Tensor,
        model: torch.Tensor, target: torch.Tensor,
        reduction='mean',
    ):
    """
       more like df loss
    """

    bs, ps = R_pred.size(0), R_pred.size(1)

    Rps = R_pred.view(-1, 3, 3)
    tps = t_pred.reshape(-1, 3)

    pred_transform = torch.bmm(model, Rps.transpose(2, 1)) + tps.unsqueeze(1)

    loss = torch.mean(torch.norm(pred_transform - target, dim=2), dim=1)

    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()

    return loss


def global_pred_quat_to_rot(bquats: torch.Tensor, base=1e-15):
    """
        bquats: Bs x 4 x Ps
        base: avoids zero-div error w/ norms when converting to unit quaternion

        output: Bs x Ps x 3 x 3
    """

    bs = bquats.size(0)

    bquats += torch.tensor(base)
    bquats = bquats / torch.norm(bquats, dim=1).view(bs, 1)

    r11 = 1 - 2*(bquats[:, 2]**2 + bquats[:, 3]**2)
    r12 = 2*(bquats[:, 1]*bquats[:, 2] - bquats[:, 0]*bquats[:, 3])
    r13 = 2*(bquats[:, 1]*bquats[:, 3] + bquats[:, 0]*bquats[:, 2])

    r21 = 2*(bquats[:, 1]*bquats[:, 2] + bquats[:, 0]*bquats[:, 3])
    r22 = 1 - 2*(bquats[:, 1]**2 + bquats[:, 3]**2)
    r23 = 2*(bquats[:, 2]*bquats[:, 3] - bquats[:, 0]*bquats[:, 1])

    r31 = 2*(bquats[:, 1]*bquats[:, 3] - bquats[:, 0]*bquats[:, 2])
    r32 = 2*(bquats[:, 2]*bquats[:, 3] + bquats[:, 0]*bquats[:, 1])
    r33 = 1 - 2*(bquats[:, 1]**2 + bquats[:, 2]**2)

    R = torch.stack([r11, r12, r13, r21, r22, r23, r31, r32, r33])
    R = R.transpose(1, 0)
    R = R.view(bs, 3, 3)
    return R

def densefusion_symmetry_aware_loss(
        R_pred: torch.Tensor, t_pred: torch.Tensor, c_pred: torch.Tensor,
        model: torch.Tensor, target: torch.Tensor,
        w=0.015, reduction='mean'
    ):
    return densefusion_loss(
        R_pred, t_pred, c_pred,
        model, target,
        symmetry_aware=True,
        w=w, reduction=reduction
    )

def densefusion_symmetry_unaware_loss(
        R_pred: torch.Tensor, t_pred: torch.Tensor, c_pred: torch.Tensor,
        model: torch.Tensor, target: torch.Tensor,
        w=0.015, reduction='mean'
    ):
    return densefusion_loss(
        R_pred, t_pred, c_pred,
        model, target,
        symmetry_aware=False,
        w=w, reduction=reduction
    )

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]