import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class SimpleRefinerFusingPointNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ptcloud covolutions
        self.conv1_pt = torch.nn.Conv1d(3, 64, 1)
        self.conv2_pt = torch.nn.Conv1d(64, 128, 1)

        # per-pixel color embedding convolutions
        self.conv1_rgb = torch.nn.Conv1d(32, 64, 1)
        self.conv2_rgb = torch.nn.Conv1d(64, 128, 1)

        # per-pixel feature convolutions
        self.conv1_comb = torch.nn.Conv1d(384, 512, 1)
        self.conv2_comb = torch.nn.Conv1d(512, 1024, 1)

        # pointnet uses maxpool for perm. invariance
        # densefusion uses avgpool instead, intuitively
        #   because correct orientation of an object is
        #   partially determined by overall surface texture
        self.pooling = torch.nn.AvgPool1d

    # densefuse point and color embeddings
    def forward(self, pts, cemb):
        num_pts = pts.size(-1)

        pts = F.relu(self.conv1_pt(pts))
        cemb = F.relu(self.conv1_rgb(cemb))
        pix_feat_1 = torch.cat((pts, cemb), 1)

        pts = F.relu(self.conv2_pt(pts))
        cemb = F.relu(self.conv2_rgb(cemb))
        pix_feat_2 = torch.cat((pts, cemb), 1)

        pix_feat_3 = torch.cat([pix_feat_1, pix_feat_2], dim=1)

        comb = F.relu(self.conv1_comb(pix_feat_3))
        comb = F.relu(self.conv2_comb(comb))

        return self.pooling(num_pts)(comb).squeeze(-1)


class DenseRefinerFuseNet(nn.Module):
    def __init__(self, num_objs):
        super().__init__()

        self.num_objs = num_objs

        self.pointnet = SimpleRefinerFusingPointNet()

        # similar to regular densefusion implementation,
        # but on global feat so no confidence vals
        self.lin1_r = nn.Linear(1024, 512, 1)
        self.lin2_r = nn.Linear(512, 128, 1)
        self.lin3_r = nn.Linear(128, num_objs*4, 1) # quat

        self.lin1_t = nn.Linear(1024, 512, 1)
        self.lin2_t = nn.Linear(512, 128, 1)
        self.lin3_t = nn.Linear(128, num_objs*3, 1) # translation

    def forward(self, pts, cemb, obj_idx, ret_cemb=False):
        batches = pts.size(0)

        pix_wise_global = self.pointnet(pts, cemb)

        # per-point estimation of Rs, ts, cs
        rx = F.relu(self.lin1_r(pix_wise_global))
        rx = F.relu(self.lin2_r(rx))
        rx = F.relu(self.lin3_r(rx))

        tx = F.relu(self.lin1_t(pix_wise_global))
        tx = F.relu(self.lin2_t(tx))
        tx = F.relu(self.lin3_t(tx))

        rx = rx.view(batches, self.num_objs, 4)
        tx = tx.view(batches, self.num_objs, 3)

        rout = torch.stack([rx[b][obj_idx[b]] for b in range(obj_idx.size(0))]).squeeze(1).contiguous()
        tout = torch.stack([tx[b][obj_idx[b]] for b in range(obj_idx.size(0))]).squeeze(1).contiguous()

        if ret_cemb:
            return rout, tout, cemb.detach()

        # 4-dim quat, 3-dim t, 1-dim c
        return rout, tout
    