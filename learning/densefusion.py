import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from pspnet.pspnet import PSPNet

class SimpleFusingPointNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ptcloud covolutions
        self.conv1_pt = torch.nn.Conv1d(3, 64, 1)
        self.conv2_pt = torch.nn.Conv1d(64, 128, 1)

        # per-pixel color embedding convolutions
        self.conv1_rgb = torch.nn.Conv1d(32, 64, 1)
        self.conv2_rgb = torch.nn.Conv1d(64, 128, 1)

        # per-pixel feature convolutions
        self.conv1_comb = torch.nn.Conv1d(256, 512, 1)
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

        comb = F.relu(self.conv1_comb(pix_feat_2))
        comb = F.relu(self.conv2_comb(comb))

        global_feat = self.pooling(num_pts)(comb)
        gf_repeat = global_feat.repeat(1, 1, num_pts)

        dense_fused = torch.cat((pix_feat_1, pix_feat_2, gf_repeat), 1)

        return dense_fused


class DenseFuseNet(nn.Module):
    def __init__(self, num_objs):
        super().__init__()

        self.num_objs = num_objs

        # Note that DenseFusion uses this pre-existing implementation from
        # https://github.com/Lextal/pspnet-pytorch/tree/master
        # I use the same resnet18-based PSPNet as them for speed in training
        self.psp_net = PSPNet(sizes=(1, 2, 3, 6), n_classes=32, psp_size=512, deep_features_size=256, backend='resnet18', pretrained=False)
        # self.psp_net = nn.parallel.DataParallel(self.psp_net)

        self.pointnet = SimpleFusingPointNet()

        # same as in my PointNet implementation, but with larger starting dims
        self.conv1_r = nn.Conv1d(1408, 512, 1)
        self.conv2_r = nn.Conv1d(512, 256, 1)
        self.conv3_r = nn.Conv1d(256, 128, 1)
        self.conv4_r = nn.Conv1d(128, num_objs*4, 1) # quat

        self.conv1_t = nn.Conv1d(1408, 512, 1)
        self.conv2_t = nn.Conv1d(512, 256, 1)
        self.conv3_t = nn.Conv1d(256, 128, 1)
        self.conv4_t = nn.Conv1d(128, num_objs*3, 1) # translation

        self.conv1_c = nn.Conv1d(1408, 512, 1)
        self.conv2_c = nn.Conv1d(512, 256, 1)
        self.conv3_c = nn.Conv1d(256, 128, 1)
        self.conv4_c = nn.Conv1d(128, num_objs*1, 1) # confidence

    def forward(self, pts, rgb, choose, obj_idx, ret_cemb=False):

        num_pts = pts.size(-1)

        cemb, _ = self.psp_net(rgb)
        batches, emb_ds = cemb.size(0), cemb.size(1)
        cemb = cemb.view(batches, emb_ds, -1)

        cemb = cemb.transpose(2, 1)
        cemb_keep = [cemb[b][torch.nonzero(choose[b])].squeeze(1) for b in range(batches)]
        cemb = pad_sequence(cemb_keep, batch_first=True, padding_value=0)
        cemb = F.pad(cemb, (0, 0, 0, num_pts - cemb.size(1)), 'constant', 0)
        cemb = cemb.transpose(2, 1)

        pix_wise_fuse = self.pointnet(pts, cemb)

        # per-point estimation of Rs, ts, cs
        rx = F.relu(self.conv1_r(pix_wise_fuse))
        rx = F.relu(self.conv2_r(rx))
        rx = F.relu(self.conv3_r(rx))
        rx = F.relu(self.conv4_r(rx))

        tx = F.relu(self.conv1_t(pix_wise_fuse))
        tx = F.relu(self.conv2_t(tx))
        tx = F.relu(self.conv3_t(tx))
        tx = F.relu(self.conv4_t(tx))

        cx = F.relu(self.conv1_c(pix_wise_fuse))
        cx = F.relu(self.conv2_c(cx))
        cx = F.relu(self.conv3_c(cx))
        cx = F.relu(self.conv4_c(cx))

        rx = rx.view(batches, self.num_objs, 4, num_pts)
        tx = tx.view(batches, self.num_objs, 3, num_pts)
        cx = torch.sigmoid(cx).view(batches, self.num_objs, 1, num_pts)

        rout = torch.stack([rx[b][obj_idx[b]] for b in range(obj_idx.size(0))]).squeeze(1).contiguous().transpose(2, 1).contiguous()
        tout = torch.stack([tx[b][obj_idx[b]] for b in range(obj_idx.size(0))]).squeeze(1).contiguous().transpose(2, 1).contiguous()
        cout = torch.stack([cx[b][obj_idx[b]] for b in range(obj_idx.size(0))]).squeeze(1).contiguous().transpose(2, 1).contiguous()

        if ret_cemb:
            return rout, tout, cout, cemb.detach()

        # 4-dim quat, 3-dim t, 1-dim c
        return rout, tout, cout
