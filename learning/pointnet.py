import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k: int):
        super().__init__()

        self.k = k

        # 1st convolution + "mlp"
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        # relu called directly

        # 2nd convolution + "mlp"
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.batchnorm2 = nn.BatchNorm1d(128)
        # relu called directly

        # 3rd convolution + "mlp"
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        # relu called directly

        # max pool called directly

        # 1st fully connected layer after max pool
        self.fc1 = nn.Linear(1024, 512)
        self.batchnorm4 = nn.BatchNorm1d(512)
        # relu called directly

        # 2nd fully connected layer after max pool
        self.fc2 = nn.Linear(512, 256)
        self.batchnorm5 = nn.BatchNorm1d(256)
        # relu called directly

        # generate final output tensor for matmul
        self.fc3 = nn.Linear(256, self.k * self.k)

    def forward(self, input):
        batch_size = input.size(0)

        # 1st convolution
        x = F.relu(self.batchnorm1(self.conv1(input)))
        # 2nd convolution
        x = F.relu(self.batchnorm2(self.conv2(x)))
        # 3rd convolution
        x = F.relu(self.batchnorm3(self.conv3(x)))

        # max pool
        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)

        # 1st fully connected layer after max pool
        x = F.relu(self.batchnorm4(self.fc1(x)))
        # 2nd fully connected layer after max pool
        x = F.relu(self.batchnorm5(self.fc2(x)))

        # final fc before reshaping to output matrix
        x = self.fc3(x)

        # init matrix to identity for orthogonality
        matrix = torch.eye(self.k, requires_grad=True).flatten().repeat(batch_size, 1)
        if torch.cuda.is_available():
            matrix = matrix.cuda()
        # add to last fc layer
        matrix = x + matrix
        # reshape to batch_size x 3 x 3
        matrix = matrix.view(batch_size, self.k, self.k)
        
        return matrix


class PointNet(nn.Module):
    def __init__(self, num_objs):
        super().__init__()
        
        self.num_objs = num_objs

        # Input Transform TNet
        self.input_tnet = TNet(3)

        # 1st shared mlp between input and feature transform steps
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.batchnorm1 = nn.BatchNorm1d(64)

        # Feature Trasformation TNet
        self.feature_tnet = TNet(64)

        # 2nd shared mlp, 1st convolution
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.batchnorm2 = nn.BatchNorm1d(128)

        # 2nd shared mlp, 2nd convolution
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.batchnorm3 = nn.BatchNorm1d(1024)

        ### convert to rot

        # output will be quat (4 vals), translation (3 val), and confidence (1 val)
        # note: loss fn will converted quat to rot mat before computing

        self.conv1_r = nn.Conv1d(1088, 512, 1)
        self.conv2_r = nn.Conv1d(512, 256, 1)
        self.conv3_r = nn.Conv1d(256, 128, 1)
        self.conv4_r = nn.Conv1d(128, num_objs*4, 1) # quat

        self.conv1_t = nn.Conv1d(1088, 512, 1)
        self.conv2_t = nn.Conv1d(512, 256, 1)
        self.conv3_t = nn.Conv1d(256, 128, 1)
        self.conv4_t = nn.Conv1d(128, num_objs*3, 1) # translation

        self.conv1_c = nn.Conv1d(1088, 512, 1)
        self.conv2_c = nn.Conv1d(512, 256, 1)
        self.conv3_c = nn.Conv1d(256, 128, 1)
        self.conv4_c = nn.Conv1d(128, num_objs*1, 1) # confidence

    def forward(self, input, obj_idx):
        batches = input.size(0)
        num_pts = input.size(-1)

        # input transformation
        mat3x3 = self.input_tnet(input)

        x = input.transpose(2, 1)       # align dims
        x = torch.bmm(x, mat3x3)
        x = x.transpose(2, 1)           # put channels back in correct spot

        # 1st shared mlp between input and feature transform steps
        x = F.relu(self.batchnorm1(self.conv1(x)))

        # feature transformation
        mat64x64 = self.feature_tnet(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, mat64x64)
        x = x.transpose(2, 1)

        feature_matrix = x

        # 2nd shared mlp convolutions
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))

        # Max Pool for symmmetric func / perm invariance
        global_feat = nn.MaxPool1d(x.size(-1))(x)
        gf_repeat = global_feat.repeat(1, 1, num_pts)

        per_pt_feat = torch.cat((feature_matrix, gf_repeat), 1)

        # per-point estimation of Rs, ts, cs
        rx = F.relu(self.conv1_r(per_pt_feat))
        rx = F.relu(self.conv2_r(rx))
        rx = F.relu(self.conv3_r(rx))
        rx = F.relu(self.conv4_r(rx))

        tx = F.relu(self.conv1_t(per_pt_feat))
        tx = F.relu(self.conv2_t(tx))
        tx = F.relu(self.conv3_t(tx))
        tx = F.relu(self.conv4_t(tx))

        cx = F.relu(self.conv1_c(per_pt_feat))
        cx = F.relu(self.conv2_c(cx))
        cx = F.relu(self.conv3_c(cx))
        cx = F.relu(self.conv4_c(cx))

        rx = rx.view(batches, self.num_objs, 4, num_pts)
        tx = tx.view(batches, self.num_objs, 3, num_pts)
        cx = torch.sigmoid(cx).view(batches, self.num_objs, 1, num_pts)

        rout = torch.stack([rx[b][obj_idx[b]] for b in range(obj_idx.size(0))]).squeeze()
        tout = torch.stack([tx[b][obj_idx[b]] for b in range(obj_idx.size(0))]).squeeze()
        cout = torch.stack([cx[b][obj_idx[b]] for b in range(obj_idx.size(0))]).squeeze()

        # 4-dim quat, 3-dim t, 1-dim c
        return rout, tout, cout

