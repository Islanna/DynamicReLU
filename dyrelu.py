import torch
from torch import nn


class DyReLUA(nn.Module):
    def __init__(self, channels, reduction=4, k=2):
        super(DyReLUA, self).__init__()
        self.channels = channels
        self.k = k

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.channels = channels
        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def forward(self, x):
        assert x.shape[1] == self.channels
        # BxCxL
        theta = torch.mean(x, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1

        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)
        return result


class DyReLUB(nn.Module):
    def __init__(self, channels, reduction=4, k=2):
        super(DyReLUB, self).__init__()
        self.channels = channels
        self.k = k

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def forward(self, x):
        assert x.shape[1] == self.channels
        # BxCxL
        theta = torch.mean(x, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxBxCx1
        x_perm = x.permute(2, 0, 1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        # LxBxCx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].permute(1, 2, 0)
        return result
