import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Edge_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_c0 = Mlp(in_features=64, hidden_features=128, out_features=64)
        self.mlp_c1 = Mlp(in_features=64, hidden_features=128, out_features=64)
        self.mlp_c2 = Mlp(in_features=128, hidden_features=256, out_features=128)
        self.mlp_c3 = Mlp(in_features=256, hidden_features=512, out_features=256)

        self.up_sample_c1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_sample_c2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4, )
        self.up_sample_c3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=4)

        self.linear_fuse = nn.Conv2d(in_channels=64 * 4, out_channels=32, kernel_size=1)

    def forward(self, c0, c1, c2, c3):
        # 不同时相的不同尺度的特征图谱分别通过mlp

        # 获取输入特征图谱的形状
        batch_size, channels, height, width = c0.shape
        # 将输入特征图谱转换为2D张量
        c0 = c0.contiguous().view(batch_size, channels, -1).transpose(1, 2)  # Shape: [8, 16384, 64]
        # 通过MLP层进行前向传播
        c0 = self.mlp_c0(c0, height, width)
        # 将输出特征张量恢复为原始形状
        c0 = c0.transpose(1, 2).contiguous().view(batch_size, channels, height, width)

        # 获取输入特征图谱的形状
        batch_size, channels, height, width = c1.shape
        # 将输入特征图谱转换为2D张量
        c1 = c1.contiguous().view(batch_size, channels, -1).transpose(1, 2)  # Shape: [8, 262144, 64]
        # 通过MLP层进行前向传播
        c1 = self.mlp_c1(c1, height, width)
        # 将输出特征张量恢复为原始形状
        c1 = c1.transpose(1, 2).contiguous().view(batch_size, channels, height, width)

        # 获取输入特征图谱的形状
        batch_size, channels, height, width = c2.shape
        # 将输入特征图谱转换为2D张量
        c2 = c2.contiguous().view(batch_size, channels, -1).transpose(1, 2)  # Shape: [8, 1024, 128]
        # 通过MLP层进行前向传播
        c2 = self.mlp_c2(c2, height, width)

        # 将输出特征张量恢复为原始形状
        c2 = c2.transpose(1, 2).contiguous().view(batch_size, channels, height, width)

        # 获取输入特征图谱的形状
        batch_size, channels, height, width = c3.shape
        # 将输入特征图谱转换为2D张量
        c3 = c3.contiguous().view(batch_size, channels, -1).transpose(1, 2)  # Shape: [8, 256, 256]
        # 通过MLP层进行前向传播
        c3 = self.mlp_c3(c3, height, width)
        # 将输出特征张量恢复为原始形状
        c3 = c3.transpose(1, 2).contiguous().view(batch_size, channels, height, width)

        c1_up = self.up_sample_c1(c1)
        c2_up = self.up_sample_c2(c2)
        c3_up = self.up_sample_c3(c3)

        # 连接+融合
        ct1_edge = self.linear_fuse(torch.cat([c3_up, c2_up, c1_up, c0], dim=1))

        return ct1_edge


class Edge_Decoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=256, output_nc=2, decoder_softmax=False):


        # MLP
        super().__init__()
        self.linear = Mlp(in_features=in_channels, hidden_features=256, out_features=embedding_dim)

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final prediction
        self.change_probability = nn.Conv2d(embedding_dim, output_nc, kernel_size=3, stride=1, padding=1)
        self.output_softmax = decoder_softmax
        self.active = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        x = x.contiguous().view(batch_size, channels, -1).transpose(1, 2)

        x = self.linear(x, height, width)
        x = x.transpose(1, 2).contiguous().view(batch_size, 8*channels, height, width)

        x = self.upsample(x)

        cp = self.change_probability(x)
        if self.output_softmax:
            cp = self.active(cp)
        return cp


if __name__ == '__main__':
    # a = Edge_Encoder()
    # b = torch.ones(8, 3, 256, 256)
    # c = torch.ones(8, 64, 128, 128)
    # d = torch.ones(8, 64, 64, 64)
    # e = torch.ones(8, 128, 32, 32)
    # f = torch.ones(8, 256, 32, 32)
    # out = a(c, d, e, f)
    # print(out.shape)
    a = Edge_Decoder(in_channels=32)
    b = torch.ones(8, 32, 128, 128)
    out = a(b)
    print(out.shape)
