import math
import torch
import torch.nn as nn
import torch.nn.functional as function


class HighFrequencyModule(nn.Module):
    def __init__(self, input_channel, the_filter='Isotropic_Sobel', mode='filtering', parameter_a=1, parameter_k=0.5,
                 smooth=False):
        """High Frequency Module for CNN.

        Extract the high frequency component of features.

        Args:
            input_channel (int): Number of channels inputted.
            the_filter (str): Decide which filter to use ('Isotropic_Sobel','Krisch','Laplacian_1,_2 and _3','LOG'.).
            mode (str): Decide which mode this module to work in ('filtering' or 'high_boost_filtering'.).
            parameter_a (float): When the module work in the high boost filtering mode, the parameter_a decide the
                                strength of the original features.
            parameter_k (float): When the module work in the high boost filtering mode, the parameter_a decide the
                                strength of the high frequency features extracted from the original features.
        """
        super(HighFrequencyModule, self).__init__()
        self.filter = the_filter
        self.mode = mode
        self.channel = input_channel
        self.A = parameter_a
        self.K = parameter_k
        self.smooth = smooth
        # Gaussian Smooth
        kernel_gaussian_smooth = [[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]]
        kernel_smooth = torch.FloatTensor(kernel_gaussian_smooth).expand(self.channel, self.channel, 3, 3)
        self.weight_smooth = nn.Parameter(data=kernel_smooth, requires_grad=False)
        # Isotropic Sobel
        kernel_isotropic_sobel_direction_1 = [[1, math.sqrt(2), 1],
                                              [0, 0, 0],
                                              [-1, -math.sqrt(2), -1]]
        kernel_isotropic_sobel_direction_2 = [[0, 1, math.sqrt(2)],
                                              [-1, 0, 1],
                                              [-math.sqrt(2), -1, 0]]
        kernel_isotropic_sobel_direction_3 = [[-1, 0, 1],
                                              [-math.sqrt(2), 0, math.sqrt(2)],
                                              [-1, 0, 1]]
        kernel_isotropic_sobel_direction_4 = [[math.sqrt(2), 1, 0],
                                              [1, 0, -1],
                                              [0, -1, -math.sqrt(2)]]
        # kernel_isotropic_sobel_direction_5 = -1 * kernel_isotropic_sobel_direction_1
        # kernel_isotropic_sobel_direction_6 = -1 * kernel_isotropic_sobel_direction_2
        # kernel_isotropic_sobel_direction_7 = -1 * kernel_isotropic_sobel_direction_3
        # kernel_isotropic_sobel_direction_8 = -1 * kernel_isotropic_sobel_direction_4
        # Krisch
        kernel_krisch_direction_1 = [[5, 5, 5],
                                     [-3, 0, -3],
                                     [-3, -3, -3]]
        kernel_krisch_direction_2 = [[-3, 5, 5],
                                     [-3, 0, 5],
                                     [-3, -3, -3]]
        kernel_krisch_direction_3 = [[-3, -3, 5],
                                     [-3, 0, 5],
                                     [-3, -3, 5]]
        kernel_krisch_direction_4 = [[-3, -3, -3],
                                     [-3, 0, 5],
                                     [-3, 5, 5]]
        kernel_krisch_direction_5 = [[-3, -3, -3],
                                     [-3, 0, -3],
                                     [5, 5, 5]]
        kernel_krisch_direction_6 = [[-3, -3, -3],
                                     [5, 0, -3],
                                     [5, 5, -3]]
        kernel_krisch_direction_7 = [[5, -3, -3],
                                     [5, 0, -3],
                                     [5, -3, -3]]
        kernel_krisch_direction_8 = [[5, 5, -3],
                                     [5, 0, -3],
                                     [-3, -3, -3]]
        # Laplacian
        kernel_laplacian_1 = [[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]]
        kernel_laplacian_2 = [[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]]
        kernel_laplacian_3 = [[1, -2, 1],
                              [-2, 4, -2],
                              [1, -2, 1]]
        # LOG
        kernel_log = [[-2, -4, -4, -4, -2],
                      [-4, 0, 8, 0, -4],
                      [-4, 8, 24, 8, -4],
                      [-4, 0, 8, 0, -4],
                      [-2, -4, -4, -4, -2]]
        if self.filter == 'Isotropic_Sobel':
            kernel_1 = torch.FloatTensor(kernel_isotropic_sobel_direction_1).expand(self.channel, self.channel, 3, 3)
            kernel_2 = torch.FloatTensor(kernel_isotropic_sobel_direction_2).expand(self.channel, self.channel, 3, 3)
            kernel_3 = torch.FloatTensor(kernel_isotropic_sobel_direction_3).expand(self.channel, self.channel, 3, 3)
            kernel_4 = torch.FloatTensor(kernel_isotropic_sobel_direction_4).expand(self.channel, self.channel, 3, 3)
            kernel_5 = -1 * kernel_1
            kernel_6 = -1 * kernel_2
            kernel_7 = -1 * kernel_3
            kernel_8 = -1 * kernel_4
            self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)
            self.weight_2 = nn.Parameter(data=kernel_2, requires_grad=False)
            self.weight_3 = nn.Parameter(data=kernel_3, requires_grad=False)
            self.weight_4 = nn.Parameter(data=kernel_4, requires_grad=False)
            self.weight_5 = nn.Parameter(data=kernel_5, requires_grad=False)
            self.weight_6 = nn.Parameter(data=kernel_6, requires_grad=False)
            self.weight_7 = nn.Parameter(data=kernel_7, requires_grad=False)
            self.weight_8 = nn.Parameter(data=kernel_8, requires_grad=False)
        elif self.filter == 'Krisch':
            kernel_1 = torch.FloatTensor(kernel_krisch_direction_1).expand(self.channel, self.channel, 3, 3)
            kernel_2 = torch.FloatTensor(kernel_krisch_direction_2).expand(self.channel, self.channel, 3, 3)
            kernel_3 = torch.FloatTensor(kernel_krisch_direction_3).expand(self.channel, self.channel, 3, 3)
            kernel_4 = torch.FloatTensor(kernel_krisch_direction_4).expand(self.channel, self.channel, 3, 3)
            kernel_5 = torch.FloatTensor(kernel_krisch_direction_5).expand(self.channel, self.channel, 3, 3)
            kernel_6 = torch.FloatTensor(kernel_krisch_direction_6).expand(self.channel, self.channel, 3, 3)
            kernel_7 = torch.FloatTensor(kernel_krisch_direction_7).expand(self.channel, self.channel, 3, 3)
            kernel_8 = torch.FloatTensor(kernel_krisch_direction_8).expand(self.channel, self.channel, 3, 3)
            self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)
            self.weight_2 = nn.Parameter(data=kernel_2, requires_grad=False)
            self.weight_3 = nn.Parameter(data=kernel_3, requires_grad=False)
            self.weight_4 = nn.Parameter(data=kernel_4, requires_grad=False)
            self.weight_5 = nn.Parameter(data=kernel_5, requires_grad=False)
            self.weight_6 = nn.Parameter(data=kernel_6, requires_grad=False)
            self.weight_7 = nn.Parameter(data=kernel_7, requires_grad=False)
            self.weight_8 = nn.Parameter(data=kernel_8, requires_grad=False)
        elif self.filter == 'Laplacian_1':
            kernel_1 = torch.FloatTensor(kernel_laplacian_1).expand(self.channel, self.channel, 3, 3)
            self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)
        elif self.filter == 'Laplacian_2':
            kernel_1 = torch.FloatTensor(kernel_laplacian_2).expand(self.channel, self.channel, 3, 3)
            self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)
        elif self.filter == 'Laplacian_3':
            kernel_1 = torch.FloatTensor(kernel_laplacian_3).expand(self.channel, self.channel, 3, 3)
            self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)
        elif self.filter == 'LOG':
            kernel_1 = torch.FloatTensor(kernel_log).expand(self.channel, self.channel, 5, 5)
            self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)

    def forward(self, x):
        # pretreatment
        if self.smooth:
            x = function.conv2d(x, self.weight_smooth, stride=1, padding=1)
            x = x / 16
        x_result = x
        x_high_frequency = x
        # filter choose
        if self.filter == 'Isotropic_Sobel' or 'Krisch':
            x1 = function.conv2d(x, self.weight_1, stride=1, padding=1)
            x2 = function.conv2d(x, self.weight_2, stride=1, padding=1)
            x3 = function.conv2d(x, self.weight_3, stride=1, padding=1)
            x4 = function.conv2d(x, self.weight_4, stride=1, padding=1)
            x5 = function.conv2d(x, self.weight_5, stride=1, padding=1)
            x6 = function.conv2d(x, self.weight_6, stride=1, padding=1)
            x7 = function.conv2d(x, self.weight_7, stride=1, padding=1)
            x8 = function.conv2d(x, self.weight_8, stride=1, padding=1)
            x_high_frequency = (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8) / 8
        elif self.filter == 'Laplacian_1' or 'Laplacian_2' or 'Laplacian_3':
            x_high_frequency = function.conv2d(x, self.weight_1, stride=1, padding=1)
        elif self.filter == 'LOG':
            x_high_frequency = function.conv2d(x, self.weight_1, stride=1, padding=2)
        # mode choose
        if self.mode == 'filtering':
            x_result = x_high_frequency
        elif self.mode == 'high_boost_filtering':
            x_result = self.A * x + self.K * x_high_frequency
        return x_result


class HighFrequencyEnhancementStage(nn.Module):
    def __init__(self, input_channel, input_size, ratio=0.5):
        super(HighFrequencyEnhancementStage, self).__init__()
        self.input_channel = input_channel
        self.input_size = input_size
        self.ratio_channel = int(ratio * input_channel)
        self.Global_pooling = nn.AvgPool2d(self.input_size)
        self.FC_1 = nn.Linear(self.input_channel, int(self.input_channel * ratio))
        self.ReLU = nn.PReLU(int(self.input_channel * ratio))
        self.FC_2 = nn.Linear(int(self.input_channel * ratio), self.input_channel)
        self.Sigmoid = nn.Sigmoid()
        self.HighFre = HighFrequencyModule(input_channel=self.input_channel, smooth=True)
        self.Channelfusion = nn.Conv2d(2 * self.input_channel, self.input_channel, kernel_size=1, stride=1)

    # ChannelAttention +HighFrequency
    def forward(self, x):
        residual = x  # residual & x's shape [batch size, channel, input size, input size]
        x_hf = self.HighFre(residual)
        x = self.Global_pooling(x)  # x's shape [batch size, channel, 1, 1]
        x = x.view(-1, self.input_channel)  # x's shape [batch size, channel]
        x = self.FC_1(x)  # x's shape [batch size, ratio channel]
        x = self.ReLU(x)
        x = self.FC_2(x)  # x's shape [batch size, channel]
        x = self.Sigmoid(x)
        x = torch.unsqueeze(x, dim=2)  # x's shape [batch size, channel, 1]
        residual_0 = residual.view(-1, self.input_channel, self.input_size ** 2)
        residual_0 = torch.mul(residual_0, x)
        residual_0 = residual_0.contiguous().view(-1, self.input_channel, self.input_size, self.input_size)
        x_output = residual + residual_0
        x_output = torch.cat((x_output, x_hf), dim=1)
        x_output = self.Channelfusion(x_output)
        return x_output


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Conv2d(channel, int(channel // reduction), kernel_size=1,
                                                          stride=1, padding=0, bias=False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(int(channel // reduction), channel, kernel_size=1,
                                                          stride=1, padding=0, bias=False),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        chn_se = self.avg_pool(x)
        chn_se = self.channel_excitation(chn_se)
        chn_se = torch.mul(x, chn_se)
        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, spa_se, alpha=1)


class HFAB(nn.Module):
    def __init__(self, input_channel, input_size, ratio=0.5, reduction=16):
        super(HFAB, self).__init__()
        self.SA = SCSEBlock(channel=input_channel, reduction=reduction)
        self.HF = HighFrequencyEnhancementStage(input_channel=input_channel,
                                                input_size=input_size,
                                                ratio=ratio)

    def forward(self, x):
        x = self.SA(x)
        x = self.HF(x)
        return x


if __name__ == '__main__':
    a = torch.ones(8, 16, 256, 256)
    b = HFAB(input_channel=16, input_size=256)
    print(b(a).shape)
