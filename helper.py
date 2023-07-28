import torch
import torch.nn as nn
import torch.nn.functional as F

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))

class ChannelLayerNorm(nn.Module):
    # layer norm on channels
    def __init__(self, in_features):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.norm(x)
        x = shift_dim(x, -1, 1)
        return x

class NormReLU(nn.Module):

    def __init__(self, channels, relu=True, affine=True):
        super().__init__()

        self.relu = relu
        self.norm = ChannelLayerNorm(channels)

    def forward(self, x):
        x_float = x.float()
        x_float = self.norm(x_float)
        x = x_float.type_as(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, filters, stride, use_projection=False):
        super().__init__()

        if use_projection:
            self.proj_conv = nn.Conv3d(in_channels, filters, kernel_size=1,
                                       stride=stride, bias=False)
            self.proj_bnr = NormReLU(filters, relu=False)

        self.conv1 = nn.Conv3d(in_channels, filters, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.bnr1 = NormReLU(filters)

        self.conv2 = nn.Conv3d(filters, filters, kernel_size=3,
                               stride=1, bias=False, padding=1)
        self.bnr2 = NormReLU(filters)

        self.use_projection = use_projection

    def forward(self, x):
        shortcut = x
        if self.use_projection:
            shortcut = self.proj_bnr(self.proj_conv(x))
        x = self.bnr1(self.conv1(x))
        x = self.bnr2(self.conv2(x))

        return F.relu(x + shortcut, inplace=True)


class ResNet(nn.Module):

    def __init__(self, in_channels, layers, width_multiplier,
                 stride, resnet_dim=240, cifar_stem=True):
        super().__init__()
        self.width_multiplier = width_multiplier
        self.resnet_dim = resnet_dim

        assert all([int(math.log2(d)) == math.log2(d) for d in stride]), stride
        n_times_downsample = np.array([int(math.log2(d)) for d in stride])

        if cifar_stem:
            self.stem = nn.Sequential(
                nn.Conv3d(in_channels, 64 * width_multiplier,
                          kernel_size=3, padding=1, bias=False),
                NormReLU(64 * width_multiplier)
            )
        else:
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            n_times_downsample -= 1  # conv
            n_times_downsample[-2:] = n_times_downsample[-2:] - 1  # pooling
            self.stem = nn.Sequential(
                nn.Conv3d(in_channels, 64 * width_multiplier,
                          kernel_size=7, stride=stride, bias=False,
                          padding=3),
                NormReLU(64 * width_multiplier),
                nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
            )

        self.group1 = BlockGroup(64 * width_multiplier, 64 * width_multiplier,
                                 blocks=layers[0], stride=1)

        stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
        n_times_downsample -= 1
        self.group2 = BlockGroup(64 * width_multiplier, 128 * width_multiplier,
                                 blocks=layers[1], stride=stride)

        stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
        n_times_downsample -= 1
        self.group3 = BlockGroup(128 * width_multiplier, 256 * width_multiplier,
                                 blocks=layers[2], stride=stride)

        stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
        n_times_downsample -= 1
        self.group4 = BlockGroup(256 * width_multiplier, resnet_dim,
                                 blocks=layers[3], stride=stride)
        assert all([d <= 0 for d in n_times_downsample]), f'final downsample {n_times_downsample}'

    def forward(self, x):
        x = self.stem(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = shift_dim(x, 1, -1)

        return x

class GroupNorm(nn.Module):

    def __init__(self, in_channels, filters, blocks, stride):
        super().__init__()

        self.start_block = ResidualBlock(in_channels, filters, stride, use_projection=True)
        in_channels = filters

        self.blocks = []
        for _ in range(1, blocks):
            self.blocks.append(ResidualBlock(in_channels, filters, 1))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.start_block(x)
        x = self.blocks(x)
        return x