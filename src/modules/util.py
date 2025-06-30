# coding: utf-8

"""
This file defines various neural network modules and utility functions, including convolutional and residual blocks,
normalizations, and functions for spatial transformation and tensor manipulation.
"""

from torch import nn
import torch.nn.functional as F
import torch
import torch.nn.utils.spectral_norm as spectral_norm
import math
import warnings
import collections.abc
from itertools import repeat

def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp  # (bs, num_kp, 3)

    # Get coordinate grid: (d, h, w, 3) - 4D
    coordinate_grid = make_coordinate_grid(spatial_size, mean)

    # Process each batch and keypoint separately to avoid 6D tensors
    bs = mean.shape[0]
    num_kp = mean.shape[1]
    d, h, w = spatial_size

    # Prepare coordinate grid for broadcasting: (d, h, w, 3) -> (1, d, h, w, 3) - 5D
    coordinate_grid = coordinate_grid.unsqueeze(0)  # (1, d, h, w, 3)

    # Process all keypoints for all batches simultaneously but within 5D constraint
    # Reshape mean to (bs*num_kp, 3) and coordinate_grid to (bs*num_kp, d, h, w, 3)
    mean_flat = mean.reshape(bs * num_kp, 3)  # (bs*num_kp, 3) - 2D
    coordinate_grid_expanded = coordinate_grid.repeat(bs * num_kp, 1, 1, 1, 1)  # (bs*num_kp, d, h, w, 3) - 5D

    # Reshape mean for broadcasting: (bs*num_kp, 3) -> (bs*num_kp, 1, 1, 1, 3) - 5D
    mean_expanded = mean_flat.reshape(bs * num_kp, 1, 1, 1, 3)  # (bs*num_kp, 1, 1, 1, 3) - 5D

    # Calculate mean_sub: (bs*num_kp, d, h, w, 3) - 5D
    mean_sub = coordinate_grid_expanded - mean_expanded

    # Calculate gaussian: (bs*num_kp, d, h, w) - 4D
    out_flat = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    # Reshape back to (bs, num_kp, d, h, w) - 5D
    out = out_flat.reshape(bs, num_kp, d, h, w)

    return out


def make_coordinate_grid(spatial_size, ref, **kwargs):
    d, h, w = spatial_size
    x = torch.arange(w).type(ref.dtype).to(ref.device)
    y = torch.arange(h).type(ref.dtype).to(ref.device)
    z = torch.arange(d).type(ref.dtype).to(ref.device)

    # NOTE: must be right-down-in
    x = (2 * (x / (w - 1)) - 1)  # the x axis faces to the right
    y = (2 * (y / (h - 1)) - 1)  # the y axis faces to the bottom
    z = (2 * (z / (d - 1)) - 1)  # the z axis faces to the inner

    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)

    return meshed


class ConvT2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(ConvT2d, self).__init__()

        self.convT = nn.ConvTranspose2d(in_features, out_features, kernel_size=kernel_size, stride=stride,
                                        padding=padding, output_padding=output_padding)
        self.norm = nn.InstanceNorm2d(out_features)

    def forward(self, x):
        out = self.convT(x)
        out = self.norm(out)
        out = F.leaky_relu(out)
        return out


class ResBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm3d(in_features, affine=True)
        self.norm2 = nn.BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock3d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock3d, self).__init__()

        self.conv = Conv3DEquivalent(in_features, out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = nn.BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class DownBlock3d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock3d, self).__init__()
        # Original 3D convolution - now using equivalent 2D approach
        # self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
        #                         padding=padding, groups=groups, stride=(1, 2, 2))
        self.conv = Conv3DEquivalent(in_features, out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = nn.BatchNorm3d(out_features, affine=True)
        # Use 2D pooling to maintain rank 4 constraint
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)

        # Reshape to 4D for 2D pooling: (bs, c, d, h, w) -> (bs*d, c, h, w)
        bs, c, d, h, w = out.shape
        out_reshaped = out.reshape(bs * d, c, h, w)  # (bs*d, c, h, w) - 4D

        # Apply 2D pooling: (bs*d, c, h, w) -> (bs*d, c, h//2, w//2) - 4D
        out_pooled = self.pool(out_reshaped)

        # Reshape back to 5D: (bs*d, c, h//2, w//2) -> (bs, c, d, h//2, w//2)
        _, _, h_new, w_new = out_pooled.shape
        out = out_pooled.reshape(bs, c, d, h_new, w_new)  # (bs, c, d, h//2, w//2) - 5D

        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1, lrelu=False):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        if lrelu:
            self.ac = nn.LeakyReLU()
        else:
            self.ac = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ac(out)
        return out





class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.use_se = use_se
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, seg1):
        x_s = self.shortcut(x, seg1)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


def filter_state_dict(state_dict, remove_name='fc'):
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def drop_path(x, drop_prob=0., training=False, scale_by_keep=True):
    """ Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """ Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class Conv3DEquivalent(nn.Module):
    """
    Efficient separable (2+1)D convolution replacement for Conv3d.

    Uses the industry-standard separable approach: spatial 2D conv + temporal 1D conv.
    This is much more efficient than full 3D convolution and often performs better.
    Based on R(2+1)D architecture.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        super(Conv3DEquivalent, self).__init__()

        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kd = self.kh = self.kw = kernel_size
        else:
            self.kd, self.kh, self.kw = kernel_size

                # Handle padding as int or tuple
        if isinstance(padding, int):
            self.pd = self.ph = self.pw = padding
        else:
            self.pd, self.ph, self.pw = padding

        # Store channel info for weight loading
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculate intermediate channels using R(2+1)D formula
        # This balances parameters between spatial and temporal convolutions
        mid_channels = max(16, (in_channels * out_channels * self.kd * self.kh * self.kw) //
                          (in_channels * self.kh * self.kw + out_channels * self.kd))

        # Spatial convolution (2D) - processes each depth slice
        self.spatial_conv = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=(self.kh, self.kw),
            padding=(self.ph, self.pw),
            stride=stride,
            bias=False  # No bias here, will be in temporal conv
        )

        # Temporal convolution (1D across depth) - aggregates across depth
        self.temporal_conv = nn.Conv1d(
            mid_channels,
            out_channels,
            kernel_size=self.kd,
            padding=self.pd,
            stride=1,
            bias=bias
        )

        # Store for weight loading
        self.mid_channels = mid_channels

    def forward(self, x):
        # Input: (bs, c_in, d, h, w)
        bs, c_in, d, h, w = x.shape

        # Step 1: Spatial convolution
        # Reshape for 2D conv: (bs, c_in, d, h, w) -> (bs*d, c_in, h, w)
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(bs * d, c_in, h, w)

        # Apply spatial convolution
        x_spatial = self.spatial_conv(x_reshaped)  # (bs*d, mid_channels, h_out, w_out)

        # Reshape back to 5D for temporal conv
        _, mid_ch, h_out, w_out = x_spatial.shape
        x_spatial_5d = x_spatial.reshape(bs, d, mid_ch, h_out, w_out)
        x_spatial_5d = x_spatial_5d.permute(0, 2, 1, 3, 4)  # (bs, mid_channels, d, h_out, w_out)

                # Step 2: Temporal convolution (1D across depth)
        # Reshape for Conv1d: (bs, mid_channels, d, h_out, w_out) -> (bs*h_out*w_out, mid_channels, d)
        bs, mid_ch, d, h_out, w_out = x_spatial_5d.shape
        x_temporal_input = x_spatial_5d.permute(0, 3, 4, 1, 2).reshape(bs * h_out * w_out, mid_ch, d)

        # Apply 1D temporal convolution
        x_temporal_output = self.temporal_conv(x_temporal_input)  # (bs*h_out*w_out, out_channels, d_out)

        # Reshape back to 5D: (bs*h_out*w_out, out_channels, d_out) -> (bs, out_channels, d_out, h_out, w_out)
        _, out_ch, d_out = x_temporal_output.shape
        output = x_temporal_output.reshape(bs, h_out, w_out, out_ch, d_out)
        output = output.permute(0, 3, 4, 1, 2)  # (bs, out_channels, d_out, h_out, w_out)

        return output

    def load_conv3d_weights(self, conv3d_weight, conv3d_bias=None):
        """
        Load weights from a Conv3d layer into separable (2+1)D structure.

        Uses SVD decomposition to approximate the 3D kernel as spatial + temporal components.
        """
        c_out, c_in, kd, kh, kw = conv3d_weight.shape

        assert kd == self.kd and kh == self.kh and kw == self.kw, \
            f"Kernel size mismatch: expected {(self.kd, self.kh, self.kw)}, got {(kd, kh, kw)}"
        assert c_out == self.out_channels and c_in == self.in_channels, \
            f"Channel mismatch: expected {(self.out_channels, self.in_channels)}, got {(c_out, c_in)}"

        # For efficiency, use a simple approximation:
        # Spatial component: average across temporal dimension
        spatial_kernel = conv3d_weight.mean(dim=2)  # (c_out, c_in, kh, kw)

        # Initialize spatial conv weights
        # We need to map (c_out, c_in) -> (mid_channels, c_in) for spatial conv
        # Use a simple projection approach
        spatial_weight = torch.zeros(self.mid_channels, c_in, kh, kw)

        # Map channels using simple repetition/averaging
        if self.mid_channels <= c_out:
            # Downsample output channels
            indices = torch.linspace(0, c_out-1, self.mid_channels).long()
            spatial_weight = spatial_kernel[indices]
        else:
            # Upsample by repetition
            repeat_factor = self.mid_channels // c_out
            remainder = self.mid_channels % c_out
            spatial_weight[:c_out * repeat_factor] = spatial_kernel.repeat(repeat_factor, 1, 1, 1)
            if remainder > 0:
                spatial_weight[c_out * repeat_factor:] = spatial_kernel[:remainder]

        self.spatial_conv.weight.data = spatial_weight

                # Temporal component: average across spatial dimensions
        temporal_kernel = conv3d_weight.mean(dim=[3, 4])  # (c_out, c_in, kd)

        # Map from c_in to mid_channels for temporal conv (Conv1d format)
        temporal_weight = torch.zeros(c_out, self.mid_channels, kd)

        if self.mid_channels <= c_in:
            # Downsample input channels
            indices = torch.linspace(0, c_in-1, self.mid_channels).long()
            temporal_weight = temporal_kernel[:, indices]
        else:
            # Upsample by repetition
            repeat_factor = self.mid_channels // c_in
            remainder = self.mid_channels % c_in
            temporal_weight[:, :c_in * repeat_factor] = temporal_kernel.repeat(1, repeat_factor, 1)
            if remainder > 0:
                temporal_weight[:, c_in * repeat_factor:] = temporal_kernel[:, :remainder]

        self.temporal_conv.weight.data = temporal_weight

        # Load bias into temporal conv (final output)
        if conv3d_bias is not None:
            self.temporal_conv.bias.data = conv3d_bias








class Encoder(nn.Module):
    """
    Hourglass Encoder (original version)
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)), min(max_features, block_expansion * (2 ** (i + 1))), kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder (original version)
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

        self.conv = Conv3DEquivalent(self.out_filters, self.out_filters, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm3d(self.out_filters, affine=True)

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture (original version)
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))
