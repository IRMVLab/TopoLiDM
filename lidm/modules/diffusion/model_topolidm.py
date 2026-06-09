import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from ..basic import CircularConv2d # same as model_ldm.py

# -----------------------------------------------------------------------------
# 1. Core building blocks (K-NN graph, GraphLayer, PositionalEncoding)
# -----------------------------------------------------------------------------

# class CircularConv2d(nn.Module):
#     """
#     Circular convolution layer using circular padding for 360° LiDAR horizontal FOV.
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super().__init__()
#         self.padding = padding
#         if isinstance(padding, int):
#             self.pad_tuple = (padding, padding, padding, padding)
#         elif len(padding) == 2:
#             self.pad_tuple = (padding[1], padding[1], padding[0], padding[0])
#         elif len(padding) == 4:
#             self.pad_tuple = padding

#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)

#     def forward(self, x):
#         x = F.pad(x, self.pad_tuple, mode='circular')
#         return self.conv(x)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # (B, 2C, N, k): edge difference features (neighbor - center) concatenated with center features
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class GraphLayer(nn.Module):
    """Graph convolution layer inspired by GLiDR and DeepGCN."""
    def __init__(self, channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        # x: (B, C, N)
        graph_feat = get_graph_feature(x, k=self.k)  # (B, 2C, N, K)
        out = self.conv(graph_feat)                  # (B, C, N, K)
        out = out.max(dim=-1, keepdim=False)[0]      # (B, C, N): max-pool over neighbors
        return out


class PositionalEncoding2D(nn.Module):
    """2D absolute sinusoidal positional encoding."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    def forward(self, tensor):
        B, C, H, W = tensor.shape
        pos_x = torch.arange(W, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(H, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(0).repeat(H, 1, 1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1).repeat(1, W, 1)
        emb = torch.cat((emb_x, emb_y), dim=-1).permute(2, 0, 1).unsqueeze(0)
        
        return tensor + emb[:, :C, :, :].to(tensor.device)


class Stem(nn.Module):
    """Downsample input from (B, C, 64, 1024) to (B, D, 16, 128)."""
    def __init__(self, in_dim=2, out_dim=64):
        super().__init__()
        # Three CircularConv2d layers: /4 vertical and /8 horizontal downsampling
        self.conv1 = CircularConv2d(in_dim, out_dim // 4, kernel_size=3, stride=(2, 2), padding=1)
        self.conv2 = CircularConv2d(out_dim // 4, out_dim // 2, kernel_size=3, stride=(2, 2), padding=1)
        self.conv3 = CircularConv2d(out_dim // 2, out_dim, kernel_size=3, stride=(1, 2), padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        return x

# -----------------------------------------------------------------------------
# 2. TopoLiDM Encoder
# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Replaces the original LiDM residual-block encoder.
    Forward output: deterministic Z, L2_feat, L4_feat.
    """
    def __init__(self, in_channels=2, z_channels=16, base_channels=64, k=20, **kwargs):
        super().__init__()
        self.k = k
        self.stem = Stem(in_channels, base_channels)
        self.pos_enc = PositionalEncoding2D(base_channels)

        self.layer1 = GraphLayer(base_channels, k)
        self.layer2 = GraphLayer(base_channels, k)
        self.layer3 = GraphLayer(base_channels, k)
        self.layer4 = GraphLayer(base_channels, k)

        self.proj = nn.Conv1d(base_channels, z_channels, 1)

    def forward(self, x):
        # 1. Stem downsampling: (B, C, 64, 1024) -> (B, D, 16, 128)
        h = self.stem(x)
        # 2. Inject positional encoding
        h = self.pos_enc(h)

        B, D, H, W = h.shape
        N = H * W

        # 3. Flatten to point set format for graph layers: (B, D, N)
        h = h.view(B, D, N)

        # 4. Hierarchical Graph Encoding
        h1 = self.layer1(h)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)

        # 5. Project to latent dimension: (B, 16, N)
        z = self.proj(h4)

        # 6. Reshape back to 2D: (B, 16, 16, 128)
        z = z.view(B, -1, H, W)

        # Return Z and transpose L2/L4 features to point format (B, N, D)
        return z, h2.transpose(1, 2), h4.transpose(1, 2)

# -----------------------------------------------------------------------------
# 3. Decoder (unchanged from LiDM)
# -----------------------------------------------------------------------------

def nonlinearity(x):
    return x * torch.sigmoid(x)

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

# Decoder auxiliary modules: Upsample, ResnetBlock, AttnBlock
UPSAMPLE_STRIDE2KERNEL_DICT = {(1, 2): (1, 5), (1, 4): (1, 7), (2, 1): (5, 1), (2, 2): (3, 3)}
UPSAMPLE_STRIDE2PAD_DICT = {(1, 2): (2, 2, 0, 0), (1, 4): (3, 3, 0, 0), (2, 1): (0, 0, 2, 2), (2, 2): (1, 1, 1, 1)}

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if self.with_conv:
            k, p = UPSAMPLE_STRIDE2KERNEL_DICT[stride], UPSAMPLE_STRIDE2PAD_DICT[stride]
            self.conv = CircularConv2d(in_channels, in_channels, kernel_size=k, padding=p)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.stride, mode='bilinear', align_corners=True)
        if self.with_conv:
            x = self.conv(x)
        return x

UNIFORM_KERNEL2PAD_DICT = {(3, 3): (1, 1, 1, 1), (1, 4): (1, 2, 0, 0)}

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, kernel_size=(3, 3), conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        pad = UNIFORM_KERNEL2PAD_DICT[kernel_size]

        self.norm1 = Normalize(in_channels)
        self.conv1 = CircularConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=pad)
        
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
            
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CircularConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=pad)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CircularConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=pad)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

def make_attn(in_channels, attn_type="none"):
    if attn_type == "none":
        return nn.Identity(in_channels)
    # Expand here for vanilla attention if needed
    return nn.Identity(in_channels)

class Decoder(nn.Module):
    """
    LiDM Decoder, unchanged from the original implementation.
    """
    def __init__(self, *, ch, out_ch, ch_mult, strides, num_res_blocks, attn_levels,
                 dropout=0.0, resamp_with_conv=True, in_channels, z_channels, give_pre_end=False,
                 tanh_out=False, use_linear_attn=False, attn_type="vanilla", use_mask=False,
                 **ignorekwargs):
        super().__init__()
        stride2kernel = {(2, 2): (3, 3), (1, 2): (1, 4)}
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]

        self.conv_in = CircularConv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            stride = tuple(strides[i_level - 1]) if i_level > 0 else None
            kernel = stride2kernel[stride] if stride is not None else (1, 4)
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, kernel_size=kernel, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if i_level in attn_levels:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if stride is not None:
                up.upsample = Upsample(block_in, resamp_with_conv, stride)
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = CircularConv2d(block_in, out_ch, kernel_size=(1, 4), stride=1, padding=(1, 2, 0, 0))

    def forward(self, z):
        self.last_z_shape = z.shape
        temb = None

        h = self.conv_in(z)

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

if __name__ == '__main__':
    # Example input: batch size 2, 2-channel range image
    x = torch.randn(2, 2, 64, 1024)

    # Instantiate the Encoder
    encoder = Encoder(in_channels=2, z_channels=16, base_channels=64, k=20)

    # Forward pass
    z, L2_feat, L4_feat = encoder(x)

    print(f"Latent Z shape: {z.shape}")
    print(f"L2 feature shape (for topology loss): {L2_feat.shape}")
    print(f"L4 feature shape (for topology loss): {L4_feat.shape}")

# python model_topolidm.py
# Latent Z shape: torch.Size([2, 16, 16, 128])
# L2 feature shape: torch.Size([2, 2048, 64])
# L4 feature shape: torch.Size([2, 2048, 64])