# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock, UnetResBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.networks.blocks import Convolution


rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinTransformerBlock",
    "PatchMerging",
    "PatchMergingV2",
    "PatchMergingV3",
    "PatchExpand",
    "PatchExpandV2",
    "PatchExpandV3",
    "FinalPatchExpand_X4",
    "FinalPatchExpand_X4V2",
    "FinalPatchExpand_X4V3",
    "MERGING_MODE",
    "EXPANDING_MODE",
    "BasicLayer",
    "SwinUnet",
]


def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Sequence[int],
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Sequence[int],
            shift_size: Sequence[int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            act_layer: str = "GELU",
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()
        x = self.norm1(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def load_from(self, weights, n_block, layer):
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])  # type: ignore
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):

        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchMergingV3(nn.Module):
    """
    Patch merging layer based on UnetrBasicBlock of Monai
    """

    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm,
                 norm_name: Union[Tuple, str] = "instance", spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.reduction = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=dim,
            out_channels=2 * dim,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=True,
        )

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x = self.norm(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        x = self.reduction(x)
        x = rearrange(x, "b c d h w -> b d h w c")
        return x

MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2, "mergingv3":PatchMergingV3}


class PatchExpand(nn.Module):
    """
    Patch expanding layer based on: "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation
    <https://arxiv.org/pdf/2105.05537.pdf>"
    https://github.com/HuCaoFighting/Swin-Unet/tree/main
    """
    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3,
                 dim_scale: int = 2) -> None:
        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.expand = nn.Linear(dim, 4 * dim, bias=False) if dim_scale == 2 else nn.Identity()
            self.norm = norm_layer(dim // dim_scale)
        elif spatial_dims == 2:
            self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
            self.norm = norm_layer(dim // dim_scale)
        self.dim = dim

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x_shape = x.size()
        x = self.expand(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            x = x.view(b, d * 2, h * 2, w * 2, c // 8)
        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            x = x.view(b, h * 2, w * 2, c // 4)
        x = self.norm(x)

        return x


class PatchExpandV2(nn.Module):
    """
    Patch expanding layer based on: "STU-Net: Scalable and Transferable Medical Image Segmentation Models
    Empowered by Large-Scale Supervised Pre-training
    <https://arxiv.org/pdf/2304.06716.pdf>"
    https://github.com/Ziyan-Huang/STU-Net
    """
    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3,
                 dim_scale: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.norm = norm_layer(dim // dim_scale)
        self.dim = dim
        self.expand = nn.Conv3d(dim, dim // dim_scale, kernel_size=1)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x_shape = x.size()

        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            x = nn.functional.interpolate(x, scale_factor=self.dim_scale,
                                          mode='nearest')
            x = self.expand(x)
            x = rearrange(x, "b c d h w -> b d h w c")
        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            x = nn.functional.interpolate(x, scale_factor=(self.dim_scale * h, self.dim_scale * w), mode='nearest')
            x = self.expand(x)
        x = self.norm(x)
        x = rearrange(x, "b d h w c -> b c d h w")

        return x


class PatchExpandV3(nn.Module):
    """
    Patch expanding layer using the Transpose Convolution layers.
    """
    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3,
                 dim_scale: int = 2, norm_name="instance") -> None:
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.norm = norm_layer(dim // dim_scale)
        self.dim = dim
        self.transp_conv = get_conv_layer(
            spatial_dims,
            dim,
            dim // dim_scale,
            kernel_size=2,
            stride=2,
            conv_only=True,
            is_transposed=True,
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x_shape = x.size()

        if len(x_shape) == 5:
            x = self.transp_conv(x)
            x = rearrange(x, "b c d h w -> b d h w c")
        elif len(x_shape) == 4:
            x = self.transp_conv(x)
        x = self.norm(x)
        x = rearrange(x, "b d h w c -> b c d h w")

        return x


class FinalPatchExpand_X4(nn.Module):
    """
    Final patch expanding layer based on: "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation
    <https://arxiv.org/pdf/2105.05537.pdf>"
    https://github.com/HuCaoFighting/Swin-Unet/tree/main
    """
    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3,
                 dim_scale: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        if spatial_dims == 3:
            self.expand = nn.Linear(dim, (self.dim_scale ** 3) * dim, bias=False)
            self.norm = norm_layer(dim)
        elif spatial_dims == 2:
            self.expand = nn.Linear(dim, (self.dim_scale ** 2) * dim, bias=False)
            self.norm = norm_layer(dim)
        self.dim = dim

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x_shape = x.size()
        x = self.expand(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            x = x.view(b, d * self.dim_scale, h * self.dim_scale, w * self.dim_scale, c // (self.dim_scale ** 3))
        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            x = x.view(b, h * self.dim_scale, w * self.dim_scale, c // (self.dim_scale ** 2))
        x = self.norm(x)

        return x


class FinalPatchExpand_X4V2(nn.Module):
    """
    Final patch expanding layer based on: "STU-Net: Scalable and Transferable Medical Image Segmentation Models
    Empowered by Large-Scale Supervised Pre-training
    <https://arxiv.org/pdf/2304.06716.pdf>"
    https://github.com/Ziyan-Huang/STU-Net
    """
    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3,
                 dim_scale: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.spatial_dims = spatial_dims
        self.dim_scale = dim_scale
        self.norm = norm_layer(dim)
        if spatial_dims == 3:
            self.expand = nn.Conv3d(dim, dim, kernel_size=1)
        elif spatial_dims == 2:
            self.expand = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = nn.functional.interpolate(x, scale_factor=self.dim_scale,
                                      mode='nearest')
        if self.spatial_dims == 3:
            x = self.expand(x)
            x = rearrange(x, "b c d h w -> b d h w c")
            x = self.norm(x)
            x = rearrange(x, "b d h w c -> b c d h w")
        elif self.spatial_dims == 2:
            x = self.expand(x)
            x = rearrange(x, "b c h w -> b h w c")
            x = self.norm(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x


class FinalPatchExpand_X4V3(nn.Module):
    """
    Final patch expanding layer using the Transpose Convolution layers.
    """
    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3,
                 dim_scale: int = 2, norm_name="instance") -> None:
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.norm = norm_layer(dim)
        self.dim = dim
        self.expand = get_conv_layer(
            spatial_dims,
            dim,
            dim,
            kernel_size=dim_scale,
            stride=dim_scale,
            dropout=0,
            bias=False,
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=True,
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x_shape = x.size()
        if len(x_shape) == 5:
            x = self.expand(x)
            x = rearrange(x, "b c d h w -> b d h w c")
            x = self.norm(x)
            x = rearrange(x, "b d h w c -> b c d h w")
        elif len(x_shape) == 4:
            x = self.expand(x)
            x = rearrange(x, "b c h w -> b h w c")
            x = self.norm(x)
            x = rearrange(x, "b h w c -> b c h w")

        return x


EXPANDING_MODE = {"expand": PatchExpand, "expandV2": PatchExpandV2, "expandV3": PatchExpandV3,
                  'finalExpand': FinalPatchExpand_X4, 'finalExpandV2': FinalPatchExpand_X4V2,
                  'finalExpandV3': FinalPatchExpand_X4V3}


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            depth: int,
            num_heads: int,
            window_size: Sequence[int],
            drop_path: list,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            upsample: Optional[nn.Module] = None,
            downsample: Optional[nn.Module] = None,
            use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))
        self.upsample = upsample
        if callable(self.upsample):
            self.upsample = upsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size), dim_scale=2)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c d h w -> b d h w c")
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
                if self.downsample != PatchMergingV3:
                    x = rearrange(x, "b d h w c -> b c d h w")
            if self.upsample is not None:
                if type(self.upsample) != PatchExpand:
                    x = rearrange(x, "b d h w c -> b c d h w")
                    x = self.upsample(x)
                else:
                    x = self.upsample(x)
                    x = rearrange(x, "b d h w c -> b c d h w")
            if self.downsample is None and self.upsample is None:
                x = rearrange(x, "b d h w c -> b c d h w")

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c h w -> b h w c")
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
                x = rearrange(x, "b h w c -> b c h w")
            if self.upsample is not None:
                if type(self.upsample) != PatchExpand:
                    x = rearrange(x, "b h w c -> b c h w")
                    x = self.upsample(x)
                else:
                    x = self.upsample(x)
                    x = rearrange(x, "b h w c -> b c h w")
            if self.downsample is None and self.upsample is None:
                x = rearrange(x, "b h w c -> b c h w")
        return x

class SwinUnet(nn.Module):
    """
    3D version of the Swin-Unet model based on: "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation
    <https://arxiv.org/pdf/2105.05537.pdf>"
    """

    def __init__(
            self,
            img_size: Union[Sequence[int], int],
            in_channels: int,
            out_channels: int,
            depths: Sequence[int] = (2, 2, 2, 2),
            num_heads: Sequence[int] = (3, 6, 12, 24),
            patch_size: int = 4,
            feature_size: int = 24,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            dropout_path_rate: float = 0.0,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_name: Union[Tuple, str] = "instance",
            drop_path_rate: float = 0.0,
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            patch_norm: bool = False,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
            upsample=("expandV3", "finalExpandV3"),
            downsample="merging",
            bottleneck="default",
            # default: two Swin Transformer block, V0: No bottleneck,
            # V1: One Swin Transformer block, V2: One Residual block, V3: two residual blocks
            skip_connection="default"
            # default: residual blocks, V1: attentionGates from attentionUnet.

    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            patch_size: size of the patch embeddings.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            upsample: module used for downsampling, available options are `("expand", "finalExpand"),
                `("expandV2", "finalExpandV2"), `("expandV3", "finalExpandV3").
                The default is currently `("expandV3", "finalExpandV3")`
            downsample: module used for downsampling, available options are `mergingv3`, `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
        """

        super().__init__()

        self.upsample_type = upsample[0]
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.window_size = ensure_tuple_rep(7, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")
        self.num_layers = len(depths)
        self.embed_dim = feature_size
        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_channels,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers_down = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        self.skip_connection_type = skip_connection

        # build encoder layers
        for i_layer in range(self.num_layers):
            if (i_layer < (self.num_layers - 1)):
                layer = BasicLayer(
                    dim=int(self.embed_dim * 2 ** i_layer),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=self.window_size,
                    drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    downsample=down_sample_mod,
                    use_checkpoint=use_checkpoint,
                )
                self.layers_down.append(layer)
            else:
                if (bottleneck == "default"):
                    layer = BasicLayer(
                        dim=int(self.embed_dim * 2 ** i_layer),
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=self.window_size,
                        drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        downsample=None,
                        use_checkpoint=use_checkpoint,
                    )
                    self.layers_down.append(layer)
                elif (bottleneck == "V1"):
                    layer = BasicLayer(
                        dim=int(self.embed_dim * 2 ** i_layer),
                        depth=1,
                        num_heads=num_heads[i_layer],
                        window_size=self.window_size,
                        drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        downsample=None,
                        use_checkpoint=use_checkpoint,
                    )
                    self.layers_down.append(layer)
                elif (bottleneck in ["V2"]):
                    layer = UnetrBasicBlock(
                        spatial_dims=spatial_dims,
                        in_channels=int(self.embed_dim * 2 ** i_layer),
                        out_channels=int(self.embed_dim * 2 ** i_layer),
                        kernel_size=3,
                        stride=1,
                        norm_name=norm_name,
                        res_block=True,
                    )
                    self.layers_down.append(layer)
                elif (bottleneck in ["V3"]):
                    self.layers_down.append(UnetrBasicBlock(
                        spatial_dims=spatial_dims,
                        in_channels=int(self.embed_dim * 2 ** i_layer),
                        out_channels=int(self.embed_dim * 2 ** i_layer),
                        kernel_size=3,
                        stride=1,
                        norm_name=norm_name,
                        res_block=True,
                    ))
                    self.layers_down.append(UnetrBasicBlock(
                        spatial_dims=spatial_dims,
                        in_channels=int(self.embed_dim * 2 ** i_layer),
                        out_channels=int(self.embed_dim * 2 ** i_layer),
                        kernel_size=3,
                        stride=1,
                        norm_name=norm_name,
                        res_block=True,
                    ))


        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        #
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        up_sample_mod = look_up_option(upsample[0], EXPANDING_MODE) if isinstance(upsample[0], str) else upsample[0]
        up_final_sample_mod = look_up_option(upsample[1],
                                             EXPANDING_MODE) if isinstance(upsample[1], str) else upsample[1]
        for i_layer in range(self.num_layers):
            fs = int(self.embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            if (skip_connection == "default"):
                concat_linear = UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=2 * fs,
                    out_channels=fs,
                    kernel_size=3,
                    stride=1,
                    norm_name=norm_name,
                    res_block=True,
                ) # Skip-connection
            if i_layer == 0:
                layer_up = up_sample_mod(
                    dim=int(self.embed_dim * 2 ** (self.num_layers - i_layer - 1)), dim_scale=2, norm_layer=norm_layer,
                    spatial_dims=spatial_dims)
            else:
                layer_up = BasicLayer(
                    dim=int(self.embed_dim * 2 ** (self.num_layers - i_layer - 1)),
                    depth=depths[(self.num_layers - i_layer)],
                    num_heads=num_heads[(self.num_layers - i_layer - 1)],
                    window_size=self.window_size,
                    drop_path=dpr[sum(depths[:self.num_layers - i_layer - 1]): sum(
                        depths[: self.num_layers - i_layer])],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    upsample=up_sample_mod if (i_layer < (self.num_layers - 1)) else None,
                    use_checkpoint=use_checkpoint,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.up = up_final_sample_mod(dim=self.embed_dim, dim_scale=self.patch_size[0], spatial_dims=spatial_dims)
        self.output = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.embed_dim, out_channels=out_channels)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

        # Encoder and Bottleneck

    def forward_features(self, x, normalize=True):
        x_downsample = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.proj_out(x, normalize)
        x_downsample.append(x)
        for inx, layer_down in enumerate(self.layers_down):
            x = layer_down(x.contiguous())
            x = self.proj_out(x, normalize)
            x_downsample.append(x)

        return x, x_downsample

    def forward(self, x, normalize=True):
        enc0 = self.encoder1(x)
        x, x_downsample = self.forward_features(x, normalize)
        for inx, layer_up in enumerate(self.layers_up):
            if (self.skip_connection_type == "default"):
                x = torch.cat([x,
                               x_downsample[self.num_layers - 1 - inx],
                               ], 1)

                x = self.concat_back_dim[inx](x)
            if inx == 0:
                if self.upsample_type == 'expand':
                    x = rearrange(x, "b c d h w -> b d h w c")
                    x = layer_up(x)
                    x = rearrange(x, "b d h w c -> b c d h w")
                else:
                    x = layer_up(x)
            else:
                x = layer_up(x)
            x = self.proj_out(x, normalize)
        if self.upsample_type == 'expand':
            x = rearrange(x, "b c d h w -> b d h w c")
            x = self.up(x)
            x = rearrange(x, "b d h w c -> b c d h w")
        else:
            x = self.up(x)
        x = torch.cat([x, enc0], 1)
        x = self.encoder2(x)

        x = self.output(x)
        return x
