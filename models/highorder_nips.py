import torch.nn.functional as F
from collections import OrderedDict

from math import exp
# from .utils.CDC import cdcconv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import InvertibleConv1x1
from .refine import Refine, CALayer
import torch.nn.init as init
import os
import cv2
import numbers
from einops import rearrange
import numpy
from typing import Tuple, List
from einops.layers.torch import Rearrange

class OminiInteraction(nn.Module):
    def __init__(self, channelin, channelout):
        super(OminiInteraction, self).__init__()
        self.reflashFused1 = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )
        self.reflashFused2 = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )

        self.reflashInfrared1 = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )
        self.reflashInfrared2 = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )



        self.convf = nn.Sequential(
            nn.Conv2d(2 * channelout, channelout, 1)
        )

        self.convout = nn.Sequential(
            nn.Conv2d(channelout, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )

        self.convatten = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(1, 3), stride=1, padding=(0, 1))
        )

        self.norm1 = LayerNorm(channelout, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(channelout, LayerNorm_type='WithBias')

    def forward(self, vis, inf, i, j):
        B, C, H, W = vis.size()

        vis_fft = torch.fft.rfft2(vis.float())
        inf_fft = torch.fft.rfft2(inf.float())

        atten = vis_fft * inf_fft
        atten = torch.fft.irfft2(atten, s=(H, W))
        atten = self.norm1(atten)


        fused_OneOrderSpa = atten * inf

        theta = vis.float().view(B, C, H * W)
        phi = inf.float().view(B, C, H * W).permute(0, 2, 1)
        g = fused_OneOrderSpa.view(B, C, H * W)

        attention = torch.matmul(theta, phi)
        attention = torch.softmax(attention, dim=-1)

        out = torch.matmul(attention, g)
        out = out.view(B, C, H, W)
        out = self.convout(out)+vis

        fused_OneOrderSpa = self.reflashFused1(atten)
        fused_OneOrderSpa = self.norm2(fused_OneOrderSpa)
        infraredReflash1 = self.reflashInfrared1(out)
        fused_twoOrderSpa = fused_OneOrderSpa * infraredReflash1

        attention1 = attention.unsqueeze(1)
        attention1 = self.convatten(attention1)
        attention1 = torch.softmax(attention1, dim=-1)
        # print(attention1.shape)
        # print(fused_twoOrderSpa.shape)
        #(b,1,c,c)
        #(b,c,h,w)
        fused_twoOrderSpa = fused_OneOrderSpa.view(B, C, (H*W))
        out1 = torch.matmul(attention1[:,0], fused_twoOrderSpa)
        out1 = out1.view(B, C, H, W)


        fused = self.convf(
            torch.cat([out, out1], dim=1)) + vis

        inf = self.reflashFused2(inf)

        return fused, inf

class ResGroup(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int=3,
                 global_kernel_size: int = 11,
                 lr_space: str = "linear",
                 topk: int = 1,
                 recursive: int = 2,
                 use_shuffle: bool = True):
        super().__init__()

        self.RME_block = RME(in_ch=in_ch,
                               num_experts=num_experts,
                               use_shuffle=use_shuffle,
                               lr_space=lr_space,
                               topk=topk,
                               recursive=recursive)
        self.SME_block = SME(in_ch=in_ch,
                                kernel_size=global_kernel_size)

        self.CME_block = CME(in_ch=in_ch,
                                kernel_size=global_kernel_size)
        self.fusion11 = nn.Sequential(nn.Conv2d(4 * in_ch, in_ch, kernel_size=3, padding=1),
                                          nn.GELU(),
                                          nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1))

        self.fusion12 = nn.Sequential(nn.Conv2d(4 * in_ch, in_ch, kernel_size=3, padding=1),
                                           nn.GELU(),
                                          nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1))

        self.SelectBlock1 = SelectBlock(select_dim=2*in_ch, select_len=5, select_size=32, lin_dim=2*in_ch)

    def forward(self, x, y):
        # print('x.shape', x.shape)
        RME_block1, RME_block2 = self.CME_block(x, y)
        SME_block1, SME_block2 = self.RME_block(RME_block1, RME_block2)
        CME_block1, CME_block2 = self.SME_block(SME_block1, SME_block2)
        # print('channel_block')
        select_x1, select_y1 = self.SelectBlock1(torch.cat([x, y], dim=1)).chunk(2, dim=1)
        # select_y1 = self.SelectBlock2(torch.cat([x, y], dim=1))

        fusion11 = self.fusion11(torch.cat([select_x1, RME_block1, SME_block1, CME_block1],dim=1))+x
        # SME_fusion21 = self.SME_fusion21(torch.cat([select_x2, SME_block1], dim=1))
        # CME_fusion31 = self.CME_fusion31(torch.cat([select_x3, CME_block1], dim=1))

        fusion12 = self.fusion12(torch.cat([select_y1, RME_block2, SME_block2, CME_block2], dim=1))+y
        # SME_fusion22 = self.SME_fusion22(torch.cat([select_y2, SME_block2], dim=1))
        # CME_fusion32 = self.CME_fusion32(torch.cat([select_y3, CME_block2], dim=1))

        # fusion1 = self.fusion1(torch.cat([RME_fusion11, SME_fusion21, CME_fusion31], dim=1))
        # fusion2 = self.fusion2(torch.cat([RME_fusion12, SME_fusion22, CME_fusion32], dim=1))

        # print('fusion.shape', fusion.shape)
        return fusion11, fusion12




class SelectBlock(nn.Module):
    def __init__(self, select_dim=128, select_len=5, select_size=32, lin_dim=192):
        super(SelectBlock, self).__init__()
        self.select_param = nn.Parameter(torch.rand(1, select_len, select_dim, select_size, select_size))
        self.linear_layer = nn.Linear(lin_dim, select_len)
        self.conv3x3 = nn.Conv2d(select_dim, select_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        select_weights = F.softmax(self.linear_layer(emb), dim=1)
        select = select_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.select_param.unsqueeze(0).repeat(B, 1,
                                                                                                                  1, 1,
                                                                                                                  1,
                                                                                                                  1).squeeze(
            1)
        select = torch.sum(select, dim=1)
        select = F.interpolate(select, (H, W), mode="bilinear")
        select = self.conv3x3(select)

        return select


class Channel_MOE(nn.Module):
    def __init__(self, in_ch, mlp_ratio=8):
        super(Channel_MOE, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        hidden_dim = int(in_ch // 2 * mlp_ratio)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_ch // 2, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_ch // 2, 1, 1, 0),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_ch // 2, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_ch // 2, 1, 1, 0),
        )
        self.proj1 = nn.Conv2d(in_ch // 2, in_ch, 3,1,1)
        self.proj2 = nn.Conv2d(in_ch // 2, in_ch, 3,1,1)

    def forward(self, x, y):
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = y.chunk(2, dim=1)

        x1 = self.avg_pool1(x1)
        x1 = self.fc1(x1)
        out1 = self.proj1(x1 * y1)

        y2 = self.avg_pool2(y2)
        y2 = self.fc2(y2)
        out2 = self.proj2(x2 * y2)

        return out1, out2

#############################
# Channel Block
#############################
class CME(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int = 11):
        super().__init__()

        self.norm_11 = LayerNorm_(in_ch, data_format='channels_first')
        self.norm_12 = LayerNorm_(in_ch, data_format='channels_first')
        self.block = Channel_MOE(in_ch=in_ch)

        self.norm_21 = LayerNorm_(in_ch, data_format='channels_first')
        self.norm_22 = LayerNorm_(in_ch, data_format='channels_first')
        self.ffn1 = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())
        self.ffn2 = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())

    def forward(self, x, y):
        x_norm_11 = self.norm_11(x)
        y_norm_11 = self.norm_12(y)

        x_block, y_block = self.block(x_norm_11, y_norm_11)

        x_block = x_block+x
        y_block = y_block+y

        x_norm_21 = self.norm_21(x_block)
        y_norm_22 = self.norm_22(y_block)

        x_ffn = self.ffn1(x_norm_21) + x_block
        y_ffn = self.ffn2(y_norm_22) + y_block

        return x_ffn, y_ffn

#############################
# Local Block
#############################
class SME(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int = 11):
        super().__init__()

        self.norm_11 = LayerNorm_(in_ch, data_format='channels_first')
        self.norm_12 = LayerNorm_(in_ch, data_format='channels_first')
        self.block = StripedConvFormer(in_ch=in_ch, kernel_size=kernel_size)

        self.norm_21 = LayerNorm_(in_ch, data_format='channels_first')
        self.norm_22 = LayerNorm_(in_ch, data_format='channels_first')
        self.ffn1 = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())
        self.ffn2 = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())

    def forward(self, x, y):
        x_norm_11 = self.norm_11(x)
        y_norm_11 = self.norm_12(y)

        x_block, y_block = self.block(x_norm_11, y_norm_11)

        x_block = x_block+x
        y_block = y_block+y

        x_norm_21 = self.norm_21(x_block)
        y_norm_22 = self.norm_22(y_block)

        x_ffn = self.ffn1(x_norm_21) + x_block
        y_ffn = self.ffn2(y_norm_22) + y_block

        return x_ffn, y_ffn

class StripedConvFormer(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.proj1 = nn.Conv2d(in_ch, in_ch, 3,1,1)
        self.proj2 = nn.Conv2d(in_ch, in_ch, 3,1,1)
        self.to_qv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, padding=0),
            nn.GELU(),
        )

        self.to_qv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, padding=0),
            nn.GELU(),
        )

        self.attn1 = StripedConv2d(in_ch, kernel_size=kernel_size, depthwise=True)
        self.attn2 = StripedConv2d(in_ch, kernel_size=kernel_size, depthwise=True)

    def forward(self, x, y):
        q1, v1 = self.to_qv1(x).chunk(2, dim=1)
        q2, v2 = self.to_qv2(y).chunk(2, dim=1)
        q1 = self.attn1(q1)
        x1 = self.proj1(q1 * v2)

        q2 = self.attn2(q2)
        y1 = self.proj2(q2 * v1)
        return x1, y1


#############################
# Global Blocks
#############################
class RME(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 topk: int,
                 lr_space: str = "linear",
                 recursive: int = 2,
                 use_shuffle: bool = False, ):
        super().__init__()

        self.norm_11 = LayerNorm_(in_ch, data_format='channels_first')
        self.norm_12 = LayerNorm_(in_ch, data_format='channels_first')
        self.block = MoEBlock(in_ch=in_ch, num_experts=num_experts, topk=topk, use_shuffle=use_shuffle,
                              recursive=recursive, lr_space=lr_space, )

        self.norm_21 = LayerNorm_(in_ch, data_format='channels_first')
        self.norm_22 = LayerNorm_(in_ch, data_format='channels_first')
        self.ffn1 = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())
        self.ffn2 = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())

    def forward(self, x, y):
        x_norm_11 = self.norm_11(x)
        y_norm_11 = self.norm_12(y)

        x_block, y_block = self.block(x_norm_11, y_norm_11)

        x_block = x_block+x
        y_block = y_block+y

        x_norm_21 = self.norm_21(x_block)
        y_norm_22 = self.norm_22(y_block)

        x_ffn = self.ffn1(x_norm_21) + x_block
        y_ffn = self.ffn2(y_norm_22) + y_block

        return x_ffn, y_ffn
#################
# MoE Layer
#################
class MoEBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 topk: int,
                 use_shuffle: bool = False,
                 lr_space: str = "linear",
                 recursive: int = 2):
        super().__init__()
        self.use_shuffle = use_shuffle
        self.recursive = recursive

        self.conv_11 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, 2 * in_ch, kernel_size=1, padding=0)
        )

        self.conv_12 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, 2 * in_ch, kernel_size=1, padding=0)
        )

        self.agg_conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=4, groups=in_ch),
            nn.GELU())

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
        )

        self.agg_conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=4, groups=in_ch),
            nn.GELU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
        )

        self.conv_21 = nn.Sequential(
            StripedConv2d(in_ch, kernel_size=3, depthwise=True),
            nn.GELU())

        self.conv_22 = nn.Sequential(
            StripedConv2d(in_ch, kernel_size=3, depthwise=True),
            nn.GELU())

        if lr_space == "linear":
            grow_func = lambda i: i + 2
        elif lr_space == "exp":
            grow_func = lambda i: 2 ** (i + 1)
        elif lr_space == "double":
            grow_func = lambda i: 2 * i + 2
        else:
            raise NotImplementedError(f"lr_space {lr_space} not implemented")

        self.moe_layer1 = MoELayer(
            experts=[Expert(in_ch=in_ch, low_dim=grow_func(i)) for i in range(num_experts)],
            # add here multiple of 2 as low_dim
            gate=Router(in_ch=in_ch, num_experts=num_experts),
            num_expert=topk,
        )

        self.moe_layer2 = MoELayer(
            experts=[Expert(in_ch=in_ch, low_dim=grow_func(i)) for i in range(num_experts)],
            # add here multiple of 2 as low_dim
            gate=Router(in_ch=in_ch, num_experts=num_experts),
            num_expert=topk,
        )

        self.proj1 = nn.Conv2d(in_ch, in_ch, 3,1,1)
        self.proj2 = nn.Conv2d(in_ch, in_ch, 3,1,1)

    def calibrate1(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x

        for _ in range(self.recursive):
            x = self.agg_conv1(x)
        x = self.conv1(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return res + x

    def calibrate2(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x

        for _ in range(self.recursive):
            x = self.agg_conv2(x)
        x = self.conv2(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return res + x
    def forward(self, x, y):
        x = self.conv_11(x)
        y = self.conv_12(y)

        if self.use_shuffle:
            x = channel_shuffle(x, groups=2)
            y = channel_shuffle(y, groups=2)
        x, k1 = torch.chunk(x, chunks=2, dim=1)
        y, k2 = torch.chunk(y, chunks=2, dim=1)

        x = self.conv_21(x)
        k1 = self.calibrate1(k1)
        y = self.conv_22(y)
        k2 = self.calibrate2(k2)

        x = self.moe_layer1(x, k2)
        y = self.moe_layer2(y, k1)

        x = self.proj1(x)
        y = self.proj2(y)
        return x, y


class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_expert: int = 1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert = num_expert

    def forward(self, inputs: torch.Tensor, k: torch.Tensor):
        out = self.gate(inputs)
        weights = F.softmax(out, dim=1, dtype=torch.float).to(inputs.dtype)
        topk_weights, topk_experts = torch.topk(weights, self.num_expert)
        out = inputs.clone()


        # training
        exp_weights = torch.zeros_like(weights)
        exp_weights.scatter_(1, topk_experts, weights.gather(1, topk_experts))
        for i, expert in enumerate(self.experts):
            out += expert(inputs, k) * exp_weights[:, i:i + 1, None, None]

        # testing
        # selected_experts = [self.experts[i] for i in topk_experts.squeeze(dim=0)]
        # for i, expert in enumerate(selected_experts):
        #     out += expert(inputs, k) * topk_weights[:, i:i + 1, None, None]

        return out


class Expert(nn.Module):
    def __init__(self,
                 in_ch: int,
                 low_dim: int, ):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv2d(low_dim, in_ch, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(k) * x  # here no more sigmoid
        x = self.conv_3(x)
        return x


class Router(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(in_ch, num_experts, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


#################
# Utilities
#################
class StripedConv2d(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int,
                 depthwise: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(1, self.kernel_size), padding=(0, self.padding),
                      groups=in_ch if depthwise else 1),
            nn.Conv2d(in_ch, in_ch, kernel_size=(self.kernel_size, 1), padding=(self.padding, 0),
                      groups=in_ch if depthwise else 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x


class GatedFFN(nn.Module):
    def __init__(self,
                 in_ch,
                 mlp_ratio,
                 kernel_size,
                 act_layer, ):
        super().__init__()
        mlp_ch = in_ch * mlp_ratio

        self.fn_1 = nn.Sequential(
            nn.Conv2d(in_ch, mlp_ch, kernel_size=1, padding=0),
            act_layer,
        )
        self.fn_2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
            act_layer,
        )

        self.gate = nn.Conv2d(mlp_ch // 2, mlp_ch // 2,
                              kernel_size=kernel_size, padding=kernel_size // 2, groups=mlp_ch // 2)

    def feat_decompose(self, x):
        s = x - self.gate(x)
        x = x + self.sigma * s
        return x

    def forward(self, x: torch.Tensor):
        x = self.fn_1(x)
        x, gate = torch.chunk(x, 2, dim=1)

        gate = self.gate(gate)
        x = x * gate

        x = self.fn_2(x)
        return x


class LayerNorm_(nn.Module):
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
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, gc)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


class DenseBlockMscale(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier'):
        super(DenseBlockMscale, self).__init__()
        self.ops = DenseBlock(channel_in, channel_out, init)
        self.fusepool = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channel_out,channel_out,1,1,0),nn.LeakyReLU(0.1,inplace=True))
        self.fc1 = nn.Sequential(nn.Conv2d(channel_out,channel_out,1,1,0),nn.LeakyReLU(0.1,inplace=True))
        self.fc2 = nn.Sequential(nn.Conv2d(channel_out, channel_out, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True))
        self.fc3 = nn.Sequential(nn.Conv2d(channel_out, channel_out, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True))
        self.fuse = nn.Conv2d(3*channel_out,channel_out,1,1,0)

    def forward(self, x):
        x1 = x
        x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
        x3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear')
        x1 = self.ops(x1)
        x2 = self.ops(x2)
        x3 = self.ops(x3)
        x2 = F.interpolate(x2, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x3 = F.interpolate(x3, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        xattw = self.fusepool(x1+x2+x3)
        xattw1 = self.fc1(xattw)
        xattw2 = self.fc2(xattw)
        xattw3 = self.fc3(xattw)
        # x = x1*xattw1+x2*xattw2+x3*xattw3
        x = self.fuse(torch.cat([x1*xattw1,x2*xattw2,x3*xattw3],1))

        return x


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlockMscale(channel_in, channel_out, init)
            else:
                return DenseBlockMscale(channel_in, channel_out)
            # return UNetBlock(channel_in, channel_out)
        else:
            return None
    return constructor


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out
    
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
    

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)





class spatialInteraction(nn.Module):
    def __init__(self, channelin, channelout):
        super(spatialInteraction, self).__init__()
        self.ResGroup1 = ResGroup(in_ch=channelin)
        self.ResGroup2 = ResGroup(in_ch=channelin)
        self.ResGroup3 = ResGroup(in_ch=channelin)

    def forward(self, vis, inf, i, j):
        _, C, H, W = vis.size()

        vis1, inf1 = self.ResGroup1(vis, inf)
        vis2, inf2 = self.ResGroup2(vis1, inf1)
        vis3, inf3 = self.ResGroup3(vis2, inf2)

        return vis3, inf3


class highOrderInteraction(nn.Module):
    def __init__(self, channelin, channelout):
        super(highOrderInteraction, self).__init__()
        self.spatial = spatialInteraction(channelin, channelout)
        # self.channel = channelInteraction(channelin, channelout)

    def forward(self, vis_y, inf, i, j):

        vis_spa, inf_spa = self.spatial(vis_y, inf, i, j)
        # vis_cha, inf_cha = self.channel(vis_spa, inf_spa, i, j)
        
        return vis_spa, inf_spa
    

# class EdgeBlock(nn.Module):
#     def __init__(self, channelin, channelout):
#         super(EdgeBlock, self).__init__()
#         self.process = nn.Conv2d(channelin,channelout,3,1,1)
#         self.Res = nn.Sequential(nn.Conv2d(channelout,channelout,3,1,1),
#             nn.ReLU(),nn.Conv2d(channelout, channelout, 3, 1, 1))
#         self.CDC = cdcconv(channelin, channelout)

#     def forward(self,x):

#         x = self.process(x)
#         out = self.Res(x) + self.CDC(x)

#         return out
import torch.nn.functional as F
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        # input = torch.cat([x,resi],dim=1)
        # out = self.conv_3(input)
        return x+resi

class Net(nn.Module):
    def __init__(self, num_channels=None,base_filter=None,args=None):
        super(Net,self).__init__()

        vis_channels=1
        inf_channels=1
        n_feat=16
        self.vis = nn.Sequential(nn.Conv2d(vis_channels,n_feat,3,1,1),HinResBlock(n_feat,n_feat),HinResBlock(n_feat,n_feat),HinResBlock(n_feat,n_feat),HinResBlock(n_feat,n_feat))
        self.inf = nn.Sequential(nn.Conv2d(inf_channels,n_feat,3,1,1),HinResBlock(n_feat,n_feat),HinResBlock(n_feat,n_feat),HinResBlock(n_feat,n_feat),HinResBlock(n_feat,n_feat))

        self.interaction1 = highOrderInteraction(channelin=n_feat, channelout=n_feat)
        self.interaction2 = highOrderInteraction(channelin=n_feat, channelout=n_feat)
        self.interaction3 = highOrderInteraction(channelin=n_feat, channelout=n_feat)

        self.postprocess = nn.Sequential(InvBlock(DenseBlock, 3 * n_feat, 3 * n_feat // 2),
                                         nn.Conv2d(3 * n_feat, n_feat, 1, 1, 0))
        
        self.reconstruction = Refine(n_feat, out_channel=vis_channels)

        # self.ResGroup = ResGroup(in_ch=n_feat)

        self.i = 0

    def forward(self, ms,_,pan):
        vis_y = ms
        inf = pan
        vis_y = self.vis(vis_y)
        inf = self.inf(inf)

        # inf_ = self.ResGroup(inf)

        vis_y_feat, inf_feat = self.interaction1(vis_y, inf, self.i, j=1)
        vis_y_feat2, inf_feat2 = self.interaction2(vis_y_feat, inf_feat, self.i, j=2)
        vis_y_feat3, inf_feat3 = self.interaction3(vis_y_feat2, inf_feat2, self.i, j=3)

        fused = self.postprocess(torch.cat([vis_y_feat, vis_y_feat2, vis_y_feat3], 1))

        fused = self.reconstruction(fused)

        self.i += 1

        return fused
