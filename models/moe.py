# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from mamba_ssm.modules.mamba_simple import Mamba
import numbers
import scipy
from . import thops
from .fusion_strategy import attention_fusion_weight
from .vmamba_efficient import CrossVSSBlock_Base
from .functions import LayerNorm


class CVLoss(nn.Module):
    def __init__(self, loss_weight=100.0,reduction='mean'):
        super(CVLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    def forward(self,logits):
        # print(torch.mean(logits,dim=1).shape)
        cv = torch.std(logits,dim=1)/torch.mean(logits,dim=1)
        # print('cv={}'.format(cv))
        return self.loss_weight*torch.mean(cv**2)


class GateNetwork(nn.Module):
    def __init__(self, input_size, num_experts, top_k):
        super(GateNetwork, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.fc0 = nn.Linear(input_size,num_experts)
        self.fc1 = nn.Linear(input_size, num_experts)
        self.relu1 = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        init.zeros_(self.fc1.weight)
        self.sp = nn.Softplus()

    def forward(self, x):
        # Flatten the input tensor
        x = self.gmp(x)+self.gap(x)
        x = x.view(-1, self.input_size)
        inp = x
        # Pass the input through the gate network layers
        x = self.fc1(x)
        x= self.relu1(x)
        noise = self.sp(self.fc0(inp))
        noise_mean = torch.mean(noise,dim=1)
        noise_mean = noise_mean.view(-1,1)
        std = torch.std(noise,dim=1)
        std = std.view(-1,1)
        noram_noise = (noise-noise_mean)/std
        # Apply topK operation to get the highest K values and indices along dimension 1 (columns)
        topk_values, topk_indices = torch.topk(x+noram_noise, k=self.top_k, dim=1)
        # Set all non-topK values to -inf to ensure they are not selected by softmax
        mask = torch.zeros_like(x).scatter_(dim=1, index=topk_indices, value=1.0)
        x[~mask.bool()] = float('-inf')
        # Pass the masked tensor through softmax to get gating coefficients for each expert network
        gating_coeffs = self.softmax(x)
        return gating_coeffs


class FuseNet(nn.Module):
    def __init__(self, channels, num_experts=4, k=2, drop_path_rate=0.1):
        super(FuseNet, self).__init__()
        self.gate = GateNetwork(channels, num_experts, k)
        self.num_experts = num_experts
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]
        self.expert_networks_d = nn.ModuleList(
            [CrossVSSBlock_Base(
                hidden_dim=channels, 
                drop_path=dpr,
                norm_layer=nn.LayerNorm,
                channel_first=False,
                ssm_d_state=16,
                ssm_ratio=2.0,
                ssm_dt_rank="auto",
                ssm_act_layer=nn.SiLU,
                ssm_conv=3,
                ssm_conv_bias=True,
                ssm_drop_rate=0.0,
                ssm_init="v0",
                forward_type="v2",
                use_checkpoint=False,
            ) for i in range(num_experts)])
        self.pre_fuse = nn.Sequential(InvBlock(HinResBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))

    def forward(self, a, b):
        # a=a.permute(0,3,1,2)
        # b=b.permute(0,3,1,2)
        x = self.pre_fuse(torch.cat((a, b), dim=1))
        cof = self.gate(x)
        # print('cof={}'.format(cof))
        # print('cof.shape={}'.format(cof.shape))
        out = torch.zeros_like(x).to(x.device)
        for idx in range(self.num_experts):
            if cof[:,idx].all()==0:
                continue
            # print('mask_all={}'.format(torch.where(cof[:,idx]>0)))
            mask = torch.where(cof[:,idx]>0)[0]
            # print('mask={}'.format(mask))
            expert_layer = self.expert_networks_d[idx]
            # print('a[mask].shape={}'.format(a[mask].shape))
            expert_out = expert_layer(a[mask], b[mask]).permute(0,3,1,2)
            cof_k = cof[mask,idx].view(-1,1,1,1)
            # print(cof_k)
            # print('out[mask].shape={}, expert_out.shape={}, cof_k.shape={}'.format(out[mask].shape, expert_out.shape, cof_k.shape))
            out[mask]+=expert_out*cof_k
        # print('cof_k={}'.format(cof_k))
        return out, cof


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

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


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            pixels = thops.pixels(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float()\
                              .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = thops.sum(self.log_s) * thops.pixels(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


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
        return x+resi


class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
    # def __init__(self,basefilter,is_2d=False) -> None:
        super().__init__()
        self.nc = basefilter
        # self.is_2d = is_2d
    def forward(self, x, x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x
        # if self.is_2d:
        #     f0 = x[0].transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])
        #     f1 = x[1].transpose(1, 2).view(B, self.nc, x_size[0], x_size[1]).transpose(2, 3).flip(2)
        #     f2 = x[2].flip(1).transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])
        #     f3 = x[3].flip(1).transpose(1, 2).view(B, self.nc, x_size[0], x_size[1]).transpose(2, 3).flip(2)
        #     return (f0+f1+f2+f3)/4
        # else:
        #     x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        #     return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    # def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True, is_2d=False):
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')
        # self.is_2d = is_2d

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            # x = self.norm(x)
        return x
        # if self.is_2d:
        #     if self.flatten:
        #         x_reverse = x.transpose(2, 3).flip(2)
        #         x0 = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        #         x1 = x_reverse.flatten(2).transpose(1, 2)
        #         x2 = x0.flip(1)
        #         x3 = x1.flip(1)
        #     return (x0, x1, x2, x3)
        # else:
        #     if self.flatten:
        #         x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        #     # x = self.norm(x)
        #     return x


# class FuseNet(nn.Module):
#     def __init__(self, channels, num_experts, k, drop_path_rate=0.1):
#         super(FuseNet, self).__init__()
#         self.gate = GateNetwork(channels, num_experts, k)
#         self.num_experts = num_experts
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]
#         self.expert_networks_d = nn.ModuleList(
#             [CrossVSSBlock_Base(
#                 hidden_dim=channels, 
#                 drop_path=dpr,
#                 norm_layer=nn.LayerNorm,
#                 channel_first=False,
#                 ssm_d_state=16,
#                 ssm_ratio=2.0,
#                 ssm_dt_rank="auto",
#                 ssm_act_layer=nn.SiLU,
#                 ssm_conv=3,
#                 ssm_conv_bias=True,
#                 ssm_drop_rate=0.0,
#                 ssm_init="v0",
#                 forward_type="v2",
#                 use_checkpoint=False,
#             ) for i in range(num_experts)])
#         self.pre_fuse = nn.Sequential(InvBlock(HinResBlock, 2*channels, channels),
#                                          nn.Conv2d(2*channels,channels,1,1,0))

#     def forward(self, a, b):
#         a=a.permute(0,3,1,2)
#         b=b.permute(0,3,1,2)
#         x = self.pre_fuse(torch.cat((a, b), dim=1))
#         cof = self.gate(x)
#         out = torch.zeros_like(x).to(x.device)
#         for idx in range(self.num_experts):
#             if cof[:,idx].all()==0:
#                 continue
#             mask = torch.where(cof[:,idx]>0)[0]
#             # print('mask={}'.format(mask))
#             expert_layer = self.expert_networks_d[idx]
#             # print('a[mask].shape={}'.format(a[mask].shape))
#             expert_out = expert_layer(a[mask], b[mask]).permute(0,3,1,2)
#             cof_k = cof[mask,idx].view(-1,1,1,1)
#             # print('out[mask].shape={}, expert_out.shape={}, cof_k.shape={}'.format(out[mask].shape, expert_out.shape, cof_k.shape))
#             out[mask]+=expert_out*cof_k
#         return out, cof_k


# class CrossMamba(nn.Module):
#     def __init__(self, dim):
#         super(CrossMamba, self).__init__()
#         self.cross_mamba = Mamba(dim,bimamba_type="v3")
#         self.norm1 = LayerNorm(dim,'with_bias')
#         self.norm2 = LayerNorm(dim,'with_bias')
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
#     def forward(self,a,a_resi,b,H,W):
#         a_resi = a+a_resi
#         a = self.norm1(a_resi)
#         b = self.norm2(b)
#         global_f = self.cross_mamba(self.norm1(a),extra_emb=self.norm2(b))
#         B,HW,C = global_f.shape
#         a = global_f.transpose(1, 2).view(B, C,H,W)
#         # ir = global_f.transpose(1, 2).view(B, C, 256, 256)
#         a =  (self.dwconv(a)+a).flatten(2).transpose(1, 2)
#         return a,a_resi


# class CrossMambaFusion(nn.Module):
#     def __init__(self, dim, base_filter=32):
#         super(CrossMambaFusion, self).__init__()
#         self.CM_a = CrossMamba(dim=dim)
#         self.CM_b = CrossMamba(dim=dim)
#         self.a_to_token = PatchEmbed(in_chans=base_filter, embed_dim=base_filter, patch_size=1, stride=1)
#         self.b_to_token = PatchEmbed(in_chans=base_filter, embed_dim=base_filter, patch_size=1, stride=1)
#         self.patchunembe = PatchUnEmbed(base_filter)

#     def forward(self, a, b, H, W):
#         a_res, b_res = 0, 0
#         a, b = self.a_to_token(a), self.b_to_token(b)
#         a, a_res = self.CM_a(a, a_res, b, H, W)
#         b, b_res = self.CM_b(b, b_res, a, H, W)
#         a_fn = self.patchunembe(a,(H,W))
#         b_fn = self.patchunembe(b,(H,W))
#         fuse = attention_fusion_weight(a_fn, b_fn,'avg')
#         return fuse