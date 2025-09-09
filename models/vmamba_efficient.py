# Copyright (c) 2023, Tri Dao, Albert Gu.

import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
import torch.fft as fft
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# SSMODE = "mamba_ssm"
# import selective_scan_cuda

try:
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    if torch.__version__ > '2.0.0':
        from selective_scan_vmamba_pt202 import selective_scan_cuda_core
    else:
        from selective_scan_vmamba import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops


# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
  
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L  
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)



class SelectiveScan(torch.autograd.Function):
    
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        
        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
        
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False  # option to recompute out_z, not used here
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
        
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


class EfficientScan(torch.autograd.Function):
    # [B, C, H, W] -> [B, 4, C, H * W] (original)
    # [B, C, H, W] -> [B, 4, C, H/w * W/w]
    @staticmethod
    def forward(ctx, x: torch.Tensor, step_size=2): # [B, C, H, W] -> [B, 4, H/w * W/w]
        B, C, org_h, org_w = x.shape
        ctx.shape = (B, C, org_h, org_w)
        ctx.step_size = step_size

        if org_w % step_size != 0:
            pad_w = step_size - org_w % step_size
            x = F.pad(x, (0, pad_w, 0, 0))  
        W = x.shape[3]

        if org_h % step_size != 0:
            pad_h = step_size - org_h % step_size
            x = F.pad(x, (0, 0, 0, pad_h))
        H = x.shape[2]

        H = H // step_size
        W = W // step_size

        xs = x.new_empty((B, 4, C, H*W))


        xs[:, 0] = x[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)
        xs[:, 1] = x.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        xs[:, 2] = x[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        xs[:, 3] = x.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)

        xs = xs.view(B, 4, C, -1)
        return xs
    
    @staticmethod
    def backward(ctx, grad_xs: torch.Tensor): # [B, 4, H/w * W/w] -> [B, C, H, W]

        B, C, org_h, org_w = ctx.shape
        step_size = ctx.step_size

        newH, newW = math.ceil(org_h / step_size), math.ceil(org_w / step_size)
        grad_x = grad_xs.new_empty((B, C, newH * step_size, newW * step_size))
        
        grad_xs = grad_xs.view(B, 4, C, newH, newW)
        
        grad_x[:, :, ::step_size, ::step_size] = grad_xs[:, 0].reshape(B, C, newH, newW)
        grad_x[:, :, 1::step_size, ::step_size] = grad_xs[:, 1].reshape(B, C, newW, newH).transpose(dim0=2, dim1=3)
        grad_x[:, :, ::step_size, 1::step_size] = grad_xs[:, 2].reshape(B, C, newH, newW)
        grad_x[:, :, 1::step_size, 1::step_size] = grad_xs[:, 3].reshape(B, C, newW, newH).transpose(dim0=2, dim1=3)

        if org_h != grad_x.shape[-2] or org_w != grad_x.shape[-1]:
            grad_x = grad_x[:, :, :org_h, :org_w]

        return grad_x, None 

class EfficientMerge(torch.autograd.Function): # [B, 4, C, H/w * W/w] -> [B, C, H*W]
    @staticmethod
    def forward(ctx, ys: torch.Tensor, ori_h: int, ori_w: int, step_size=2):
        B, K, C, L = ys.shape
        H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)
        ctx.shape = (H, W)
        ctx.ori_h = ori_h
        ctx.ori_w = ori_w
        ctx.step_size = step_size


        new_h = H * step_size
        new_w = W * step_size

        y = ys.new_empty((B, C, new_h, new_w))


        y[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, H, W)
        y[:, :, 1::step_size, ::step_size] = ys[:, 1].reshape(B, C, W, H).transpose(dim0=2, dim1=3)
        y[:, :, ::step_size, 1::step_size] = ys[:, 2].reshape(B, C, H, W)
        y[:, :, 1::step_size, 1::step_size] = ys[:, 3].reshape(B, C, W, H).transpose(dim0=2, dim1=3)
        
        if ori_h != new_h or ori_w != new_w:
            y = y[:, :, :ori_h, :ori_w].contiguous()

        y = y.view(B, C, -1)
        return y
    
    @staticmethod
    def backward(ctx, grad_x: torch.Tensor): # [B, C, H*W] -> [B, 4, C, H/w * W/w]

        H, W = ctx.shape
        B, C, L = grad_x.shape
        step_size = ctx.step_size

        grad_x = grad_x.view(B, C, ctx.ori_h, ctx.ori_w)

        if ctx.ori_w % step_size != 0:
            pad_w = step_size - ctx.ori_w % step_size
            grad_x = F.pad(grad_x, (0, pad_w, 0, 0))  
        W = grad_x.shape[3]

        if ctx.ori_h % step_size != 0:
            pad_h = step_size - ctx.ori_h % step_size
            grad_x = F.pad(grad_x, (0, 0, 0, pad_h))
        H = grad_x.shape[2]
        B, C, H, W = grad_x.shape
        H = H // step_size
        W = W // step_size
        grad_xs = grad_x.new_empty((B, 4, C, H*W)) 

        grad_xs[:, 0] = grad_x[:, :, ::step_size, ::step_size].reshape(B, C, -1) 
        grad_xs[:, 1] = grad_x.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].reshape(B, C, -1)
        grad_xs[:, 2] = grad_x[:, :, ::step_size, 1::step_size].reshape(B, C, -1)
        grad_xs[:, 3] = grad_x.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].reshape(B, C, -1)
        
        return grad_xs, None, None, None 


def cross_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    nrows = -1,
    delta_softplus = True,
    to_dtype=True,
    step_size = 2,
):
    
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
    # H * W
    ori_h, ori_w = H, W

    xs = EfficientScan.apply(x, step_size) # [B, C, H*W] -> [B, 4, C, H//w * W//w]
    
    # H//w * W//w
    H = math.ceil(H / step_size)
    W = math.ceil(W / step_size)
    
    L = H * W

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)

    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    
    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)
    
    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
    
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, L)

    ori_h, ori_w = int(ori_h), int(ori_w)
    y = EfficientMerge.apply(ys, ori_h, ori_w, step_size) # [B, 4, C, H//w * W//w] -> [B, C, H*W]

    H = ori_h
    W = ori_w
    L = H * W

    y = y.transpose(dim0=1, dim1=2).contiguous()
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)

def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True)
    return flops


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        # ======================
        forward_type="v2",
        # ======================
        step_size=2,
        **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv

        self.step_size = step_size

        # disable z act ======================================
        self.disable_z_act = forward_type[-len("nozact"):] == "nozact"
        if self.disable_z_act:
            forward_type = forward_type[:-len("nozact")]

        # softmax | sigmoid | norm ===========================
        if forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type =======================================
        self.forward_core = dict(
            v0=self.forward_corev0,
            v0_seq=self.forward_corev0_seq,
            v1=self.forward_corev2,
            v2=self.forward_corev2,
            share_ssm=self.forward_corev0_share_ssm,
            share_a=self.forward_corev0_share_a,
        ).get(forward_type, self.forward_corev2)
        self.K = 4 if forward_type not in ["share_ssm"] else 1
        self.K2 = self.K if forward_type not in ["share_a"] else 1

        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner))) 

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # only used to run previous version
    def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float() # (b, k, d_state, l)
        Cs = Cs.float() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()) # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)
    
    def forward_corev0_seq(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float() # (b, k, d, l)
        dts = dts.contiguous().float() # (b, k, d, l)
        Bs = Bs.float() # (b, k, d_state, l)
        Cs = Cs.float() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1) # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1) # (k, d)


        out_y = []
        for i in range(4):
            yi = selective_scan(
                xs[:, i], dts[:, i], 
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)


    def forward_corev0_share_ssm(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev0_share_a(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev2(self, x: torch.Tensor, nrows=-1, channel_first=False, step_size=2):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, step_size=step_size
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x
    
    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x)) # (b, d, h, w)
        else:
            if self.disable_z_act:
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
                x = self.act(x)
            else:
                xz = self.act(xz)
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
        y = self.forward_core(x, channel_first=(self.d_conv > 1), step_size=self.step_size)
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduce_channels, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super(SqueezeExcite, self).__init__()
        self.avgp = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_channels, reduce_channels, 1, bias=True)
        self.act_fn = act_fn()
        self.conv_expand = nn.Conv2d(reduce_channels, in_channels, 1, bias=True)
        self.gate_fn = gate_fn()

    def forward(self, x):
        x_se = self.avgp(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean([1, 2], keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        # x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  

        attn = c_attn 
        out = ori_x * attn
        return out


class CrossVSSBlock_Base(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        # =============================
        ssm_d_state: int = 32,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_simple_init=False,
        forward_type="v2",
        use_checkpoint: bool = False,
        step_size=2,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.encoder = CrossSS2D(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            # =============================
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=True,
            ssm_drop_rate = ssm_drop_rate,
            ssm_simple_init=ssm_simple_init,
            forward_type=forward_type,
            step_size=step_size,
            **kwargs,
        )

    def _forward(self, input_x: torch.Tensor, input_b: torch.Tensor):
        x, b = input_x.permute(0,2,3,1), input_b.permute(0,2,3,1)
        out_x, out_b = self.encoder(x, b)
        fuse = out_x+out_b
        return fuse

    def forward(self, input_x: torch.Tensor, input_b: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input_x, input_b)
        else:
            return self._forward(input_x, input_b)


class CrossSS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=32,
        d_state=32,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        # ======================
        forward_type="v2",
        # ======================
        step_size=2,
        **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv

        self.step_size = step_size

        # disable z act ======================================
        self.disable_z_act = forward_type[-len("nozact"):] == "nozact"
        if self.disable_z_act:
            forward_type = forward_type[:-len("nozact")]

        # softmax | sigmoid | norm ===========================
        if forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type =======================================
        self.forward_core = dict(
            v0=self.forward_corev0,
            v0_seq=self.forward_corev0_seq,
            v1=self.forward_corev2,
            v2=self.forward_corev2,
            share_ssm=self.forward_corev0_share_ssm,
            share_a=self.forward_corev0_share_a,
        ).get(forward_type, self.forward_corev2)
        self.forward_core_b = dict(
            v0=self.forward_corev0,
            v0_seq=self.forward_corev0_seq,
            v1=self.forward_corev2,
            v2=self.forward_corev2,
            share_ssm=self.forward_corev0_share_ssm,
            share_a=self.forward_corev0_share_a,
        ).get(forward_type, self.forward_corev2)
        self.K = 4 if forward_type not in ["share_ssm"] else 1
        self.K2 = self.K if forward_type not in ["share_a"] else 1

        # in proj =======================================
        self.in_proj_x = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.in_proj_b = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d_x = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_b = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.out_proj_b = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_b = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner))) 

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # only used to run previous version
    def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float() # (b, k, d_state, l)
        Cs = Cs.float() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()) # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)
    
    def forward_corev0_seq(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float() # (b, k, d, l)
        dts = dts.contiguous().float() # (b, k, d, l)
        Bs = Bs.float() # (b, k, d_state, l)
        Cs = Cs.float() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1) # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1) # (k, d)


        out_y = []
        for i in range(4):
            yi = selective_scan(
                xs[:, i], dts[:, i], 
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)


    def forward_corev0_share_ssm(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev0_share_a(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev2(self, x: torch.Tensor, nrows=-1, channel_first=False, step_size=2):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, step_size=step_size
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x
    
    def forward(self, x: torch.Tensor, b: torch.Tensor, **kwargs):
        # print('x.shape={}, b.shape={}'.format(x.shape, b.shape))
        xz, bz = self.in_proj_x(x), self.in_proj_b(b)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            b, zb = bz.chunk(2, dim=-1)
            if not self.disable_z_act:
                z, zb = self.act(z), self.act(zb)
            x, b = x.permute(0, 3, 1, 2).contiguous(), b.permute(0, 3, 1, 2).contiguous()
            x, b = self.act(self.conv2d_x(x)), self.act(self.conv2d_b(b)) # (b, d, h, w)
        else:
            if self.disable_z_act:
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
                b, zb = xz.chunk(2, dim=-1)
                x, b = self.act(x), self.act(b)
            else:
                xz, bz = self.act(xz), self.act(bz)
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
                b, zb = bz.chunk(2, dim=-1)
        y = self.forward_core(x, channel_first=(self.d_conv > 1), step_size=self.step_size)
        yb = self.forward_core_b(b, channel_first=(self.d_conv > 1), step_size=self.step_size)
        # print(y.shape, z.shape)
        out = self.dropout(self.out_proj(y*z+yb*z))
        out_b = self.dropout(self.out_proj_b(y*zb+yb*zb))
        return out, out_b


class Fourier_VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_simple_init=False,
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        # =============================
        use_checkpoint: bool = False,
        step_size=2,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim, 
            d_state=ssm_d_state, 
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # bias=False,
            # ==========================
            # dt_min=0.001,
            # dt_max=0.1,
            # dt_init="random",
            # dt_scale="random",
            # dt_init_floor=1e-4,
            simple_init=ssm_simple_init,
            # ==========================
            forward_type=forward_type,
            step_size=step_size,
        )
        self.conv_branch = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*2, 3, stride=1, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim*2),
            nn.GELU(),
            nn.Conv2d(hidden_dim*2, hidden_dim*2, 1)
        )
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.se = BiAttn(hidden_dim)  
        self.drop_path = DropPath(drop_path)
        
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)

    def _forward(self, input: torch.Tensor):
        x = self.norm(input)    #(b,h,w,c)
        x_ssm = self.op(x)
        b, h, w, c = x.shape
        # print(x.shape)
        x_f = fft.rfft2(x.permute(0, 3, 1, 2))
        # print(x_f.shape)
        x_f = torch.cat([x_f.real, x_f.imag], dim=1)
        x_conv = self.conv_branch(x_f)
        x_conv = self.lrelu(x_conv)
        c = x_conv.shape[1]
        x_conv = torch.complex(x_conv[:, : c // 2], x_conv[:, c // 2 :])
        x_conv = fft.irfft2(x_conv)
        x_conv = F.interpolate(x_conv, size=[h,w], mode='bilinear', align_corners=False)
        # print(x_ssm.shape)
        # print(x_conv.shape)
        x = self.se(x_ssm) + self.se(x_conv.permute(0,2,3,1))
        x = input + self.drop_path(x)
        # print(x.shape)
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


# class VSSBlock(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int = 32,
#         drop_path: float = 0,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#         # =============================
#         ssm_d_state: int = 16,
#         ssm_ratio=2.0,
#         ssm_rank_ratio=2.0,
#         ssm_dt_rank: Any = "auto",
#         ssm_act_layer=nn.SiLU,
#         ssm_conv: int = 3,
#         ssm_conv_bias=True,
#         ssm_drop_rate: float = 0,
#         ssm_simple_init=False,
#         forward_type="v2",
#         # =============================
#         mlp_ratio=4.0,
#         mlp_act_layer=nn.GELU,
#         mlp_drop_rate: float = 0.0,
#         # =============================
#         use_checkpoint: bool = False,
#         step_size=2,
#         **kwargs,
#     ):
#         super().__init__()
#         self.use_checkpoint = use_checkpoint
#         self.norm = norm_layer(hidden_dim)
#         self.op = SS2D(
#             d_model=hidden_dim, 
#             d_state=ssm_d_state, 
#             ssm_ratio=ssm_ratio,
#             ssm_rank_ratio=ssm_rank_ratio,
#             dt_rank=ssm_dt_rank,
#             act_layer=ssm_act_layer,
#             # ==========================
#             d_conv=ssm_conv,
#             conv_bias=ssm_conv_bias,
#             # ==========================
#             dropout=ssm_drop_rate,
#             # bias=False,
#             # ==========================
#             # dt_min=0.001,
#             # dt_max=0.1,
#             # dt_init="random",
#             # dt_scale="random",
#             # dt_init_floor=1e-4,
#             simple_init=ssm_simple_init,
#             # ==========================
#             forward_type=forward_type,
#             step_size=step_size,
#         )
#         self.sss2d = SpatialSS2D(
#             d_model=hidden_dim, 
#             d_state=ssm_d_state, 
#             ssm_ratio=ssm_ratio,
#             ssm_rank_ratio=ssm_rank_ratio,
#             dt_rank=ssm_dt_rank,
#             act_layer=ssm_act_layer,
#             # ==========================
#             d_conv=ssm_conv,
#             conv_bias=ssm_conv_bias,
#             # ==========================
#             dropout=ssm_drop_rate,
#             # bias=False,
#             # ==========================
#             # dt_min=0.001,
#             # dt_max=0.1,
#             # dt_init="random",
#             # dt_scale="random",
#             # dt_init_floor=1e-4,
#             simple_init=ssm_simple_init,
#             # ==========================
#             forward_type=forward_type,
#             step_size=step_size,
#         )
#         self.css2d = ChannelSS2D(
#             d_model=hidden_dim, 
#             d_state=ssm_d_state, 
#             ssm_ratio=ssm_ratio,
#             ssm_rank_ratio=ssm_rank_ratio,
#             dt_rank=ssm_dt_rank,
#             act_layer=ssm_act_layer,
#             # ==========================
#             d_conv=ssm_conv,
#             conv_bias=ssm_conv_bias,
#             # ==========================
#             dropout=ssm_drop_rate,
#             # bias=False,
#             # ==========================
#             # dt_min=0.001,
#             # dt_max=0.1,
#             # dt_init="random",
#             # dt_scale="random",
#             # dt_init_floor=1e-4,
#             simple_init=ssm_simple_init,
#             # ==========================
#             forward_type=forward_type,
#             step_size=step_size,
#         )
#         self.conv_branch = nn.Sequential(
#             nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, groups=hidden_dim),
#             nn.BatchNorm2d(hidden_dim),
#             nn.GELU(),
#             nn.Conv2d(hidden_dim, hidden_dim, 1)
#         )
#         self.drop_path = DropPath(drop_path)
        
#         self.mlp_branch = mlp_ratio > 0
#         if self.mlp_branch:
#             self.norm2 = norm_layer(hidden_dim)
#             mlp_hidden_dim = int(hidden_dim * mlp_ratio)
#             self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)

#     def _forward(self, input: torch.Tensor):
#         x = self.norm(input)
#         x_ssm = self.op(x)
#         x_conv = self.conv_branch(x.permute(0,3,1,2)).permute(0,2,3,1)
#         x = self.css2d(self.sss2d(x_ssm+x_conv))
#         x = input + self.drop_path(x)
#         if self.mlp_branch:
#             x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
#         return x

#     def forward(self, input: torch.Tensor):
#         if self.use_checkpoint:
#             return checkpoint.checkpoint(self._forward, input)
#         else:
#             return self._forward(input)


# class SpatialSS2D(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int = 32,
#         drop_path: float = 0,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#         # =============================
#         ssm_d_state: int = 16,
#         ssm_ratio=2.0,
#         ssm_rank_ratio=2.0,
#         ssm_dt_rank: Any = "auto",
#         ssm_act_layer=nn.SiLU,
#         ssm_conv: int = 3,
#         ssm_conv_bias=True,
#         ssm_drop_rate: float = 0,
#         ssm_simple_init=False,
#         forward_type="v2",
#         step_size=2,
#         **kwargs,
#     ):
#         super(SpatialSS2D, self).__init__()
#         self.norm = norm_layer(hidden_dim)
#         self.preconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
#                         nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim))
#         self.op = SS2D(
#             d_model=hidden_dim, 
#             d_state=ssm_d_state, 
#             ssm_ratio=ssm_ratio,
#             ssm_rank_ratio=ssm_rank_ratio,
#             dt_rank=ssm_dt_rank,
#             act_layer=ssm_act_layer,
#             # ==========================
#             d_conv=ssm_conv,
#             conv_bias=ssm_conv_bias,
#             # ==========================
#             dropout=ssm_drop_rate,
#             # bias=False,
#             # ==========================
#             # dt_min=0.001,
#             # dt_max=0.1,
#             # dt_init="random",
#             # dt_scale="random",
#             # dt_init_floor=1e-4,
#             simple_init=ssm_simple_init,
#             # ==========================
#             forward_type=forward_type,
#             step_size=step_size,
#         )
#         self.drop_path = DropPath(drop_path)

#     def forward(self, a: torch.Tensor):
#         ax = self.preconv(self.norm(a).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # (b, h, w, c)
#         a_ssm = self.op(ax)
#         out_a = a + self.drop_path(a_ssm)
#         return out_a


# class ChannelSS2D(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int = 32,
#         drop_path: float = 0,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#         # =============================
#         ssm_d_state: int = 16,
#         ssm_ratio=2.0,
#         ssm_rank_ratio=2.0,
#         ssm_dt_rank: Any = "auto",
#         ssm_act_layer=nn.SiLU,
#         ssm_conv: int = 3,
#         ssm_conv_bias=True,
#         ssm_drop_rate: float = 0,
#         ssm_simple_init=False,
#         forward_type="v2",
#         step_size=2,
#         **kwargs,
#     ):
#         super(ChannelSS2D, self).__init__()
#         self.norm = norm_layer(hidden_dim)
#         self.preconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
#                         nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim))
#         self.op = SS2D(
#             d_model=hidden_dim*2, 
#             d_state=ssm_d_state, 
#             ssm_ratio=ssm_ratio,
#             ssm_rank_ratio=ssm_rank_ratio,
#             dt_rank=ssm_dt_rank,
#             act_layer=ssm_act_layer,
#             # ==========================
#             d_conv=ssm_conv,
#             conv_bias=ssm_conv_bias,
#             # ==========================
#             dropout=ssm_drop_rate,
#             # bias=False,
#             # ==========================
#             # dt_min=0.001,
#             # dt_max=0.1,
#             # dt_init="random",
#             # dt_scale="random",
#             # dt_init_floor=1e-4,
#             simple_init=ssm_simple_init,
#             # ==========================
#             forward_type=forward_type,
#             step_size=step_size,
#         )
#         self.drop_path = DropPath(drop_path)
#         self.reconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
#                                     nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim))
#         self.lrelu = nn.LeakyReLU(inplace=True)
#     def forward(self, a: torch.Tensor):
#         ax = self.preconv(self.norm(a).permute(0, 3, 1, 2))# (b, c, h, w)
#         b, c, h, w = ax.shape
#         ax_fft = fft.rfft2(ax)
#         ax_fft = torch.cat([ax_fft.real, ax_fft.imag], dim=1).permute(0,2,3,1)
#         a_ssm = self.op(ax_fft)
#         a_ssm = self.lrelu(a_ssm.permute(0,3,1,2))
#         c = a_ssm.shape[1]
#         a_ifft = torch.complex(a_ssm[:, : c//2], a_ssm[:, c//2 :])
#         a_ifft = fft.irfft2(a_ifft)
#         a_ifft = F.interpolate(a_ifft, size=[h,w], mode='bilinear', align_corners=False)
#         out_a = a + self.drop_path(a_ifft.permute(0,2,3,1))
#         res_a = out_a + self.reconv(out_a.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
#         return res_a


# class CrossVSSBlock(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int = 32,
#         drop_path: float = 0,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#         # =============================
#         ssm_d_state: int = 16,
#         ssm_ratio=1.0,
#         ssm_rank_ratio=2.0,
#         ssm_dt_rank: Any = "auto",
#         ssm_act_layer=nn.SiLU,
#         ssm_conv: int = 3,
#         ssm_conv_bias=True,
#         ssm_drop_rate: float = 0,
#         ssm_simple_init=False,
#         forward_type="v2",
#         use_checkpoint: bool = False,
#         step_size=2,
#         **kwargs,
#     ):
#         super().__init__()
#         self.use_checkpoint = use_checkpoint
#         self.sss2d = SpatialCrossSS2D(
#             hidden_dim=hidden_dim,
#             drop_path=drop_path,
#             norm_layer=norm_layer,
#             # =============================
#             ssm_d_state=ssm_d_state,
#             ssm_ratio=ssm_ratio,
#             ssm_rank_ratio=ssm_rank_ratio,
#             ssm_dt_rank=ssm_dt_rank,
#             ssm_act_layer=ssm_act_layer,
#             ssm_conv=ssm_conv,
#             ssm_conv_bias=True,
#             ssm_drop_rate = ssm_drop_rate,
#             ssm_simple_init=ssm_simple_init,
#             forward_type=forward_type,
#             step_size=step_size,
#             **kwargs,
#         )
#         self.css2d = ChannelCrossSS2D(
#             hidden_dim=hidden_dim,
#             drop_path=drop_path,
#             norm_layer=norm_layer,
#             # =============================
#             ssm_d_state=ssm_d_state,
#             ssm_ratio=ssm_ratio,
#             ssm_rank_ratio=ssm_rank_ratio,
#             ssm_dt_rank=ssm_dt_rank,
#             ssm_act_layer=ssm_act_layer,
#             ssm_conv=ssm_conv,
#             ssm_conv_bias=True,
#             ssm_drop_rate = ssm_drop_rate,
#             ssm_simple_init=ssm_simple_init,
#             forward_type=forward_type,
#             step_size=step_size,
#             tensor_shape=tensor_shape,
#             **kwargs,
#         )

#     def _forward(self, input_x: torch.Tensor, input_b: torch.Tensor):
#         x, b = input_x.permute(0,2,3,1), input_b.permute(0,2,3,1)
#         out_x, out_b = self.sss2d(x, b)
#         res_a, res_b = self.css2d(out_x, out_b)
#         fuse = res_a+res_b
#         return fuse

#     def forward(self, input_x: torch.Tensor, input_b: torch.Tensor):
#         if self.use_checkpoint:
#             return checkpoint.checkpoint(self._forward, input_x, input_b)
#         else:
#             return self._forward(input_x, input_b)


# class SpatialCrossSS2D(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int = 32,
#         drop_path: float = 0,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#         # =============================
#         ssm_d_state: int = 16,
#         ssm_ratio=2.0,
#         ssm_rank_ratio=2.0,
#         ssm_dt_rank: Any = "auto",
#         ssm_act_layer=nn.SiLU,
#         ssm_conv: int = 3,
#         ssm_conv_bias=True,
#         ssm_drop_rate: float = 0,
#         ssm_simple_init=False,
#         forward_type="v2",
#         step_size=2,
#         **kwargs,
#     ):
#         super(SpatialCrossSS2D, self).__init__()
#         self.norm = norm_layer(hidden_dim)
#         self.preconv_a = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
#                         nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim))
#         self.preconv_b = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
#                         nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim))
#         self.op = CrossSS2D(
#             d_model=hidden_dim, 
#             d_state=ssm_d_state, 
#             ssm_ratio=ssm_ratio,
#             ssm_rank_ratio=ssm_rank_ratio,
#             dt_rank=ssm_dt_rank,
#             act_layer=ssm_act_layer,
#             # ==========================
#             d_conv=ssm_conv,
#             conv_bias=ssm_conv_bias,
#             # ==========================
#             dropout=ssm_drop_rate,
#             # bias=False,
#             # ==========================
#             # dt_min=0.001,
#             # dt_max=0.1,
#             # dt_init="random",
#             # dt_scale="random",
#             # dt_init_floor=1e-4,
#             simple_init=ssm_simple_init,
#             # ==========================
#             forward_type=forward_type,
#             step_size=step_size,
#         )
#         self.drop_path_a = DropPath(drop_path)
#         self.drop_path_b = DropPath(drop_path)

#     def forward(self, a: torch.Tensor, b: torch.Tensor):
#         ax = self.preconv_a(self.norm(a).permute(0,3,1,2)).permute(0,2,3,1)
#         bx = self.preconv_b(self.norm(b).permute(0,3,1,2)).permute(0,2,3,1) #(b, h, w, c)
#         a_ssm, b_ssm = self.op(ax, bx)  #(b, h, w, c)
#         # print(a.shape, a_ssm.shape)
#         out_a, out_b = a+self.drop_path_a(a_ssm), b+self.drop_path_b(b_ssm)
#         return out_a, out_b


# class ChannelCrossSS2D(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int = 32,
#         drop_path: float = 0,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#         # =============================
#         ssm_d_state: int = 16,
#         ssm_ratio=2.0,
#         ssm_rank_ratio=2.0,
#         ssm_dt_rank: Any = "auto",
#         ssm_act_layer=nn.SiLU,
#         ssm_conv: int = 3,
#         ssm_conv_bias=True,
#         ssm_drop_rate: float = 0,
#         ssm_simple_init=False,
#         forward_type="v2",
#         step_size=2,
#         **kwargs,
#     ):
#         super(ChannelCrossSS2D, self).__init__()
#         self.norm = norm_layer(hidden_dim)
#         self.preconv_a = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
#                         nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim))
#         self.preconv_b = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
#                         nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim))
#         # print('self.shape={}'.format(self.shape))
#         self.op = CrossSS2D(
#             d_model=hidden_dim*2,
#             d_state=ssm_d_state, 
#             ssm_ratio=ssm_ratio,
#             ssm_rank_ratio=ssm_rank_ratio,
#             dt_rank=ssm_dt_rank,
#             act_layer=ssm_act_layer,
#             # ==========================
#             d_conv=ssm_conv,
#             conv_bias=ssm_conv_bias,
#             # ==========================
#             dropout=ssm_drop_rate,
#             # bias=False,
#             # ==========================
#             # dt_min=0.001,
#             # dt_max=0.1,
#             # dt_init="random",
#             # dt_scale="random",
#             # dt_init_floor=1e-4,
#             simple_init=ssm_simple_init,
#             # ==========================
#             forward_type=forward_type,
#             step_size=step_size,
#         )
#         self.drop_path_a = DropPath(drop_path)
#         self.drop_path_b = DropPath(drop_path)
#         self.reconv_a = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
#                                         nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim))
#         self.reconv_b = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
#                                         nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim))
#         self.lrelu = nn.LeakyReLU(inplace=True)

#     def forward(self, a: torch.Tensor, b: torch.Tensor):
#         ax = self.preconv_a(self.norm(a).permute(0,3,1,2))
#         bx = self.preconv_b(self.norm(b).permute(0,3,1,2)) #(b, c, h, w)
#         b, c, h, w = ax.shape
#         ax_fft, bx_fft = fft.rfft2(ax), fft.rfft2(bx)
#         ax_fft = torch.cat([ax_fft.real, ax_fft.imag], dim=1).permute(0,2,3,1)
#         bx_fft = torch.cat([bx_fft.real, bx_fft.imag], dim=1).permute(0,2,3,1)
#         a_ssm, b_ssm = self.op(ax_fft, bx_fft)
#         a_ssm, b_ssm = self.lrelu(a_ssm.permute(0,3,1,2)), self.lrelu(b_ssm.permute(0,3,1,2))#(B,C,H,W)
#         c = a_ssm.shape[1]
#         a_ifft = torch.complex(a_ssm[:, : c//2], a_ssm[:, c//2 :])#(B,C,H,W)
#         b_ifft = torch.complex(b_ssm[:, : c//2], b_ssm[:, c//2 :])
#         a_ifft, b_ifft = fft.irfft2(a_ifft), fft.irfft2(b_ifft)
#         a_ifft = F.interpolate(a_ifft, size=[h,w], mode='bilinear', align_corners=False)
#         b_ifft = F.interpolate(b_ifft, size=[h,w], mode='bilinear', align_corners=False)
#         out_a, out_b = a+self.drop_path_a(a_ifft.permute(0,2,3,1)), b+self.drop_path_b(b_ifft.permute(0,2,3,1))
#         res_a = out_a+self.reconv_a(out_a.permute(0,3,1,2)).permute(0,2,3,1)
#         res_b = out_b+self.reconv_b(out_b.permute(0,3,1,2)).permute(0,2,3,1)
#         return res_a, res_b