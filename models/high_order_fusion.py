import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms
from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm import Mamba
# from .vmamba import VSSBlock
from .moe import HinResBlock, FuseNet
from .vmamba_efficient import Fourier_VSSBlock as EVSS
import copy
import torch.fft as fft
from .functions import LayerNorm, to_2d, to_3d


class Net(nn.Module):
    def __init__(self, num_channels=1, base_filter=32, num_experts=4, topk=2):
        super(Net, self).__init__()
        self.base_filter = base_filter
        self.stride=1
        self.patch_size=1
        self.embed_dim = base_filter*self.stride*self.patch_size
        self.shared_encoder = nn.Sequential(nn.Conv2d(num_channels,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.vis_decoder = InvSplit(self.embed_dim*2, self.embed_dim)
        self.ir_decoder = InvSplit(self.embed_dim*2, self.embed_dim)
        self.base_feature_extraction = nn.Sequential(*[EVSS_Block(self.embed_dim) for i in range(2)])
        self.detail_feature_extraction = nn.Sequential(*[EVSS_Block(self.embed_dim) for i in range(2)])
        self.cross_fusion = FuseNet(channels=base_filter, num_experts=num_experts, k=topk)
        self.fuse_decoder = InvSplit(self.embed_dim*2, self.embed_dim)
        self.decoder = nn.Conv2d(base_filter,num_channels,kernel_size=1)
        

    def forward(self,ir,vis,scale=1,decompose=True,fuse=True):
        ir_bic = F.interpolate(input=ir, scale_factor=scale)  
        ir_f = self.shared_encoder(ir_bic)
        # b,c,h,w = ir_f.shape
        # print('ir_f.shape={}'.format(ir_f.shape))
        vis_f = self.shared_encoder(vis)
        vis_b, vis_d = self.vis_decoder(vis_f)
        ir_b, ir_d = self.ir_decoder(ir_f)
        if not fuse:
            return ir_b, ir_d, vis_b, vis_d
        else:
            # print('ir_b.shape={}'.format(ir_b.shape), 'vis_b.shape={}'.format(vis_b.shape))
            base_f = self.base_feature_extraction((ir_b+vis_b).permute(0,2,3,1)).permute(0,3,1,2)
            detail_f = self.detail_feature_extraction((ir_d+vis_d).permute(0,2,3,1)).permute(0,3,1,2)
            fuse_f, cof_k = self.cross_fusion(base_f, detail_f)
            fuse_img = self.decoder(fuse_f)
            if decompose:
                # print(fuse_f.shape)
                fuse_b, fuse_d = self.fuse_decoder(fuse_f)
                return fuse_img, fuse_b, fuse_d, base_f, detail_f, ir_b, ir_d, vis_b, vis_d, cof_k
            else:
                return fuse_img


# 解耦
class InvSplit(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=0.8):
        super(InvSplit,self).__init__()
        self.encoder = nn.Conv2d(channel_split_num,channel_num,1,1,0)
        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2
        self.clamp = clamp
        self.predict = UNetConvBlock(self.split_len1, self.split_len2)
        self.update = UNetConvBlock(self.split_len1, self.split_len2)
        # self.low_out = nn.Conv2d(self.split_len1,3,1,1,0)
        # self.high_out = nn.Conv2d(self.split_len1,3,1,1,0)
    def forward(self,x):
        x = self.encoder(x)
        # print(x.shape)
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)) #x_1 low,x_2 high
        # print(x1.shape, x2.shape)
        x2 = x2-self.predict(x1)
        x1 = x1+self.update(x2)
        # x_low = torch.sigmoid(self.low_out(x1))
        # x_high = torch.sigmoid(self.high_out(x2))
        x_low = torch.sigmoid(x1)
        x_high = torch.sigmoid(x2)
        return x_low, x_high
        # return (x_low,x_low),(x_high,x_high)
        # return x_low, x_high


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


# 空频
class EVSS_Block(nn.Module):
    def __init__(self, MlpRatio=0, dim=32, drop_path_rate=0.1):
        super(EVSS_Block, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]
        self.encoder = EVSS(
        hidden_dim=dim, 
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
        mlp_ratio=MlpRatio,  # 选择在Vanilla VSS模块后添加FFN层，默认值为4.0，如果不加就设为0
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        use_checkpoint=False,)

    def forward(self, x):
        return self.encoder(x)


# class Net(nn.Module):
#     def __init__(self, num_channels=1, base_filter=32, num_experts=4, topk=2):
#         super(Net, self).__init__()
#         self.base_filter = base_filter
#         self.stride=1
#         self.patch_size=1
#         self.embed_dim = base_filter*self.stride*self.patch_size
#         # self.vis_encoder = nn.Sequential(nn.Conv2d(num_channels,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
#         # self.ir_encoder = nn.Sequential(nn.Conv2d(num_channels,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
#         self.shared_encoder = nn.Sequential(nn.Conv2d(num_channels,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
#         self.vis_feature_extraction = nn.Sequential(*[EVSS_Block(self.embed_dim) for i in range(2)])
#         self.ir_feature_extraction = nn.Sequential(*[EVSS_Block(self.embed_dim) for i in range(2)])
#         self.cross_fusion = FuseNet(channels=base_filter, num_experts=num_experts, k=topk)
#         self.decoder = nn.Conv2d(base_filter,num_channels,kernel_size=1)
        
#         self.vis_decoder = InvSplit(self.embed_dim*2, self.embed_dim)
#         self.ir_decoder = InvSplit(self.embed_dim*2, self.embed_dim)
#         self.fuse_decoder = InvSplit(self.embed_dim*2, self.embed_dim)

#     def forward(self,ir,vis,scale=1,decompose=True):
#         # if self.mode in ['VIF', 'MED']:
#         #     ir = torch.cat([ir] * 3, dim=1)
#         #     vis = torch.cat([vis] * 3, dim=1)
#         ir_bic = F.interpolate(input=ir, scale_factor=scale)  
#         # print("ir_bic.shape={}".format(ir_bic.shape))
#         # ir_f = self.ir_encoder(ir_bic)
#         ir_f = self.shared_encoder(ir_bic)
#         # print('ir_f.shape={}'.format(ir_f.shape))
#         b,c,h,w = ir_f.shape
#         # vis_f = self.vis_encoder(vis)
#         vis_f = self.shared_encoder(vis_bic)
#         ir_f = self.ir_feature_extraction(ir_f.permute(0,2,3,1))
#         vis_f = self.vis_feature_extraction(vis_f.permute(0,2,3,1))
#         # print("ir_f.shape={}".format(ir_f.shape))
#         fuse_f, cof_k = self.cross_fusion(ir_f, vis_f)
#         # fuse_f = self.feature_extraction(fuse_f.permute(0,2,3,1))
#         fuse_img = self.decoder(fuse_f)
#         # if self.mode in ['VIF', 'MED']:
#         #     fuse_f = torch.mean(fuse_f, dim=1)
#         if decompose:
#             # print(fuse_f.shape)
#             fuse_l, fuse_h = self.fuse_decoder(fuse_f)
#             vis_l, vis_h = self.vis_decoder(vis_f.permute(0,3,1,2))
#             ir_l, ir_h = self.ir_decoder(ir_f.permute(0,3,1,2))
#             return fuse_img, fuse_l, fuse_h, ir_l, ir_h, vis_l, vis_h
#         else:
#             return fuse_img


# class CU_EVSS(nn.Module):
#     def __init__(self, MlpRatio=0, dim=32, drop_path_rate=0.1):
#         super(CU_EVSS, self).__init__()
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]
#         self.encoder = EVSS(
#         hidden_dim=dim, 
#         drop_path=dpr,
#         norm_layer=nn.LayerNorm,
#         channel_first=False,
#         ssm_d_state=16,
#         ssm_ratio=2.0,
#         ssm_dt_rank="auto",
#         ssm_act_layer=nn.SiLU,
#         ssm_conv=3,
#         ssm_conv_bias=True,
#         ssm_drop_rate=0.0,
#         ssm_init="v0",
#         forward_type="v2",
#         mlp_ratio=MlpRatio,  # 选择在Vanilla VSS模块后添加FFN层，默认值为4.0，如果不加就设为0
#         mlp_act_layer=nn.GELU,
#         mlp_drop_rate=0.0,
#         gmlp=False,
#         use_checkpoint=False,)

#     def forward(self, x):
#         return self.encoder(x)


# class SpatialSSM(nn.Module):
#     def __init__(self, dim):
#         super(SpatialSSM, self).__init__()
#         self.norm = LayerNorm(dim, 'with_bias')
#         self.conv = nn.Conv2d(dim, dim, kernel_size=1)
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, groups=dim)
#         self.SelectiveSSM = Mamba(dim,bimamba_type=None)

#     def forward(self, x):
#         residual = x
#         x = self.dwconv(self.conv(self.norm(x)))
#         x = self.SelectiveSSM(to_3d(x)) + residual
#         return x


# class ChannelSSM(nn.Module):
#     def __init__(self, dim):
#         super(ChannelSSM, self).__init__()
#         self.norm = LayerNorm(dim, 'with_bias')
#         self.conv = nn.Conv2d(dim, dim, kernel_size=1)
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, groups=dim)
#         self.SelectiveSSM = Mamba(dim,bimamba_type=None)
#         self.dwconv_x2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, groups=dim),
#                                         nn.Conv2d(dim, dim, kernel_size=3, stride=1, groups=dim))

#     def forward(self, x):
#         residual = x
#         x = self.dwconv(self.conv(self.norm(x)))
#         x = self.SelectiveSSM(to_3d(x).transpose(1, 2)).transpose(1, 2) + residual
#         residual = x
#         x = self.dwconv_x2(x) + residual
#         return x


# class CUMambaBlock(nn.Module):
#     def __init__(self, dim):
#         super(CUMambaBlock, self).__init__()
#         self.cssm = ChannelSSM(dim)
#         self.sssm = SpatialSSM(dim)

#     def forward(self, x):
#         return(self.cssm(self.sssm(x)))


# class NonLocalAttention(nn.Module):
#     def __init__(self, in_channels) -> None:
#         super().__init__()
#         # self.conv = nn.Conv2d(in_channels, 1, 1)
#         # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]
#         # self.softmax = nn.Softmax(dim=1)
#         self.vmamba = VanillaVSSBlock(dim=in_channels)

#     def forward(self, feat) -> torch.Tensor:
#         b, c, h, w = feat.shape
#         # print(feat.shape)
#         # out_feat = self.conv(feat)  #(1,1,h,w)
#         # out_feat = rearrange(out_feat, 'b c h w -> b (h w) c')
#         # out_feat = torch.unsqueeze(out_feat, -1)
#         # out_feat = self.softmax(out_feat)
#         # out_feat = torch.squeeze(out_feat, -1)
#         out_feat = rearrange(feat, 'b c h w -> b h w c')
#         # out_feat = rearrange(out_feat, 'b c h w -> b h w c')    #(1,h,w,1)
#         out_feat = self.vmamba(out_feat)    #(1,h,w,1)
#         out_feat = rearrange(out_feat, 'b h w c -> b c h w')
#         # identity = rearrange(feat, 'b c h w -> b c (h w)')
#         # out_feat = torch.matmul(identity, out_feat)
#         # print(out_feat.shape)
#         # out_feat = torch.unsqueeze(out_feat, -1)
#         return out_feat #(b,c,1,1)


# class NonLocalAttentionBlock(nn.Module):
#     """simplified Non-Local attention network"""

#     def __init__(self, in_channels) -> None:
#         super().__init__()
#         self.nonlocal_attention = NonLocalAttention(in_channels)
#         self.global_transform = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, 1),
#             nn.LeakyReLU(inplace=True),
#         )

#     def forward(self, feat):
#         out_feat = self.nonlocal_attention(feat)    #(b,c,1,1)
#         # print('0={}'.format(feat.shape))
#         # print('1={}'.format(out_feat.shape))
#         out_feat = self.global_transform(out_feat)
#         # print('2={}'.format(out_feat.shape))
#         # print('3={}'.format((feat + out_feat).shape))
#         return feat + out_feat


# class SpectralTransformer(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         # self.conv = nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1)
#         # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]
#         self.vmamba = VanillaVSSBlock(dim=in_channels)
#         self.lrelu = nn.LeakyReLU(inplace=True)

#     def forward(self, feat: torch.Tensor) -> torch.Tensor:
#         b, c, h, w = feat.shape
#         out_feat = fft.rfft2(feat)
#         out_feat = torch.cat([out_feat.real, out_feat.imag], dim=1)
#         # print(out_feat.shape)
#         out_feat = rearrange(out_feat, 'b c h w -> b h w c')
#         # out_feat = self.conv(out_feat)
#         out_feat = self.vmamba(out_feat)
#         out_feat = rearrange(out_feat, 'b h w c -> b c h w')
#         out_feat = self.lrelu(out_feat)
#         c = out_feat.shape[1]
#         out_feat = torch.complex(out_feat[:, : c // 2], out_feat[:, c // 2 :])
#         # print(out_feat.shape)
#         out_feat = fft.irfft2(out_feat)
#         out_feat = F.interpolate(out_feat, size=[h,w], mode='bilinear', align_corners=False)
#         # print(out_feat.shape)
#         return out_feat


# class FourierConvolutionBlock(nn.Module):
#     def __init__(self, in_channels=32):
#         super().__init__()
#         self.half_channels = in_channels // 2
#         self.func_g_to_g = SpectralTransformer(in_channels)
#         self.func_g_to_l = nn.Sequential(
#             nn.Conv2d(self.half_channels, self.half_channels, kernel_size=3, padding=1),
#             nn.LeakyReLU(inplace=True),
#         )
#         self.func_l_to_g = nn.Sequential(
#             nn.Conv2d(self.half_channels, self.half_channels, kernel_size=1),
#             NonLocalAttentionBlock(self.half_channels),
#         )
#         self.func_l_to_l = nn.Sequential(
#             nn.Conv2d(self.half_channels, self.half_channels, kernel_size=3, padding=1),
#             nn.LeakyReLU(inplace=True),
#         )

#     def forward(self, feat: torch.Tensor) -> torch.Tensor:
#         # print('feat.shape={}'.format(feat.shape))
#         global_feat = feat[:, self.half_channels :]
#         local_feat = feat[:, : self.half_channels]
#         # print(global_feat.shape,local_feat.shape)
#         out_global_feat = self.func_l_to_g(local_feat) + self.func_g_to_g(global_feat)
#         # print(out_global_feat.shape)
#         out_local_feat = self.func_g_to_l(global_feat) + self.func_l_to_l(local_feat)
#         return torch.cat([out_global_feat, out_local_feat], 1)


# class VanillaVSSBlock(nn.Module):
#     def __init__(self, MlpRatio=0, dim=32, drop_path_rate=0.1):
#         super(VanillaVSSBlock, self).__init__()
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]
#         self.encoder = VSSBlock(
#         hidden_dim=dim, 
#         drop_path=dpr,
#         norm_layer=nn.LayerNorm,
#         channel_first=False,
#         ssm_d_state=16,
#         ssm_ratio=2.0,
#         ssm_dt_rank="auto",
#         ssm_act_layer=nn.SiLU,
#         ssm_conv=3,
#         ssm_conv_bias=True,
#         ssm_drop_rate=0.0,
#         ssm_init="v0",
#         forward_type="v2",
#         mlp_ratio=MlpRatio,  # 选择在Vanilla VSS模块后添加FFN层，默认值为4.0，如果不加就设为0
#         mlp_act_layer=nn.GELU,
#         mlp_drop_rate=0.0,
#         gmlp=False,
#         use_checkpoint=False,)

#     def forward(self, x):
#         return self.encoder(x)


# class SingleMambaBlock(nn.Module):
#     def __init__(self, dim):
#         super(SingleMambaBlock, self).__init__()
#         self.encoder = Mamba(dim,bimamba_type=None)
#         self.norm = LayerNorm(dim,'with_bias')
#         # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
#     def forward(self,ipt):
#         x,residual = ipt
#         residual = x+residual
#         x = self.norm(residual)
#         return (self.encoder(x),residual)


# # 前后向扫描
# class BidirectionalMamba(nn.Module):
#     def __init__(self, dim):
#         super(BidirectionalMamba, self).__init__()
#         self.bimamba = Mamba(dim,bimamba_type="v1")
#         self.norm = LayerNorm(dim,'with_bias')
#     def forward(self,ipt):
#         a, a_res = ipt
#         a_res = a+a_res
#         a_norm = self.norm(a_res)
#         return (self.bimamba(a_norm), a_res)