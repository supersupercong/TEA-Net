import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F


class FFTLoss(nn.Module):
    def __init__(self,loss_weight=0.05,reduction='mean'):
        super(FFTLoss, self).__init__()
        self.cri_l1 = nn.L1Loss(reduction=reduction)
        self.loss_weight = loss_weight
    def forward(self,target,pred):
        gt_fft = torch.fft.rfft2(target, dim=(-2, -1))
        gt_amp = torch.abs(gt_fft)
        gt_phase = torch.angle(gt_fft)
        label_fft = torch.stack((gt_fft.real, gt_fft.imag), -1)
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        pred_amp = torch.abs(pred_fft)
        pred_phase = torch.angle(pred_fft)
        fuse_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        l_fft = self.cri_l1(label_fft, fuse_fft)
        l_phase = self.cri_l1(gt_phase, pred_phase)
        # l_inv = self.cri_l1(target,pred)
        return self.loss_weight*(l_fft+l_phase)


class FFTV10Loss(nn.Module):
    def __init__(self,loss_weight=0.05,reduction='mean'):
        super(FFTV10Loss, self).__init__()
        self.cri_l1 = nn.L1Loss(reduction=reduction)
        self.loss_weight = loss_weight
    def forward(self,target,pred):
        gt_fft = torch.fft.rfft2(target, dim=(-2, -1))
        gt_amp = torch.abs(gt_fft)
        gt_phase = torch.angle(gt_fft)
        label_fft = torch.stack((gt_fft.real, gt_fft.imag), -1)
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        pred_amp = torch.abs(pred_fft)
        pred_phase = torch.angle(pred_fft)
        fuse_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        l_fft = self.cri_l1(fuse_fft, label_fft)
        l_phase = self.cri_l1(pred_phase, gt_phase)
        l_amp = self.cri_l1(pred_amp,gt_amp)
        # l_inv = self.cri_l1(y1,target)
        return self.loss_weight*(l_fft+l_phase+l_amp)


class FFTV10Y1Loss(nn.Module):
    def __init__(self,loss_weight=0.05,reduction='mean'):
        super(FFTV10Y1Loss, self).__init__()
        self.cri_l1 = nn.L1Loss(reduction=reduction)
        self.loss_weight = loss_weight
    def forward(self,target,pred):
        gt_fft = torch.fft.rfft2(target, dim=(-2, -1))
        gt_amp = torch.abs(gt_fft)
        gt_phase = torch.angle(gt_fft)
        label_fft = torch.stack((gt_fft.real, gt_fft.imag), -1)
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        pred_amp = torch.abs(pred_fft)
        pred_phase = torch.angle(pred_fft)
        fuse_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        l_fft = self.cri_l1(fuse_fft, label_fft)
        l_phase = self.cri_l1(pred_phase, gt_phase)
        l_amp = self.cri_l1(pred_amp,gt_amp)
        l_total = 0.5*self.cri_l1(pred,target)
        # l_inv = self.cri_l1(y1,target)
        return self.loss_weight*(l_fft+l_phase+l_amp+l_total)