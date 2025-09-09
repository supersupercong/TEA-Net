
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.transforms.functional as TF
from models.loss_ssim import ssim

class L_color(nn.Module):
        return k

class L_Grad(nn.Module):
        return Loss_gradient

class Sobelxy(nn.Module):
        return torch.abs(sobelx)+torch.abs(sobely)
class L_SSIM(nn.Module):
        return Loss_SSIM
class L_Intensity(nn.Module):
        return Loss_intensity

class fusion_loss_mef(nn.Module):
    def __init__(self):
        super(fusion_loss_mef, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

        # print(1)
    def forward(self, image_A, image_B, image_fused):
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM