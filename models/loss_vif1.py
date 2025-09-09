from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
from utils.utils_color import RGB_HSV, RGB_YCbCr
from models.loss_ssim import ssim
import torchvision.transforms.functional as TF
from .vgg import *

def features_grad(features):
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.cuda()
    _, c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    return feat_grads

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient
        
class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused, weight_list):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        Loss_SSIM = weight_list[:,0] * ssim(image_A, image_fused) + weight_list[:,1] * ssim(image_B, image_fused)
        return Loss_SSIM

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):        
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity

class Perception_Weight(nn.Module):
    def __init__(self):
        super(Perception_Weight, self).__init__()
        self.feature_model = vgg16().cuda()
        self.feature_model.load_state_dict(torch.load('/home/zrj/workshop/TGRS-muti-focus/models/vgg16.pth'))
    
    def forward(self, tensorA, tensorB):
        with torch.no_grad():
            featA = torch.cat((tensorA, tensorA, tensorA), dim=1)
            featA = self.feature_model(featA)
            featB = torch.cat((tensorB, tensorB, tensorB), dim=1)
            featB = self.feature_model(featB)

            for i in range(len(featA)):
                m1 = torch.mean(features_grad(featA[i]).pow(2), dim=[1, 2, 3])
                m2 = torch.mean(features_grad(featB[i]).pow(2), dim=[1, 2, 3])
                if i == 0:
                    w1 = torch.unsqueeze(m1, dim=-1)
                    w2 = torch.unsqueeze(m2, dim=-1)
                else:
                    w1 = torch.cat((w1, torch.unsqueeze(m1, dim=-1)), dim=-1)
                    w2 = torch.cat((w2, torch.unsqueeze(m2, dim=-1)), dim=-1)
            weight_1 = torch.mean(w1, dim=-1) / 3500
            weight_2 = torch.mean(w2, dim=-1) / 3500
            weight_list = torch.cat((weight_1.unsqueeze(-1), weight_2.unsqueeze(-1)), -1)
            weight_list = F.softmax(weight_list, dim=-1)
        return weight_list
        
class fusion_loss_vif1(nn.Module):
    def __init__(self):
        super(fusion_loss_vif1, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        self.Perception_Weight = Perception_Weight()

        # print(1)
    def forward(self, image_A, image_B, image_fused):
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
        weight_list = self.Perception_Weight(image_A, image_B)
        loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused, weight_list))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM