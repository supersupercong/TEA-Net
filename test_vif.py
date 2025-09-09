import os
import argparse
import numpy as np
from Datasets.datasets import *
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torchvision
import json
import shutil
import os
import copy
import yaml
from easydict import EasyDict
from models.models import MODELS
# from models.functions import upsample
from utils.optimizer import Lion
from torchvision.utils import save_image
from utils.imgqual_utils import PSNR, SSIM
import collections
from PIL import Image
from torchvision.transforms import ToTensor

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('--gpu_ids', default='1,2,3', type=str)
    parser.add_argument('-c', '--config', default='./configs/Train_ct.json', type=str,
                        help='Path to the config file')
    parser.add_argument('--ckpt', default='/data6/weiwang/medical/code/TCSVT-Mamba/Experiments/MED/CT/EXP0/best_model-epoch:119-Q:20.1044.ckpt', type=str)
    parser.add_argument('--TestSet', default='./data/TNO', type=str,
                        help='./data/TNO, ./data/VIFB')

    global args
    args = parser.parse_args()
    global config
    config = json.load(open(args.config))
    device_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
    FusionNet = MODELS[config["net"]](1,1,32)
    if len(device_ids)>1:
        FusionNet = nn.DataParallel(FusionNet, device_ids=device_ids)  # 使用DataParallel来使用多个GPU
        ckpt = torch.load(args.ckpt, map_location='cuda:{}'.format(args.gpu_ids))
    FusionNet = FusionNet.cuda()  # 将模型移动到GPU上
    # with open(args.ckpt, 'rb') as file:
    #     ckpt_bytes = file.read()
    # buffer = BytesIO(ckpt_bytes)
    # ckpt = torch.load(buffer, map_location=args.gpu_id)
    # buffer.close()
    ckpt = torch.load(args.ckpt, map_location='cuda:0')
    new_state_dict = collections.OrderedDict()
    for k in ckpt['state_dict']:
        print(k)
        if k[:15] != 'highorder_nips.':
            continue
        name = 'module.'+k[15:]
        new_state_dict[name] = ckpt['state_dict'][k]
    # print(new_state_dict)

    FusionNet.load_state_dict(new_state_dict, strict=True)
    FusionNet.eval()
    outputdir = os.path.join('./TestEXP_____', args.ckpt.split('/')[-1].replace('.ckpt', ''))
    os.makedirs(outputdir, exist_ok=True)
    vis_folder = os.path.join(args.TestSet, 'VIS')
    ir_folder = os.path.join(args.TestSet, 'IR')
    numpy2tensor = ToTensor()
    for i in os.listdir(vis_folder):
        vis_sorc = os.path.join(vis_folder, i)
        ir_sorc = os.path.join(ir_folder, i.replace('VIS', 'IR'))
        vis_img = Image.open(vis_sorc).convert('L')
        ir_img = Image.open(ir_sorc).convert('L')
        print('ir_img.shape={}'.format(ir_img.size))
        vis_tensor = numpy2tensor(vis_img).float().unsqueeze(0).cuda()
        ir_tensor = numpy2tensor(ir_img).float().unsqueeze(0).cuda()
        fusion = FusionNet(vis_tensor, ir_tensor)*255
        print(f'Fusion tensor 最小值: {torch.min(fusion)}, 最大值: {torch.max(fusion)}')
        fusion_numpy = fusion.squeeze(0).squeeze(0).clamp(0, 255).byte().cpu().detach().numpy()
        print(fusion_numpy.shape)
        fusion_image = Image.fromarray(fusion_numpy)
        fusion_image.save(os.path.join(outputdir, i.replace('VIS', '')))


if __name__ == '__main__':
    print('-----------------------------------------test.py testing-----------------------------------------')
    main()
