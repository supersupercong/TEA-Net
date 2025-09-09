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
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('-c', '--config', default='./configs/Train_MMnet.json', type=str,
                        help='Path to the config file')
    parser.add_argument('--ckpt', default='./Experiments/same/LLVIP/EXP2/psnr28_6850ssim0_7821.ckpt', type=str)
    parser.add_argument('-d', '--device', default='0,1', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--TestSet', default='./data/TNO_os', type=str,
                        help='./data/TNO_os, ./data/VIFB_os')

    global args
    args = parser.parse_args()
    global config
    config = json.load(open(args.config))
    device = torch.device('cuda:{}'.format(args.gpu_id))
    FusionNet = MODELS[config["fusion_net"]](config).to(device)
    # with open(args.ckpt, 'rb') as file:
    #     ckpt_bytes = file.read()
    # buffer = BytesIO(ckpt_bytes)
    # ckpt = torch.load(buffer, map_location=args.gpu_id)
    # buffer.close()
    ckpt = torch.load(args.ckpt, map_location=device)

    new_state_dict = collections.OrderedDict()
    for k in ckpt['state_dict']:
        # print(k)
        if k[:10] != 'FusionNet.':
            continue
        name = k[10:]
        new_state_dict[name] = ckpt['state_dict'][k]
    FusionNet.load_state_dict(new_state_dict, strict=True)
    outputdir = os.path.join('./TestEXP', args.ckpt.split('/')[-1].replace('.ckpt', ''))
    os.makedirs(outputdir, exist_ok=True)
    vis_folder = os.path.join(args.TestSet, 'VIS')
    ir_folder = os.path.join(args.TestSet, 'IR_ds_x4')
    numpy2tensor = ToTensor()
    for i in os.listdir(vis_folder):
        vis_sorc = os.path.join(vis_folder, i)
        ir_sorc = os.path.join(ir_folder, i.replace('VIS', 'IR'))
        vis_img = Image.open(vis_sorc).convert('L')
        ir_img = Image.open(ir_sorc).convert('L')
        print('ir_img.shape={}'.format(ir_img.size))
        vis_tensor = numpy2tensor(vis_img).float().unsqueeze(0).to(device)
        ir_tensor = numpy2tensor(ir_img).float().unsqueeze(0).to(device)
        fusion = FusionNet(vis_tensor, ir_tensor)*255
        print(f'Fusion tensor 最小值: {torch.min(fusion)}, 最大值: {torch.max(fusion)}')
        fusion_numpy = fusion.squeeze(0).squeeze(0).clamp(0, 255).byte().cpu().detach().numpy()
        print(fusion_numpy.shape)
        fusion_image = Image.fromarray(fusion_numpy)
        fusion_image.save(os.path.join(outputdir, i.replace('VIS', '')))


if __name__ == '__main__':
    print('-----------------------------------------test.py testing-----------------------------------------')
    main()
