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
from torchvision.utils import save_image
from utils.imgqual_utils import PSNR, SSIM
from utils.utils_color import rgb2ycbcr, ycbcr2rgb
import collections
from PIL import Image
from torchvision.transforms import ToTensor
import time
from thop import profile

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('--gpu_num', default=2, type=int)
    parser.add_argument('--gpu_ids', default='0,3', type=str)
    parser.add_argument('-c', '--config', default='./configs/Train_vif.json', type=str,
                        help='Path to the config file')
    parser.add_argument('--ckpt', default='./Experiments/VIF/MSRS/wop4seg/best_model-epoch:42-quality:2.2397-det:0.0078.ckpt', type=str)
    parser.add_argument('--vdt', default='val', type=str, help='val, det, test')
    # parser.add_argument('--TestSet', default='./data/TNO', type=str,
    #                     help='./data/TNO, ./data/VIFB')

    global args
    args = parser.parse_args()
    global config
    config = json.load(open(args.config))
    FusionNet = MODELS[config["net"]](num_experts=config["MambaFusion"]["num_experts"], topk=config["MambaFusion"]["top_k"])
    if args.gpu_num >1:
        device_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
        device = torch.device('cuda:{}'.format(device_ids[0]))
        FusionNet = nn.DataParallel(FusionNet, device_ids=device_ids)  # 使用DataParallel来使用多个GPU
        FusionNet = FusionNet.cuda(device_ids[0])  # 将模型移动到GPU上
        ckpt = torch.load(args.ckpt, map_location='cuda:{}'.format(device_ids[0]))
        new_state_dict = collections.OrderedDict()
        for k in ckpt['state_dict']:
            # print(k)
            if k[:12] != 'MambaFusion.':
                continue
            name = 'module.'+k[12:]
            new_state_dict[name] = ckpt['state_dict'][k]
        # print(new_state_dict)

    else:
        device=torch.device('cuda:{}'.format(args.gpu_ids))
        FusionNet = FusionNet.to(device)
        ckpt = torch.load(args.ckpt, map_location='cuda:{}'.format(args.gpu_ids))
        new_state_dict = collections.OrderedDict()
        for k in ckpt['state_dict']:
            # print(k)
            if k[:12] != 'MambaFusion.':
                continue
            name = k[12:]
            new_state_dict[name] = ckpt['state_dict'][k]

    FusionNet.load_state_dict(new_state_dict, strict=True)
    FusionNet.eval()

    __dataset__ = {config["train_dataset"]: Data}
    test_dataset = __dataset__[config["train_dataset"]](config, is_train=False, is_label=False, is_grayA=True, val_det_test=args.vdt)
    outputdir = os.path.join('./TestEXP', config['train_dataset'], args.ckpt.split('/')[-2]+'_'+args.ckpt.split('/')[-1].replace('.ckpt', '')+'_'+args.vdt)
    os.makedirs(outputdir, exist_ok=True)
    # test_dataset = __dataset__[config["train_dataset"]](config, is_train=False, is_label=False, is_grayA=True)
    test_batchsize = config["train_batch_size"]
    num_workers = config["num_workers"]
    test_loader = data.DataLoader(
            test_dataset,
            batch_size=test_batchsize,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )
    counter = 0
    for i, test_data in enumerate(test_loader):
        img_a = test_data[0].to(device)
        img_b = test_data[1].to(device)
        b_Y, b_Cb, b_Cr = rgb2ycbcr(img_b)
        print('b_Y.shape={}'.format(b_Y.shape))
        b_Y.to(device)
        b_Cb.to(device)
        b_Cr.to(device)
        file = test_data[2]

        start = time.time()
        # inference
        with torch.no_grad():
            fusion = FusionNet(ir=img_a, vis=b_Y, is_train=False)
        end = time.time()
        counter += (end-start)
        fusion = ycbcr2rgb(fusion, b_Cb, b_Cr)
        print('fusion.shape={}'.format(fusion.shape))
        fusion_numpy = (fusion*255).squeeze(0).permute(1,2,0).clamp(0, 255).byte().cpu().detach().numpy()
        # print(fusion_numpy.shape)
        fusion_image = Image.fromarray(fusion_numpy)
        fusion_image.save(os.path.join(outputdir, file[0]))
        lente = i
    print('avg_runtime={}'.format(counter/lente))

if __name__ == '__main__':
    print('-----------------------------------------test.py testing-----------------------------------------')
    main()
