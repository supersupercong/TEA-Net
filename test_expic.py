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
from torchvision.transforms import ToTensor, ToPILImage
import time
from thop import profile

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('--gpu_num', default=1, type=int)
    parser.add_argument('--gpu_ids', default='1', type=str)
    parser.add_argument('-c', '--config', default='./configs/Train_pet.json', type=str,
                        help='Path to the config file')
    parser.add_argument('--ckpt',
                        default='/data6/weiwang/medical/code/PET/Experiments/MED/PET/EXP0/best_model-epoch:79-Q:17.9848.ckpt',
                        type=str)
    # parser.add_argument('--TestSet',
    #                     default='/data6/weiwang/medical/data/fusion_data/Med/Havard-Medical-Image-Fusion-Datasets-main/MyDatasets/PET-MRI/test',
    #                     type=str,
    #                     help='./data/TNO, ./data/VIFB')

    global args
    args = parser.parse_args()
    global config
    config = json.load(open(args.config))
    FusionNet = MODELS[config["net"]]()
    if args.gpu_num >1:
        device_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
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
        device_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
        FusionNet = FusionNet.cuda()
        ckpt = torch.load(args.ckpt, map_location='cuda:{}'.format(device_ids[0]))
        new_state_dict = collections.OrderedDict()
        for k in ckpt['state_dict']:
            # print(k)
            if k[:12] != 'MambaFusion.':
                continue
            name = k[12:]
            new_state_dict[name] = ckpt['state_dict'][k]

    FusionNet.load_state_dict(new_state_dict, strict=True)
    FusionNet.eval()

    outputdir = os.path.join('./TestEXP_pic/517', config['train_dataset'], args.ckpt.split('/')[-1].replace('.ckpt', ''))
    os.makedirs(outputdir, exist_ok=True)

    __dataset__ = {config["train_dataset"]: Data}
    test_dataset = __dataset__[config["train_dataset"]](config, is_train=False, is_ycbcrA=True)
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
    to_image = ToPILImage()
    to_tensor = ToTensor()
    for i, test_data in enumerate(test_loader):
        # print('test_data[0].shape={}'.format(test_data[0].shape))
        img_a = test_data[0].cuda()
        img_b = to_image(test_data[1].squeeze(0)).convert('YCbCr')
        img_b_Y, img_b_Cb, img_b_Cr = img_b.split()
        img_b = to_tensor(img_b_Y).unsqueeze(0)
        img_b = img_b.cuda()
        # print(img_b.shape)
        file = test_data[2]

        start = time.time()
        # inference
        with torch.no_grad():
            fusion = FusionNet(img_a,None, img_b)
        end = time.time()
        # if i ==0:
        #     print("The thop result")
        #     flops, params = profile(FusionNet, inputs=(img_a, img_b))
        #     print('flops:{:.6f}, params:{:.6f}'.format(flops/(1e9), params/(1e5)))
        counter += (end-start)
        fusion_numpy = fusion.squeeze(0).squeeze(0).clamp(0, 255).byte().cpu().detach().numpy()
        # print(fusion_numpy.shape)
        fusion_image = Image.fromarray(fusion_numpy)
        fusion_image_3c = Image.merge("YCbCr", [fusion_image, img_b_Cb, img_b_Cr])
        fusion_image_3c = fusion_image_3c.convert("RGB")
        print(os.path.join(outputdir, file[0]))
        fusion_image_3c.save(os.path.join(outputdir, file[0]))
        lente = i
    print('avg_runtime={}'.format(counter/lente))


if __name__ == '__main__':
    print('-----------------------------------------test.py testing-----------------------------------------')
    main()
