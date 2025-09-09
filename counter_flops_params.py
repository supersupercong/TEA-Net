
import  sys
from thop import profile
import importlib, torch
import math
import time
from models.models import MODELS
import re, yaml, os  
import sys
import json
import collections

if __name__ == "__main__":
    
    #cfg = 'Train_'+model_name+'.json'
    # config  = json.load(open(test_config_path+cfg))
    model =  MODELS['MambaFusion']().cuda()
    ckpt_path = './Experiments/VIF/MSRS/EXP6/best_model-epoch:01-quality:0.7144.ckpt'
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    new_state_dict = collections.OrderedDict()
    for k in ckpt['state_dict']:
        # print(k)
        if k[:12] != 'MambaFusion.':
            continue
        name = k[12:]
        new_state_dict[name] = ckpt['state_dict'][k]

    #input = torch.randn(1, 4, 32, 32)
    input1 = torch.randn(1, 1, 640, 480).cuda()
    input2 = torch.randn(1, 1, 640, 480).cuda()
    print("The thop result")
    flops, params = profile(model, inputs=(input1, input2))
    print('flops:{:.6f}, params:{:.6f}'.format(flops/(1e9), params/(1e5)))
    print('===========================================================================')