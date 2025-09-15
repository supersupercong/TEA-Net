import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

import argparse
import numpy as np
from Datasets.datasets import *
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import json
import shutil
import os
import copy
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from easydict import EasyDict
from models.models import MODELS
from models.models import LOSSES
# from models.functions import get_gradient
from utils.optimizer import Lion
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from utils.imgqual_utils import PSNR, SSIM
import collections
from utils.ema import EMA as EMACallback
from utils.utils_color import rgb2ycbcr, ycbcr2rgb


# create floder
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# select dataset type
__dataset__ = {
    "PET": Data
}


class CoolSystem(pl.LightningModule):
    def __init__(self):
        """初始化训练的参数"""
        super(CoolSystem, self).__init__()
        # train datasets
        self.train_datasets = __dataset__[config["train_dataset"]](config, is_train=True, is_grayA=True, is_ycbcrB=True)
        self.train_batchsize = config["train_batch_size"]
        # val datasets
        self.validation_datasets = __dataset__[config["train_dataset"]](config, is_train=False, is_label=False, is_grayA=False, is_ycbcrB=True)
        self.val_batchsize = config["train_batch_size"]
        self.num_workers = config["num_workers"]
        self.save_path = config["save_path"]
        ensure_dir(self.save_path)
        # set mode stype
        # self.MMSR = MODELS[config["fusion_net"]](config)
        self.MambaFusion = MODELS[config["net"]](1,1,32)

        # loss
        self.loss = LOSSES[config["loss"]]()

        # Resume from pth ...
        if args.resume is not None:
            print("Loading from existing MambaFusion chekpoint")
            ckpt = torch.load(args.resume)
            new_state_dict = collections.OrderedDict()
            for k in ckpt['state_dict']:
                # print(k)
                if k[:12] != 'MambaFusion.':
                    continue
                name = k[12:]
                new_state_dict[name] = ckpt['state_dict'][k]

            self.MambaFusion.load_state_dict(new_state_dict, strict=True)

        print(PATH)
        # print model summary.txt
        import sys
        original_stdout = sys.stdout
        with open(PATH + "/" + "model_summary.txt", 'w+') as f:
            sys.stdout = f
            print(f'\n{self.MambaFusion}\n')
            sys.stdout = original_stdout
            # shutil.copy(f'./models/{config["model"]}.py',PATH+"/"+"model.py")
        self.automatic_optimization = False

    def train_dataloader(self):
        train_loader = data.DataLoader(
            self.train_datasets,
            batch_size=self.train_batchsize,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        train_loader = data.DataLoader(
            self.validation_datasets,
            batch_size=self.train_batchsize,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return train_loader

    def configure_optimizers(self):
        """配置优化器和学习率的调整策略"""
        # Setting up optimizer.
        self.initlr = config["optimizer"]["args"]["lr"]  # initial learning
        self.weight_decay = config["optimizer"]["args"]["weight_decay"]  # optimizers weight decay
        self.momentum = config["optimizer"]["args"]["momentum"]
        if config["optimizer"]["type"] == "SGD":
            optimizer = optim.SGD(
                self.MambaFusion.parameters(),
                lr=self.initlr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif config["optimizer"]["type"] == "ADAM":
            optimizer = optim.Adam(
                self.MambaFusion.parameters(),
                lr=self.initlr,
                weight_decay=self.weight_decay,
                betas=[0.9, 0.999]
            )

        elif config["optimizer"]["type"] == "ADAMW":
            optimizer = optim.AdamW(
                self.MambaFusion.parameters(),
                lr=self.initlr,
                weight_decay=self.weight_decay,
                betas=[0.9, 0.999]
            )
        elif config["optimizer"]["type"] == "Lion":
            optimizer = Lion(filter(lambda p: p.requires_grad, self.MambaFusion.parameters()),
                             lr=self.initlr,
                             betas=[0.9, 0.99],
                             weight_decay=0)

        else:
            exit("Undefined optimizer type")

        # Learning rate shedule
        if config["optimizer"]["sheduler"] == "StepLR":
            step_size = config["optimizer"]["sheduler_set"]["step_size"]
            gamma = config["optimizer"]["sheduler_set"]["gamma"]
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif config["optimizer"]["sheduler"] == "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.initlr, max_lr=1.2 * self.initlr,
                                                          cycle_momentum=False)
        elif config["optimizer"]["sheduler"] == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=config["trainer"]["total_epochs"] * 200,
                                                                   eta_min=self.initlr * 1e-2)
        # Sheduler == None时
        else:
            scheduler = None
        return [optimizer], [scheduler]

    def training_step(self, data):
        """optimize the training"""
        opt = self.optimizers()
        opt.zero_grad()
        device = next(self.MambaFusion.parameters()).device

        """trainning step"""
        # Reading data.
        source, guide = data   # source是未经采样的红外图像，guide是可见光图像
        # print("guide.shape={}".format(guide.shape))
        fusion = self.MambaFusion(source,None, guide)

        '''compute loss function'''
        # z = (fusion - source)
        # z = torch.abs(z)
        fusion_loss, loss_gradient, loss_l1, loss_SSIM = self.loss(guide, source, fusion)
        ######### Computing loss #########
        # loss = Compute_loss(config,out,PAN_image,gt)
        self.log('train_loss', fusion_loss, prog_bar=True)
        # self.log('lr',self.trainer.optimizers[0].state_dict()['param_groups'][0]['lr'],sync_dist=True,prog_bar=True)

        '''clip gradients'''
        self.manual_backward(fusion_loss)
        # self.clip_gradients(opt1, gradient_clip_val=10, gradient_clip_algorithm="norm")
        # self.clip_gradients(opt2, gradient_clip_val=10, gradient_clip_algorithm="norm")
        opt.step()

        # mutiple schedulers
        sch = self.lr_schedulers()
        sch.step()

        return {'loss': fusion_loss}

    def validation_step(self, data, batch_idx):
        self.MambaFusion.eval()

        """validation step"""
        source, guide, file = data #source是未采样红外图像，ds_B是下采样红外图像
        # print(guide.shape)
        # reconstrucion network
        # b_Y.to(device)
        # b_Cb.to(device)
        # b_Cr.to(device)
        b_Y, b_Cb, b_Cr = rgb2ycbcr(source)
        fusion = self.MambaFusion(b_Y,None,guide)
        quality = self.loss(guide, b_Y, fusion)[0]

        fusion = ycbcr2rgb(fusion, b_Cb, b_Cr)
        fusion_numpy = (fusion*255).squeeze(0).permute(1,2,0).clamp(0, 255).byte().cpu().detach().numpy()
        fusion_image = Image.fromarray(fusion_numpy)
        fusion_image.save(os.path.join(self.save_path, file[0]))

        # cv2.imwrite(os.path.join(self.save_path, file[0].split(".")[0]+'.png'),fusion_numpy)
        # save_image(fusion, os.path.join(self.save_path, file[0].split(".")[0]+'.png'))

        self.log('quality', quality, sync_dist=True, prog_bar=True)

        return {"quality": quality}


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='./configs/Train_pet.json', type=str,
                        help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default='1', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-v', '--val', default=True, type=bool,
                        help='Valdation')
    parser.add_argument('-val_path',
                        default='/data6/weiwang/medical/code/224-c-g-s/Experiments/MED/PET/EXP0/model.ckpt',
                        type=str, help='Path to the val path')

    global args
    args = parser.parse_args()
    # set resmue
    global config
    config = json.load(open(args.config))

    # Set seeds.
    seed = 42  # Global seed set to 42
    seed_everything(seed)

    # wandb log init
    global wandb_logger
    import wandb
    # output_dir = './TensorBoardLogs'
    logger = WandbLogger(project=config['name'] + "-" + config["train_dataset"])
    # logger = TensorBoardLogger(name=config['name']+"_"+config["train_dataset"],save_dir = output_dir )

    # Setting up path
    global PATH
    PATH = "./" + config["experim_name"] + "/" + config["train_dataset"] + "/" + str(config["tags"])
    ensure_dir(PATH + "/")
    shutil.copy2(args.config, PATH)

    # init pytorch-litening
    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
    model = CoolSystem()

    # set checkpoint mode and init ModelCheckpointHook
    checkpoint_callback = ModelCheckpoint(
        monitor='quality',
        dirpath=PATH,
        filename='best_model-epoch:{epoch:02d}-Q:{quality:.4f}',
        auto_insert_metric_name=False,
        every_n_epochs=config["trainer"]["test_freq"],
        save_on_train_epoch_end=True,
        save_top_k=config["trainer"]["save_top_k"],
        mode="max"
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    ema_callback = EMACallback(decay=0.995, every_n_steps=1)

    trainer = pl.Trainer(
        strategy=ddp,
        max_epochs=config["trainer"]["total_epochs"],
        accelerator='gpu', devices=args.device,
        logger=logger,
        # amp_backend="apex",
        # amp_level='01',
        # accelerator='ddp',
        # precision='16-mixed',
        callbacks=[checkpoint_callback, lr_monitor_callback, ema_callback],
        check_val_every_n_epoch=config["trainer"]["test_freq"],
        log_every_n_steps=10,
        # fast_dev_run=True,
    )

    if args.val == True:
        trainer.validate(model, ckpt_path=args.val_path)
    else:
        # resume from ckpt pytorch lightening
        # trainer.fit(model,ckpt_path=resume_checkpoint_path)
        # resume from pth pytorch
        trainer.fit(model)


if __name__ == '__main__':
    print('-----------------------------------------train_pl.py training-----------------------------------------')
    main()
