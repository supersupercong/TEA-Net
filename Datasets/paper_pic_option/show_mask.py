from torchvision import transforms
from PIL import Image

# 读取图像
ir_path = "/data/LLVIP_with_downsample_infrared/visible/train/080216.jpg"
vis_path = "/data/LLVIP_with_downsample_infrared/infrared_downsample_x4/train/080216.jpg"
ir_save_path = "/data/paper_pic/mask_ir.jpg"
vis_save_path = "/data/paper_pic/mask_vis.jpg"

def mask_image(irpath, vispath, compliment, irsavepath, vissavepath, scaleir, scalevis):
    ir, vis = Image.open(irpath), Image.open(vispath)
    # 可以选择应用一些预处理操作，如缩放、裁剪等
    preprocess = transforms.Compose([
        transforms.ToTensor()  # 将图像转换为张量
    ])
    # 应用预处理操作
    irtensor, vistensor = preprocess(ir), preprocess(vis)
    H, W = irtensor.shape[2:]
    mask = torch.ones([1, 1, int(H / scaleir), int(W / scaleir)]) * 0.3
    mask = torch.bernoulli(mask).to(device)
    upsampleir = torch.nn.Upsample(scale_factor=scaleir, mode='nearest')
    upsamplevis = torch.nn.Upsample(scale_factor=scalevis, mode='nearest')
    Maskir = upsampleir(mask)
    if compliment is True:
        mask = 1.0 - mask
    Maskvis = upsamplevis(mask)
    ir_mm = irtensor * Maskir
    vis_mm = vistensor * Maskvis
    # 将张量转换回图像
    irimage = transforms.ToPILImage()(ir_mm)
    visimage = transforms.ToPILImage()(vis_mm)
    # 保存图像
    irimage.save(irsavepath)
    visimage.save(vissavepath)


mask_image(ir_path, vis_path, compliment='True', irsavepath=ir_save_path, vissavepath=vis_save_path, scaleir=4, scalevis=16)
