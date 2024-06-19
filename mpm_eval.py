import torch
import torch.nn.functional as F
from tqdm import tqdm #进度条
import numpy as np
import cv2

def eval_net(net, loader, device, criterion, writer, global_step):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    # mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    total_error = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, mpms_gt = batch['img'], batch['mpm']
            '''
                将图像移动到指定的device中，并将数据类型转为dtype指定的类型
                ==》 这一步通常用于将数据从CPU移动到GPU上，以加速计算过程。
            '''
            imgs = imgs.to(device=device, dtype=torch.float32)
            mpms_gt = mpms_gt.to(device=device, dtype=torch.float32)

            '''
                用于指定代码块中的操作不计算梯度（代码块内有效）
                ==》这通常用在模型的推理（inference）阶段，即在模型训练完成后，使用模型进行预测或测试时。
                ==》相对应地，model.train()会将模型设置为训练模式，此时默认会计算梯度
            '''
            with torch.no_grad():
                mpms_pred = net(imgs) # 用训练好的模型去预测数据

            total_error += criterion(mpms_pred, mpms_gt).item()

            pbar.update()

        # print(imgs.shape)
        writer.add_images('images/1', imgs[:, :1], global_step)
        writer.add_images('images/2', imgs[:, 1:], global_step)

        writer.add_images('mpms/true', mpms_gt, global_step)
        writer.add_images('mpms/pred', mpms_pred, global_step)

        mags_gt = mpms_gt.pow(2).sum(dim=1, keepdim=True).sqrt()
        mags_pred = mpms_pred.pow(2).sum(dim=1, keepdim=True).sqrt()

        writer.add_images('mags/true', mags_gt, global_step)
        writer.add_images('mags/pred', mags_pred, global_step)

    net.train() # 将模型设置为训练模式，此时默认会计算梯度
    return total_error / n_val
