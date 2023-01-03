'''
Author: Dylan8527 vvm8933@gmail.com
Date: 2022-12-03 15:12:42
LastEditors: Dylan8527 vvm8933@gmail.com
LastEditTime: 2022-12-17 22:59:54
FilePath: \code\main.py
Description: 

Copyright (c) 2022 by Dylan8527 vvm8933@gmail.com, All Rights Reserved. 
'''
import torch
import numpy as np
import dataset
from tqdm import tqdm
import SimpleITK as sitk
import commentjson as json
from torch.utils import data
from torch.optim import lr_scheduler

from network import NeRFNetwork

import cv2
import os

from utils import psnr, ssim

def train(config_path):

    # load config
    # -----------------------
    with open(config_path) as config_file:
        config = json.load(config_file)

    # file
    # -----------------------
    proj_data_dir = config["file"]["proj_data_dir"]
    proj_pose_data_dir = config["file"]["proj_pose_data_dir"]
    workspace = config["file"]["workspace"]
    h, w, SOD = config["file"]["h"], config["file"]["w"], config["file"]["SOD"]
    num_angle, num_det = sitk.GetArrayFromImage(sitk.ReadImage(proj_data_dir)).shape

    # parameter
    # -----------------------
    lr = config["train"]["lr"]
    gpu = config["train"]["gpu"]
    epoch = config["train"]["epoch"]
    save_epoch = config["train"]["save_epoch"]
    lr_decay_epoch = config["train"]["lr_decay_epoch"]
    lr_decay_coefficient = config["train"]["lr_decay_coefficient"]
    batch_size = config["train"]["batch_size"]
    num_sample_ray = config["train"]["num_sample_ray"]

    # data loader
    # -----------------------
    train_loader = data.DataLoader(
        dataset=dataset.TrainData(proj_path=proj_data_dir, proj_pos_path=proj_pose_data_dir,
                                  num_sample_ray=num_sample_ray, num_angle=num_angle, SOD=SOD),
        batch_size=batch_size,
        shuffle=True)
    test_loader = data.DataLoader(
        dataset=dataset.TestData(h=(2*SOD), w=(2*SOD)),
        batch_size=1,
        shuffle=False)

    # model
    # -----------------------
    device = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))

    #############################################################################
    # TODO: Define your neural network and loss function here                   #
    # Hint: Use hash encoding and try different loss functions                  #
    #############################################################################
    network = NeRFNetwork(encoding='hashgrid', num_layers=2, hidden_dim=64, bound=1.).to(device)
    loss_func = torch.nn.MSELoss()
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    optimizer = torch.optim.Adam(params=network.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch, gamma=lr_decay_coefficient)

    loop_tqdm = tqdm(range(epoch), leave=False)
    eval_interval = int(epoch * 0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(workspace, exist_ok=True)

    for e in loop_tqdm:

        #############################################################################
        # TODO: Implement the training and testing processes here                   #
        #############################################################################
        # train
        network.train()
        average_psnr = 0
        average_ssim = 0
        for ray_sample, proj_sample in train_loader:
            pts =(ray_sample).to(device).float() # [B, N, T, 2]
            gts =(proj_sample).to(device).float()  # [B, N]
            # grad = 0
            optimizer.zero_grad()       
            # forward
            density = network(pts, perturb=True) # [B, N, T]
            # get loss
            preds = torch.sum(density, dim=-1) # [B, N]
            loss = loss_func(preds, gts)
            # backward
            loss.backward()
            optimizer.step()

            pred = preds.detach().cpu().numpy()
            gt = gts.detach().cpu().numpy()

            _psnr = psnr(pred, gt)
            _ssim = ssim(pred, gt)
            average_psnr += _psnr
            average_ssim += _ssim

            loop_tqdm.set_description(f'loss: {loss:.6f} psnr: {_psnr:.2f} ssim: {_ssim}')

        average_psnr /= len(train_loader)
        average_ssim /= len(train_loader)

        scheduler.step()

        if (e+1) % eval_interval == 0:
            # save psnr and ssim and epoch and loss into txt
            with open(os.path.join(workspace, 'log.txt'), 'a') as f:
                f.write(f'[EPOCH {e+1}] loss: {loss:.6f} psnr: {average_psnr:.6f} ssim: {average_ssim:.6f}\n')

            # test
            network.eval()
            with torch.no_grad():
                for ray_sample in test_loader:
                    h=(2*SOD)
                    w=(2*SOD)
                    pts = (ray_sample).to(device).float() #[H*W, 2]
                    pts = pts.view(1, h, w, 2)

                    pred = torch.zeros((h, w)).to(device)
                    for i in range(h):
                        pred[i] = network(pts[:, i:i+1, :, :])[0]
                    
                    pred = pred.cpu().numpy()
                    # pred = (pred - pred.min()) / (pred.max() - pred.min())
                    pred /= pred.max()
                    pred = (pred * 255).astype(np.uint8)
     
                    cv2.imwrite(os.path.join(workspace, f'{e+1}.png'), pred)    
    
    # save network after training
    torch.save(network.state_dict(), os.path.join(workspace, 'network.pth'))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

if __name__ == '__main__':
    # train('config.json')
    train('./configs/sparse90.json')
    train('./configs/dense90.json')
    train('./configs/sparse180.json')
    train('./configs/dense180.json')
    train('./configs/sparse45.json')
    train('./configs/dense45.json')
    train('./configs/sparse360.json')
