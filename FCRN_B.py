# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 10:30:42 2019

@author: lenovo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib.image import imsave
from torchvision import transforms


# from utils import count_param
# from torchsummary import summary


class conv_uint(nn.Module):
    '''(conv => BN => ReLU => pooling)'''
    def __init__(self,in_ch,out_ch):
        super(conv_uint,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_uint_without_pool(nn.Module):
    '''(conv => BN => ReLU)'''
    def __init__(self,in_ch,out_ch):
        super(conv_uint_without_pool,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_uint_without_pool_5k(nn.Module):
    '''(conv => BN => ReLU)'''
    def __init__(self,in_ch,out_ch):
        super(conv_uint_without_pool_5k,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_trans_uint_for_odd(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_trans_uint_for_odd, self).__init__()

        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 5,2,2,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_trans(x)
        return x

class conv_trans_uint(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_trans_uint, self).__init__()

        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 5,2,2,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_trans(x)
        return x


class conv_trans_uint_to_odd(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_trans_uint_to_odd, self).__init__()

        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3,2,0,output_padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_trans(x)
        return x


class conv_trans_uint_for_pad(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_trans_uint_for_pad, self).__init__()

        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3,2,0,output_padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_trans(x)
        return x



class FCRN_B_Net(nn.Module):
    def __init__(self, n_channels, n_classes,dataset,mode):
        super(FCRN_B_Net, self).__init__()
        self.dataset=dataset
        self.mode = mode

        self.inc = conv_uint_without_pool(n_channels, 32)
        self.down1 = conv_uint(32, 64)
        self.down2 = conv_uint_without_pool(64, 128)
        self.down3 = conv_uint(128, 256)
        self.down4 = conv_uint_without_pool_5k(256, 256)

        if mode ==  "density":

            if self.dataset=="BM":
                self.up1 = conv_trans_uint_to_odd(256, 256)
                self.up2 = conv_trans_uint(256, n_classes)

            elif self.dataset=="BCFM" or self.dataset=="PDL1-1":
                self.up1 = conv_trans_uint(256, 256)
                self.up2 = conv_trans_uint(256, n_classes)

            elif self.dataset=="PDL1-2":
                self.up1 = conv_trans_uint(256, 256)
                self.up2 = conv_trans_uint(256, n_classes)

            elif self.dataset=="PSU":
                self.up1 = conv_trans_uint(256, 256)
                self.up2 = conv_trans_uint(256, n_classes)



        if mode ==  "proximity":
            if self.dataset == "BCFM":
                self.up1 = conv_trans_uint(256, 256)
                self.up2 = conv_trans_uint(256, n_classes)

            if self.dataset=="BM" or self.dataset == "PSU":
                self.up1 = conv_trans_uint_to_odd(256, 256)
                self.up2 = conv_trans_uint(256, n_classes)

            if self.dataset=="PDL1-1":
                self.up1 = conv_trans_uint(256, 256)
                self.up2 = conv_trans_uint(256, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5)
        x7 = self.up2(x6)
        # x8 = self.up3(x7)
        # x = self.outc(x)
        return x7
        # return x8


# net=UNet(3,2)
# param = count_param(net)
# print(param)
# check keras-like model summary using torchsummary
'''
summary(model, input_size=(3, 64, 64))
'''
# net=FCRN_B_Net(3,1,"PSU","density")
# print(net)
# input=torch.rand(8,3,100,100)
# out=net(input)
# print(out.shape)
# #
# x_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
#
# input=np.load(r'C:\Users\zya\Desktop\ACCESS\BCFM\定位（figure7）\0.npy')
#
# input=x_transforms(input)
# input=input.unsqueeze(dim=0)
# model = UNet(3,1)
# output=model(input)
# # print(model)
# print(output.shape)
# output=output.cpu()
# output=output.detach().numpy()
# imsave('check_unet.png',output[0,0,:,:],cmap='jet')