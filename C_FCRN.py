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


class _conv_bn_relu_maxp(nn.Module):
    '''(conv => BN => ReLU => pooling)'''
    def __init__(self,in_ch,out_ch):
        super(_conv_bn_relu_maxp,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class _conv_bn_relu(nn.Module):
    '''(conv => BN => ReLU)'''
    def __init__(self,in_ch,out_ch):
        super(_conv_bn_relu,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class _conv_bn_relu_1x1(nn.Module):
    '''(conv => BN => ReLU)'''
    def __init__(self,in_ch,out_ch):
        super(_conv_bn_relu_1x1,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class _conv_bn_relu_from_odd(nn.Module):
    '''(conv => BN => ReLU)'''
    def __init__(self,in_ch,out_ch):
        super(_conv_bn_relu_from_odd,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_trans_uint_from_odd(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_trans_uint_from_odd, self).__init__()

        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3,2,1,output_padding=0),
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


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()

        self.conv_trans = _conv_bn_relu(in_ch,out_ch)

    def forward(self, x1,x2):
        x1 = nn.functional.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)

        x = self.conv_trans(x)
        return x

class up_from_odd(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_from_odd, self).__init__()

        self.conv_trans = _conv_bn_relu_from_odd(in_ch,out_ch)

    def forward(self, x1,x2):
        x1 = nn.functional.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)

        # x = self.conv_trans(x)
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


class C_FCRN(nn.Module):
    def __init__(self, n_channels, n_classes,dataset,mode):
        super(C_FCRN, self).__init__()

        self.dataset = dataset
        self.mode = mode
        if mode=="density":
            self.down0 = _conv_bn_relu(n_channels, 32)
            self.pool0 = nn.MaxPool2d(2)
            self.down1 = _conv_bn_relu(32, 64)
            self.pool1 = nn.MaxPool2d(2)
            self.down2 = _conv_bn_relu(64, 128)
            self.pool2 = nn.MaxPool2d(2)
            self.down3 = _conv_bn_relu(128, 512)
            self.conv1 = nn.Sequential(
                _conv_bn_relu(512,128),
                _conv_bn_relu_1x1(128,1)
            )

            if dataset=="PSU":
                self.up1 = up(640, 64)
                self.conv2 = nn.Sequential(
                    _conv_bn_relu(64, 32),
                    _conv_bn_relu_1x1(32,1)
                )
                self.up2 = up(128, 32)
                self.conv3 = nn.Sequential(
                    _conv_bn_relu(32, 16),
                    _conv_bn_relu_1x1(16, 1)
                )
                self.up3 = up(64, 32)
                self.up4 = _conv_bn_relu_1x1(32, n_classes)

            elif dataset=="pannuke":
                self.up1 = up(640, 64)
                self.conv2 = nn.Sequential(
                    _conv_bn_relu(64, 32),
                    _conv_bn_relu_1x1(32, 1)
                )
                self.up2 = up(128, 32)
                self.conv3 = nn.Sequential(
                    _conv_bn_relu(32, 16),
                    _conv_bn_relu_1x1(16, 1)
                )
                self.up3 = up(64, 32)
                self.up4 = _conv_bn_relu_1x1(32, n_classes)

            elif dataset=="consep":
                self.up1 = up(640, 64)
                self.conv2 = nn.Sequential(
                    _conv_bn_relu(64, 32),
                    _conv_bn_relu_1x1(32, 1)
                )
                self.up2 = up(128, 32)
                self.conv3 = nn.Sequential(
                    _conv_bn_relu(32, 16),
                    _conv_bn_relu_1x1(16, 1)
                )
                self.up3 = up(64, 32)
                self.up4 = _conv_bn_relu_1x1(32, n_classes)

            elif dataset=="kaggle":
                self.up1 = up(640, 64)
                self.conv2 = nn.Sequential(
                    _conv_bn_relu(64, 32),
                    _conv_bn_relu_1x1(32, 1)
                )
                self.up2 = up(128, 32)
                self.conv3 = nn.Sequential(
                    _conv_bn_relu(32, 16),
                    _conv_bn_relu_1x1(16, 1)
                )
                self.up3 = up(64, 32)
                self.up4 = _conv_bn_relu_1x1(32, n_classes)

        # if mode=="proximity":
        #     self.inc = conv_uint(n_channels, 32)
        #     self.down1 = conv_uint(32, 64)
        #     self.down2 = conv_uint(64, 128)
        #     self.down3 = conv_uint_without_pool(128, 512)
        #
        #     if dataset == "BCFM":
        #         self.up1 = conv_trans_uint(512, 128)
        #         self.up2 = conv_trans_uint(128, 64)
        #         self.up3 = conv_trans_uint(64, 32)
        #         self.up4 = conv_uint_without_pool(32, n_classes)
        #
        #     if dataset == "BM":
        #         self.up1 = conv_trans_uint(512, 128)
        #         self.up2 = conv_trans_uint_to_odd(128, 64)
        #         self.up3 = conv_trans_uint(64, 32)
        #         self.up4 = conv_uint_without_pool(32, n_classes)
        #
        #     elif dataset=="PDL1-1":
        #         self.up1 = conv_trans_uint_to_odd(512, 128)
        #         self.up2 = conv_trans_uint(128, 64)
        #         self.up3 = conv_trans_uint(64, 32)
        #         self.up4 = conv_uint_without_pool(32, n_classes)
        #
        #     elif dataset=="PSU":
        #         self.up1 = conv_trans_uint(512, 128)
        #         self.up2 = conv_trans_uint_to_odd(128, 64)
        #         self.up3 = conv_trans_uint(64, 32)
        #         self.up4 = conv_uint_without_pool(32, n_classes)



    def forward(self, x):
        x1_1 = self.down0(x)
        x1_2 = self.pool0(x1_1)
        x2_1 = self.down1(x1_2)
        x2_2 = self.pool1(x2_1)
        x3_1 = self.down2(x2_2)
        x3_2 = self.pool2(x3_1)
        x4 = self.down3(x3_2)


        x5 = self.up1(x4,x3_1)
        x6 = self.up2(x5,x2_1)
        x7 = self.up3(x6,x1_1)
        x8 = self.up4(x7)
        # x = self.outc(x)

        x4 = self.conv1(x4)
        x5 = self.conv2(x5)
        x6 = self.conv3(x6)

        # return x8
        return x4,x5,x6,x8


# net=UNet(3,2)
# param = count_param(net)
# print(param)
# check keras-like model summary using torchsummary
'''
summary(model, input_size=(3, 64, 64))
'''
net=C_FCRN(3,1,"consep","density")
print(net)
input=torch.rand(8,3,64,64)
out1,out2,out3,out4=net(input)
# print(out1.shape)
# print(out2.shape)
# print(out3.shape)
print(out4.shape)
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