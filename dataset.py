
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import os
import numpy as np
from matplotlib.image import imsave

class valDataSet(Dataset):
    """
    root: 图像存放根目录
    augment: 是否需要数据增强
    """

    def __init__(self, root, start, end, label=True, augment=None, transform=None, target_transform=None):
        # 获取数据地址
        self.label=label
        data_path = []
        for i in range(start, end):
            img = os.path.join(root + str(i) + '.npy')
            if label=='mask':
                mask = os.path.join(root + str(i) + '_mask.npy')
                data_path.append((img, mask))
            else:
                if label=='pred_label':
                    mask = os.path.join(root + str(i) + '_pred_label.npy')
                    data_path.append((img, mask))
                else:
                    data_path.append(img)
        self.data_path = data_path

        # 是否进行数据增强
        #self.augment = augment
        #
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        if self.label:
            x_path, y_path = self.data_path[index]
            x = np.load(x_path)  # .reshape(3,128,128)
            x = x.astype('uint8')
            y = np.load(y_path)  # .reshape(1,128,128)
            y = y * 1000
        else:
            x_path=self.data_path[index]
            x=np.load(x_path)
            x=x.astype('uint8')

        if self.transform is not None:
            x = self.transform(x)
        if self.label and self.target_transform is not None:
            if self.label=='mask':
                y = self.target_transform(y)
            if self.label=='pred_label':
                y = y.astype(np.float64)
                # print(u_mask.shape)
                y = torch.from_numpy(y)
        if self.label:
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.data_path)


class MyDataSet(Dataset):
    """
    root: 图像存放根目录
    augment: 是否需要数据增强
    """

    def __init__(self, root, start, end, label=True, augment=None, transform=None, target_transform=None):
        # 获取数据地址
        self.label=label
        data_path = []
        for i in range(start, end):
            img = os.path.join(root + str(i) + '.npy')
            if label=='mask':
                mask = os.path.join(root + str(i) + '_mask.npy')
                data_path.append((img, mask))
            else:
                if label=='pred_label':
                    mask = os.path.join(root + str(i) + '_pred_label.npy')
                    data_path.append((img, mask))
                else:
                    data_path.append(img)
        self.data_path = data_path

        # 是否进行数据增强
        #self.augment = augment
        #
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        if self.label:
            x_path, y_path = self.data_path[index]
            x = np.load(x_path)  # .reshape(3,128,128)
            # print(x.shape)
            # x = np.pad(x,((14,14),(14,14),(0,0)))

            x = x.astype('uint8')
            y = np.load(y_path)  # .reshape(1,128,128)
            # print(y.shape)
            # y = np.pad(y, ((14, 14), (14, 14)))
            y = y * 1000
        else:
            x_path=self.data_path[index]
            # print(x_path)
            x=np.load(x_path)
            # print(x.shape)
            x=x[:,:,:3]
            # x = np.pad(x, ((14, 14), (14, 14), (0, 0)))
            x=x.astype('uint8')
            # imsave("check.png",x)

        if self.transform is not None:
            x = self.transform(x)
        if self.label and self.target_transform is not None:
            if self.label=='mask':
                y = self.target_transform(y)
            if self.label=='pred_label':
                y = y.astype(np.float64)
                # print(u_mask.shape)
                y = torch.from_numpy(y)
        if self.label:
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.data_path)
