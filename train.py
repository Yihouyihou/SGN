from net.UNet_pytorch import UNet

import torch

import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import total_Loss
import numpy as np
from tqdm import tqdm
from dataset import MyDataSet
from LE_LOSS import LE_LOSS
from net import layers
from Regularizer import get_countvalue,get_nbrs,normalized_s,get_s
from total_Loss import MyLoss
from DLP import LP
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#----------------------------- ready the dataset------------------------------
# 将数据转化为张量
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transforms = transforms.ToTensor()

label_train_dataset = MyDataSet(root='C:/Users/zya/Desktop/dataset/dataset_aug/', start=0, end=1536,label=True,
                              transform=x_transforms, target_transform=y_transforms)
unlabel_train_dataset=MyDataSet(root='C:/Users/zya/Desktop/dataset/dataset_aug/',start=333,end=1515,label=None,
                                transform=x_transforms,target_transform=None)

val_dataset = MyDataSet(root='C:/Users/zya/Desktop/dataset/dataset_aug/', start=1600, end=3200,
                            transform=x_transforms, target_transform=y_transforms)


batch_size = 10
patch_num=3200
alfa = 0.2
epochs = 55
patch_size = 32
n_neighbors = 5
lamda=0.5
nita=0.1
epoch_loss = np.zeros(epochs)
#直接进行分patch 每个patch 32*32

# ----------------------------- creat the UNet and training------------------------------

model = UNet(3,1).to(device)

checkpoint = torch.load('D:/zhuyuang/results/MRRN_results/pretrain_2.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
#model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = MyLoss()
#criterion = LE_LOSS()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# 直接进行分patch 每个patch 32*32
# 一共12800的个patch,取百分之25进行训练,也就是3200个patch 其中有20%有标记样本(640) 无标记(2560)
l_index = [i for i in range(640)]        #存放所有有标记样本patch的索引

l_length = len(l_index)
u_index = [i for i in range(640, 3200)]       #存放所有无标记样本patch的索引

u_length = len(u_index)
k = int(patch_num / batch_size)          #patch_num为patch总数,每次取batch_size个样本进行训练,需要k=160次遍历全部样本
root1 = 'D:/zhuyuang/datasets/BCFM/32_img/'
root2 = 'D:/zhuyuang/datasets/SMRRN_dataset_10_1/'
img_path='D:/zhuyuang/datasets/BCFM/32_img_canny/'
B_save_path='D:/zhuyuang/projects/SMRRN/myproject/pred_labels_bina_1/'

iteration=0

while u_length>0:
    random.shuffle(l_index)
    random.shuffle(u_index)
    print('无标记样本数量:'+str(u_length))            # 当存放无标记样本的列表还有值时,就继续训练
    print('有标记样本数量:'+str(l_length))
    #print('无标记样本索引:',u_index)
    #print('有标记样本索引:',l_index)
    nothing=-1
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        train_loss = 0.0
        #final_count_value = torch.zeros(10, 1,dtype=torch.float64)                   # final_count_value存放所有样本的估计值 最终大小为(patch_num,1) 
        for K in range(k):
            if (K+1)%80==0 or K==0:
                print('此时正在取出第'+str(K+1)+'批样本')
            l_result = torch.zeros(1, 3, patch_size, patch_size)
            l_maps=torch.zeros(1,1,patch_size,patch_size)
            u_result = torch.zeros(1, 3, patch_size, patch_size)
            u_maps = torch.zeros(1, 1, patch_size, patch_size)
            for i in range(K * (l_length // k), (K + 1) * (l_length // k)):  # 当k为0时 i in range(0,1)
                                                                           # 当K为1时, i in range(1,2)
                l_img = np.load(root1 + str(l_index[i]) + '.npy')  # 32*32*3
                l_img=l_img.astype('uint8')
                l_img=x_transforms(l_img).unsqueeze(0)             # 1*3*32*32
                l_mask = np.load(root2 + str(l_index[i]) + '_mask.npy')  # 32*32*1
                if l_mask.shape==(patch_size,patch_size,1):
                    l_mask = y_transforms(l_mask).unsqueeze(0)  # 1*1*32*32
                else:
                    l_mask = l_mask.astype(np.float64)
                    # print(u_mask.shape)
                    l_mask = torch.from_numpy(l_mask)
                #l_mask=y_transforms(l_mask).unsqueeze(0)            # 1*1*32*32
                
                if i == K*(l_length // k):
                    l_result = l_img  # 如果是第一张图或者只需要取出一张图,那么result就为此图
                    l_maps=l_mask
                else:
                    l_result = torch.cat((l_result, l_img), dim=0)  # 如果不是第一张图,则把此图与前面的图拼接
                    l_maps=torch.cat((l_maps,l_mask),dim=0)
            # l_result为每次取出的有标记样本的总和 如果每次取出一个 那么l_result为(1,3,32,32)
            # l_maps为每次取出的有标记样本的标签的总和 如果每次取出一个 那么l_maps为(1,1,32,32)
            for j in range(K * (u_length // k), (K + 1) * (u_length // k)):  # 当K为0时 i in range(0,9)
                                                                        # 当K为1时, i in range(9,18)
                u_img = np.load(root1 + str(u_index[j]) + '.npy')  # 32*32*3
                u_img=u_img.astype('uint8')
                u_img=x_transforms(u_img).unsqueeze(0)  # 1*1*32*32
                u_mask = np.load(root2 + str(u_index[j]) + '_pred_label.npy')  # 1*1*32*32
                u_mask=u_mask.astype(np.float64)
                #print(u_mask.shape)
                u_mask = torch.from_numpy(u_mask)  # 1*1*32*32
                #u_mask=u_mask.unsqueeze(0)
                #print(u_mask.shape)
                if j == K * (u_length // k):
                    u_result = u_img  # 如果是第一张图或者只需要取出一张图,那么result就为此图
                    u_maps = u_mask
                    #print(u_maps.shape)
                else:
                    u_result = torch.cat((u_result, u_img), dim=0)  # 如果不是第一张图,则把此图与前面的图拼接 最终为9*3*32*32
                    u_maps = torch.cat((u_maps, u_mask), dim=0)
                    #print(u_maps.shape)
    
            # u_result为每次取出的无标记样本的总和 如果每次取出9个 那么u_result为(9,3,32,32)
            # u_maps为每次取出的无标记样本的伪标签的总和 如果每次取出9个 那么u_maps为(9,1,32,32)
            imgs = torch.cat((l_result, u_result), dim=0)        # (10,3,32,32)
            #print(l_maps.shape)
            #print(u_maps.shape)
            maps = torch.cat((l_maps, u_maps), dim=0)          # (10,1,32,32)

            imgs=imgs.float().cuda()
            maps=maps.float().cuda()
            #print(imgs.shape)
            #print(maps.shape)
            outputs = model(imgs)            #(10,1,32,32)
            #print(outputs.shape)
            #print(outputs)
           
            if epoch==epochs-1:
                for bs in range(l_length // k,batch_size):           # (3,10)      (0-11)  (12-23) ()
                    a = outputs[bs, :, :, :].unsqueeze(0).cpu()     
                    a=a.detach().numpy()
                    order=1+bs-(l_length//k)
                    #print(nothing+order)
                    np.save(root2+str(u_index[nothing+order])+'_pred_label.npy',a)
                nothing=nothing+order
            
            count_value=get_countvalue(outputs)
            #if epoch==epochs-1:
                #print(outputs.shape)
                #print(count_value)

            indices=get_nbrs(n_neighbors,batch_size,outputs)
            #print(outputs.requires_grad)

            loss = criterion(outputs, imgs, maps,count_value,
                             lamda, alfa, l_length // k, u_length // k,n_neighbors)            # 计算loss
          
            #print(train_loss)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()  # loss求导
            optimizer.step()  # 更新参数

            #if (i + 1) % 8 == 0:
                #print('-', end='')
        # 当这个for循环读完意味着所有的样本都输入了网络,可以结束第一个epoch
        print('EPOCH: {:.1f}-------------------TRAIN LOSS: {:.6f}'.format(epoch+1, train_loss /k))
        if u_length==320:
            epoch_loss[epoch] = train_loss /k
        #print(model.state_dict())
    #print('final_pred:',final_pred.shape)
    #print('final_img:',final_img.shape)
    #print('final_count:',final_count)

    print('标签扩散进行中...')

    l_index,u_index,l_length,u_length=LP(img_path,root2,B_save_path,l_index,u_index,nita)
    print('扩散完成')
    # LP函数过后将产生新的有/无标记样本索引列表 以及有/无标记样本的数量(也就是列表长度)
    # 根据设置的nita可以得出需要的总轮次为((1/nita)*epochs)

# ----------------------------- Save training model------------------------------
#torch.save(model, 'C:/Users/zya/Desktop/dataset/pretrain.pth')
    torch.save({'epoch': epochs,'alfa':alfa,'lamda':lamda, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
               './train_result/SMRRN_PDL1_three/SMRRN_PDL1_'+str(iteration)+'.pth.tar')
    np.save('./train_result/SMRRN_PDL1_three/SMRRN_PDL1_'+str(iteration)+'_loss.npy', epoch_loss)
    iteration=iteration+1



