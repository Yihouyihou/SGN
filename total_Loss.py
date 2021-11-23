from Regularizer import regularizeloss,MESA
import torch
import torch.nn as nn
import numpy as np

mse = nn.MSELoss()

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    def forward(self, pred_vector, img_vector, maps_vector,count_value,lamda, alfa,l_batch_size,u_batch_size,n_neighbors):
        patch_num = l_batch_size+u_batch_size
        regularizer = regularizeloss(pred_vector, maps_vector, img_vector,count_value,patch_num, n_neighbors)
        #print('有标签样本mse:',torch.sum(mse(maps_vector[:l_batch_size], pred_vector[:l_batch_size])))
        #print('无标签样本mse:',torch.sum(mse(maps_vector[l_batch_size:], pred_vector[l_batch_size:])))
        #print('流形:',regularizer)

        if u_batch_size==0 or alfa==1:
            #print('All labeled')
            result = torch.sum(mse(maps_vector, pred_vector)+regularizer * lamda)
        elif l_batch_size==0:
            print('All unlabeled')
            result = torch.sum(2*alfa*mse(maps_vector,pred_vector)+regularizer * lamda)
        else:
            #for i in range()
            result = torch.sum(mse(maps_vector[:l_batch_size], pred_vector[:l_batch_size])) +\
                 torch.sum(alfa*mse(maps_vector[l_batch_size:],pred_vector[l_batch_size:])+regularizer * lamda)
            
        return 10*result




class MESA_LOSS(nn.Module):
    def __init__(self):
        super(MESA_LOSS, self).__init__()

    def forward(self, pred, maps):
        # result = torch.sum(pred_patch-label_patch)+lamda
        # print(result)

        batch_size = 8
        patch_size = 64
        patch_num = int(pow(128 / patch_size, 2)) * batch_size
        loss = MESA(pred, maps, batch_size, patch_size, patch_num)
        loss = torch.from_numpy(np.array(loss))
        result = torch.sum(mse(maps, pred) + loss.float())

        return result
