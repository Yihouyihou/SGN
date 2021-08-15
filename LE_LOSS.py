# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:21:56 2019

@author: ZHENGYUXIN
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

def splitPredPatch(pred,batch_size,patch_num,patch_size,img_size):    
    index = 0
    s = patch_size
    patch = np.zeros((patch_num,s,s))
    for i in range(pred.shape[0]):
        image = pred[i,:,:]
        for m in range(0,img_size,s):
            for n in range(0,img_size,s):
                image_patch = image[m:m+s,n:n+s]
                patch[index,:,:] = image_patch
                index +=1
    return patch
'''
def splitImgPatch(img,batch_size,patch_num,patch_size):    
    index = 0
    s = patch_size
    patch = np.zeros((patch_num,3,s,s))
    for i in range(img.shape[0]):
        image = img[i,:,:,:]
        for m in range(0,64,s):
            for n in range(0,64,s):
                image_patch = image[:,m:m+s,n:n+s]
                patch[index,:,:,:] = image_patch
                index +=1
    return patch
'''

def DataProcessing(pred,maps,batch_size,patch_size,patch_num,img_size):
    s = patch_size
    pred = pred.cpu()
    pred = pred.detach().numpy()
    maps = maps.cpu()
    maps = maps.detach().numpy()
    #img = img.cpu()
    #img = img.detach().numpy()
    #print("pred shape: ",pred.shape)
    ## LOAD SIMCEP LABEL AND FLATTEN AS VECTOR
    ### (num,128,128) --> (num,16384)
    
    data_len = pred.shape[0]
    pred = pred.reshape(data_len,img_size,img_size)
    maps = maps.reshape(data_len,img_size,img_size)
    #img = img.reshape(data_len,3,64,64)
    #print(data.shape)
    pred_patch = splitPredPatch(pred,batch_size,patch_num,patch_size,img_size)
    maps_patch = splitPredPatch(maps,batch_size,patch_num,patch_size,img_size)
    #img_patch = splitImgPatch(img,batch_size,patch_num,patch_size)
    #print("patch shape: ",patch.shape)
    c = pred_patch.shape[0]

    pred_vector = np.zeros((c,s*s))
    maps_vector = np.zeros((c,s*s))
    #img_vector = np.zeros((c,s*s*3))
    for i in range(c):
        pred_vector[i] = pred_patch[i].flatten()/100
        maps_vector[i] = maps_patch[i].flatten()/100
        #img_vector[i] = img_patch[i].flatten()
        
    return pred_vector, maps_vector

### Inter-Patch Manifold
def RegularizeLoss1(maps_vector,pred_vector,W,n_neighbors,patch_num,bandwidth):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(maps_vector)
    distances, indices = nbrs.kneighbors(maps_vector) #indices(n,n_neighbors)
    for i in range(patch_num):
      base = maps_vector[i]
      for j in range(0,n_neighbors):
        D_l2 = np.linalg.norm(base-maps_vector[indices[i,j]])
        D_l1 = np.linalg.norm(base-maps_vector[indices[i,j]],1)+1e-8
        W[i,indices[i,j]] = np.exp(-1*D_l2/D_l1)
        
    loss = np.zeros((patch_num,patch_num))
    for i in range(patch_num):
        for j in range(patch_num):
            loss[i,j] = np.linalg.norm(pred_vector[i]-pred_vector[j])*W[i,j]
    loss = np.sum(loss)/(patch_num*patch_num)
    
    return loss
    
def LE(pred,maps,img,batch_size,patch_size,patch_num,img_size,n_neighbors):
   
    pred_vector, maps_vector = DataProcessing(pred,maps,batch_size,patch_size,patch_num,img_size)
    #print(data_vector.shape)
    #print("data_vector shape: ",data_vector.shape)   

    W = np.zeros((patch_num,patch_num))
    
    loss_2 = RegularizeLoss1(maps_vector,pred_vector,W,n_neighbors,patch_num,2)

    
   
    return loss_2
         
    
mse = nn.MSELoss()

class LE_LOSS(nn.Module): 
    def __init__(self):
        super(LE_LOSS,self).__init__()
    def forward(self,pred,maps,img,lamda,batch_size,patch_size,img_size,n_neighbors):
        
        
        patch_num = int(pow(img_size/patch_size,2))*batch_size
        loss_2 = LE(pred,maps,img,batch_size,patch_size,patch_num,img_size,n_neighbors)

            
        #regular_b = lamda*mean_squared_error(Weigth,label_patch)
        #regu = (lamda*M).clone().detach().requires_grad_(True)
        result = torch.sum(mse(maps,pred)+loss_2*lamda)
        
        return result
  