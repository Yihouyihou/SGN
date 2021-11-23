import numpy as np
import matplotlib.pyplot as plt
from Regularizer import get_s, new_normalized_s
import os
import matplotlib
import torch
import cv2
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#c=np.load('C:/Users/zya/Desktop/BCFM/pretrain_labels/)
def get_canny(index,path,B_save_path,C_save_path,mode,type,B_minthreshold,B_maxthreshold,C_minthreshold,C_maxthreshold):
    if type=='pred_label':
        get_bina(index,path,B_save_path,B_minthreshold,B_maxthreshold)
        img = cv2.imread(B_save_path+str(index)+'_pred_label.png', 0)  # 由于Canny只能处理灰度图，所以将读取的图像转成灰度图
    else:
        img = cv2.imread(path+str(index)+'.png', 0)
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯平滑处理原图像降噪。若效果不好可调节高斯核大小
    canny = cv2.Canny(img, C_minthreshold, C_maxthreshold)  # 调用Canny函数，指定最大和最小阈值，其中apertureSize默认为3。
    if mode=='show':
        cv2.imshow('Canny', canny)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if mode=='save':
        if C_save_path:
            np.save(C_save_path + str(index) + '_pred_label.npy', canny)
            matplotlib.image.imsave(C_save_path + str(index) + '_pred_label.png', canny,cmap='Greys_r')
        else:
            print('save_path should not be empty')
    if mode=='train':
        return canny

        #print(im.shape)
'''
for i in range(12800):
    A=np.load('C:/Users/zya/Desktop/32_mask/'+str(i)+'_mask.npy')
    #A=A*100
    #A=A.astype('uint8')
    A=A[:,:,0]
    #plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(a)
    matplotlib.image.imsave('C:/Users/zya/Desktop/32_mask_jet/'+str(i)+'_mask.png', A, cmap='jet')
'''
def get_bina(index,path,bina_save_path,minthreshold,maxthreshold):
    img=np.load(path + str(index) + '_pred_label.npy')
    #print(img.shape)
    img=img[0,0,:,:]
    img=img*255/100
    retval, dst = cv2.threshold(img, minthreshold, maxthreshold, cv2.THRESH_BINARY)
    dst=dst.astype('uint8')
    #dst=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    #print(dst)
    matplotlib.image.imsave(bina_save_path+str(index)+'_pred_label.png', dst,cmap='Greys_r')

def canny_similarity(img_path,path,B_save_path,C_save_path,index):
    mse = nn.MSELoss()
    a = np.load(img_path+str(index)+'.npy')
    canny=get_canny(index,path,B_save_path,C_save_path,'train','pred_label',20,255,100,150)
    a = a.astype('float')
    a = torch.from_numpy(a / 255).cuda()
    #print('a', a.dtype)
    canny = canny.astype('float')
    canny = torch.from_numpy(canny / 255).cuda()
    return mse(canny,a)

#test=canny_similarity(img_path,path,B_save_path,None,0)
#print(test)


def old_LP(root, l_index, u_index, l_patch_num, u_patch_num, final_pred_vector,
       final_img_vector, final_count_value, n_neighbors, nita):  # indices(3200,15)
    patch_num = len(u_index) + len(l_index)  # final_pred_vector(3200,32*32)
    k = int(nita * patch_num)
    up_patch_index = []
    indices_ul = []  # 定义一个indices_ul对于每一个无标签样本对所有有标签样本的最近邻
    indices_lu = []  # 定义一个indices_lu对于每一个有标签样本对所有无标签样本的最近邻
    for i in range(len(u_index)):
        dic = {}
        for j in range(len(l_index)):
            index_i = i + (1 + i // u_patch_num) * l_patch_num
            index_j = j + (j // l_patch_num) * u_patch_num
            # print(index_i,index_j)
            dic[j] = get_s(final_pred_vector[index_i], final_pred_vector[index_j])
        sorted_key_list = sorted(dic, key=lambda x: dic[x])
        indices_ul.append(sorted_key_list[:n_neighbors])
        if (i + 1) % 512 == 0:
            print('已经为第' + str(i + 1) + '个无标记样本建立好indices')
    for i in range(len(l_index)):
        dic = {}
        for j in range(len(u_index)):
            index_j = j + (1 + j // u_patch_num) * l_patch_num
            index_i = i + (i // l_patch_num) * u_patch_num
            dic[j] = get_s(final_pred_vector[index_i], final_pred_vector[index_j])
        sorted_key_list = sorted(dic, key=lambda x: dic[x], reverse=True)
        indices_lu.append(sorted_key_list[:n_neighbors])

    dic = {}
    for i in range(len(l_index)):
        for j in range(len(u_index)):
            if j in indices_lu[i] or i in indices_ul[j]:
                dic[(i, j)] = new_normalized_s(i, j, indices_lu, indices_ul, l_patch_num, u_patch_num,
                                               final_pred_vector) \
                              * new_normalized_s(i, j, indices_lu, indices_ul, l_patch_num, u_patch_num,
                                                 final_img_vector) \
                              * new_normalized_s(i, j, indices_lu, indices_ul, l_patch_num, u_patch_num,
                                                 final_count_value)
            else:
                dic[(i, j)] = 0

    sorted_key_list = sorted(dic, key=lambda x: dic[x], reverse=True)
    index = 0
    while len(up_patch_index) < k:
        if u_index[sorted_key_list[index][1]] in up_patch_index:
            index = index + 1
        else:
            up_patch_index.append(u_index[sorted_key_list[index][1]])
            os.rename(root + str(u_index[sorted_key_list[index][1]]) + '_pred_label.npy',
                      root + str(u_index[sorted_key_list[index][1]]) + '_mask.npy')
            # a1=np.load(root+str(u_index[sorted_key_list[index][1]])+'_pred_label.npy')
            # np.save(root+str(u_index[sorted_key_list[index][1]])+'_mask.npy',a1)
            index = index + 1

    l_index = list(set(l_index) | set(up_patch_index))
    u_index = list(set(u_index) - set(up_patch_index))
    return l_index, u_index, len(l_index), len(u_index)


def LP(img_path,path,B_save_path,l_index,u_index,nita):
    patch_num = len(l_index)+len(u_index)  # final_pred_vector(3200,32*32)
    k = int(nita * patch_num)
    up_patch_index = []
    dic = {}
    for i in range(len(u_index)):
        index=u_index[i]
        dic[index]=canny_similarity(img_path,path,B_save_path,None,index)

    sorted_key_list = sorted(dic, key=lambda x: dic[x])
    #print(dic)
    #print(sorted_key_list)
    index=0
    while len(up_patch_index) < k:
        if sorted_key_list[index] in up_patch_index:
            index = index + 1
        else:
            up_patch_index.append(sorted_key_list[index])
            os.rename(path + str(sorted_key_list[index]) + '_pred_label.npy',
                      path + str(sorted_key_list[index]) + '_mask.npy')
            # a1=np.load(root+str(u_index[sorted_key_list[index][1]])+'_pred_label.npy')
            # np.save(root+str(u_index[sorted_key_list[index][1]])+'_mask.npy',a1)
            index = index + 1
    print(up_patch_index,len(up_patch_index))
    l_index = list(set(l_index) | set(up_patch_index))
    u_index = list(set(u_index) - set(up_patch_index))
    return l_index, u_index, len(l_index), len(u_index)

#a,b,c,d=LP(img_path,path,B_save_path,l_index,u_index,10,10,0.1)
#print(a,b,c,d)
"""
import time
start = time.clock()



end = time.clock()
print(str(end-start))

start1 = time.clock()

end1 = time.clock()
print(str(end1-start1))


"""