import numpy as np
import torch

def get_s(arg1,arg2):
    '''
    x_norm = np.linalg.norm(x, ord=None, axis=None, keepdims=False)

    axis=1 表示按行向量处理,求多个行向量的范数
    axis=0 表示按列向量处理，求多个列向量的范数
    axis=None 表示矩阵范数
    keepding：是否保持矩阵的二维特性 True表示保持矩阵的二维特性，False相反

    '''
    arg1=arg1.cpu().detach()
    arg2=arg2.cpu().detach()
    args_norm = np.linalg.norm(arg1-arg2)
    arg1_norm = np.linalg.norm(arg1)
    arg2_norm = np.linalg.norm(arg2)
    if arg1_norm==0 or arg2_norm==0:
        return 0
    return np.exp(-1*args_norm/arg1_norm/arg2_norm)

def new_get_s(arg1,arg2):
    '''
    x_norm = np.linalg.norm(x, ord=None, axis=None, keepdims=False)

    axis=1 表示按行向量处理,求多个行向量的范数
    axis=0 表示按列向量处理，求多个列向量的范数
    axis=None 表示矩阵范数
    keepding：是否保持矩阵的二维特性 True表示保持矩阵的二维特性，False相反

    '''

    args_norm = torch.norm(arg1-arg2)
    arg1_norm = torch.norm(arg1)
    arg2_norm = torch.norm(arg2)
    if arg1_norm==0 or arg2_norm==0:
        return 0
    return torch.exp(-1*args_norm/arg1_norm/arg2_norm)

# image_size:64  patch_size:32  patch_num=((image_size/patch_size)**2)*batch_size
def splitPredPatch(pred, patch_num, patch_size, img_size):
    index = 0
    s = patch_size
    patch = np.zeros((patch_num, s, s))
    for i in range(pred.shape[0]):
        image = pred[i, :, :]
        for m in range(0, img_size, s):
            for n in range(0, img_size, s):
                image_patch = image[m:m + s, n:n + s]
                patch[index, :, :] = image_patch
                index += 1
    return patch

def splitImgPatch(img, patch_num, patch_size, img_size):
    index = 0
    s = patch_size
    patch = np.zeros((patch_num,3,s,s))
    for i in range(img.shape[0]):
        image = img[i,:,:,:]
        for m in range(0,img_size,s):
            for n in range(0,img_size,s):
                image_patch = image[:,m:m+s,n:n+s]
                patch[index,:,:,:] = image_patch
                index +=1
    return patch

def DataProcessing(pred, maps,final_maps,img, patch_size):
    s = patch_size
    pred = pred.cpu()
    pred = pred.detach().numpy()
    maps = maps.cpu()
    maps = maps.detach().numpy()
    final_maps=final_maps.cpu().detach().numpy()
    img = img.cpu()
    img = img.detach().numpy()
    # print("pred shape: ",pred.shape)
    ## LOAD SIMCEP LABEL AND FLATTEN AS VECTOR
    ### (num,128,128) --> (num,16384)

    data_len = pred.shape[0]
    patch_num=final_maps.shape[0]
    pred = pred.reshape(data_len, patch_size, patch_size)
    maps = maps.reshape(data_len, patch_size, patch_size)
    final_maps=final_maps.reshape(patch_num,patch_size,patch_size)
    #img = img.reshape(data_len,3,img_size,img_size)
    # print(data.shape)
    '''
    pred_patch = splitPredPatch(pred, patch_num, patch_size, img_size) #32*32*32
    maps_patch = splitPredPatch(maps, patch_num, patch_size, img_size)
    img_patch = splitImgPatch(img, patch_num, patch_size, img_size)
    '''
    c = pred.shape[0] #c=32 s=32

    pred_vector = np.zeros((c, s * s))
    maps_vector = np.zeros((c, s * s))
    final_maps_vector=np.zeros((patch_num,s*s))
    img_vector = np.zeros((c,s*s*3))
    count_value=np.zeros((c,s))
    for i in range(c):
        pred_vector[i] = pred[i].flatten() / 100
        maps_vector[i] = maps[i].flatten() / 100
        img_vector[i] = img[i].flatten()
    for j in range(patch_num):
        final_maps_vector[j]=final_maps_vector[j].flatten()/100

    count_value=np.sum(pred_vector,axis=1)
    return pred_vector, maps_vector,final_maps_vector,count_value

def ConvertToVector(pred,img,maps):
    pred = pred.cpu()
    pred = pred.detach().numpy()
    maps = maps.cpu()
    maps = maps.detach().numpy()
    #final_maps = final_maps.cpu().detach().numpy()
    img = img.cpu()
    img = img.detach().numpy()

    c = pred.shape[0]
    s = pred.shape[-1]
    pred = pred.reshape(c, s, s)
    maps = maps.reshape(c ,s, s)
    img=img.reshape(c,3,s,s)

    pred_vector = np.zeros((c, s * s))
    maps_vector = np.zeros((c, s * s))
    img_vector = np.zeros((c, s * s * 3))
    count_value = np.zeros((c, s))
    for i in range(c):
        pred_vector[i] = pred[i].flatten()
        maps_vector[i] = maps[i].flatten()
        img_vector[i] = img[i].flatten()

    count_value = np.sum(pred_vector, axis=1)

    return pred_vector, img_vector,maps_vector,count_value

def get_countvalue(pred):
    #print(pred.shape)
    pred = pred.cpu()
    pred = pred.detach().numpy()
    c = pred.shape[0]
    s = pred.shape[-1]
    pred = pred.reshape(c, s, s)
    pred_vector = np.zeros((c, s * s))
    #print('pred_vector:',pred_vector.shape)
    for i in range(c):
        pred_vector[i] = pred[i].flatten()
    count_value = np.sum(pred_vector, axis=1)
    count_value=torch.from_numpy(count_value)
    count_value=count_value.unsqueeze(1)
    return count_value/1000

def new_get_countvalue(pred):
    #print(pred.shape)
    c = pred.shape[0]
    s = pred.shape[-1]
    pred = pred.squeeze(1).resize(c, s * s)

    #print('pred_vector:',pred_vector.shape)
    count_value = torch.sum(pred, axis=1)
    count_value=count_value.unsqueeze(1)
    return count_value/100

def trace(matrix):
    l1=len(matrix)
    l2=len(matrix[0])
    if l1!=l2:
        return 'require a matrix having same number of rows and columns'
    sum=0
    for i in range(l1):
        sum=sum+matrix[i][i]
    return sum

def calLaplacianMatrix(adjacentMatrix):

    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

def new_calLaplacianMatrix(adjacentMatrix):

    # compute the Degree Matrix: D=sum(A)
    # degreeMatrix = np.sum(adjacentMatrix, axis=1)
    degreeMatrix = torch.sum(adjacentMatrix, dim=1)

    # compute the Laplacian Matrix: L=D-A
    # laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix
    laplacianMatrix = torch.diag(degreeMatrix) - adjacentMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    # sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    sqrtDegreeMatrix = torch.diag(1.0 / (degreeMatrix ** (0.5)))
    # return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
    return torch.matmul(torch.matmul(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

def new_normalized_s(index1,index2,indices_lu,indices_ul,l_patch_num,u_patch_num,vector):
    sum1=sum2=0
    f_index1 = index1 + (index1 // l_patch_num) * u_patch_num
    f_index2 = index2 + (1 + index2 // u_patch_num) * l_patch_num

    for i in range(len(indices_lu[index1])):
        f_index=indices_lu[index1][i]+(1 + indices_lu[index1][i] // u_patch_num) * l_patch_num
        sum1=sum1+get_s(vector[f_index1],vector[f_index])
    for j in range(len(indices_ul[index2])):
        f_index = indices_ul[index2][j] + (indices_ul[index2][j] // l_patch_num) * u_patch_num
        sum2=sum2+get_s(vector[f_index2],vector[f_index])
    if sum1==0 or sum2 ==0:
        return 0
    return get_s(vector[index1],vector[index2])/sum1/sum2

def get_nbrs(n_neighbors,batch_size,pred):
    '''
    n_neighbors:int  最近邻的数量
    pred_vector: 用于计算相似度的预测patch
    patch_num: 预测patch的总数
    '''
    indices=[]                 # 定义一个indices列表存储最后返回的最近邻的索引
    for i in range(batch_size):
        dic={}                     # 对每一个patch建立一个字典存储键值对
        for j in range(batch_size):            # 对于所有patch都需要遍历一遍
            #print(i,j)
            dic[j]=get_s(pred[i],pred[j])            # 按照{0:S(pi,p0),1:S(pi,p0)}的形式存储
        sorted_key_list = sorted(dic, key=lambda x: dic[x],reverse=True)
        #print(dic)
        indices.append(sorted_key_list[:n_neighbors])          #indices应为一个二维列表
    return indices

#pred_vector (32,32*32) so is maps_vector
def normalized_s(index1,index2,indices,vector):
    sum1=sum2=0
    for i in range(len(indices[index1])):
        sum1=sum1+get_s(vector[index1],vector[indices[index1][i]])
    for j in range(len(indices[index2])):
        sum2=sum2+get_s(vector[index2],vector[indices[index2][j]])
    if sum1==0 or sum2 ==0:
        return 0
    return get_s(vector[index1],vector[index2])/sum1/sum2

def get_RegularizeLoss(pred,img,maps,count_value,W, n_neighbors, batch_size):
    indices= get_nbrs(n_neighbors,batch_size,pred)

    for i in range(batch_size):
        for j in range(0, n_neighbors):
            W[i][indices[i][j]] = normalized_s(i,indices[i][j],indices,maps)\
                                  *normalized_s(i,indices[i][j],indices,img)\
                                  *normalized_s(i,indices[i][j],indices,count_value)

    regularizer = np.zeros((batch_size, batch_size))
    pred=pred.cpu().detach()
    for i in range(batch_size):
        for j in range(batch_size):
            if W[i][j]==0:
                regularizer[i][j]=0
            else:
                regularizer[i][j] = np.linalg.norm(pred[i] - pred[j]) * W[i][j]
    regularizer = np.sum(regularizer)

    return regularizer

def new_get_RegularizeLoss(pred,img,maps,count_value,W, n_neighbors, batch_size):
    indices= get_nbrs(n_neighbors,batch_size,pred)

    for i in range(batch_size):
        for j in range(0, n_neighbors):
            W[i][indices[i][j]] = normalized_s(i,indices[i][j],indices,maps)\
                                  *normalized_s(i,indices[i][j],indices,img)\
                                  *normalized_s(i,indices[i][j],indices,count_value)

    #regularizer = torch.zeros((batch_size, batch_size))
    L=new_calLaplacianMatrix(W)
    L = L.float().cuda()
    pred = pred.squeeze(1).resize(10, 32 * 32)
    pred_T = pred.transpose(1, 0)
    # print(pred.dtype,L.dtype)
    regularizer = torch.matmul(torch.matmul(pred_T, L), pred)

    regularizer = trace(regularizer)

    regularizer = torch.sum(regularizer)

    return regularizer

def regularizeloss(pred, img,maps,count_value, batch_size, n_neighbors):
    W = torch.zeros((batch_size, batch_size))

    regularize_loss = get_RegularizeLoss(pred, img,maps, count_value,W, n_neighbors, batch_size)
    #new_regularize_loss=new_get_RegularizeLoss(pred, img,maps, count_value,W, n_neighbors, batch_size)

    return regularize_loss


def MESA(pred, maps, batch_size, patch_size, patch_num):
    pred_vector, maps_vector = DataProcessing(pred, maps, batch_size, patch_size, patch_num)
    c = pred_vector.shape[0]
    maxcount = 0
    for i in range(c):
        temp = np.absolute(np.sum(pred_vector[i] - maps_vector[i]))

        if maxcount < temp:
            maxcount = temp
    return maxcount

