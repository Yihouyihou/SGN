import torch
from UNet_pytorch import UNet
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from MyLoss import LE_LOSS
from dataset import MyDataSet
from DLP import LP
import random
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# ----------------------------- ready the dataset------------------------------
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transforms = transforms.ToTensor()

val_dataset = MyDataSet(root = 'UNLABELED_PATH',start=START,end=END,label=None,
                              transform=x_transforms,target_transform=y_transforms)

val_dataloader = DataLoader(val_dataset, batch_size=1)

batch_size = 16

unlabel_ratio=4

label_ratio=12

alfa = 0.2

epochs=[20]*20

epochs[0]=100

patch_size = 64

regularise_patch_size = 32

n_neighbors = 8

lamda = 0.2

beita=1

nita = 1/20

# ----------------------------- creat the UNet and training------------------------------

model = UNet(3, 1).to(device)
checkpoint = torch.load('MODEL_PATH')
model.load_state_dict(checkpoint)

model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = LE_LOSS()

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

l_index = [_ for _ in range(L_START,U_END)]

u_index = [_ for _ in range(U_START,U_END)]

l_length = len(l_index)

u_length = len(u_index)


root1 = 'LABELED_PATH'
root2 = 'UNLABELED_PATH'
canny_path = 'CANNY_PATH'
B_save_path = 'BINARY_PATH'

iteration = 0

while u_length > 0:
    model.train()
    min_error = float('inf')
    epoch_loss = np.zeros(epochs[iteration])
    random.shuffle(l_index)
    random.shuffle(u_index)

    print('unlabeled:' + str(u_length))
    print('labeled:' + str(l_length))

    k = int(u_length / unlabel_ratio)

    for epoch in range(epochs[iteration]):
        model.train()
        print('epoch {}'.format(epoch + 1))
        train_loss = 0.0

        for K in tqdm(range(k)):

            l_result = torch.zeros(1, 3, patch_size, patch_size)
            l_maps = torch.zeros(1, 1, patch_size, patch_size)
            u_result = torch.zeros(1, 3, patch_size, patch_size)
            u_maps = torch.zeros(1, 1, patch_size, patch_size)
            i_index = random.sample(l_index,label_ratio)
            for i in i_index:
                l_img = np.load(root1 + str(i) + '.npy')
                l_img = l_img.astype('uint8')
                l_img = x_transforms(l_img).unsqueeze(0)
                l_mask = np.load(root2 + str(i) + '_mask.npy')
                if l_mask.shape == (patch_size, patch_size, 1):
                    l_mask=l_mask
                    l_mask = y_transforms(l_mask).unsqueeze(0)
                else:
                    l_mask = l_mask.astype(np.float64)

                    l_mask = torch.from_numpy(l_mask)


                if i == i_index[0]:
                    l_result = l_img
                    l_maps = l_mask
                else:
                    l_result = torch.cat((l_result, l_img), dim=0)

                    l_maps = torch.cat((l_maps, l_mask), dim=0)

            for j in range(K * unlabel_ratio, (K + 1) *unlabel_ratio):

                u_img = np.load(root1 + str(u_index[j]) + '.npy')
                u_img = u_img.astype('uint8')
                u_img = x_transforms(u_img).unsqueeze(0)
                u_mask = np.load(root2 + str(u_index[j]) + '_pred_label.npy')
                u_mask = u_mask.astype(np.float64)

                u_mask = torch.from_numpy(u_mask)

                if j == K * unlabel_ratio:
                    u_result = u_img
                    u_maps = u_mask

                else:
                    u_result = torch.cat((u_result, u_img), dim=0)
                    u_maps = torch.cat((u_maps, u_mask), dim=0)


            imgs = torch.cat((l_result, u_result), dim=0)

            maps = torch.cat((l_maps, u_maps), dim=0)

            imgs = imgs.float().cuda()
            maps = maps.float().cuda()

            outputs=model(imgs)

            loss = criterion(outputs, maps, alfa, lamda,label_ratio,unlabel_ratio,regularise_patch_size,patch_size,n_neighbors)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('EPOCH: {:.1f}-------------------TRAIN LOSS: {:.6f}'.format(epoch + 1, train_loss / k))
        epoch_loss[epoch] = train_loss / k

    model.eval()
    for jj, val_data in enumerate(val_dataloader, 0):
        imgs = val_data
        imgs = imgs.float()

        imgs = imgs.cuda()


        pred = model(imgs)

        c = pred.cpu()
        c = c.detach().numpy()
        np.save(root2 + str(jj) + '_pred_label.npy', c)



    l_index, u_index, l_length, u_length = LP(canny_path, root2, B_save_path, l_index, u_index, nita)



    # ----------------------------- Save training model------------------------------
    np.save('SAVE_PATH' + str(iteration) + '_loss.npy', epoch_loss)
    torch.save(model.state_dict(),'SAVE_PATH' + str(iteration) + '.pth')
    iteration = iteration + 1

