import cv2
import sys
import os
import pickle
import numpy as np
from tqdm import tqdm 
import math
from statistics import mean
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.functional import threshold, normalize
from custom_build_sam import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

global labels 

# import pre-trained weight
sam_checkpoint = "/home/fisher/Peoples/suyeon/segment-anything-main/segment_anything/sam_vit_b_01ec64.pth"
model_type = "vit_b"

# Define device
device = "cuda:1"
        
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()

def apply_keypoint(keypoint): # 11,2,2
    coords = []
    for a, b in keypoint:
        if None not in a:
            coords.append(list(a))
        if None not in b:
            coords.append(list(b))
    
    return np.array(coords)

# mask-image를 시각화를 위한 함수 정의
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    
class SamDatahandler(Dataset):
    def __init__(self,
                 label: list,
                 train_data_path: str,
                 mask_data_path: str,
                 batch_size: int,
                 model
                 ):
        
        self.train_data_path = train_data_path
        self.train_data_path_list = sorted(os.listdir(self.train_data_path))[1:]
        self.mask_data_path = mask_data_path
        self.mask_data_path_list = sorted(os.listdir(mask_data_path))
        
        self.label_list = list(label.values())
        self.batch_size = batch_size
        self.model = model
        self.resize_transform = ResizeLongestSide(model.image_encoder.img_size)
        
        self.train_dataset = []
        
    def apply_keypoint(self, keypoint): # 11,2,2
        coords = []
        for a, b in keypoint:
            if None not in a:
                coords.append(list(a))
            if None not in b:
                coords.append(list(b))
        
        return np.array(coords)
        
    def prepare_image(self, image, transform, device):
        image = transform.apply_image(image)
        image = torch.as_tensor(image, device=device.device) 
        # print(image)
        return image.permute(2, 0, 1).contiguous()
    
    def __getitem__(self, idx):
            
        train_data = {}
        
        # Import image
        image = cv2.imread(self.train_data_path + self.train_data_path_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create batched_input
        train_data['original_image'] = image
        train_data['image'] = self.prepare_image(image, resize_transform, sam)
        mask_image = np.where(np.load(self.mask_data_path + self.mask_data_path_list[idx]) != 0, 1, 0)
        train_data['mask_image'] = torch.Tensor(mask_image).to(device)
        train_data['filename'] = self.train_data_path + self.train_data_path_list[idx]
        train_data['original_size'] = image.shape[:2]

        return train_data
    
    def __len__(self):
        
        return len(os.listdir(self.train_data_path)[1:])
    
    
label_data_path = "/home/fisher/DATA/GMISSION/annotations/annotation_v3.pkl"
train_data_path = "/home/fisher/DATA/GMISSION/images/"
mask_data_path = "/home/fisher/DATA/GMISSION/masks/"

with open(label_data_path,"rb") as fr:
            labels = pickle.load(fr)
            
label_list = list(labels.values())        

batch_size = 1

dataset = SamDatahandler(label=labels, 
                         train_data_path=train_data_path, 
                         mask_data_path=mask_data_path,
                         batch_size=batch_size,
                         model=sam)

train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=False)

# print(dataset[666])


# ******** Train ********
wd = 0
lr = 1e-4
mask_threshold = 0.0
num_epochs = 50
optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
loss_fn = torch.nn.MSELoss()
losses = []

save_count = 0
pred_loss = math.inf

for epoch in tqdm(range(num_epochs)):
    epoch_losses = []
    
    sam.train()
    for idx, batched_inputs in enumerate(train_loader):
        
        loss_sum = 0.0
        
        # Create batch dataset
        start_idx = idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        
        batch_input = []
        labels = label_list[start_idx:end_idx]
        
        idx_size = end_idx - start_idx
        for i in range(idx_size):
            data = {}
            data['image'] = batched_inputs['image'][i]
            data['original_size'] = (batched_inputs['original_size'][0][i], batched_inputs['original_size'][1][i])
            
            data['point_coords'] = apply_keypoint(labels[i]['keypoint'])
            data['point_coords'] = resize_transform.apply_coords(data['point_coords'], batched_inputs['original_image'][i].shape[:2])
            
            data['point_coords'] = torch.as_tensor(data['point_coords'], dtype=torch.float, device=device)
            data['point_labels'] = torch.as_tensor(np.ones(data['point_coords'].shape[0]), dtype=torch.int, device=device)
            
            data['point_coords'], data['point_labels'] = data['point_coords'][None, :, :], data['point_labels'][None, :]
            batch_input.append(data)
        
        ##################################################################################################                
        batched_output = sam(batch_input, multimask_output=True)
                
        
        # Loss function
        pred_binary_masks = []
        binary_masks = []

        for i in range(idx_size):
            pred_binary_masks.append(normalize(threshold(batched_output[i]['masks'][0][0].float().to(device), 0.0, 0)))
            binary_masks.append(batched_inputs['mask_image'][i])
    
        
        for predictions, targets in zip(pred_binary_masks, binary_masks):
            predictions.requires_grad_(True)
            # targets.requires_grad_(True)
            loss_sum += loss_fn(predictions.to(device), targets.to(device))
        
        
        average_loss = loss_sum/batch_size
        
        f3 = open(f"/home/fisher/Peoples/suyeon/segment-anything-main/weight/loss.txt", 'a')
        f3.writelines(f'{idx} || average_loss = {average_loss}\n')
        f3.close()
        
        print(average_loss)
        
        optimizer.zero_grad()
        average_loss.backward()
        optimizer.step()
        epoch_losses.append(average_loss.item())
        # print(f"Average_loss = {average_loss}")
        
        if pred_loss > average_loss:    
            torch.save(sam.state_dict(), f"/home/fisher/Peoples/suyeon/segment-anything-main/weight/save_weight{save_count}.pth")
            pred_loss = average_loss
            save_count = save_count + 1
    scheduler.step()
    losses.append(epoch_losses)
    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')

plt.figure(figsize=(10, 6))
for idx, epoch_loss in enumerate(losses):
    plt.plot(epoch_loss, label=f'Epoch {idx}')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid()
    plt.savefig('/home/fisher/Peoples/suyeon/segment-anything-main/loss_image/loss_curves.png') 
    plt.show()