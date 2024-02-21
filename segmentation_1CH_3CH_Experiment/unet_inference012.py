import os
os.environ['MPLBACKEND'] = 'TkAgg'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import time
import cv2
import pickle
# import matplotlib
from PIL import Image
from torchmetrics.classification import Dice,BinaryAccuracy
from torchmetrics import JaccardIndex
from sklearn.metrics import f1_score, roc_auc_score,confusion_matrix
from torchvision import transforms
from torch.nn import Softmax
from torchmetrics.classification import Dice, BinaryAccuracy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import segmentation_models_pytorch as smp
from tqdm import tqdm

global labels
device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data 불러오기
label_data_path = "./annotation_v3.pkl"
test_data_path = "./img/"
mask_data_path = "./mask/"
model_path = "./epoch_25.pth"

datasize = len(os.listdir(test_data_path))

def accuracy(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask==pred_mask)
    acc = np.mean(xor/(union + xor - intersect))
    return round(acc, 3)

def dice_coef(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)
    dice = np.mean(2*intersect/total_sum)
    return round(dice, 3) #round up to 3 decimal places

def iou(groundtruth_mask, pred_mask):
    intersection = np.logical_and(groundtruth_mask, pred_mask)
    union = np.logical_or(groundtruth_mask, pred_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return round(iou_score, 3)


class UnetDatahandler(Dataset):
    def __init__(self,
                 label: list,
                 test_data_path: str,
                 mask_data_path: str,
                 batch_size: int,
                 model
                 ):
        
        self.test_data_path = test_data_path
        self.test_data_path_list = sorted(os.listdir(self.test_data_path))
        self.filename_list = [t.split('.JPG')[0] for t in self.test_data_path_list if True]
        # temp 
        # self.filename_list = self.filename_list[1:]
        self.mask_data_path = mask_data_path
        
        self.label_list = list(label.values())
        
        self.batch_size = batch_size
        self.model = model
        self.test_dataset = []
    
    def __getitem__(self, idx):
            
        test_data = {}
        
        # Import image
        # filename = self.label_list[idx]['filename']
        image = cv2.imread(self.test_data_path + self.filename_list[idx] + '.JPG')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pytorch = image.transpose((2, 0, 1))
        # NumPy 배열을 PyTorch 텐서로 변환
        image_tensor = torch.from_numpy(image_pytorch).float()

        # 이미지를 정규화 (옵션)
        image_tensor /= 255.0
        
        # Create batched_input
        test_data['original_image'] = image
        test_data['image'] = image_tensor
        test_data['original_size'] = image.shape[:2]
        
        # Create batched mask input data
        mask = np.load(self.mask_data_path + self.filename_list[idx] + '.npy')
        mask = np.where(mask != 0, 1, mask)
        
        test_data['mask'] = mask
        test_data['mask_size'] = mask.shape[:2]

        return test_data
    
    def __len__(self):
        print(f'Batched input length: {self.batch_size}')
        print(f'Test dataset size: {len(os.listdir(self.test_data_path))}')
        
        return len(os.listdir(self.test_data_path))

model = smp.Unet(encoder_weights = 'imagenet',
                 in_channels=3,            # 입력 이미지 채널 수 (예: RGB 이미지는 3)
                 classes=3)

model = torch.load(model_path)

# 모델의 파라미터를 디바이스로 이동
model.to(device)

with open(label_data_path,"rb") as fr:
            labels = pickle.load(fr)
            
label_list = list(labels.values())        


batch_size = 32
dataset = UnetDatahandler(label=labels, 
                          test_data_path = test_data_path, 
                          mask_data_path = mask_data_path,
                          batch_size = batch_size, 
                          model=model)
test_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=False)

for param in model.parameters():
    param.requires_grad = True


threshold = 0.5

acc = BinaryAccuracy().to(torch.device('cuda:2'))
dice= Dice(average='micro').to(device)
jaccard = JaccardIndex(task='binary', num_classes=1).to(device)

total_jaccard = 0
total_dice = 0
total_acc = 0
total_f1 = 0
count = 0

model.eval()
with torch.no_grad():
    for idx, batched_inputs in tqdm(enumerate(test_loader), desc='inner', position=1, leave=False):
        input_image = batched_inputs['image']
        input_mask = batched_inputs['mask']
        
        input_image = input_image.to(torch.float)
        input_mask = input_mask.to(torch.float)
        
        input_image = input_image.to(device)
        input_mask = input_mask.to(device)
        
        outputs = model(input_image)
        
        for n in range(outputs.shape[0]):
            count = count + 1
            GT_mask = input_mask[n]
            outputs = (outputs >= threshold).float()
            # print( "1: "+ str(outputs[n].shape))
            output = outputs[n].permute((1, 2, 0))
            
            channel_0 = output[:, :, 0]
            channel_1 = output[:, :, 1]
            channel_2 = output[:, :, 2]
            
            # %matplotlib inline
            # plt.imshow(channel_1, cmap='gray')  # 'cmap' 매개변수로 색상 맵 설정 (흑백 이미지의 경우 'gray')
            # plt.axis('off')  # 축 제거 (선택 사항)

            # # 이미지를 화면에 보여주기
            # plt.show()
            
            # print("2: "+ str(GT_mask.shape))
            # print("3: " + str(channel_1.shape))
            
            target = GT_mask.cpu().numpy()
            pred = channel_1.cpu().numpy()
            
            jaccard_score = iou(target, pred)
            dice_score = dice_coef(target, pred)
            acc_score = accuracy(target, pred)
            f1 = f1_score(target, pred, average='macro', zero_division=1).item()
            
            total_jaccard = total_jaccard + jaccard_score
            total_dice = total_dice + dice_score
            total_acc = total_acc + acc_score
            total_f1 = total_f1 + f1
            
final_jaccard = total_jaccard/count
final_dice = total_dice/count
final_acc = total_acc/count
final_f1 = total_f1/count

f1 = open("./Unet_compare_01_012_result.txt", 'w')
f1.write(f"jaccard = {final_jaccard}\n")
f1.write(f"dice = {final_dice}\n")
f1.write(f"acc = {final_acc}\n")
f1.write(f"f1 = {final_f1}\n")
f1.close()

print(f"jaccard = {final_jaccard}\n")
print(f"dice = {final_dice}\n")
print(f"acc = {final_acc}\n")
print(f"f1 = {final_f1}\n")
