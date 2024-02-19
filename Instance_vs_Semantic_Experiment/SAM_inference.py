"""
이 파일은 Paper Task2(Segmentation 비교)를 위해 짜여진 코드
prompt는 bbox로 들어간다.
사용된 image는 full size image를 사용해 inference한다. 

"""

import cv2
import pickle
import os
import numpy as np
from tqdm import tqdm 
from statistics import mean
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.functional import threshold, normalize
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from sklearn.metrics import f1_score, roc_auc_score,confusion_matrix
from PIL import Image

global labels

label_data_path = "/home/fisher/DATA/GMISSION/annotations/annotation_v3.pkl"
train_data_path = "/home/fisher/Peoples/suyeon/Paper/DATA/Test_data/img/"
mask_data_path = "/home/fisher/Peoples/suyeon/Paper/DATA/Test_data/mask/"
full_data_path = "/home/fisher/DATA/GMISSION/images/"
full_mask_path = "/home/fisher/DATA/GMISSION/masks/"
crop_mask_data_path = "/home/fisher/DATA/GMISSION/object_images/"
save_path = "/home/fisher/Peoples/suyeon/Paper/SAM/sam_task2_output/"

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    
    # 수정
    pos_points = pos_points[0]
    
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    

def apply_keypoint(keypoint): # 11,2,2
    coords = []
    for a in keypoint:
        if None not in a:
            coords.append(list(a))
    
    return np.array(coords)

def prepare_image(image, transform, device):
        image = transform.apply_image(image)
        image = torch.as_tensor(image, device=device.device) 
        # print(image)
        return image.permute(2, 0, 1).contiguous()
    
def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imshow(mask_image)
    
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

import pickle

# pickle 파일을 읽기 모드로 열기
with open('/home/fisher/Peoples/suyeon/Paper/test_bbox_annotation.pkl', 'rb') as file:
    # pickle 파일에서 객체 로드
    label_bbox = pickle.load(file)


# import pre-trained weight
sam_checkpoint = "/home/fisher/Peoples/suyeon/segment-anything-main/segment_anything/sam_vit_b_01ec64.pth"
model_type = "vit_b"

# Define device
device = "cuda:0"
        
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

sam.load_state_dict(torch.load('/home/fisher/Peoples/suyeon/segment-anything-main/notebooks/save_sam_weight/sam_weight_700'))

test_filename = sorted(os.listdir(full_data_path))[1:]
test17_filename = [filename.split('.JPG')[0] for filename in test_filename if "20220817" in filename]
test19_filename = [filename.split('.JPG')[0] for filename in test_filename if "20220819" in filename]
test_filename = test17_filename + test19_filename

# input datas 생성
from collections import defaultdict

global labels

images = {}
point_labels = {}
original_size = {}

transformed_data = defaultdict(dict)
print("Data Loading...")
for filename in test_filename:
    # input image 생성
    image = cv2.imread(full_data_path + filename + '.JPG')    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = resize_transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    
    input_image = sam.preprocess(transformed_image)
    original_image_size = image.shape[:2]
    input_size = tuple(transformed_image.shape[-2:])
    
    images[filename] = input_image
    
    # original size 생성
    original_size[filename] = image.shape[:2]
    
    
    # input_data 
    transformed_data[filename]['original_image'] = image
    transformed_data[filename]['image'] = input_image
    transformed_data[filename]['input_size'] = input_size
    transformed_data[filename]['original_image_size'] = original_image_size

print("Data Loading Success!")
# GT Mask data
ground_truth_masks = {}
for filename in images.keys():
    path = full_mask_path + filename + '.npy'
    gt_grayscale = np.load(path)
    ground_truth_masks[filename] = gt_grayscale

    
total_iou = 0
total_dice = 0
total_acc = 0
total_f1 = 0
count = 0

for filename in tqdm(test_filename):
    
    box = torch.tensor(label_bbox[filename], device=sam.device)
    image1 = transformed_data[filename]['original_image']
    original_size = transformed_data[filename]['original_image_size']
    
    batched_input = [
        {
            'image': prepare_image(image1, resize_transform, sam),
            'boxes': resize_transform.apply_boxes_torch(box, image1.shape[:2]),
            'original_size': image1.shape[:2]
        }
    ]

    batched_output = sam(batched_input, multimask_output=False)
    
    # Crop original image, ground truth mask image and predicted ground truth mask image 
    # current_image = transformed_data[filename]['original_image']
    gt_mask_image = ground_truth_masks[filename]
    for object_num, (object_coord, mask_image) in enumerate(zip(box, batched_output[0]['masks'])):
        count += 1
        x_min = int(object_coord[0])
        y_min = int(object_coord[1])
        x_max = int(object_coord[2])
        y_max = int(object_coord[3])
        
        # 1. Crop predicted mask image output from SAM -> binary_crop_mask_image
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        current_mask_image = mask_image.cpu().numpy()
        h, w = current_mask_image.shape[-2:]
        current_mask_image = current_mask_image.reshape(h, w, 1) * color.reshape(1, 1, -1)
        crop_current_mask_image = current_mask_image[y_min:y_max, x_min:x_max]
        # 1-1. Change demension 3D -> 2D
        temp = crop_current_mask_image.copy()
        temp = np.delete(temp, [0,1,2], axis=2)
        temp = np.squeeze(temp, axis=2)
        bool_temp = np.where(temp != 0, True, False) 
        binary_crop_mask_image = bool_temp*1
        
        # 2. Crop ground truth original image -> cv_image
        # pil_image = Image.fromarray(current_image)
        # crop_image = pil_image.crop((x_min, y_min, x_max, y_max))
        # numpy_image=np.array(crop_image)
        # cv_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        
        # 3. Crop ground truth mask image -> crop_gt_mask_image
        crop_gt_mask_image = gt_mask_image[y_min:y_max, x_min:x_max]
        # print(f"object_num+1 = {object_num+1}")
        # print(f"unique crop_gt_mask_image = {np.unique(crop_gt_mask_image)}")
        crop_gt_mask_image = np.where(crop_gt_mask_image == object_num+1, 1, 0)
        
        # Evaluate performance         
        target = crop_gt_mask_image
        prediction = binary_crop_mask_image
        
        iou_score = iou(target, prediction)
        dice_score = dice_coef(target, prediction)
        acc_score = accuracy(target, prediction)
        f1Score = f1_score(target, prediction, average='macro', zero_division=1).item()
        
        total_iou = total_iou + iou_score
        total_dice = total_dice + dice_score
        total_acc = total_acc + acc_score
        total_f1 = total_f1 + f1Score
        
        # Draw GT mask image and predicted mask image
        
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.set_title('target')
        ax1.imshow(target)
        ax1.axis('off')
        ax2.set_title('prediction')
        ax2.imshow(prediction)
        # draw_point = apply_keypoint(points)
        # show_points(draw_point, point_label.cpu().numpy().reshape(-1,), plt.gca())
        ax2.axis('off')
        
        # plt.show()
        f.savefig(f"{save_path}{filename}-{object_num}")      
final_iou = total_iou/count
final_dice = total_dice/count
final_acc = total_acc/count
final_f1 = total_f1/count

f1 = open("/home/fisher/Peoples/suyeon/segment-anything-main/notebooks/SAM_performance.txt", 'w')
f1.write(f"iou = {final_iou}\n")
f1.write(f"dice = {final_dice}\n")
f1.write(f"acc = {final_acc}\n")
f1.write(f"f1 = {final_f1}\n")
f1.close()

print(f"iou = {final_iou}\n")
print(f"dice = {final_dice}\n")
print(f"acc = {final_acc}\n")
print(f"f1 = {final_f1}\n")