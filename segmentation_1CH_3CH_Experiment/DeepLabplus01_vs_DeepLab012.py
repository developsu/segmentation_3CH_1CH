from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Softmax
from torchmetrics.classification import Dice,BinaryAccuracy
from torchmetrics import JaccardIndex
from sklearn.metrics import f1_score, roc_auc_score,confusion_matrix
import time
# img_path ='/home/fisher/Peoples/hseung/Full/img/20220816_100GOPRO_G0020073.JPG'
# '/home/fisher/Peoples/hseung/NEW/Train/img_same_no_pad/20220816_100GOPRO_G0050536_3.jpg'
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# mask_files = 

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
start = time.time()
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

deeplab = torch.load(
                   r'/home/fisher/Peoples/hseung/NEW/1st_Trial/dataset수정/no_pad_3class_3_zero_to_three_dataset3.pt',map_location=torch.device('cuda:2'))
deeplabv3plus = torch.load(r'/home/fisher/Peoples/hseung/SMP_PYROCH/exp/deeplabplus_30.pt',map_location=torch.device('cuda:2'))
deeplabv3plus.eval()
deeplab.eval()

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Softmax
from torchmetrics.classification import Dice,BinaryAccuracy
from torchmetrics import JaccardIndex
from sklearn.metrics import f1_score, roc_auc_score,confusion_matrix
import os


img_folder =r'/home/fisher/Peoples/hseung/Full/img' 
os.chdir(img_folder)
filenames = os.listdir()
test_file = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), filenames))) 
# len(test_file)
mask_folder = r'/home/fisher/Peoples/hseung/Full/mask'
os.chdir(mask_folder)
mask_names = os.listdir()
mask_file = sorted(list(filter(lambda x: ('20220817' in x) or ('20220819' in x), mask_names)))

metric_df = pd.DataFrame(columns=['File Name','Deep Lab Accuarcy', 'Deep Plus Accuarcy', 'Deep DIce', 'Plus Dice', 
                                  'Deep Jaccard', 'Plus Jaccard', 'Deep f1', 'Plus F1'])# metric_df = pd.DataFrame(columns = ['Deep Lab Accuarcy','Mask Accuarcy', 'Deep DIce', 'Mask Dice', 'Deep Jaccard', 'Mask Jaccard', 'Deep f1', 'Mask F1'])
for n in range(len(test_file)):
     metric_dict = {}
     img_path = img_folder + "/"+ test_file[n]
     mask_path = mask_folder +"/" + mask_file[n]
     mask_npy = np.load(mask_path)
     mask=mask_npy
     obj_ids = np.unique(mask)
     # first id is the background, so remove it
     obj_ids = obj_ids[1:]

     # split the color-encoded mask into a set
     # of binary masks
     masks = mask == obj_ids[:, None, None]


     # get bounding box coordinates for each mask
     num_objs = len(obj_ids)
     # save_name =test_file[n].split('.')[0]
     boxes = []
     for i in range(num_objs):
          pos = np.where(masks[i])
          xmin = np.min(pos[1])
          xmax = np.max(pos[1])
          ymin = np.min(pos[0])
          ymax = np.max(pos[0])
          boxes.append([xmin, ymin, xmax, ymax])
          # 박스 불러오기
     #이건 전체 이미지
     img = Image.open(img_path).convert('RGB')
     image_array = np.array(img)
     for m in range(num_objs):
        boxes[m]
        target_size = (img.size[1],img.size[0])
        rcnn_trial = np.zeros(target_size,dtype=int)
        rcnn_trial[boxes[m][1]:boxes[m][3],boxes[m][0]:boxes[m][2]]=1
        mask_trial = np.stack([rcnn_trial,rcnn_trial,rcnn_trial], axis=-1)
        
        #여기선 box로 줬어
        masked_image = image_array * mask_trial


        masked_images = Image.fromarray(masked_image.astype('uint8'))
        img_trans = transforms.ToTensor()(masked_images).unsqueeze(0)
        deeplabv3plus.to(torch.device('cuda:2'))
        deeplab.to(torch.device('cuda:2'))

        coords = boxes[m]

        ground_truth = Image.fromarray((masks[m]).astype(np.uint8))
        ground_mask = ground_truth.crop(coords)

        new_image = masked_images.crop(coords)
        new_image = new_image.resize((256,256))
        # new_image= new_image.convert('RGB')
        newimg_trans = transforms.ToTensor()(new_image).unsqueeze(0)
        with torch.no_grad():
            b = deeplab(newimg_trans.to(torch.device('cuda:2')))
            c = deeplabv3plus(newimg_trans.to(torch.device('cuda:2')))
        bb = np.argmax(np.squeeze(b['out']).cpu().numpy(), 0)
        cc = c.cpu().numpy()
        deeplab_output = (bb==1)
        original_output = cc[0][0]>0.5
        image_list = [new_image,ground_mask.resize((256,256)),original_output, deeplab_output]
        ground_resize = ground_mask.resize((256,256))
        ground_numpy= np.array(ground_resize)
        # crop_original = np.array(original_output)
        # 제목 리스트
        titles = ['Crop_Image','Crop_Ground','plus','1_channel']
        num_subplots = len(image_list)

        # 서브플롯 배치 설정
        rows = 1
        cols = num_subplots

        # 서브플롯 생성
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
        name = test_file[n]

        #   각 이미지와 제목을 서브플롯에 추가
        for i, ax in enumerate(axes):
            ax.imshow(image_list[i])
            ax.set_title(titles[i])
            ax.axis('off')
            ax.set_adjustable('box')  # 이미지 비율 유지
          
# # 제목 간 간격 조정
        plt.subplots_adjust(top=0.8)

        # plt.savefig(f'/home/fisher/Peoples/hseung/NEW/segment outputs/deeplabv3plus_30_01/{name}_{m}.png')
          
        #metric time
        deep_pred= np.where(deeplab_output ==True, 1 ,0)
        truee = np.where(ground_numpy>0, 1,0)
        # mask_true= np.where(original_output==True,1,0)
        mask_pred= np.where(original_output==True,1,0)
        
        # mask_pred= np.where(crop_rcnn_numpy==True, 1, 0)
        pred_t = deep_pred
        truee_t = truee
        # truee_mask = torch.tensor(mask_true)
        mask_pred_t = mask_pred
        acc = BinaryAccuracy().to(torch.device('cuda:2'))
        # print('Deep lab의 구성 : ', np.unique(deep_pred))

        metric_dict = {
            'File Name': f'{name}_{m}',
            'Deep Lab Accuarcy': accuracy(truee_t, pred_t ).item(),
            'Deep Plus Accuarcy': accuracy(truee_t,mask_pred_t).item(),
            'Deep DIce': dice_coef(truee_t, pred_t).item(),
            'Plus Dice': dice_coef(truee_t, mask_pred_t).item(),
            'Deep Jaccard': iou(truee_t,pred_t).item(),
            'Plus Jaccard': iou(truee_t, mask_pred_t).item(),
            'Deep f1': f1_score(truee, deep_pred, average='macro',zero_division=1).item(),
            'Plus F1': f1_score(truee_t, mask_pred, average='macro',zero_division=1).item()
        }
        metric_df = pd.concat([metric_df, pd.DataFrame([metric_dict])], ignore_index=True)

        # plt.show()
     # metric_df = metric_df.reset_index(True)
metric_df.to_csv('/home/fisher/Peoples/hseung/NEW/deeplabv3plus_output_metric/deeplabplus_vs_deeplab_metric_new.csv', index=False)
          