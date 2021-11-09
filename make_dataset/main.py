################
#
# 処理の流れ
# (in preprocess.py)
# 1. データセットをTwitterから取得→X pxにreshape
# 2. 
#
################
import os
import glob

import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations

import preprocess

def draw_res(image,results):
    for r in results:
        bbox = r['bbox']
        if not bbox:continue
        cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),thickness=5)
    return image

model = get_model("resnet50_2020-07-20", max_size=2048)
model.eval()

#path = "Dataset\kurusu/nijimasu_rinrin-1423211787718062084-20210805_181826-img1.jpg"
#path_no_face = "Dataset/kurusu/nijimasu_rinrin-1028493444971343872-20180812_130840-img1.jpg"


path = "Dataset/kurusu"
path = preprocess.folder_check(path)

dst_folder = "facedet/kurusu"
dst_folder = preprocess.folder_check(dst_folder)

image_path = glob.glob(path+ "*.jpg")

for img in tqdm(image_path):
    image = cv2.imread(img)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict_jsons(image)
    image_out = draw_res(image,results)
    dst_path = dst_folder + os.path.basename(img)
    cv2.imwrite(dst_path, image_out)