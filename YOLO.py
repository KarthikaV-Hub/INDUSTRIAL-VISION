import os
from glob import glob
from PIL import Image
import shutil

base = "/content/products"
os.makedirs(base, exist_ok=True)
os.makedirs(f"{base}/images/train", exist_ok=True)
os.makedirs(f"{base}/images/val", exist_ok=True)
os.makedirs(f"{base}/labels/train", exist_ok=True)
os.makedirs(f"{base}/labels/val", exist_ok=True)

imgs = glob("/content/product_images/*.jpg")
labels = glob("/content/product_labels/*.txt")

split = int(0.8 * len(imgs))

for i, img in enumerate(imgs):
    name = os.path.basename(img)
    if i < split:
        shutil.copy(img, f"{base}/images/train/{name}")
        shutil.copy(f"/content/product_labels/{name.replace('.jpg','.txt')}", f"{base}/labels/train/{name.replace('.jpg','.txt')}")
    else:
        shutil.copy(img, f"{base}/images/val/{name}")
        shutil.copy(f"/content/product_labels/{name.replace('.jpg','.txt')}", f"{base}/labels/val/{name.replace('.jpg','.txt')}")

data = """
train: /content/products/images/train
val: /content/products/images/val
nc: 4
names: ['bottle','box','packet','can']
"""
open("/content/products.yaml","w").write(data)

%cd /content/yolov5
!python train.py --img 640 --batch 16 --epochs 40 --data /content/products.yaml --weights yolov5s.pt --project product_train --name run1
