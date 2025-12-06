import cv2, os, numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import random

src_dir = "/content/unit5_src"
os.makedirs(src_dir, exist_ok=True)

def make_img(path="kart.png", size=256):
    im = Image.new("RGB", (size, size), (160, 160, 160))
    dr = ImageDraw.Draw(im)
    m = 30
    dr.rounded_rectangle([m, m, size - m, size - m], radius=24, fill=(200, 200, 200))
    im.save(path)
    return path

base_img = make_img()

out_dir = "/content/unit5_gen"
os.makedirs(out_dir, exist_ok=True)
light_map = {"low": 0.6, "mid": 1.0, "high": 1.4}
rot_list = [0, 10, -10]

c = 0
for ln, fac in light_map.items():
    for ang in rot_list:
        im = Image.open(base_img).convert("RGB")
        im = ImageEnhance.Brightness(im).enhance(fac)
        im = im.rotate(ang, expand=False, fillcolor=(40, 40, 40))
        name = os.path.join(out_dir, f"item_{ln}_r{ang}.png")
        im.save(name)
        c += 1
print("Generated:", c)

def prep(x):
    x = x.astype(np.float32)
    m = x.mean(axis=(0, 1))
    x2 = x * (m.mean() / m)
    x2 = np.clip(x2, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(x2, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clh.apply(l)
    lab2 = cv2.merge((l2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return out

imgs = sorted(os.listdir(out_dir))
plt.figure(figsize=(12, 4))
for i, f in enumerate(imgs):
    im = cv2.cvtColor(cv2.imread(os.path.join(out_dir, f)), cv2.COLOR_BGR2RGB)
    p = prep(im)
    plt.subplot(3, 3, i + 1)
    plt.imshow(p)
    plt.title(f)
    plt.axis('off')
plt.show()
