import os, random
from PIL import Image, ImageDraw, ImageFilter
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

DIR_OUT = "/unit3/segment"
os.makedirs(DIR_OUT, exist_ok=True)
IMG_W = 256

def synth_img(n=IMG_W, flag=True):
    base = Image.new("L", (n, n), 120)
    draw = ImageDraw.Draw(base)
    if flag:
        px = random.randint(n//4, n//2)
        py = random.randint(n//8, n//8 + 10)
        pts = [(px, py)]
        k = random.randint(20, 35)
        for _ in range(k):
            px += random.randint(-8, 8)
            py += random.randint(6, 12)
            pts.append((max(0, min(n-1, px)), max(0, min(n-1, py))))
        d2 = ImageDraw.Draw(base)
        for j in range(len(pts)-1):
            d2.line([pts[j], pts[j+1]], fill=30, width=random.randint(2, 4))
    base = base.filter(ImageFilter.GaussianBlur(0.6))
    m = Image.fromarray((np.array(base) < 80).astype(np.uint8) * 255)
    return base, m

for sect in ["train", "val"]:
    p_img = os.path.join(DIR_OUT, sect, "images")
    p_msk = os.path.join(DIR_OUT, sect, "masks")
    os.makedirs(p_img, exist_ok=True)
    os.makedirs(p_msk, exist_ok=True)
    ct = 250 if sect == "train" else 60
    for i in range(ct):
        im, mk = synth_img(flag=(random.random() < 0.9))
        im.save(os.path.join(p_img, f"{sect}_{i:04d}.png"))
        mk.save(os.path.join(p_msk, f"{sect}_{i:04d}.png"))

class CrackData(Dataset):
    def __init__(self, root, sec="train"):
        self.root = root
        self.sec = sec
        self.names = sorted(os.listdir(os.path.join(root, sec, "images")))
        self.tf = T.Compose([T.Resize((IMG_W, IMG_W)), T.ToTensor()])
    def __len__(self):
        return len(self.names)
    def __getitem__(self, idx):
        p0 = os.path.join(self.root, self.sec)
        img = Image.open(os.path.join(p0, "images", self.names[idx])).convert("L")
        msk = Image.open(os.path.join(p0, "masks", self.names[idx])).convert("L")
        img = self.tf(img)
        msk = self.tf(msk)
        msk = (msk > 0.5).float()
        return img, msk

class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.mod(x)

class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = Block(1, 32)
        self.e2 = Block(32, 64)
        self.e3 = Block(64, 128)
        self.pl = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d2 = Block(64 + 128, 64)
        self.d1 = Block(32 + 64, 32)
        self.out = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        c1 = self.e1(x)
        x = self.pl(c1)
        c2 = self.e2(x)
        x = self.pl(c2)
        x = self.e3(x)
        x = self.up(x)
        x = torch.cat([x, c2], dim=1)
        x = self.d2(x)
        x = self.up(x)
        x = torch.cat([x, c1], dim=1)
        x = self.d1(x)
        return torch.sigmoid(self.out(x))

ds_tr = CrackData(DIR_OUT, "train")
ds_va = CrackData(DIR_OUT, "val")
ld_tr = DataLoader(ds_tr, batch_size=8, shuffle=True)
ld_va = DataLoader(ds_va, batch_size=4)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = MiniUNet().to(dev)
optm = torch.optim.Adam(net.parameters(), lr=1e-3)
crit = nn.BCELoss()

for ep in range(8):
    net.train()
    acc = 0.0
    for a, b in tqdm(ld_tr, desc=f"Epoch {ep+1}"):
        a, b = a.to(dev), b.to(dev)
        pr = net(a)
        ls = crit(pr, b)
        optm.zero_grad()
        ls.backward()
        optm.step()
        acc += ls.item()
    print("train loss:", acc / len(ld_tr))
    net.eval()
    with torch.no_grad():
        vv_i, vv_m = next(iter(ld_va))
        vv_i = vv_i.to(dev)
        vv_p = net(vv_i).cpu().numpy()
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(vv_i.cpu()[0, 0], cmap='gray')
        plt.title("Image")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(vv_m[0, 0], cmap='gray')
        plt.title("GT")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(vv_p[0, 0] > 0.5, cmap='gray')
        plt.title("Pred")
        plt.axis('off')
        plt.show()

torch.save(net.state_dict(), "/content/unet_crack.pth")
print("Saved /content/unet_crack.pth")

