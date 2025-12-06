import os, random
import torch, torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter

pA = "/content/camA"
pB = "/content/camB"
os.makedirs(pA, exist_ok=True)
os.makedirs(pB, exist_ok=True)

def build_set(outdir, num=200, mode="A"):
    for cls in ["ok", "bad"]:
        tgt = os.path.join(outdir, cls)
        os.makedirs(tgt, exist_ok=True)
        for i in range(num):
            img = Image.open(base_path).convert("RGB")
            if mode == "A":
                img = ImageEnhance.Color(img).enhance(1.0)
            else:
                img = ImageEnhance.Color(img).enhance(0.6)
                img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
            if cls == "bad":
                dr = ImageDraw.Draw(img)
                dr.line((50, 120, 206, 130), fill=(20, 20, 20), width=3)
            img.save(os.path.join(tgt, f"{cls}_{i:04d}.png"))

build_set(pA, num=150, mode="A")
build_set(pB, num=80, mode="B")

trf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_set = ImageFolder(pA, transform=trf)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

net = torchvision.models.resnet18(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, 2)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(dev)

opt = torch.optim.Adam(net.parameters(), lr=1e-4)
crit = nn.CrossEntropyLoss()

for ep in range(4):
    net.train()
    s = 0
    for x, y in train_loader:
        x, y = x.to(dev), y.to(dev)
        out = net(x)
        ls = crit(out, y)
        opt.zero_grad()
        ls.backward()
        opt.step()
        s += ls.item()
    print("Epoch", ep+1, "loss", s / len(train_loader))

test_set = ImageFolder(pB, transform=trf)
test_loader = DataLoader(test_set, batch_size=32)

net.eval()
hit = 0
tot = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(dev), y.to(dev)
        pred = net(x).argmax(1)
        hit += (pred == y).sum().item()
        tot += y.size(0)

print("Test on camB acc:", hit / tot)
