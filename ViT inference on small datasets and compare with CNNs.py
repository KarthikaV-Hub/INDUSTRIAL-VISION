
!pip install -q torch torchvision

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Common transforms: resize to 224x224 for ResNet/ViT
train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

test_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

train_full = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=test_transform)

num_classes = 10

# ---- create small train subset: 2000 images total ----
np.random.seed(42)
indices = np.random.permutation(len(train_full))[:2000]
train_small = Subset(train_full, indices)

batch_size = 64

train_loader = DataLoader(train_small, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

len(train_small), len(test_dataset)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total

from torchvision.models import resnet18, ResNet18_Weights

# Load pretrained ResNet-18
cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Freeze backbone
for param in cnn.parameters():
    param.requires_grad = False

# Replace final FC layer for CIFAR-10
in_features = cnn.fc.in_features
cnn.fc = nn.Linear(in_features, num_classes)

cnn = cnn.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.fc.parameters(), lr=1e-3)

num_epochs = 10   # you can increase to 10–15 for better accuracy

cnn_train_accs, cnn_test_accs = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(cnn, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(cnn, test_loader, criterion)

    cnn_train_accs.append(train_acc)
    cnn_test_accs.append(test_acc)

    print(f"[CNN][Epoch {epoch+1}/{num_epochs}] "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

from torchvision.models import vit_b_16, ViT_B_16_Weights

# Load pretrained ViT-B/16
vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

# Freeze backbone
for param in vit.parameters():
    param.requires_grad = False

# Replace head for CIFAR-10
in_features = vit.heads.head.in_features
vit.heads.head = nn.Linear(in_features, num_classes)

vit = vit.to(device)

criterion_vit = nn.CrossEntropyLoss()
optimizer_vit = optim.Adam(vit.heads.head.parameters(), lr=1e-3)

num_epochs_vit = 5   # match CNN epochs for fairness

vit_train_accs, vit_test_accs = [], []

for epoch in range(num_epochs_vit):
    train_loss, train_acc = train_one_epoch(vit, train_loader, criterion_vit, optimizer_vit)
    test_loss, test_acc = evaluate(vit, test_loader, criterion_vit)

    vit_train_accs.append(train_acc)
    vit_test_accs.append(test_acc)

    print(f"[ViT][Epoch {epoch+1}/{num_epochs_vit}] "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

print(f"Final CNN Test Accuracy : {cnn_test_accs[-1]:.4f}")
print(f"Final ViT Test Accuracy : {vit_test_accs[-1]:.4f}")

import matplotlib.pyplot as plt

epochs_cnn = range(1, len(cnn_test_accs) + 1)
epochs_vit = range(1, len(vit_test_accs) + 1)

plt.figure()
plt.plot(epochs_cnn, cnn_test_accs, label="CNN (ResNet-18)")
plt.plot(epochs_vit, vit_test_accs, label="ViT-B/16")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("CNN vs ViT on Small CIFAR-10 Subset")
plt.legend()
plt.grid(True)
plt.show()
