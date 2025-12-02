
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

batch_size = 128
num_epochs = 5
lr = 1e-3
num_classes = 10    # CIFAR-10 = 10 classes
val_split = 0.1

train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
test_ds    = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

# Split train → train + val
n_total = len(full_train)
n_val   = int(0.1 * n_total)
n_train = n_total - n_val

train_ds, val_ds = random_split(full_train, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total, correct, running_loss = 0, 0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        running_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return running_loss/total, correct/total


def evaluate(model, loader, loss_fn):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)

            running_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return running_loss/total, correct/total

def get_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model.to(device)

def get_mobilenet():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, num_classes)
    return model.to(device)

def run(model, name="model"):
    print(f"\n===== Training {name} =====")
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0
    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, loss_fn)
        v_loss, v_acc = evaluate(model, val_loader, loss_fn)

        if v_acc > best_val:
            best_val = v_acc
            best_model = model.state_dict()

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Acc: {tr_acc:.4f} | Val Acc: {v_acc:.4f}")

    # Load best model weights
    model.load_state_dict(best_model)

    # Final test accuracy
    te_loss, te_acc = evaluate(model, test_loader, loss_fn)
    print(f"[{name}] Test Accuracy = {te_acc:.4f}")

    return te_acc

resnet = get_resnet18()
mobile = get_mobilenet()

acc_resnet = run(resnet, "ResNet18")
acc_mobile = run(mobile, "MobileNetV2")

print("\nFinal Comparison:")
print("ResNet18 Test Acc  :", acc_resnet)
print("MobileNetV2 Test Acc:", acc_mobile)
