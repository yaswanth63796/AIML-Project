# train.py
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(data_root="Data_frames", epochs=6, batch_size=16, lr=1e-4, out_model="models/child_labour_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")

    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
    val_ds = datasets.ImageFolder(val_dir, transform=transform_val)

    print("[INFO] Classes:", train_ds.classes)
    print("[INFO] class_to_idx:", train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / max(1, len(train_loader.dataset))

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total>0 else 0.0

        print(f"[INFO] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_model)
            print(f"[INFO] Saved best model to {out_model} with val_acc={val_acc:.4f}")

    # Save labels mapping (index->class name)
    labels_map = {v:k for k,v in train_ds.class_to_idx.items()}
    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)
    labels_json = os.path.join(os.path.dirname(out_model), "labels.json")
    with open(labels_json, "w") as f:
        json.dump(labels_map, f)
    print("[INFO] Saved labels mapping to", labels_json)
    print("[INFO] Training complete. Best val acc:", best_val_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Data_frames")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="models/child_labour_model.pth")
    args = parser.parse_args()
    train_model(data_root=args.data, epochs=args.epochs, batch_size=args.batch, lr=args.lr, out_model=args.out)
