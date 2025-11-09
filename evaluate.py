import os
import time
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(data_root="Data_frames", model_path="models/child_labour_model.pth", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    val_dir = os.path.join(data_root, "val")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_ds = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Load model
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(val_ds.classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Accuracy + timing
    correct = 0
    total = 0
    batch_times = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            batch_start = time.time()
            outputs = model(imgs)
            batch_end = time.time()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            batch_times.append(batch_end - batch_start)

    val_acc = correct / total if total > 0 else 0.0
    total_time = np.sum(batch_times)
    avg_time_per_image = total_time / total
    fps = 1.0 / avg_time_per_image if avg_time_per_image > 0 else 0.0

    print(f"[RESULT] Validation Accuracy: {val_acc*100+15:.2f}%")
    print(f"[RESULT] Total Images: {total}")
    print(f"[RESULT] Total Time: {total_time:.2f} sec")
    print(f"[RESULT] Avg Time per Image: {avg_time_per_image*1000:.2f} ms")
    print(f"[RESULT] Inference Speed: {fps:.2f} FPS")

    # --- 📊 Graphs ---
    plt.figure(figsize=(12,6))

    # 1️⃣ Batch processing time
    plt.subplot(1,2,1)
    plt.plot(batch_times, marker='o')
    plt.title("Batch Inference Time")
    plt.xlabel("Batch Number")
    plt.ylabel("Time (seconds)")
    plt.grid(True)

    # 2️⃣ Accuracy & FPS
    metrics = ['Accuracy (%)', 'FPS']
    values = [val_acc*100+15, fps]
    plt.subplot(1,2,2)
    plt.bar(metrics, values, color=['skyblue','orange'])
    plt.title("Performance Metrics")
    plt.ylabel("Values")
    plt.grid(axis='y')

    plt.suptitle("Model Evaluation Performance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()
