import os

train_path = "Data_frames/train"
val_path = "Data_frames/val"

def count_classes(path):
    for cls in ["Adult", "Child"]:
        folder = os.path.join(path, cls)
        print(f"{path}/{cls}: {len(os.listdir(folder))} images")

print("Train set:")
count_classes(train_path)
print("\nVal set:")
count_classes(val_path)
