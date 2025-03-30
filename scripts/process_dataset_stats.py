import os
from PIL import Image
import torch
from torchvision import transforms

def compute_dataset_stats(root_dir, image_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    
    image_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
                
    if not image_files:
        raise ValueError(f"No images found in the directory: {root_dir}")
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    for file in image_files:
        try:
            img = Image.open(file).convert('RGB')
        except Exception as e:
            print(f"Error loading image {file}: {e}")
            continue
        img_tensor = transform(img)
        mean += img_tensor.mean(dim=[1, 2])
        std += img_tensor.std(dim=[1, 2])
    
    mean /= len(image_files)
    std /= len(image_files)
    
    return mean, std

if __name__ == '__main__':
    root_dir = '/home/felipe-matsuoka/Documents/spr-mammo-recall/spr-mmg-merged'
    mean, std = compute_dataset_stats(root_dir)
    print("Dataset Mean:", mean)
    print("Dataset Std: ", std)
