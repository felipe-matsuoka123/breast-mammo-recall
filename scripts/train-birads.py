import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm


class BIRADSMammoDataset(Dataset):
    def __init__(self, df, base_img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.base_img_dir = base_img_dir
        self.transform = transform
        self.label_map = {
            "BI-RADS 1": 0,
            "BI-RADS 2": 1,
            "BI-RADS 3": 2,
            "BI-RADS 4": 3,
            "BI-RADS 5": 4,
        }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row['study_id']
        image_id = row['image_id']
        label_str = row['breast_birads']
        
        if label_str in self.label_map:
            label = self.label_map[label_str]
        else:
            raise ValueError(f"Label '{label_str}' não encontrado no mapeamento.")
            
        img_path = os.path.join(self.base_img_dir, study_id, f"{image_id}.png")
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, study_id

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

df = pd.read_csv('/media/felipe-matsuoka/FelipeSSD/datasets/physionet.org/files/vindr-mammo/1.0.0/breast-level_annotations.csv')

base_img_dir = '/media/felipe-matsuoka/FelipeSSD/datasets/vindr-mammo-png'
dataset = BIRADSMammoDataset(df, base_img_dir, transform=data_transforms)

total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
num_classes = 5

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")

from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Barra de progresso para o treinamento
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        for images, labels, _ in train_pbar:
            if (labels < 0).any() or (labels >= num_classes).any():
                print("Labels fora do intervalo:", labels)
                continue  
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            train_pbar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Avaliação na validação com barra de progresso
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False)
        with torch.no_grad():
            for images, labels, _ in val_pbar:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix(loss=loss.item())
        
        val_loss /= len(val_loader.dataset)
        val_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    return model


trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

evaluate_model(trained_model, test_loader, criterion)
