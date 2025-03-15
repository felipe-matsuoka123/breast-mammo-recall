import torch
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

def train_one_epoch(model, optimizer, dataloader, criterion, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc='Training', leave=True)
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().flatten().tolist())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds)
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.5  # default if only one class present
    return epoch_loss, epoch_f1, epoch_auc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        # Add tqdm progress bar
        pbar = tqdm(dataloader, desc='Validating', leave=True)
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().numpy().flatten().tolist())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds)
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.5
    return epoch_loss, epoch_f1, epoch_auc

def inference_on_loader(model, dataloader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            # Make sure labels have shape [batch_size, 1] and are floats
            labels = labels.to(device).unsqueeze(1).float()
            # Forward pass: get raw logits
            outputs = model(images)
            # Compute probabilities
            probs = torch.sigmoid(outputs)
            # Compute predictions based on threshold 0.5
            preds = (probs > 0.5).int()
            losses = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none').squeeze(1)
            
            images_cpu = images.cpu()
            labels_cpu = labels.cpu().numpy().flatten()
            probs_cpu = probs.cpu().numpy().flatten()
            losses_cpu = losses.cpu().numpy().flatten()
            
            for i in range(len(labels_cpu)):
                results.append((images_cpu[i], labels_cpu[i], probs_cpu[i], losses_cpu[i]))
    return results


def get_transforms(IMAGE_SIZE):
    transform_pos = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

    transform_neg = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transform_pos, transform_neg, val_transform

def inference_per_study_auc(model, dataloader, device):
    model.eval()
    study_preds = {}
    study_truths = {}

    with torch.no_grad():
        for images, labels, study_ids in dataloader:
            images = images.to(device)
            outputs = model(images)  # raw logits
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()  # predicted probability per image
            labels = labels.cpu().numpy()

            # Group predictions and ground truth by study_id
            for study_id, prob, label in zip(study_ids, probs, labels):
                if study_id not in study_preds:
                    study_preds[study_id] = []
                    study_truths[study_id] = []
                study_preds[study_id].append(prob)
                study_truths[study_id].append(label)

    all_avg_probs = []
    all_truths = []
    for study_id in study_preds:
        avg_prob = np.mean(study_preds[study_id])
        true_label = study_truths[study_id][0]  # assume all images in a study share the same ground truth
        all_avg_probs.append(avg_prob)
        all_truths.append(true_label)
    
    auc = roc_auc_score(all_truths, all_avg_probs)
    return auc