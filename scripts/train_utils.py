import torch
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from dataloaders import create_dataloader, load_and_split_data_cv

# Funções de treinamento já existentes
def train_one_epoch(model, optimizer, dataloader, criterion, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc='Training', leave=True)
    # Desempacota: images, labels, study_id, patient_id, laterality
    for images, labels, study_id, patient_id, laterality in pbar:
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
        epoch_auc = 0.5  # caso só haja uma classe
    return epoch_loss, epoch_f1, epoch_auc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating', leave=True)
        for images, labels, study_id, patient_id, laterality in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
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
        epoch_auc = 0.5
    return epoch_loss, epoch_f1, epoch_auc

def inference_on_loader(model, dataloader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for images, labels, study_id, patient_id, laterality in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            outputs = model(images)
            probs = torch.sigmoid(outputs)
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
        for images, labels, study_id, patient_id, laterality in dataloader:
            images = images.to(device)
            outputs = model(images)  # logits
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            labels = labels.cpu().numpy()

            # Agrupa as predições por estudo (AccessionNumber)
            for study, prob, label in zip(study_id, probs, labels):
                if study not in study_preds:
                    study_preds[study] = []
                    study_truths[study] = []
                study_preds[study].append(prob)
                study_truths[study].append(label)

    all_avg_probs = []
    all_truths = []
    for study in study_preds:
        avg_prob = np.mean(study_preds[study])
        true_label = study_truths[study][0]  # assume consistência
        all_avg_probs.append(avg_prob)
        all_truths.append(true_label)
    
    auc = roc_auc_score(all_truths, all_avg_probs)
    return auc

# Novo loop de treinamento com Cross Validation
def cross_validation_training(model_factory, optimizer_factory, criterion, csv_path, base_img_dir,
                              transform_pos, transform_neg, val_transform, device, n_epochs=10, n_splits=5,
                              batch_size=8, num_workers=4, upsample=True):

    cv_splits, test_df = load_and_split_data_cv(csv_path, test_split=0.1, n_splits=n_splits)
    fold_metrics = []
    
    for fold_idx, (train_df, val_df) in enumerate(cv_splits):
        print(f"\nIniciando Fold {fold_idx+1}/{n_splits}")
        model = model_factory().to(device)
        optimizer = optimizer_factory(model.parameters())
        
        # Cria os dataloaders para treino e validação
        train_loader = create_dataloader(train_df, base_img_dir, transform_neg, transform_pos,
                                         batch_size=batch_size, shuffle=True, num_workers=num_workers, upsample=upsample)
        val_loader = create_dataloader(val_df, base_img_dir, val_transform, val_transform,
                                       batch_size=batch_size, shuffle=False, num_workers=num_workers, upsample=False)
        
        best_val_auc = 0.0
        best_epoch = -1
        for epoch in range(1, n_epochs+1):
            train_loss, train_f1, train_auc = train_one_epoch(model, optimizer, train_loader, criterion, device)
            val_loss, val_f1, val_auc = validate(model, val_loader, criterion, device)
            print(f"Fold {fold_idx+1} - Epoch {epoch}/{n_epochs}: "
                  f"Train Loss: {train_loss:.4f} | F1: {train_f1:.4f} | AUC: {train_auc:.4f} || "
                  f"Val Loss: {val_loss:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch

        fold_metrics.append({'fold': fold_idx+1, 'best_val_auc': best_val_auc, 'best_epoch': best_epoch})
        print(f"Finalizado Fold {fold_idx+1}. Melhor Val AUC: {best_val_auc:.4f} na Epoch {best_epoch}")
    
    # Opcional: Avaliação final no conjunto de teste (fixo)
    test_loader = create_dataloader(test_df, base_img_dir, val_transform, val_transform,
                                    batch_size=batch_size, shuffle=False, num_workers=num_workers, upsample=False)
    # Aqui, para exemplificar, utilizamos o modelo do último fold para inferência (ideal é combinar ou retrenar com todos os dados)
    test_loss, test_f1, test_auc = validate(model, test_loader, criterion, device)
    print(f"\nDesempenho no Conjunto de Teste: Loss: {test_loss:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}")
    
    return fold_metrics, test_auc

