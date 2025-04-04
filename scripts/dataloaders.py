import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import Sampler
import pandas as pd

class MammoDataset(Dataset):
    def __init__(self, df, base_img_dir=None, transform_neg=None, transform_pos=None, pos_aug_factor=3, upsample=True):
        self.df = df.reset_index(drop=True)
        self.base_img_dir = base_img_dir  # if image_path in CSV is relative, set base_img_dir; otherwise, leave as None.
        self.transform_neg = transform_neg
        self.transform_pos = transform_pos
        self.upsample = upsample
        
        if self.upsample:
            self.indices = []
            for i, row in self.df.iterrows():
                if row['target'] == 1:
                    self.indices.extend([i] * pos_aug_factor)
                else:
                    self.indices.append(i)
        else:
            self.indices = list(range(len(self.df)))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        row = self.df.iloc[actual_idx]
        target = row['target']
        
        # Use the image_path from CSV. If base_img_dir is provided, join it.
        if self.base_img_dir:
            img_path = os.path.join(self.base_img_dir, row['image_path'])
        else:
            img_path = row['image_path']
        
        image = Image.open(img_path).convert("RGB")
        
        if target == 1 and self.transform_pos:
            image = self.transform_pos(image)
        elif target == 0 and self.transform_neg:
            image = self.transform_neg(image)
        
        # Return extra metadata: AccessionNumber, PatientID, and Laterality.
        return image, target, row['AccessionNumber'], row['PatientID'], row['Laterality']

class RatioBatchSampler(Sampler):
    def __init__(self, labels, batch_size=8, pos_count=1, neg_count=7):
        assert pos_count + neg_count == batch_size, "batch_size must equal pos_count + neg_count"
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.pos_count = pos_count
        self.neg_count = neg_count
        
        self.positive_indices = np.where(self.labels == 1)[0]
        self.negative_indices = np.where(self.labels == 0)[0]
        
    def __iter__(self):
        num_batches = len(self.positive_indices)
        for _ in range(num_batches):
            pos_idx = np.random.choice(self.positive_indices, size=self.pos_count, replace=True)
            neg_idx = np.random.choice(self.negative_indices, size=self.neg_count, replace=True)
            batch = np.concatenate([pos_idx, neg_idx])
            np.random.shuffle(batch)
            yield batch.tolist()
            
    def __len__(self):
        return len(self.positive_indices)
    
def create_dataloader(df, base_img_dir=None, transform_neg=None, transform_pos=None, batch_size=8, shuffle=True, num_workers=4, sampler=None, upsample=True):
    dataset = MammoDataset(df, base_img_dir=base_img_dir, transform_neg=transform_neg, transform_pos=transform_pos, upsample=upsample)
    if sampler is not None:
        dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def load_and_split_data(csv_path: str, train_split: float = 0.8, val_split: float = 0.1, random_seed: int = 42) -> tuple:
    df = pd.read_csv(csv_path)
    
    # Split by unique PatientID to prevent patient-level leakage.
    unique_patients = df['PatientID'].unique()
    np.random.seed(random_seed)
    np.random.shuffle(unique_patients)
    n = len(unique_patients)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    
    train_patients = unique_patients[:n_train]
    val_patients = unique_patients[n_train:n_train+n_val]
    test_patients = unique_patients[n_train+n_val:]
    
    train_df = df[df['PatientID'].isin(train_patients)].reset_index(drop=True)
    val_df = df[df['PatientID'].isin(val_patients)].reset_index(drop=True)
    test_df = df[df['PatientID'].isin(test_patients)].reset_index(drop=True)
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, val_df, test_df

