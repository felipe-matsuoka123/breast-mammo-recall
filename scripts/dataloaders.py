import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import Sampler
import pandas as pd

class MammoDataset(Dataset):
    def __init__(self, df, base_img_dir, transform_neg=None, transform_pos=None, pos_aug_factor=3, upsample=True):
        self.df = df.reset_index(drop=True)
        self.base_img_dir = base_img_dir
        self.transform_neg = transform_neg
        self.transform_pos = transform_pos
        self.upsample = upsample
        
        if self.upsample:
            self.indices = []
            for i, row in self.df.iterrows():
                if row['label'] == 1:
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
        study_id = row['study_id']
        image_id = row['image_id']
        label = row['label']
        img_path = os.path.join(self.base_img_dir, study_id, f"{image_id}.png")
        image = Image.open(img_path).convert("RGB")
        
        # Apply the transform based on the label
        if label == 1 and self.transform_pos:
            image = self.transform_pos(image)
        elif label == 0 and self.transform_neg:
            image = self.transform_neg(image)
        return image, label, study_id
    

    

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
    
def create_dataloader(df, base_img_dir, transform_neg, transform_pos, batch_size, shuffle=True, num_workers=4, sampler=None, upsample=True):
    dataset = MammoDataset(df, base_img_dir, transform_neg=transform_neg, transform_pos=transform_pos, upsample=upsample)
    if sampler is not None:
        dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def map_birads_to_label(birads_str):
    birads_str = birads_str.strip()
    if birads_str in ['BI-RADS 1', 'BI-RADS 2']:
        return 0
    elif birads_str in ['BI-RADS 0', 'BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5']:
        return 1
    else:
        return None
    
def load_and_split_data(csv_path: str, train_split: float = 0.8, random_seed: int = 42) -> tuple:
    df = pd.read_csv(csv_path)
    df['label'] = df['breast_birads'].apply(map_birads_to_label)
    df = df[df['label'].notnull()].copy()
    df['label'] = df['label'].astype(int)

    holdout_df = df[df['split'] == 'test'].reset_index(drop=True)
    train_pool_df = df[df['split'] == 'training'].reset_index(drop=True)
    
    unique_patients = train_pool_df['study_id'].unique()
    np.random.seed(random_seed)
    num_val = int((1-train_split) * len(unique_patients))
    val_patients = np.random.choice(unique_patients, size=num_val, replace=False)
    
    train_df = train_pool_df[~train_pool_df['study_id'].isin(val_patients)].reset_index(drop=True)
    val_df = train_pool_df[train_pool_df['study_id'].isin(val_patients)].reset_index(drop=True)
    
    # Print dataset statistics
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Holdout test set: {len(holdout_df)} samples")
    
    return train_df, val_df, holdout_df
