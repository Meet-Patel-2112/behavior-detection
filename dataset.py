import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class DCSASSVideoDataset(Dataset):
    def __init__(self, dataframe, root_dir, num_frames=32, img_size=224):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            return torch.zeros((3, self.num_frames, self.img_size, self.img_size))

        if len(frames) < self.num_frames:
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        elif len(frames) > self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
            frames = [frames[i] for i in indices]

        frames = np.array(frames) 
        frames = torch.tensor(frames).permute(3, 0, 1, 2) 
        return frames

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_full_path = os.path.join(self.root_dir, row['rel_path'])
        label = torch.tensor(row['multiclass_label'], dtype=torch.long)

        if not os.path.exists(video_full_path):
            return torch.zeros((3, self.num_frames, self.img_size, self.img_size)), label

        video_tensor = self.load_video(video_full_path)
        return video_tensor, label


def prepare_dcsass_splits(root_dir, val_size=0.15, random_state=42):
    # Directory crawler
    classes = [
        "Abuse", "Arson", "Assault", "Burglary", "Fighting", 
        "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"
    ]
    
    # 2. Map micro classes into macro classes
    macro_map = {
        "Assault": 1,     # Violence
        "Fighting": 1,    # Violence
        "Shooting": 1,    # Violence
        "Burglary": 2,    # Theft
        "Robbery": 2,     # Theft
        "Shoplifting": 2, # Theft
        "Stealing": 2,    # Theft
        "Arson": 3,       # Property Damage
        "Vandalism": 3,   # Property Damage
        "Abuse": 4,       # Harassment
        "Harass": 4       # Harassment
    }
    
    parsed_records = []

    for cls_name in classes:
        csv_path = os.path.join(root_dir, "labels", f"{cls_name}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping...")
            continue
            
        df = pd.read_csv(csv_path, header=None)
        df = df.dropna(subset=[2])
        
        for _, row in df.iterrows():
            segment_name = str(row[0])
            binary_flag = int(float(row[2]))
            
            base_folder = segment_name.rpartition('_')[0]
            rel_path = os.path.join(cls_name, f"{base_folder}.mp4", f"{segment_name}.mp4")
            
            if binary_flag == 0:
                macro_label = 0  
            else:
                macro_label = macro_map[cls_name] 
            
            parsed_records.append({
                "rel_path": rel_path,
                "multiclass_label": macro_label
            })

    master_df = pd.DataFrame(parsed_records)
    
    # Theft Soft Cap for Class Imbalance
    theft_pool = master_df[master_df['multiclass_label'] == 2]
    other_pool = master_df[master_df['multiclass_label'] != 2]
    
    if len(theft_pool) > 2500:
        # Randomly sample exactly 2500 rows to maintain background variety without over-biasing
        theft_pool = theft_pool.sample(n=2500, random_state=random_state)
        
    master_df = pd.concat([theft_pool, other_pool]).reset_index(drop=True)
    print(f"Data Imbalance Corrected: Theft class soft-capped at {len(theft_pool)} samples.")
    
    train_df, val_df = train_test_split(
        master_df, 
        test_size=val_size, 
        stratify=master_df['multiclass_label'], 
        random_state=random_state
    )
    
    print(f"Dataset Prepared: {len(train_df)} train, {len(val_df)} val (5 Macro Classes).")
    return train_df, val_df