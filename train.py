import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DCSASSVideoDataset, prepare_dcsass_splits
from models.slowfast_model import SuspiciousActivityModel
from utils import pack_pathway_output

# Create checkpoints directory safely
os.makedirs("checkpoints", exist_ok=True)

# SYNC CONFIGURATIONS
DATASET_ROOT = "data/"     # Dataset path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4             # Physical batch size
ACCUMULATION_STEPS = 4     # Effective batch size = 16
EPOCHS = 20
NUM_CLASSES = 5            # Number of Macro Classes

def main():
    # Split generation
    train_df, val_df = prepare_dcsass_splits(DATASET_ROOT)
    train_dataset = DCSASSVideoDataset(train_df, DATASET_ROOT)
    val_dataset = DCSASSVideoDataset(val_df, DATASET_ROOT)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Architecture and weights
    model = SuspiciousActivityModel(num_classes=NUM_CLASSES).to(DEVICE)

    # Calculate class Weights for imbalance
    class_counts = train_df['multiclass_label'].value_counts().sort_index().values
    class_weights = len(train_df) / (len(class_counts) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(EPOCHS):
        # ### TRAINING PHASE ###
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")

        for batch_idx, (videos, labels) in loop:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            inputs = pack_pathway_output(videos)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                # Scale loss by accumulation steps
                loss = criterion(outputs, labels) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Metrics for display
            running_loss += (loss.item() * ACCUMULATION_STEPS)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            loop.set_postfix(loss=loss.item()*ACCUMULATION_STEPS, acc=100.*correct/total)

        # ### VALIDATION PHASE ###
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for val_videos, val_labels in val_loader:
                val_videos, val_labels = val_videos.to(DEVICE), val_labels.to(DEVICE)
                v_inputs = pack_pathway_output(val_videos)
                with torch.amp.autocast('cuda'):
                    v_outputs = model(v_inputs)
                    _, v_pred = v_outputs.max(1)
                    val_total += val_labels.size(0)
                    val_correct += v_pred.eq(val_labels).sum().item()

        print(f"Validation Accuracy: {100.0 * val_correct / val_total:.2f}%")
        torch.save(model.state_dict(), f"checkpoints/slowfast_dcsass_e{epoch+1}.pth")

if __name__ == '__main__':
    main()