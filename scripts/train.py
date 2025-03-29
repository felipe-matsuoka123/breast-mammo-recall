import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dataloaders import create_dataloader, load_and_split_data, RatioBatchSampler
from train_utils import train_one_epoch, validate, inference_on_loader, get_transforms, inference_per_study_auc
from model import MammoModel, ConvnextMammoModel
from error_analysis import get_confusion_matrix_fig, get_roc_curve_fig, wandb_table_top_losses
from torch.optim import lr_scheduler

# ----- CONFIGURATION -----
CSV_PATH = '/home/felipe-matsuoka/Desktop/breast-screening/scripts/spr_train_image_level.csv'
BASE_IMG_DIR = '/home/felipe-matsuoka/Documents/spr-mammo-recall/spr-mmg-merged'
BATCH_SIZE = 8
NUM_EPOCHS = 5
LR = 1e-4
NUM_WORKERS = 4
IMAGE_SIZE = 224
DROP_OUT_RATE = 0.3
TRAIN_SPLIT = 0.8 
RUN_NAME = 'SPR_v0'

def main():
    wandb.init(project="VinDrMammo", 
               name=RUN_NAME,           
               config={
                   "batch_size": BATCH_SIZE,
                   "learning_rate": LR,
                   "epochs": NUM_EPOCHS,
                   "model": "efficientnet_b0",
                   "image_size": IMAGE_SIZE,
                   "loss_function": "BCEWithLogitsLoss",
                   "optimizer": "Adam",
                   "dropout_rate": DROP_OUT_RATE,
                   "dataset": "3-channel (microcal, whole_range, soft_tissue)"
               })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CSV and split data based on unique PatientIDs
    train_df, val_df, holdout_df = load_and_split_data(
        csv_path=CSV_PATH,
        train_split=TRAIN_SPLIT  # default val_split=0.1, test=remaining 10%
    )

    # Get transforms for positive, negative, and validation/test
    transform_pos, transform_neg, val_transform = get_transforms(IMAGE_SIZE)
    
    # Create a ratio sampler to ensure a fixed ratio of positives/negatives in training batches
    train_labels = train_df['target'].values
    ratio_sampler = RatioBatchSampler(train_labels, batch_size=BATCH_SIZE, pos_count=1, neg_count=7)

    train_loader = create_dataloader(
        train_df, 
        BASE_IMG_DIR, 
        transform_neg, 
        transform_pos, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        sampler=ratio_sampler
    )

    val_loader = create_dataloader(
        val_df, 
        BASE_IMG_DIR, 
        val_transform, 
        val_transform, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        upsample=False
    )
    holdout_loader = create_dataloader(
        holdout_df, 
        BASE_IMG_DIR, 
        val_transform, 
        val_transform, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        upsample=False
    )

    model = MammoModel(dropout_rate=DROP_OUT_RATE)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max', 
        factor=0.1,  
        patience=2,  
        min_lr=1e-7
    )

    best_val_auc = 0.0
    best_model_path = f"/home/felipe-matsuoka/Desktop/breast-screening/models/{RUN_NAME}_best_model.pth"
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_f1, train_auc = train_one_epoch(model, optimizer, train_loader, criterion, device)
        val_loss, val_f1, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with validation AUC: {val_auc:.4f}")
            wandb.log({"best_val_auc": val_auc})

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train -> Loss: {train_loss:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        print(f"Val   -> Loss: {val_loss:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        current_lr = scheduler.get_last_lr()[0]
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_f1": train_f1,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "val_auc": val_auc,
            "lr": current_lr
        })
    
    # Load the best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))

    holdout_loss, holdout_f1, holdout_auc = validate(model, holdout_loader, criterion, device)
    print("-----------------------------")
    print("Holdout Test Set Metrics:")
    print(f"Loss: {holdout_loss:.4f}, F1: {holdout_f1:.4f}, AUC: {holdout_auc:.4f}")
    wandb.log({
        "holdout_loss": holdout_loss,
        "holdout_f1": holdout_f1,
        "holdout_auc": holdout_auc
    })

    torch.save(model.state_dict(), "/home/felipe-matsuoka/Desktop/breast-screening/models/efficientnet_b0_mammo.pth")
    wandb.save("efficientnet_b0_mammo.pth")
    
    # Inference returns tuples of:
    # (image, label, probability, loss, study (AccessionNumber), patient (PatientID), laterality)
    results = inference_on_loader(model, holdout_loader, device)
    y_true = [r[1] for r in results]
    y_probs = [r[2] for r in results]
    y_pred = [1 if p > 0.5 else 0 for p in y_probs]

    cm_fig = get_confusion_matrix_fig(y_true, y_pred, title_prefix="Holdout Test Set Confusion Matrix")
    roc_fig = get_roc_curve_fig(y_true, y_probs, title_prefix="Holdout Test Set ROC Curve")
    top_losses = wandb_table_top_losses(results, top_n=10)
    auc = inference_per_study_auc(model, holdout_loader, device)

    wandb.log({
        "Confusion Matrix": wandb.Image(cm_fig),
        "ROC Curve": wandb.Image(roc_fig),
        "Top Losses": top_losses,
        "Study Level AUC": auc
    })
    
    wandb.finish()


if __name__ == '__main__':
    main()
