import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
from dataloaders import create_dataloader, load_and_split_data_cv, RatioBatchSampler
from train_utils import train_one_epoch, validate, inference_on_loader, get_transforms, inference_per_study_auc
from model import MammoModel  # ou ConvnextMammoModel, se preferir
from error_analysis import get_confusion_matrix_fig, get_roc_curve_fig, wandb_table_top_losses
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, roc_auc_score

# ----- CONFIG -----
CSV_PATH = '/home/felipe-matsuoka/Desktop/breast-screening/scripts/spr_train_image_level.csv'
BASE_IMG_DIR = '/home/felipe-matsuoka/Documents/spr-mammo-recall/spr-mmg-merged'
BATCH_SIZE = 8
NUM_EPOCHS = 2
LR = 1e-4
NUM_WORKERS = 4
IMAGE_SIZE = 224
DROP_OUT_RATE = 0.3
RUN_NAME = 'SPR_v0_CV_Ensemble'
NUM_FOLDS = 4

def main():
    wandb.init(project="SPR_Breast_Recall", 
               name=RUN_NAME,           
               config={
                   "batch_size": BATCH_SIZE,
                   "learning_rate": LR,
                   "epochs": NUM_EPOCHS,
                   "num_folds": NUM_FOLDS,
                   "model": "efficientnet_b0",
                   "image_size": IMAGE_SIZE,
                   "loss_function": "BCEWithLogitsLoss",
                   "optimizer": "Adam",
                   "dropout_rate": DROP_OUT_RATE,
                   "dataset": "3-channel (microcal, whole_range, soft_tissue)"
               })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cv_splits, holdout_df = load_and_split_data_cv(
        csv_path=CSV_PATH,
        test_split=0.1,
        n_splits=NUM_FOLDS,
        random_seed=42
    )

    transform_pos, transform_neg, val_transform = get_transforms(IMAGE_SIZE)
    
    fold_best_model_paths = []
    fold_best_metrics = []

    for fold_idx, (train_df, val_df) in enumerate(cv_splits):
        print(f"\nIniciando Fold {fold_idx+1}/{len(cv_splits)}")
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
        best_epoch = -1
        best_model_path = f"/home/felipe-matsuoka/Desktop/breast-screening/models/{RUN_NAME}_fold{fold_idx+1}_best_model.pth"
        
        for epoch in range(NUM_EPOCHS):
            train_loss, train_f1, train_auc = train_one_epoch(model, optimizer, train_loader, criterion, device)
            val_loss, val_f1, val_auc = validate(model, val_loader, criterion, device)
            scheduler.step(val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                wandb.log({f"Fold{fold_idx+1}_best_val_auc": best_val_auc})
                print(f"Novo melhor modelo salvo no Fold {fold_idx+1} com AUC: {val_auc:.4f}")
            
            print(f"Fold {fold_idx+1} - Epoch {epoch+1}/{NUM_EPOCHS}")
            print(f"Train -> Loss: {train_loss:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
            print(f"Val   -> Loss: {val_loss:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
            wandb.log({
                "fold": fold_idx+1,
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_f1": train_f1,
                "train_auc": train_auc,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "lr": scheduler.get_last_lr()[0]
            })
            
        print(f"Finished Fold {fold_idx+1}. Best AUC na Val: {best_val_auc:.4f} at Epoch {best_epoch}")
        fold_best_metrics.append({"fold": fold_idx+1, "best_val_auc": best_val_auc, "best_epoch": best_epoch})
        fold_best_model_paths.append(best_model_path)

    holdout_loader = create_dataloader(
        holdout_df, 
        BASE_IMG_DIR, 
        val_transform, 
        val_transform, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS/2,
        upsample=False
    )
    
    ensemble_probs = None
    ensemble_results = None  
    
    for fold_path in fold_best_model_paths:
        model = MammoModel(dropout_rate=0.0)
        model.load_state_dict(torch.load(fold_path))
        model.to(device)
        model.eval()
        results = inference_on_loader(model, holdout_loader, device)

        fold_probs = np.array([r[2] for r in results])

        if ensemble_results is None:
            ensemble_results = results  
        if ensemble_probs is None:
            ensemble_probs = fold_probs
        else:
            ensemble_probs += fold_probs
    
    ensemble_probs /= len(fold_best_model_paths)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    y_true = np.array([r[1] for r in ensemble_results])
    
    # Calcula métricas com o ensemble
    ensemble_f1 = f1_score(y_true, ensemble_preds)
    try:
        ensemble_auc = roc_auc_score(y_true, ensemble_probs)
    except ValueError:
        ensemble_auc = 0.5
    
    print("-----------------------------")
    print("Holdout test set metrics (Ensemble):")
    print(f"F1: {ensemble_f1:.4f}, AUC: {ensemble_auc:.4f}")
    wandb.log({
        "holdout_ensemble_f1": ensemble_f1,
        "holdout_ensemble_auc": ensemble_auc
    })
    
    # Geração das figuras e análise de erros usando as predições do ensemble
    cm_fig = get_confusion_matrix_fig(y_true, ensemble_preds, title_prefix="Holdout Ensemble Confusion Matrix")
    roc_fig = get_roc_curve_fig(y_true, ensemble_probs, title_prefix="Holdout Ensemble ROC Curve")
    #top_losses = wandb_table_top_losses(ensemble_results, top_n=10)
    #study_auc = inference_per_study_auc(model, holdout_loader, device)
    
    wandb.log({
        "Ensemble Confusion Matrix": wandb.Image(cm_fig),
        "Ensemble ROC Curve": wandb.Image(roc_fig)
    })

    final_ensemble_info = {"fold_best_model_paths": fold_best_model_paths, "fold_best_metrics": fold_best_metrics}
    wandb.save("final_ensemble_info.json")
    
    wandb.finish()

if __name__ == '__main__':
    main()
