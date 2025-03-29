import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import wandb

def wandb_table_top_losses(results, top_n=10):
    # Check if results include extra metadata (Study, Patient, Laterality)
    if results and len(results[0]) >= 7:
        columns = ["Image", "GT", "Probability", "Loss", "Study", "Patient", "Laterality"]
    else:
        columns = ["Image", "GT", "Probability", "Loss"]
    
    results_sorted = sorted(results, key=lambda x: x[3], reverse=True)
    top_results = results_sorted[:top_n]
    table = wandb.Table(columns=columns)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for res in top_results:
        if len(res) >= 7:
            img, label, prob, loss, study, patient, laterality = res
        else:
            img, label, prob, loss = res
        img_denorm = img.clone().detach() * std + mean
        img_denorm = img_denorm.clamp(0, 1)
        np_img = img_denorm.permute(1, 2, 0).cpu().numpy()
        wb_img = wandb.Image(np_img, caption=f"GT: {int(label)}, Prob: {prob:.2f}, Loss: {loss:.2f}")
        if len(res) >= 7:
            table.add_data(wb_img, int(label), prob, loss, study, patient, laterality)
        else:
            table.add_data(wb_img, int(label), prob, loss)
        
    return table

def get_confusion_matrix_fig(y_true, y_pred, title_prefix="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"{title_prefix}")
    plt.tight_layout()
    return fig

def get_roc_curve_fig(y_true, y_probs, title_prefix="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], 'k--', label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title_prefix}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig
