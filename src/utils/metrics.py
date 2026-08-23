import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Computes classification metrics for binary classification.
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    targets_np = targets.cpu().numpy()
    
    try:
        roc_auc = roc_auc_score(targets_np, probs)
    except ValueError:
        roc_auc = 0.5 # Default if only one class is present in batch
        
    try:
        pr_auc = average_precision_score(targets_np, probs)
    except ValueError:
        pr_auc = 0.0
        
    f1 = f1_score(targets_np, preds, zero_division=0)
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1': f1
    }
