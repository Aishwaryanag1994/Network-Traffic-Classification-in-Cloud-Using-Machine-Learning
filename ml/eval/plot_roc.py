from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc(y_true, y_score, out_path: str = "roc_curve.png", title: str = "ROC Curve"):
    """
    y_true: array-like of shape (n_samples,) with binary labels {0,1}
    y_score: array-like of shape (n_samples,) with model scores/probabilities for the positive class
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    return out_path, roc_auc

if __name__ == "__main__":
  
    import numpy as np
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(size=200)
    out, auc_val = plot_roc(y_true, y_score)
    print(f"Saved {out} with AUC={auc_val:.3f}")