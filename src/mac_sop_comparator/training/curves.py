"""Training-curve figure: losses and validation ROC-AUC per epoch."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402


def plot_training_curve(history: list[dict], best_epoch: int, path: str | Path, title: str = "") -> Path:
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax_loss.set(xlabel="epoch", ylabel="weighted cross-entropy", title="Loss")
    ax_auc.plot(epochs, [h["val_auc"] for h in history], color="tab:green")
    ax_auc.set(xlabel="epoch", ylabel="validation ROC-AUC", title="Validation ROC-AUC")
    for ax in (ax_loss, ax_auc):
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best epoch ({best_epoch})")
        ax.grid(alpha=0.3)
    ax_loss.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
