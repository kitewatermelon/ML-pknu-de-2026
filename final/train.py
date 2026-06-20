import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torchmetrics
import wandb

from dataset import train_loader, val_loader, test_loader
from model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_CLASSES = 5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_metrics():
    return {
        "acc":  torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE),
        "auc":  torchmetrics.AUROC(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE),
        "f1":   torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(DEVICE),
    }


def train_epoch(model, loader, criterion, optimizer, metrics):
    model.train()
    for m in metrics.values():
        m.reset()
    total_loss, total = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        probs = outputs.softmax(dim=1)
        metrics["acc"].update(probs, labels)
        metrics["auc"].update(probs, labels)
        metrics["f1"].update(probs, labels)
    return (
        total_loss / total,
        metrics["acc"].compute().item(),
        metrics["auc"].compute().item(),
        metrics["f1"].compute().item(),
    )


@torch.no_grad()
def eval_epoch(model, loader, criterion, metrics):
    model.eval()
    for m in metrics.values():
        m.reset()
    total_loss, total = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        probs = outputs.softmax(dim=1)
        metrics["acc"].update(probs, labels)
        metrics["auc"].update(probs, labels)
        metrics["f1"].update(probs, labels)
    return (
        total_loss / total,
        metrics["acc"].compute().item(),
        metrics["auc"].compute().item(),
        metrics["f1"].compute().item(),
    )


def run(pretrained: bool, epochs: int, patience: int):
    tag = "pretrained" if pretrained else "scratch"
    ckpt_path = f"best_model_{tag}.pth"

    set_seed(SEED)
    wandb.init(
        project="retinamnist-resnet18",
        name=f"resnet18-{tag}",
        config={
            "epochs": epochs, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "batch_size": 32, "model": "resnet18",
            "pretrained": pretrained, "seed": SEED, "patience": patience,
        },
        reinit=True,
    )

    model = get_model(pretrained=pretrained).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_metrics = make_metrics()
    val_metrics = make_metrics()

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_auc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, train_metrics)
        vl_loss, vl_acc, vl_auc, vl_f1 = eval_epoch(model, val_loader, criterion, val_metrics)

        print(
            f"[{tag}] Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  AUC: {tr_auc:.4f}  F1: {tr_f1:.4f} | "
            f"Val   Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}  AUC: {vl_auc:.4f}  F1: {vl_f1:.4f}"
        )
        wandb.log({
            "epoch": epoch,
            "train/loss": tr_loss, "train/acc": tr_acc, "train/auc": tr_auc, "train/f1": tr_f1,
            "val/loss":   vl_loss, "val/acc":   vl_acc, "val/auc":   vl_auc, "val/f1":   vl_f1,
        })

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  => best model saved → {ckpt_path} (val_acc={vl_acc:.4f})")
        else:
            no_improve += 1
            print(f"  => no improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path))
    te_loss, te_acc, te_auc, te_f1 = eval_epoch(model, test_loader, criterion, make_metrics())
    print(f"\n[{tag}] Test Loss: {te_loss:.4f}  Acc: {te_acc:.4f}  AUC: {te_auc:.4f}  F1: {te_f1:.4f}\n")
    wandb.log({"test/loss": te_loss, "test/acc": te_acc, "test/auc": te_auc, "test/f1": te_f1})
    wandb.finish()


if __name__ == "__main__":
    run(pretrained=True,  epochs=20,  patience=5)
    run(pretrained=False, epochs=100, patience=20)
