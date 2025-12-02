# classifier/train_clf.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import os

from .models_clf import build_efficientnet
from .utils_clf import load_clf_config, build_dataloaders, DATA_ROOT, PROJECT_ROOT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one(model_name: str, cfg):
    batch_size   = cfg["batch_size"]
    epochs       = cfg["epochs"]
    lr           = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    seed         = cfg["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, class_names = build_dataloaders(batch_size)
    num_classes = len(class_names)

    model = build_efficientnet(model_name, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    save_root = DATA_ROOT / cfg["save_dir"]
    save_root.mkdir(parents=True, exist_ok=True)
    best_path = save_root / f"{model_name}_best.pth"

    best_val_acc = 0.0

    print(f"\n======================================================================")
    print(f"   Train classifier: {model_name}")
    print(f"   save_dir={best_path}")
    print(f"   DEVICE={DEVICE}")
    print("======================================================================")

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        run_loss = 0.0
        run_correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            run_correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = run_loss / total
        train_acc  = run_correct / total

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc  = val_correct / val_total

        print(
            f"[{model_name}] Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(
                f"  ({model_name}) Best updated! "
                f"val_acc={best_val_acc:.4f}, saved to {best_path}"
            )

    return model_name, best_val_acc, best_path, class_names


def main():
    cfg = load_clf_config()
    print(f"[INFO] DEVICE = {DEVICE}")

    summary = []
    for m in cfg["models"]:
        name, best_acc, path, classes = train_one(m, cfg)
        summary.append((name, best_acc, path))

    print("\n All EfficientNet models finished!")
    print("=== Summary (val_acc) ===")
    for name, acc, path in summary:
        print(f"{name:<15} | best_val_acc = {acc:.4f} | ckpt = {path}")
    print(f"\nClass order (for all models): {classes}")


if __name__ == "__main__":
    main()
