# classifier/utils_clf.py
from pathlib import Path
import os
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "driver"
DATA_ROOT = Path(os.environ.get("DDD_DATA_ROOT", str(DEFAULT_DATA_ROOT)))

CONFIG_DIR = PROJECT_ROOT / "configs"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def load_clf_config():
    cfg_path = CONFIG_DIR / "clf_config.json"
    with open(cfg_path, "r") as f:
        return json.load(f)

def build_dataloaders(batch_size: int, num_workers: int = 2):
    train_dir = DATA_ROOT / "cls_crops_v2" / "train"
    val_dir   = DATA_ROOT / "cls_crops_v2" / "valid"

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, train_ds.classes
