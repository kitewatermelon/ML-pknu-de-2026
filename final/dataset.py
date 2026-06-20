import torch
from torch.utils.data import DataLoader
from medmnist import RetinaMNIST
from torchvision.transforms import (
    RandomResizedCrop, ColorJitter, RandomHorizontalFlip,
    RandomRotation, Compose, ToTensor, Normalize
)

_mean = [0.485, 0.456, 0.406]
_std  = [0.229, 0.224, 0.225]

train_transform = Compose([
    RandomResizedCrop(224, scale=(0.8, 1.0)),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    RandomRotation(15),
    ToTensor(),
    Normalize(mean=_mean, std=_std),
])

val_transform = Compose([
    ToTensor(),
    Normalize(mean=_mean, std=_std),
])

target_transform = lambda y: torch.tensor(y, dtype=torch.long).squeeze()

train_ds = RetinaMNIST(split="train", size=224, transform=train_transform, target_transform=target_transform, download=True)
val_ds   = RetinaMNIST(split="val",   size=224, transform=val_transform,   target_transform=target_transform, download=True)
test_ds  = RetinaMNIST(split="test",  size=224, transform=val_transform,   target_transform=target_transform, download=True)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
