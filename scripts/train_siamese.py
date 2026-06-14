"""Train a Siamese matching network for ref icon → canvas shape matching.

Architecture:
- Shared backbone: lightweight CNN or frozen DINOv2 + projection head
- Input: (ref_crop, canvas_crop) pair
- Output: similarity score (0-1)
- Loss: contrastive loss (positive pairs = same shape, negative = different)
"""
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SiamesePairDataset(Dataset):
    """Dataset of (ref_crop, canvas_crop, label) pairs."""
    
    def __init__(self, manifest_path: str, pairs_dir: str, img_size: int = 64):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.pairs_dir = pairs_dir
        self.img_size = img_size
    
    def __len__(self):
        return len(self.manifest)
    
    def _load_and_preprocess(self, filename: str) -> torch.Tensor:
        path = os.path.join(self.pairs_dir, filename)
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [-1, 1]
        tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 127.5 - 1.0
        return tensor
    
    def __getitem__(self, idx):
        entry = self.manifest[idx]
        ref = self._load_and_preprocess(entry["ref"])
        cand = self._load_and_preprocess(entry["candidate"])
        label = torch.tensor(entry["label"], dtype=torch.float32)
        return ref, cand, label


class SiameseEncoder(nn.Module):
    """Lightweight CNN encoder for shape crops."""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Global average pool
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.projection = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        x = self.projection(x)
        x = F.normalize(x, dim=-1)
        return x


class SiameseMatchNet(nn.Module):
    """Full Siamese network with shared encoder."""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.encoder = SiameseEncoder(embed_dim)
    
    def forward(self, ref: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """Returns cosine similarity between ref and cand embeddings."""
        ref_emb = self.encoder(ref)
        cand_emb = self.encoder(cand)
        similarity = F.cosine_similarity(ref_emb, cand_emb, dim=-1)
        return similarity
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single image."""
        return self.encoder(x)


def contrastive_loss(similarity: torch.Tensor, labels: torch.Tensor, margin: float = 0.5):
    """Contrastive loss: push matching pairs together, non-matching apart.
    
    - For positive pairs (label=1): loss = (1 - similarity)^2
    - For negative pairs (label=0): loss = max(0, similarity - margin)^2
    """
    pos_loss = labels * (1 - similarity).pow(2)
    neg_loss = (1 - labels) * F.relu(similarity - margin).pow(2)
    return (pos_loss + neg_loss).mean()


def train(
    data_dir: str = "data/siamese",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    embed_dim: int = 128,
    device: str = "cpu",
):
    manifest_path = os.path.join(data_dir, "manifest.json")
    pairs_dir = os.path.join(data_dir, "pairs")
    
    # Load dataset
    dataset = SiamesePairDataset(manifest_path, pairs_dir)
    
    # Train/val split
    n = len(dataset)
    n_val = max(1, int(n * 0.15))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Training: {n_train} pairs, Validation: {n_val} pairs")
    
    # Model
    model = SiameseMatchNet(embed_dim=embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    save_dir = os.path.join("runs", "siamese")
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for ref, cand, labels in train_loader:
            ref, cand, labels = ref.to(device), cand.to(device), labels.to(device)
            
            similarity = model(ref, cand)
            loss = contrastive_loss(similarity, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            predictions = (similarity > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += len(labels)
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for ref, cand, labels in val_loader:
                ref, cand, labels = ref.to(device), cand.to(device), labels.to(device)
                similarity = model(ref, cand)
                loss = contrastive_loss(similarity, labels)
                
                val_loss += loss.item() * len(labels)
                predictions = (similarity > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += len(labels)
        
        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100
        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total
        
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: loss={avg_train_loss:.4f} acc={train_acc:.1f}% | "
                f"Val: loss={avg_val_loss:.4f} acc={val_acc:.1f}%"
            )
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
    
    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.1f}%")
    print(f"Model saved: {save_dir}/best.pt")
    
    # Save model info
    info = {
        "embed_dim": embed_dim,
        "img_size": 64,
        "best_val_acc": best_val_acc,
        "epochs": epochs,
    }
    with open(os.path.join(save_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    train()
