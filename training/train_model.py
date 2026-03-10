import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class DiskPatchDataset(Dataset):
    """Dataset of (noisy, clean) patch pairs stored as .npy files."""

    def __init__(self, noisy_dir, clean_dir):
        self.noisy_paths = sorted(Path(noisy_dir).glob("*.npy"))
        self.clean_paths = sorted(Path(clean_dir).glob("*.npy"))
        assert len(self.noisy_paths) == len(self.clean_paths), "Mismatch between noisy and clean patches"

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        noisy = np.load(self.noisy_paths[idx]).astype(np.float32)
        clean = np.load(self.clean_paths[idx]).astype(np.float32)
        # Add channel dimension: (H, W) -> (1, H, W)
        noisy = torch.from_numpy(noisy).unsqueeze(0)
        clean = torch.from_numpy(clean).unsqueeze(0)
        return noisy, clean


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * noisy.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = criterion(output, clean)
            total_loss += loss.item() * noisy.size(0)
    return total_loss / len(loader.dataset)


def train_model(model, train_dir, val_dir, *, epochs=50, batch_size=16, lr=1e-3,
                device=None, save_path=None):
    """
    Full training loop.

    Parameters
    ----------
    model : nn.Module
    train_dir : str  — directory containing noisy/ and clean/ subdirs for training
    val_dir : str    — directory containing noisy/ and clean/ subdirs for validation
    epochs : int
    batch_size : int
    lr : float
    device : str or None
    save_path : str or None — where to save the best model weights
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    train_ds = DiskPatchDataset(Path(train_dir) / "noisy", Path(train_dir) / "clean")
    val_ds = DiskPatchDataset(Path(val_dir) / "noisy", Path(val_dir) / "clean")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:03d}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  -> saved best model (val_loss={val_loss:.6f})")

    return history
