import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    DnCNN denoiser — a residual-learning CNN that predicts the noise layer.
    Architecture: Conv+ReLU -> (Conv+BN+ReLU) x (depth-2) -> Conv
    Input/output: single-channel 2D images (B, 1, H, W).
    """

    def __init__(self, depth=17, n_channels=64, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        layers = []

        # First layer: Conv + ReLU (no batch norm)
        layers.append(nn.Conv2d(1, n_channels, kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: Conv + BN + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: Conv (predicts noise residual)
        layers.append(nn.Conv2d(n_channels, 1, kernel_size, padding=padding, bias=True))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise  # residual learning: clean = noisy - predicted_noise
