import numpy as np


def center_crop(image, crop_size=256):
    """Center-crop a 2D image to (crop_size, crop_size)."""
    h, w = image.shape
    cy, cx = h // 2, w // 2
    half = crop_size // 2
    return image[cy - half : cy + half, cx - half : cx + half]


def normalize(image):
    """Min-max normalize an image to [0, 1]."""
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val == 0:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)


def preprocess_image(image, crop_size=256):
    """Full preprocessing: center crop then normalize."""
    cropped = center_crop(image, crop_size)
    normed = normalize(cropped)
    return normed
