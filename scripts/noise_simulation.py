import numpy as np


def add_gaussian_noise(image, sigma=0.05, rng=None):
    """Add additive Gaussian noise to simulate thermal / receiver noise."""
    rng = np.random.default_rng(rng)
    noise = rng.normal(0, sigma, size=image.shape)
    return image + noise


def add_poisson_noise(image, peak_counts=1000, rng=None):
    """Add Poisson (shot) noise scaled to a given peak photon count."""
    rng = np.random.default_rng(rng)
    scaled = image * peak_counts
    noisy = rng.poisson(np.clip(scaled, 0, None)).astype(np.float64)
    return noisy / peak_counts


def add_telescope_noise(image, gaussian_sigma=0.03, poisson_peak=1000, rng=None):
    """
    Simulate realistic telescope noise by combining:
      1. Poisson (photon shot) noise
      2. Gaussian (thermal / instrumental) noise
    Returns the noisy image (not clipped — caller decides).
    """
    rng = np.random.default_rng(rng)
    noisy = add_poisson_noise(image, peak_counts=poisson_peak, rng=rng)
    noisy = add_gaussian_noise(noisy, sigma=gaussian_sigma, rng=rng)
    return noisy


def make_training_pair(clean_image, gaussian_sigma=0.03, poisson_peak=1000, rng=None):
    """Return (noisy, clean) training pair from a clean preprocessed image."""
    noisy = add_telescope_noise(
        clean_image,
        gaussian_sigma=gaussian_sigma,
        poisson_peak=poisson_peak,
        rng=rng,
    )
    return noisy, clean_image
