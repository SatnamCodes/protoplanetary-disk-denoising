import numpy as np
from astropy.io import fits
from pathlib import Path


def load_fits_image(filepath):
    """Load a FITS file and return the squeezed 2D image array."""
    with fits.open(filepath) as hdu_list:
        data = hdu_list[0].data
    return np.squeeze(data).astype(np.float64)


def load_all_disks(raw_dir):
    """Load all FITS files from a directory and return a dict of {name: 2D array}."""
    raw_dir = Path(raw_dir)
    disk_images = {}
    for fits_path in sorted(raw_dir.glob("*.fits")):
        name = fits_path.stem  # e.g. "AS205_continuum"
        disk_images[name] = load_fits_image(fits_path)
    return disk_images
