from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 在云端不弹窗
import matplotlib.pyplot as plt
import rasterio

def read_ndvi(path: Path):
    """Read NDVI GeoTIFF -> (flat_ndvi, ndvi_2d, meta)"""
    with rasterio.open(path) as src:
        band = src.read(1).astype(float)
        meta = src.meta.copy()
    band[(band < -1.0) | (band > 1.0)] = np.nan
    flat = band[np.isfinite(band)]
    return flat, band, meta

def freedman_diaconis_bins(x: np.ndarray, default: int = 50):
    x = x[np.isfinite(x)]
    if x.size < 2:
        return default
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return default
    h = 2 * iqr * (x.size ** (-1/3))
    if h <= 0:
        return default
    bins = int(np.ceil((x.max() - x.min()) / h))
    return max(bins, 10)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
