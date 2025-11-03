# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from utils import read_ndvi, freedman_diaconis_bins, ensure_dir

DATA = Path("data/ndvi_s2_2024_growing_season.tif")
OUT  = Path("outputs")

def plot_pdf(ndvi, out_png: Path):
    bins = freedman_diaconis_bins(ndvi, 50)
    plt.figure()
    plt.hist(ndvi, bins=bins, density=True, alpha=0.6, label="Histogram")
    if ndvi.size > 100:
        kde = gaussian_kde(ndvi)
        x = np.linspace(float(ndvi.min()), float(ndvi.max()), 400)
        plt.plot(x, kde(x), label="KDE")
    plt.xlabel("NDVI"); plt.ylabel("Density")
    plt.title("NDVI PDF"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

def plot_cdf(ndvi, out_png: Path):
    x = np.sort(ndvi)
    y = np.arange(1, x.size+1) / x.size
    plt.figure()
    plt.plot(x, y, "-")
    plt.xlabel("NDVI"); plt.ylabel("Cumulative Probability")
    plt.title("NDVI ECDF"); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

def plot_NS(ndvi2d, out_png: Path):
    H, W = ndvi2d.shape
    half = H // 2
    north = ndvi2d[:half,:].ravel()
    south = ndvi2d[half:,:].ravel()
    north = north[np.isfinite(north)]
    south = south[np.isfinite(south)]
    bins = freedman_diaconis_bins(np.concatenate([north, south]), 50)
    plt.figure()
    plt.hist(north, bins=bins, alpha=0.6, density=True, label="North")
    plt.hist(south, bins=bins, alpha=0.6, density=True, label="South")
    plt.xlabel("NDVI"); plt.ylabel("Density"); plt.title("North vs South")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

def main():
    ensure_dir(OUT)
    ndvi, ndvi2d, _ = read_ndvi(DATA)
    plot_pdf(ndvi, OUT / "task05_pdf.png")
    plot_cdf(ndvi, OUT / "task05_cdf.png")
    plot_NS(ndvi2d, OUT / "task05_region_compare.png")
    print("✅ Task 5 done → outputs/")

if __name__ == "__main__":
    main()
