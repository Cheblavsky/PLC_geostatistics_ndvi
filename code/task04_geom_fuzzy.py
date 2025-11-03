# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils import read_ndvi, ensure_dir

DATA = Path("data/ndvi_s2_2024_growing_season.tif")
OUT  = Path("outputs")

# 参数
T = 0.5          # 几何概率阈值：NDVI>=T
BLOCK = 25       # 25x25 像元 ≈ 250 m
FUZZY_A = 0.20   # 隶属函数下端（<=A → 0）
FUZZY_B = 0.60   # 隶属函数上端（>=B → 1）

def block_prob_ge(ndvi2d, t=0.5, block=25):
    H, W = ndvi2d.shape
    h = (H // block) * block
    w = (W // block) * block
    a = ndvi2d[:h, :w]
    # reshape 为 [nBlockRows, block, nBlockCols, block]
    A = a.reshape(h//block, block, w//block, block)
    # 每个块的概率：>=t 的比例
    ge = (A >= t).sum(axis=(1,3)) / np.isfinite(A).sum(axis=(1,3))
    return ge

def fuzzy_membership(ndvi2d, a=0.2, b=0.6):
    mu = (ndvi2d - a) / (b - a)
    mu = np.clip(mu, 0, 1)
    mu[~np.isfinite(ndvi2d)] = np.nan
    return mu

def main():
    ensure_dir(OUT)
    _, ndvi2d, _ = read_ndvi(DATA)

    # Crisp geom prob
    prob = block_prob_ge(ndvi2d, T, BLOCK)
    plt.figure()
    im = plt.imshow(prob, vmin=0, vmax=1, interpolation="nearest")
    plt.title(f"Crisp Geometrical Probability P(NDVI ≥ {T}) ~{BLOCK*10} m grid")
    plt.axis("off"); plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.tight_layout(); plt.savefig(OUT / "task04_geom_prob_map.png", dpi=220); plt.close()

    # Fuzzy
    mu = fuzzy_membership(ndvi2d, FUZZY_A, FUZZY_B)
    plt.figure()
    im = plt.imshow(mu, vmin=0, vmax=1, interpolation="nearest")
    plt.title(f"Fuzzy Membership μ_veg (A={FUZZY_A}, B={FUZZY_B})")
    plt.axis("off"); plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.tight_layout(); plt.savefig(OUT / "task04_fuzzy_map.png", dpi=220); plt.close()

    print("✅ Task 4 done → outputs/: task04_geom_prob_map.png, task04_fuzzy_map.png")

if __name__ == "__main__":
    main()
