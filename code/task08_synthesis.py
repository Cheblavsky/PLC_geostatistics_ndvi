# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv, datetime
from utils import read_ndvi, ensure_dir

DATA = Path("data/ndvi_s2_2024_growing_season.tif")
OUT  = Path("outputs")
RPT  = Path("reports")

T_ROBUST = 0.60; T_RISK = 0.30

def percentiles(arr, qs=(10,25,50,75,90)):
    c = arr[np.isfinite(arr)]
    if c.size==0: return {q: np.nan for q in qs}
    v = np.percentile(c, qs)
    return {q: float(x) for q, x in zip(qs, v)}

def share_ge(arr, t): 
    c = arr[np.isfinite(arr)]
    return float(np.mean(c>=t)) if c.size else np.nan

def share_le(arr, t):
    c = arr[np.isfinite(arr)]
    return float(np.mean(c<=t)) if c.size else np.nan

def north_south(a2d):
    H,W=a2d.shape; half=H//2
    n=a2d[:half,:]; s=a2d[half:,:]
    n=n[np.isfinite(n)]; s=s[np.isfinite(s)]
    return dict(
        north_mean=float(np.mean(n)) if n.size else np.nan,
        south_mean=float(np.mean(s)) if s.size else np.nan,
        north_median=float(np.median(n)) if n.size else np.nan,
        south_median=float(np.median(s)) if s.size else np.nan,
        delta_mean=float(np.mean(n)-np.mean(s)) if (n.size and s.size) else np.nan
    )

def write_csv(p:Path, d:dict):
    import pandas as pd; pd.DataFrame([d]).to_csv(p, index=False)

def write_md(p:Path, ind:dict, r1:Path, r2:Path, ppng:Path):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    md=f"""# Task 08 — Geoscience Synthesis

**Date:** {now}

## Key Indicators
- p10/p25/p50/p75/p90: {ind['p10']:.3f} / {ind['p25']:.3f} / {ind['p50']:.3f} / {ind['p75']:.3f} / {ind['p90']:.3f}
- Robust share (NDVI ≥ {ind['T_ROBUST']:.2f}): {ind['share_robust']*100:.1f}%
- Risk   share (NDVI ≤ {ind['T_RISK']:.2f}): {ind['share_risk']*100:.1f}%
- North mean/median: {ind['north_mean']:.3f} / {ind['north_median']:.3f}
- South mean/median: {ind['south_mean']:.3f} / {ind['south_median']:.3f}
- Δ mean (N - S): {ind['delta_mean']:.3f}

## Maps
![Robust]({r1.as_posix()})
![Risk]({r2.as_posix()})
![Percentiles]({ppng.as_posix()})
"""
    p.write_text(md, encoding="utf-8")

def main():
    ensure_dir(OUT); ensure_dir(RPT)
    nd, nd2d, _ = read_ndvi(DATA)
    if nd.size<100: raise RuntimeError("Too few valid NDVI pixels.")

    P = percentiles(nd); ns = north_south(nd2d)
    ind = dict(p10=P[10], p25=P[25], p50=P[50], p75=P[75], p90=P[90],
               share_robust=share_ge(nd,T_ROBUST), share_risk=share_le(nd,T_RISK),
               T_ROBUST=T_ROBUST, T_RISK=T_RISK, **ns)

    write_csv(OUT/"task08_indicators.csv", ind)

    # Robust map
    rob = (nd2d>=T_ROBUST).astype(float); rob[~np.isfinite(nd2d)] = np.nan
    plt.figure(); im=plt.imshow(rob, vmin=0, vmax=1, interpolation="nearest")
    plt.title(f"Robust Vegetation (≥ {T_ROBUST:.2f})"); plt.axis("off")
    plt.colorbar(im, fraction=0.03, pad=0.02, label="1=Robust"); r1=OUT/"task08_robust_map.png"
    plt.tight_layout(); plt.savefig(r1, dpi=220); plt.close()

    # Risk map
    risk = (nd2d<=T_RISK).astype(float); risk[~np.isfinite(nd2d)] = np.nan
    plt.figure(); im=plt.imshow(risk, vmin=0, vmax=1, interpolation="nearest")
    plt.title(f"Degradation Risk (≤ {T_RISK:.2f})"); plt.axis("off")
    plt.colorbar(im, fraction=0.03, pad=0.02, label="1=Risk"); r2=OUT/"task08_risk_map.png"
    plt.tight_layout(); plt.savefig(r2, dpi=220); plt.close()

    # Percentiles bar
    labels=["p10","p25","p50","p75","p90"]; vals=[ind[k] for k in labels]
    plt.figure(); plt.bar(labels, vals); plt.ylim(0,1); plt.ylabel("NDVI"); plt.title("NDVI Percentiles")
    ppng=OUT/"task08_ndvi_percentiles.png"; plt.tight_layout(); plt.savefig(ppng, dpi=220); plt.close()

    write_md(RPT/"task08_report.md", ind, r1, r2, ppng)
    print("✅ Task 8 done → outputs/ & reports/")

if __name__ == "__main__":
    main()
