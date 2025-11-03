# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import read_ndvi, ensure_dir, freedman_diaconis_bins

DATA = Path("data/ndvi_s2_2024_growing_season.tif")
OUT  = Path("outputs")
RPT  = Path("reports")

def main():
    ensure_dir(OUT); ensure_dir(RPT)
    ndvi, ndvi2d, meta = read_ndvi(DATA)

    # 1) 快视图
    plt.figure()
    im = plt.imshow(ndvi2d, vmin=0, vmax=1, interpolation="nearest")
    plt.title("NDVI Quicklook"); plt.axis("off")
    plt.colorbar(im, fraction=0.03, pad=0.02, label="NDVI")
    ql = OUT / "task02_quicklook.png"
    plt.tight_layout(); plt.savefig(ql, dpi=220); plt.close()

    # 2) 直方图
    bins = freedman_diaconis_bins(ndvi, 60)
    plt.figure()
    plt.hist(ndvi, bins=bins, density=False, alpha=0.8)
    plt.xlabel("NDVI"); plt.ylabel("Count"); plt.title("NDVI Histogram")
    hist_png = OUT / "task02_hist.png"
    plt.tight_layout(); plt.savefig(hist_png, dpi=220); plt.close()

    # 3) 摘要（CSV + JSON）
    summary = dict(
        count=int(ndvi.size),
        mean=float(np.nanmean(ndvi)),
        std=float(np.nanstd(ndvi)),
        min=float(np.nanmin(ndvi)),
        p25=float(np.nanpercentile(ndvi,25)),
        median=float(np.nanmedian(ndvi)),
        p75=float(np.nanpercentile(ndvi,75)),
        max=float(np.nanmax(ndvi)),
        crs=str(meta.get("crs")),
        transform=str(meta.get("transform"))
    )
    pd.DataFrame([summary]).to_csv(OUT / "task02_summary.csv", index=False)
    (OUT / "task02_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("✅ Task 3 done → outputs/: task02_quicklook.png, task02_hist.png, task02_summary.*")

if __name__ == "__main__":
    main()
