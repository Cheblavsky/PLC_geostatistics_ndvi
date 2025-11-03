# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, geom, poisson, expon, gamma as gamma_dist, norm, lognorm, beta as beta_dist
from utils import read_ndvi, freedman_diaconis_bins, ensure_dir

DATA = Path("data/ndvi_s2_2024_growing_season.tif")
OUT  = Path("outputs")

T = 0.5; BINOM_N = 50; BOOTSTRAP = 1000
TILE_H = TILE_W = 50
GEOM_SEQ_LEN = 200; GAMMA_GROUP = 3

def stem_compat(x, y, *, label):
    try: return plt.stem(x, y, linefmt="-", markerfmt="o", basefmt=" ", label=label, use_line_collection=True)
    except TypeError: return plt.stem(x, y, linefmt="-", markerfmt="o", basefmt=" ", label=label)

def to_event(x, thr): return x >= thr

def count_tiles(band2d, thr, th, tw):
    H, W = band2d.shape; out=[]
    for r0 in range(0,H,th):
        r1=min(r0+th,H)
        for c0 in range(0,W,tw):
            c1=min(c0+tw,W)
            tile = band2d[r0:r1,c0:c1]
            tile = tile[np.isfinite(tile)]
            if tile.size==0: continue
            out.append(int(np.count_nonzero(tile>=thr)))
    return np.array(out) if out else np.array([0])

def gaps_between(events_flat):
    g=0; arr=[]
    for e in events_flat:
        if e: arr.append(g); g=0
        else: g+=1
    return np.array(arr, float)

def main():
    ensure_dir(OUT)
    ndvi, band2d, _ = read_ndvi(DATA)
    if ndvi.size<100: raise RuntimeError("Too few valid NDVI pixels.")
    events = to_event(ndvi, T)

    # Binomial
    idx = np.arange(events.size)
    ks = [int(events[np.random.choice(idx, size=BINOM_N, replace=True)].sum()) for _ in range(BOOTSTRAP)]
    ks = np.array(ks)
    k = np.arange(0,BINOM_N+1); p_hat=float(events.mean())
    pmf = binom.pmf(k, BINOM_N, p_hat)
    plt.figure()
    plt.hist(ks, bins=np.arange(-0.5,BINOM_N+1.5,1), density=True, alpha=0.6, label="Empirical")
    stem_compat(k, pmf, label=f"Binomial(n={BINOM_N}, p={p_hat:.2f})")
    plt.xlabel("Successes"); plt.ylabel("Probability"); plt.title(f"Binomial (T={T})")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT/"task06_binomial.png", dpi=200); plt.close()

    # Geometric
    N = events.size; ks=[]
    for _ in range(5000):
        seq = events[np.random.randint(0,N,size=GEOM_SEQ_LEN)]
        w = np.where(seq)[0]
        if w.size>0: ks.append(int(w[0])+1)
    ks = np.array(ks) if ks else np.array([1])
    kmax = int(max(np.percentile(ks,99),10))
    k = np.arange(1,kmax+1); pmf = geom.pmf(k, p_hat if p_hat>0 else 1e-9)
    plt.figure()
    plt.hist(ks, bins=np.arange(0.5,kmax+1.5,1), density=True, alpha=0.6, label="Empirical")
    stem_compat(k, pmf, label=f"Geometric(p={p_hat:.2f})")
    plt.xlabel("Trials until first success"); plt.ylabel("Probability")
    plt.title(f"Geometric (T={T})"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT/"task06_geometric.png", dpi=200); plt.close()

    # Poisson
    counts = count_tiles(band2d, T, TILE_H, TILE_W)
    lam = float(counts.mean()); kmax = int(max(counts.max(initial=0), lam+6*np.sqrt(max(lam,1e-9))))
    k = np.arange(0,kmax+1); pmf = poisson.pmf(k, lam)
    plt.figure()
    plt.hist(counts, bins=np.arange(-0.5,kmax+1.5,1), density=True, alpha=0.6, label="Empirical")
    stem_compat(k, pmf, label=f"Poisson(λ={lam:.2f})")
    plt.xlabel("Success count per tile"); plt.ylabel("Probability")
    plt.title(f"Poisson (tile={TILE_H}x{TILE_W})"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT/"task06_poisson.png", dpi=200); plt.close()

    # Exponential
    finite = band2d[np.isfinite(band2d)]
    gaps = gaps_between(finite>=T); pos = gaps[gaps>0]; 
    if pos.size<5: pos=gaps
    lam = 1.0/float(np.mean(pos)) if pos.size>0 else 1.0
    x = np.linspace(0, np.percentile(pos,99), 400)
    plt.figure()
    plt.hist(pos, bins=freedman_diaconis_bins(pos,50), density=True, alpha=0.6, label="Empirical gaps")
    plt.plot(x, expon(scale=1/lam).pdf(x), label=f"Exponential(λ={lam:.3f})")
    plt.xlabel("Gap length"); plt.ylabel("Density"); plt.title("Exponential")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT/"task06_exponential.png", dpi=200); plt.close()

    # Gamma
    group=GAMMA_GROUP; pos = gaps[gaps>0]
    if pos.size<group*10: pos=gaps[gaps>=0]
    n = pos.size//group
    if n<5: grouped = pos
    else:
        grouped = pos[:n*group].reshape(n,group).mean(axis=1)
    m=float(np.mean(grouped)); v=float(np.var(grouped, ddof=1)) if grouped.size>1 else 1.0
    k = (m**2)/v if v>0 else 1.0; theta = v/m if m>0 else 1.0
    x = np.linspace(0, np.percentile(grouped,99), 400)
    plt.figure()
    plt.hist(grouped, bins=freedman_diaconis_bins(grouped,50), density=True, alpha=0.6, label="Empirical")
    plt.plot(x, gamma_dist(a=k, scale=theta).pdf(x), label=f"Gamma(k={k:.2f}, θ={theta:.2f})")
    plt.xlabel("Grouped gap"); plt.ylabel("Density"); plt.title(f"Gamma (group={group})")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT/"task06_gamma.png", dpi=200); plt.close()

    # Normal / Lognormal / Beta
    nd = ndvi[(ndvi>=0)&(ndvi<=1)]
    bins = freedman_diaconis_bins(nd,60); x = np.linspace(float(nd.min()), float(nd.max()), 500)
    plt.figure(); plt.hist(nd, bins=bins, density=True, alpha=0.5, label="NDVI hist")
    mu, sigma = float(np.mean(nd)), float(np.std(nd, ddof=1)); sigma = sigma if sigma>0 else 1e-6
    plt.plot(x, norm(mu, sigma).pdf(x), label=f"Normal(μ={mu:.3f}, σ={sigma:.3f})")
    pos = nd[nd>0]
    if pos.size>3:
        s, loc, scale = lognorm.fit(pos, floc=0); mask = x>0
        plt.plot(x[mask], lognorm(s, loc, scale).pdf(x[mask]), label=f"Lognormal(σ={s:.3f})")
    eps=1e-6; beta_samp = nd[(nd>eps)&(nd<1-eps)]
    if beta_samp.size>5:
        a,b,_,_ = beta_dist.fit(beta_samp, floc=0, fscale=1)
        plt.plot(x, beta_dist(a,b,loc=0,scale=1).pdf(x), label=f"Beta(α={a:.2f}, β={b:.2f})")
    plt.xlabel("NDVI"); plt.ylabel("Density"); plt.title("Normal / Lognormal / Beta")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT/"task06_normal_lognormal_beta.png", dpi=200); plt.close()

    print("✅ Task 6 done → outputs/ (6 figures)")

if __name__ == "__main__":
    main()
