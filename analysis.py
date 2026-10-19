"""Ensemble score-based diffusion on sklearn digits:
arithmetic-mean score aggregation (baseline) vs. confidence-weighted aggregation (improved)."""
import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
SEED = 42
SEEDS = [42, 43, 44]          # ensemble training + sampling is randomized -> repeat
N_MEMBERS = 5
T = 100                       # diffusion steps (linear beta schedule)
SAMPLER_STEPS = 25            # DDIM steps (eta = 0), same for both models
N_GEN, N_PILOT, N_REAL = 1000, 200, 1000
N_PCA = 20
TEMP = 0.05                   # softmax temperature on pilot confidence (fixed a priori, not tuned)
URL = "https://scikit-learn.org/stable/datasets/toy_dataset.html"


def log(msg):
    print(f"[digits-ensemble] {msg}", flush=True)


# ---------- data: load, scale to [0,1], one stratified split ----------
data = load_digits()
X = data.data / 16.0
y = data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
N_REAL = min(N_REAL, len(X_te))
log(f"rows={len(X)} train={len(X_tr)} test={len(X_te)}")

# ---------- shared pieces: noise schedule, feature extractor, classifier ----------
betas = np.linspace(1e-4, 0.2, T)
abar = np.cumprod(1.0 - betas)
pca = PCA(n_components=N_PCA, random_state=SEED).fit(X_tr)          # fit on training data only
clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(X_tr, y_tr)  # perceptual-quality proxy
log("fitted PCA features and digit classifier on training data")


def train_ensemble(seed):
    """Train N_MEMBERS noise-prediction MLPs; each gets its own noised data draw and init."""
    members = []
    for k in range(N_MEMBERS):
        rng = np.random.default_rng(seed * 100 + k)
        X0 = np.repeat(X_tr, 8, axis=0)
        t = rng.integers(0, T, len(X0))
        eps = rng.standard_normal(X0.shape)
        xt = np.sqrt(abar[t])[:, None] * X0 + np.sqrt(1 - abar[t])[:, None] * eps
        inp = np.hstack([xt, t[:, None] / T])
        m = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=40, batch_size=256,
                         learning_rate_init=1e-3, random_state=seed * 100 + k)
        members.append(m.fit(inp, eps))
    return members


def ens_eps(members, w, x, t):
    """Weighted combination of member noise predictions (= weighted mean of scores)."""
    inp = np.hstack([x, t[:, None] / T])
    preds = np.stack([m.predict(inp) for m in members])
    return np.tensordot(w, preds, axes=1)


def sample(members, w, n, rng):
    """DDIM-style deterministic reverse sampler (eta = 0) with clipping to [0,1]."""
    x = rng.standard_normal((n, X.shape[1]))
    ts = np.linspace(T - 1, 0, SAMPLER_STEPS).round().astype(int)
    for i, t in enumerate(ts):
        eps = ens_eps(members, w, x, np.full(n, t))
        ab = abar[t]
        x0 = np.clip((x - np.sqrt(1 - ab) * eps) / np.sqrt(ab), 0, 1)
        ab_prev = abar[ts[i + 1]] if i + 1 < len(ts) else 1.0
        x = np.sqrt(ab_prev) * x0 + np.sqrt(1 - ab_prev) * eps
    return x0


def frechet(a, b):
    """Frechet distance between Gaussians fitted to two feature sets."""
    m1, m2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs = np.real(linalg.sqrtm(s1 @ s2))
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs))


def noise_mse(members, w, seed):
    """Held-out noise-prediction MSE of the ensemble on test digits at random timesteps."""
    rng = np.random.default_rng(seed + 7)
    t = rng.integers(0, T, len(X_te))
    eps = rng.standard_normal(X_te.shape)
    xt = np.sqrt(abar[t])[:, None] * X_te + np.sqrt(1 - abar[t])[:, None] * eps
    return float(np.mean((ens_eps(members, w, xt, t) - eps) ** 2))


def mean_conf(samples):
    return float(clf.predict_proba(samples).max(1).mean())


# ---------- run both models for each seed; same members, sampler, noise per seed ----------
real_feat = pca.transform(X_te[:N_REAL])
res = {k: [] for k in ["fd_b", "fd_i", "conf_b", "conf_i", "mse_b", "mse_i"]}
for seed in SEEDS:
    log(f"seed {seed}: training {N_MEMBERS} members")
    members = train_ensemble(seed)

    # Improved model only: pilot batch from each member alone -> confidence -> softmax weights
    rng_p = np.random.default_rng(seed + 1000)
    conf = np.array([mean_conf(sample(members, np.eye(N_MEMBERS)[k], N_PILOT, rng_p))
                     for k in range(N_MEMBERS)])
    e = np.exp((conf - conf.max()) / TEMP)
    w_imp = e / e.sum()
    w_base = np.full(N_MEMBERS, 1.0 / N_MEMBERS)
    log(f"seed {seed}: pilot confidences={np.round(conf, 3).tolist()} weights={np.round(w_imp, 3).tolist()}")

    # Identical initial noise for both aggregation rules
    xb = sample(members, w_base, N_GEN, np.random.default_rng(seed))
    xi = sample(members, w_imp, N_GEN, np.random.default_rng(seed))
    res["fd_b"].append(frechet(real_feat, pca.transform(xb)))
    res["fd_i"].append(frechet(real_feat, pca.transform(xi)))
    res["conf_b"].append(mean_conf(xb))
    res["conf_i"].append(mean_conf(xi))
    res["mse_b"].append(noise_mse(members, w_base, seed))
    res["mse_i"].append(noise_mse(members, w_imp, seed))
    log(f"seed {seed}: FD baseline={res['fd_b'][-1]:.3f} improved={res['fd_i'][-1]:.3f}")

metrics = {}
for k, v in res.items():
    v = np.array(v, dtype=float)
    if not np.all(np.isfinite(v)):
        log("non-finite metric; cannot produce meaningful results")
        sys.exit(1)
    metrics[k] = v
out = {
    "baseline_score": float(metrics["fd_b"].mean()),
    "improved_score": float(metrics["fd_i"].mean()),
    "baseline_score_std": float(metrics["fd_b"].std()),
    "improved_score_std": float(metrics["fd_i"].std()),
    "improvement": float(metrics["fd_b"].mean() - metrics["fd_i"].mean()),  # lower FD is better
    "baseline_mean_confidence": float(metrics["conf_b"].mean()),
    "improved_mean_confidence": float(metrics["conf_i"].mean()),
    "baseline_noise_mse": float(metrics["mse_b"].mean()),
    "improved_noise_mse": float(metrics["mse_i"].mean()),
}

results = {
    "question": "Can weighting ensemble members of a small score-based diffusion model by a "
                "perceptual-quality proxy, rather than by score-matching loss, lower the Frechet "
                "distance of generated samples relative to the arithmetic-mean ensemble?",
    "method": "Five-member MLP noise-prediction ensemble, DDIM-style sampler; baseline averages "
              "member scores uniformly, improved uses softmax weights over pilot-batch digit-classifier "
              "confidence. Frechet distance in a PCA feature space fitted on training data. "
              "Seeds 42, 43, 44.",
    "dataset": {
        "name": "scikit-learn digits (load_digits)", "source_url": URL,
        "n_rows": int(len(X)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "same_as_paper": False, "baseline_is_papers_method": False,
    },
    "metric_name": "Frechet distance (20-D PCA feature space, lower is better)",
    "higher_is_better": False,
    "baseline": "Arithmetic-mean deep-ensemble score model (uniform member weights).",
    "improved": "Same members, sampler and seeds; member weights = softmax of pilot-batch "
                "digit-classifier confidence; weighted mean of scores.",
    "metrics": out,
    "notes": f"Test set has {len(X_te)} digits, so the real reference set uses all of them rather than "
             "1,000. Frechet distance is computed on PCA features, not Inception features, so values are "
             "not comparable to FID in the paper. Only three seeds; std is across seeds. The improved "
             "model spends extra compute on pilot sampling (not matched in budget). Hyper-parameters "
             "were fixed a priori; nothing was tuned or selected on the test set. The classifier and PCA "
             "use training data only. Results are from a small toy setting.",
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log("wrote results.json")

# ---------- optional chart ----------
import os
os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(4, 3.5))
ax.bar(["baseline\n(mean)", "improved\n(weighted)"],
       [out["baseline_score"], out["improved_score"]],
       yerr=[out["baseline_score_std"], out["improved_score_std"]], capsize=5,
       color=["grey", "steelblue"])
ax.set_ylabel("Frechet distance (lower is better)")
ax.set_title("Digits ensemble: mean over seeds")
fig.tight_layout()
fig.savefig("figures/fd_comparison.png", dpi=120)
log("saved figures/fd_comparison.png")
