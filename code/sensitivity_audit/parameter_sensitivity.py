"""Sensitivity audit for Tai's assignment.

The script reuses the cached King County model, explanation points, background
points, and attribution arrays. It varies three audit parameters:

* number of explained instances,
* perturbation mask fraction,
* background/donor size.

Outputs are written to code/sensitivity_audit/outputs.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
KC_DIR = ROOT / "code" / "king_county"
KC_OUT = KC_DIR / "outputs"
OUT_DIR = ROOT / "code" / "sensitivity_audit" / "outputs"

METHODS = ["kernelshap", "samplingshap", "lime", "occlusion", "treeshap", "consensus"]
SCENARIOS = [
    ("baseline", "Baseline", 60, 0.30, 40),
    ("points_20", "20 diem", 20, 0.30, 40),
    ("points_40", "40 diem", 40, 0.30, 40),
    ("mask_20", "Che 20%", 60, 0.20, 40),
    ("mask_50", "Che 50%", 60, 0.50, 40),
    ("bg_10", "Nen 10", 60, 0.30, 10),
    ("bg_20", "Nen 20", 60, 0.30, 20),
]


@dataclass
class ScenarioContext:
    key: str
    label: str
    n_points: int
    mask_frac: float
    bg_size: int
    ex: np.ndarray
    bg: np.ndarray
    attrs: dict[str, np.ndarray]
    fx: np.ndarray
    mu: float
    gx: np.ndarray
    valid: np.ndarray
    bg_mean: np.ndarray
    scale: np.ndarray
    bg_scaled: np.ndarray
    ex_scaled: np.ndarray


def import_ensemble():
    """Import ensemble_predict with its expected working directory."""
    sys.path.insert(0, str(KC_DIR))
    old_cwd = os.getcwd()
    os.chdir(KC_DIR)
    try:
        import ensemble_predict as ensemble  # type: ignore
    finally:
        os.chdir(old_cwd)
    return ensemble


def predict_fn(ensemble, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return ensemble.predict_price(arr).astype("float64")


def load_base(seed: int):
    ex = pd.read_parquet(KC_OUT / "xai_explain.parquet").to_numpy().astype("float32")
    bg = pd.read_parquet(KC_OUT / "xai_bg.parquet").to_numpy().astype("float32")

    attrs = {}
    for method in METHODS:
        if method == "consensus":
            attrs[method] = np.load(KC_OUT / "consensus_attr.npz")["attr"].astype("float64")
        else:
            attrs[method] = np.load(KC_OUT / f"attr_{method}.npz")["attr"].astype("float64")

    rng = np.random.default_rng(seed)
    ex_perm = rng.permutation(len(ex))
    bg_perm = np.random.default_rng(seed + 1).permutation(len(bg))
    return ex, bg, attrs, ex_perm, bg_perm


def make_context(
    ensemble,
    base_ex: np.ndarray,
    base_bg: np.ndarray,
    base_attrs: dict[str, np.ndarray],
    ex_perm: np.ndarray,
    bg_perm: np.ndarray,
    scenario,
) -> ScenarioContext:
    key, label, n_points, mask_frac, bg_size = scenario
    idx = ex_perm[:n_points]
    bidx = bg_perm[:bg_size]

    ex = base_ex[idx].copy()
    bg = base_bg[bidx].copy()
    attrs = {m: base_attrs[m][idx].copy() for m in METHODS}
    fx = predict_fn(ensemble, ex)
    mu = float(predict_fn(ensemble, bg).mean())
    gx = fx - mu
    valid = np.isfinite(gx) & (np.abs(gx) > 1000.0)
    bg_mean = bg.mean(axis=0).astype("float32")

    scale = np.vstack([bg, ex]).std(axis=0).astype("float64")
    scale[scale < 1e-9] = 1.0
    bg_scaled = bg.astype("float64") / scale
    ex_scaled = ex.astype("float64") / scale

    return ScenarioContext(
        key=key,
        label=label,
        n_points=n_points,
        mask_frac=mask_frac,
        bg_size=bg_size,
        ex=ex,
        bg=bg,
        attrs=attrs,
        fx=fx,
        mu=mu,
        gx=gx,
        valid=valid,
        bg_mean=bg_mean,
        scale=scale,
        bg_scaled=bg_scaled,
        ex_scaled=ex_scaled,
    )


def donor_fill(ctx: ScenarioContext, keep: np.ndarray) -> np.ndarray:
    out = ctx.ex.copy()
    for i in range(len(ctx.ex)):
        cols = keep[i]
        if cols.any():
            diff = ctx.bg_scaled[:, cols] - ctx.ex_scaled[i, cols][None, :]
            nearest = int(np.argmin((diff * diff).sum(axis=1)))
        else:
            diff = ctx.bg_scaled - ctx.ex_scaled[i][None, :]
            nearest = int(np.argmin((diff * diff).sum(axis=1)))
        out[i, ~cols] = ctx.bg[nearest, ~cols]
    return out.astype("float32")


def evaluate_masks(ensemble, ctx: ScenarioContext, keeps: list[np.ndarray], regime: str) -> np.ndarray:
    blocks = []
    for keep in keeps:
        if regime == "marginal":
            blocks.append(np.where(keep, ctx.ex, ctx.bg_mean[None, :]).astype("float32"))
        elif regime == "on_manifold":
            blocks.append(donor_fill(ctx, keep))
        else:
            raise ValueError(f"Unknown regime: {regime}")
    stacked = np.vstack(blocks)
    pred = predict_fn(ensemble, stacked)
    return pred.reshape(len(keeps), len(ctx.ex)).T


def deletion_auc(ensemble, ctx: ScenarioContext, attr: np.ndarray, regime: str) -> np.ndarray:
    order = np.argsort(-np.abs(attr), axis=1)
    keep = np.ones_like(attr, dtype=bool)
    keeps = [keep.copy()]
    rows = np.arange(len(ctx.ex))
    for t in range(attr.shape[1]):
        keep[rows, order[:, t]] = False
        keeps.append(keep.copy())

    curves = evaluate_masks(ensemble, ctx, keeps, regime) - ctx.mu
    denom = np.where(np.abs(ctx.gx) > 1e-9, ctx.gx, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = curves / denom[:, None]
    return np.nanmean(normalized, axis=1)


def perturbation_deltas(ensemble, ctx: ScenarioContext, n_perturbations: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keeps = []
    masks = []
    for _ in range(n_perturbations):
        mask = rng.random(ctx.ex.shape) < ctx.mask_frac
        masks.append(mask)
        keeps.append(~mask)
    mask_arr = np.stack(masks, axis=0)
    pred_m = evaluate_masks(ensemble, ctx, keeps, "marginal")
    pred_o = evaluate_masks(ensemble, ctx, keeps, "on_manifold")
    delta_m = ctx.fx[:, None] - pred_m
    delta_o = ctx.fx[:, None] - pred_o
    return mask_arr, delta_m, delta_o


def calibrated_attr(ctx: ScenarioContext, attr: np.ndarray) -> np.ndarray:
    sums = attr.sum(axis=1)
    scale = np.ones(len(attr), dtype="float64")
    ok = np.abs(sums) > 1e-9
    scale[ok] = ctx.gx[ok] / sums[ok]
    return attr * scale[:, None]


def infidelity_score(ctx: ScenarioContext, attr: np.ndarray, masks: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    attr_c = calibrated_attr(ctx, attr)
    predicted_delta = np.einsum("np,tnp->nt", attr_c, masks)
    denom = np.where(np.abs(ctx.gx) > 1e-9, ctx.gx * ctx.gx, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = (predicted_delta - deltas) ** 2 / denom[:, None]
    return np.nanmean(err, axis=1)


def method_means(metric: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    return np.array([np.nanmean(metric[m][idx]) for m in METHODS], dtype="float64")


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    corr = spearmanr(a, b, nan_policy="omit").correlation
    return float(corr) if np.isfinite(corr) else np.nan


def bootstrap_rankcorr(metric_m: dict[str, np.ndarray], metric_o: dict[str, np.ndarray], valid: np.ndarray, n_boot: int, rng):
    valid_idx = np.flatnonzero(valid)
    point = safe_spearman(method_means(metric_m, valid_idx), method_means(metric_o, valid_idx))
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(valid_idx, size=len(valid_idx), replace=True)
        boots.append(safe_spearman(method_means(metric_m, sample), method_means(metric_o, sample)))
    arr = np.asarray(boots, dtype="float64")
    return {
        "mean": float(np.nanmean(arr)),
        "point": point,
        "ci_low": float(np.nanpercentile(arr, 2.5)),
        "ci_high": float(np.nanpercentile(arr, 97.5)),
        "valid_n": int(len(valid_idx)),
    }


def run_scenario(ensemble, ctx: ScenarioContext, n_perturbations: int, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    deletion_m, deletion_o = {}, {}
    for method in METHODS:
        deletion_m[method] = deletion_auc(ensemble, ctx, ctx.attrs[method], "marginal")
        deletion_o[method] = deletion_auc(ensemble, ctx, ctx.attrs[method], "on_manifold")

    masks, delta_m, delta_o = perturbation_deltas(ensemble, ctx, n_perturbations, rng)
    inf_m, inf_o = {}, {}
    for method in METHODS:
        inf_m[method] = infidelity_score(ctx, ctx.attrs[method], masks, delta_m)
        inf_o[method] = infidelity_score(ctx, ctx.attrs[method], masks, delta_o)

    boot_rng = np.random.default_rng(seed + 10_000)
    del_boot = bootstrap_rankcorr(deletion_m, deletion_o, ctx.valid, n_boot, boot_rng)
    inf_boot = bootstrap_rankcorr(inf_m, inf_o, ctx.valid, n_boot, boot_rng)

    scenario_rows = []
    for metric_name, stats in [("deletion", del_boot), ("infidelity", inf_boot)]:
        scenario_rows.append(
            {
                "setting": ctx.key,
                "label": ctx.label,
                "n_points": ctx.n_points,
                "mask_frac": ctx.mask_frac,
                "bg_size": ctx.bg_size,
                "metric": metric_name,
                **stats,
            }
        )

    method_rows = []
    for method in METHODS:
        idx = np.flatnonzero(ctx.valid)
        method_rows.append(
            {
                "setting": ctx.key,
                "label": ctx.label,
                "n_points": ctx.n_points,
                "mask_frac": ctx.mask_frac,
                "bg_size": ctx.bg_size,
                "method": method,
                "deletion_marginal": float(np.nanmean(deletion_m[method][idx])),
                "deletion_on_manifold": float(np.nanmean(deletion_o[method][idx])),
                "infidelity_marginal": float(np.nanmean(inf_m[method][idx])),
                "infidelity_on_manifold": float(np.nanmean(inf_o[method][idx])),
            }
        )
    return scenario_rows, method_rows


def summarize_parameter_sensitivity(results: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "n_points": ["points_20", "points_40", "baseline"],
        "mask_frac": ["mask_20", "baseline", "mask_50"],
        "bg_size": ["bg_10", "bg_20", "baseline"],
    }
    rows = []
    for parameter, settings in groups.items():
        sub = results[results["setting"].isin(settings)]
        for metric in ["deletion", "infidelity"]:
            vals = sub[sub["metric"] == metric]["mean"].to_numpy(dtype="float64")
            lows = sub[sub["metric"] == metric]["ci_low"].to_numpy(dtype="float64")
            highs = sub[sub["metric"] == metric]["ci_high"].to_numpy(dtype="float64")
            rows.append(
                {
                    "parameter": parameter,
                    "metric": metric,
                    "min_corr": float(np.nanmin(vals)),
                    "max_corr": float(np.nanmax(vals)),
                    "range": float(np.nanmax(vals) - np.nanmin(vals)),
                    "mean_ci_width": float(np.nanmean(highs - lows)),
                }
            )
    return pd.DataFrame(rows)


def plot_results(results: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.4), dpi=160)
    order = [s[0] for s in SCENARIOS]
    labels = [s[1] for s in SCENARIOS]
    x = np.arange(len(order))
    colors = {"deletion": "#2f5597", "infidelity": "#c55a11"}
    offsets = {"deletion": -0.08, "infidelity": 0.08}
    for metric in ["deletion", "infidelity"]:
        sub = results[results["metric"] == metric].set_index("setting").loc[order]
        mean = sub["mean"].to_numpy(dtype="float64")
        low = sub["ci_low"].to_numpy(dtype="float64")
        high = sub["ci_high"].to_numpy(dtype="float64")
        yerr = np.vstack([mean - low, high - mean])
        ax.errorbar(
            x + offsets[metric],
            mean,
            yerr=yerr,
            fmt="o-",
            color=colors[metric],
            capsize=4,
            linewidth=2,
            markersize=5,
            label=metric,
        )
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Rank-correlation M vs on-manifold (95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Sensitivity of XAI audit to evaluation parameters")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_report(results: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    base = results[results["setting"] == "baseline"].set_index("metric")
    lines = [
        "# Tai sensitivity audit",
        "",
        "Scope: King County cached XAI audit; model and attribution files are reused.",
        "",
        "Baseline rank-correlation between marginal and on-manifold evaluation:",
        f"- Deletion: {base.loc['deletion', 'mean']:.2f} "
        f"[{base.loc['deletion', 'ci_low']:.2f}, {base.loc['deletion', 'ci_high']:.2f}]",
        f"- Infidelity: {base.loc['infidelity', 'mean']:.2f} "
        f"[{base.loc['infidelity', 'ci_low']:.2f}, {base.loc['infidelity', 'ci_high']:.2f}]",
        "",
        "Parameter sensitivity ranges:",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['parameter']} / {row['metric']}: range={row['range']:.2f}, "
            f"mean CI width={row['mean_ci_width']:.2f}"
        )
    lines.extend(
        [
            "",
            "Causal interpretation:",
            "- Marginal replacement asks a weak observational question: what happens if features are broken independently?",
            "- On-manifold donor replacement is closer to a conditional/counterfactual question: what changes when a house is compared with plausible neighboring houses?",
            "- The audit is not a causal identification proof; it checks whether explanations survive more plausible interventions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=500, help="bootstrap samples for 95%% CI")
    parser.add_argument("--n-perturbations", type=int, default=24, help="random perturbations for infidelity")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensemble = import_ensemble()
    base_ex, base_bg, base_attrs, ex_perm, bg_perm = load_base(args.seed)

    all_rows = []
    all_method_rows = []
    for i, scenario in enumerate(SCENARIOS):
        ctx = make_context(ensemble, base_ex, base_bg, base_attrs, ex_perm, bg_perm, scenario)
        rows, method_rows = run_scenario(
            ensemble,
            ctx,
            n_perturbations=args.n_perturbations,
            n_boot=args.bootstrap,
            seed=args.seed + 100 * i,
        )
        all_rows.extend(rows)
        all_method_rows.extend(method_rows)
        print(f"done {ctx.label}: valid_n={int(ctx.valid.sum())}")

    results = pd.DataFrame(all_rows)
    method_results = pd.DataFrame(all_method_rows)
    summary = summarize_parameter_sensitivity(results)

    results.to_csv(OUT_DIR / "sensitivity_ci.csv", index=False)
    method_results.to_csv(OUT_DIR / "method_metric_means.csv", index=False)
    summary.to_csv(OUT_DIR / "sensitivity_summary.csv", index=False)
    plot_results(results, OUT_DIR / "parameter_sensitivity.png")
    write_report(results, summary, OUT_DIR / "parameter_sensitivity_report.md")

    print("saved:")
    print(OUT_DIR / "sensitivity_ci.csv")
    print(OUT_DIR / "sensitivity_summary.csv")
    print(OUT_DIR / "method_metric_means.csv")
    print(OUT_DIR / "parameter_sensitivity.png")
    print(OUT_DIR / "parameter_sensitivity_report.md")


if __name__ == "__main__":
    main()
