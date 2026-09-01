#!/usr/bin/env python3
"""用量反応の単調性を **パッチレベル** で測る。

slide_mean（512パッチの平均）は MIL の文脈では最も鈍い集約で、病変が面積比 f しか
占めないとき信号は f 倍に希釈される。**この希釈率は平均する枚数に依存しない**ので、
パッチを増やしても解決しない。効くのは集約関数の方。

ここではパッチごとに対照群からの Mahalanobis 距離を出し、スライド内の分布を
上位分位点や top-k 平均で集約する。「最も外れたパッチがどれだけ外れているか」を
見ることになり、疎な病変に対して平均より感度が高い。

集約 (--agg):
    mean        パッチ距離の平均（slide_mean の距離とは別物。距離の平均 ≠ 平均の距離）
    q90/q95/q99 上位分位点
    top16/top32 上位 k 枚の平均（ABMIL の max/attention プーリングに近い）
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

DOSE_ORDER = ["Control", "Low", "Middle", "High"]


def aggregate(d: np.ndarray, agg: str) -> float:
    """スライド内のパッチ距離 d (512,) を 1 値に集約する。"""
    if agg == "mean":
        return float(d.mean())
    if agg.startswith("q"):
        return float(np.percentile(d, float(agg[1:])))
    if agg.startswith("top"):
        k = int(agg[3:])
        return float(np.sort(d)[-k:].mean())
    raise ValueError(agg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--agg", default="q95")
    ap.add_argument("--axis", choices=["dose", "grade"], default="dose",
                    help="単調性を見る軸。dose=用量水準(効果の代理) / "
                         "grade=病理医が付けた重症度(効果そのもの)")
    ap.add_argument("--min-doses", type=int, default=3)
    ap.add_argument("--pca-slides", type=int, default=800,
                    help="PCA 学習に使うスライド数（全パッチを載せると重いため）")
    ap.add_argument("--cov-patches", type=int, default=200_000,
                    help="共分散推定に使う対照パッチ数")
    ap.add_argument("--out-dir", default="outputs/eval")
    ap.add_argument("--dump-scores", metavar="PATH",
                    help="スライド単位の距離を CSV に出す（所見タイプ別の解析用）")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["meta_ok"] == 1].reset_index(drop=True)
    encoder = Path(args.features).stem
    rng = np.random.default_rng(0)

    with h5py.File(args.features, "r") as h5:
        ids = [s.decode() if isinstance(s, bytes) else s for s in h5["slide_id"][:]]
        done = h5["done"][:]
        order = {s: i for i, s in enumerate(ids)}
        keep = [(i, order[s]) for i, s in enumerate(manifest["slide_id"].astype(str))
                if s in order and done[order[s]]]
        df = manifest.iloc[[i for i, _ in keep]].reset_index(drop=True)
        rows_h5 = [j for _, j in keep]
        print(f"{encoder}: {len(df):,} slides, agg={args.agg}")

        # --- PCA をパッチで学習 ---
        sample = sorted(rng.choice(rows_h5, min(args.pca_slides, len(rows_h5)),
                                   replace=False))
        pool = np.concatenate([h5["patch_feat"][j].astype(np.float32) for j in sample])
        pca = PCA(n_components=args.pca_dim, random_state=0).fit(pool)
        print(f"PCA {pool.shape[1]} -> {args.pca_dim}  "
              f"累積寄与率 {pca.explained_variance_ratio_.sum():.3f} "
              f"(パッチ {len(pool):,} 枚で学習)")
        del pool

        # --- 全パッチを射影して保持 ---
        Z = np.empty((len(rows_h5), 512, args.pca_dim), dtype=np.float32)
        for k, j in enumerate(rows_h5):
            Z[k] = pca.transform(h5["patch_feat"][j].astype(np.float32))
            if (k + 1) % 2000 == 0:
                print(f"  projected {k + 1:,}/{len(rows_h5):,}", flush=True)

    # --- 対照群セントロイド（化合物×時点、パッチをすべて使う） ---
    df["_grp"] = list(map(tuple, df[["compound", "sacrifice_period"]].astype(str).values))
    is_ctrl = (df["dose_level"] == "Control").values
    ctrl_idx = collections.defaultdict(list)
    for i, (g, c) in enumerate(zip(df["_grp"], is_ctrl)):
        if c:
            ctrl_idx[g].append(i)
    centroid = {g: Z[idxs].reshape(-1, args.pca_dim).mean(0) for g, idxs in ctrl_idx.items()}

    # --- 共分散: 対照パッチの群内残差 ---
    res = []
    for g, idxs in ctrl_idx.items():
        p = Z[idxs].reshape(-1, args.pca_dim)
        res.append(p - p.mean(0))
    res = np.concatenate(res)
    if len(res) > args.cov_patches:
        res = res[rng.choice(len(res), args.cov_patches, replace=False)]
    lw = LedoitWolf().fit(res)
    P = np.linalg.pinv(lw.covariance_)
    print(f"共分散: 対照パッチ群内残差 n={len(res):,} shrinkage={lw.shrinkage_:.3f}")

    # --- スライドごとの集約距離 ---
    score = np.full(len(df), np.nan)
    for i, (g, c) in enumerate(zip(df["_grp"], is_ctrl)):
        if g not in centroid:
            continue
        idxs = ctrl_idx[g]
        if c:
            if len(idxs) < 2:
                continue
            tot = Z[idxs].reshape(-1, args.pca_dim)
            n_self = Z[i].shape[0]
            mu = (tot.sum(0) - Z[i].sum(0)) / (len(tot) - n_self)   # leave-one-slide-out
        else:
            mu = centroid[g]
        d = Z[i] - mu
        score[i] = aggregate(np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, P, d), 0)),
                             args.agg)
    df["score"] = score

    # --- 単調性を見る軸 ---
    # dose は「効果の代理」でしかない。実際 High 用量の 52.5% は所見なしで、
    # grade を軸にすると代理を挟まずに「表現が重症度を追えているか」を直接測れる。
    if args.axis == "dose":
        df["_axis"] = df["dose_level_idx"]          # 0..3, -1 は欠損
        AXIS_LAB = ["Control", "Low", "Middle", "High"]
    else:
        df["_axis"] = df["max_grade_idx"] + 1       # 0=所見なし, 1=minimal .. 4=severe
        AXIS_LAB = ["none", "minimal", "slight", "moderate", "severe"]

    # --- Spearman ρ ---
    cmp_clean = df.groupby("compound")["is_normal"].min().eq(1)
    out_rows = []
    for g, sub in df[np.isfinite(df["score"])].groupby("_grp"):
        med = sub.groupby("_axis")["score"].median()
        med = med[med.index >= 0].sort_index()
        if len(med) < args.min_doses:
            continue
        rho, p = spearmanr(med.index.values, med.values)
        out_rows.append(dict(compound=g[0], period=g[1], n_slides=len(sub),
                             n_doses=len(med), rho=rho, p=p,
                             has_finding=int((sub["is_normal"] == 0).any()),
                             compound_clean=int(cmp_clean[g[0]]),
                             **{f"d_{AXIS_LAB[i]}": med.get(i, np.nan) for i in range(len(AXIS_LAB))}))
    res_df = pd.DataFrame(out_rows)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.axis}_response_patch_{args.agg}_{encoder}.csv"
    res_df.to_csv(out, index=False)

    dcol = df["score"].values
    if args.dump_scores:
        Path(args.dump_scores).parent.mkdir(parents=True, exist_ok=True)
        df[["slide_id", "score"]].rename(columns={"score": "score"}).to_csv(
            args.dump_scores, index=False)
        print(f"wrote {args.dump_scores}")

    # --- 軸との相関を全スライドでも見る（群ごとの Spearman は水準数が足りない組を
    #     落とすため、n が小さくなる。プールした値は落とさない）---
    ok = np.isfinite(dcol) & (df["_axis"].values >= 0)
    rho_all, p_all = spearmanr(df["_axis"].values[ok], dcol[ok])
    print(f"\n全スライドでの Spearman ρ({args.axis}, 距離): {rho_all:+.4f} "
          f"(n={ok.sum():,}, p={p_all:.2e})")
    y = (df["is_normal"].values == 0)[ok]
    if y.any() and not y.all():
        print(f"所見あり/なしの判別 AUROC: {roc_auc_score(y, dcol[ok]):.4f} "
              f"(陽性 {y.sum():,} / 陰性 {(~y).sum():,})")

    print(f"\n=== 単調性 [{encoder}, patch/{args.agg}, axis={args.axis}] ===")
    for label, sub in [("全体", res_df),
                       ("所見あり", res_df[res_df["has_finding"] == 1]),
                       ("★陰性対照", res_df[res_df["compound_clean"] == 1])]:
        print(f"  {label:<14} n={len(sub):>5}  ρ 平均 {sub['rho'].mean():+.4f}  "
              f"ρ>0 {(sub['rho'] > 0).mean():.1%}  ρ=1 {(sub['rho'] > 0.99).mean():.1%}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
