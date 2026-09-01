#!/usr/bin/env python3
"""用量反応の単調性（研究計画 §5.2(1)、Phase 0 の主要評価）。

「良い表現とは用量反応を線形に写す表現である」という主張を検証する。
アノテーション不要で、TG-GATEs の用量4水準×時点4水準がほぼ均衡していることを利用する。

手順:
  1. スライドを slide_mean で代表させ、PCA で次元を落とす。
     1024/1536 次元のまま Mahalanobis を計算すると共分散が特異になるため。
  2. 対照群セントロイドを **化合物×時点ごと** に取る。
  3. 共分散は **全対照群からプールして1つだけ** 推定する（化合物ごとには推定しない）。
     --cov-mode within : 各対照スライドの「自分の群のセントロイドからの残差」を集めて推定。
                         study 間のオフセットが除かれる古典的な pooled within-group 共分散。
     --cov-mode pooled : 対照群全体の共分散。study 間変動を含むので距離は保守的になる。
     いずれも Ledoit-Wolf 縮小推定で安定化する。
  4. 各スライドの Mahalanobis 距離を計算し、化合物×時点ごとに用量水準別の中央値を取り、
     Spearman ρ で単調性を測る。
  5. 所見を出す化合物と出さない化合物で層別する。**後者で ρ が高いなら、
     毒性ではなく「用量に相関する何か」（染色ロット、処理日）を拾っている疑いがある。**

対照スライド自身の距離は、自分を除いたセントロイド（leave-one-out）から測る。
そうしないと対照群だけ距離が構造的に小さくなり、単調性が過大評価される。
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

DOSE_ORDER = ["Control", "Low", "Middle", "High"]


def load_features(h5_path: Path, manifest: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """h5 の slide_mean を manifest の行順に合わせて返す（done のものだけ）。"""
    with h5py.File(h5_path, "r") as h5:
        ids = [s.decode() if isinstance(s, bytes) else s for s in h5["slide_id"][:]]
        feat = h5["slide_mean"][:]
        done = h5["done"][:]
        encoder = h5.attrs.get("encoder", h5_path.stem)
    order = {sid: i for i, sid in enumerate(ids)}
    keep = [i for i, sid in enumerate(manifest["slide_id"].astype(str))
            if sid in order and done[order[sid]]]
    df = manifest.iloc[keep].reset_index(drop=True)
    X = np.stack([feat[order[s]] for s in df["slide_id"].astype(str)])
    print(f"{encoder}: {len(df):,} slides ({len(manifest) - len(df):,} 未完了/欠損を除外)")
    return df, X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="outputs/features/<enc>.h5")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--cov-mode", choices=["within", "pooled"], default="within")
    ap.add_argument("--timepoint", choices=["per", "pooled"], default="per",
                    help="per=時点ごとに単調性を見る（既定） / pooled=時点を混ぜる")
    ap.add_argument("--min-doses", type=int, default=3,
                    help="Spearman を計算するのに必要な用量水準数")
    ap.add_argument("--out-dir", default="outputs/eval")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["meta_ok"] == 1].reset_index(drop=True)
    df, X = load_features(Path(args.features), manifest)
    encoder = Path(args.features).stem

    # --- 1. PCA ---
    pca = PCA(n_components=args.pca_dim, random_state=0)
    Z = pca.fit_transform(X)
    print(f"PCA {X.shape[1]} -> {args.pca_dim} 次元  累積寄与率 "
          f"{pca.explained_variance_ratio_.sum():.3f}")

    # --- 2. 対照群セントロイド（化合物×時点） ---
    grp_key = ["compound", "sacrifice_period"] if args.timepoint == "per" else ["compound"]
    df["_grp"] = list(map(tuple, df[grp_key].astype(str).values))
    is_ctrl = (df["dose_level"] == "Control").values

    centroid, ctrl_idx = {}, collections.defaultdict(list)
    for i, (g, c) in enumerate(zip(df["_grp"], is_ctrl)):
        if c:
            ctrl_idx[g].append(i)
    for g, idxs in ctrl_idx.items():
        centroid[g] = Z[idxs].mean(0)

    # --- 3. 共分散（全対照群をプールして1つ） ---
    if args.cov_mode == "within":
        resid = np.concatenate([Z[idxs] - Z[idxs].mean(0)
                                for idxs in ctrl_idx.values() if len(idxs) >= 2])
    else:
        ctrl_all = np.concatenate([Z[idxs] for idxs in ctrl_idx.values()])
        resid = ctrl_all - ctrl_all.mean(0)
    lw = LedoitWolf().fit(resid)
    P = np.linalg.pinv(lw.covariance_)
    print(f"共分散: {args.cov_mode}  n={len(resid):,}  shrinkage={lw.shrinkage_:.3f}")

    # --- 4. Mahalanobis 距離（対照スライドは leave-one-out） ---
    dist = np.full(len(df), np.nan)
    for i, (g, c) in enumerate(zip(df["_grp"], is_ctrl)):
        if g not in centroid:
            continue
        idxs = ctrl_idx[g]
        if c:
            if len(idxs) < 2:
                continue
            mu = (Z[idxs].sum(0) - Z[i]) / (len(idxs) - 1)   # 自分を除く
        else:
            mu = centroid[g]
        d = Z[i] - mu
        dist[i] = np.sqrt(max(d @ P @ d, 0.0))
    df["mahalanobis"] = dist

    ctrl_d = df.loc[is_ctrl & np.isfinite(dist), "mahalanobis"]
    print(f"対照群自身の距離（LOO）: 中央値 {ctrl_d.median():.2f}  "
          f"(理論値の目安 √{args.pca_dim} = {np.sqrt(args.pca_dim):.2f})")

    # --- 5. 化合物×時点ごとに Spearman ρ ---
    rows = []
    for g, sub in df[np.isfinite(df["mahalanobis"])].groupby("_grp"):
        med = sub.groupby("dose_level_idx")["mahalanobis"].median()
        med = med[med.index >= 0].sort_index()
        if len(med) < args.min_doses:
            continue
        rho, p = spearmanr(med.index.values, med.values)
        rows.append(dict(
            compound=g[0], period=g[1] if len(g) > 1 else "all",
            n_slides=len(sub), n_doses=len(med), rho=rho, p=p,
            has_finding=int((sub["is_normal"] == 0).any()),
            **{f"d_{DOSE_ORDER[i]}": med.get(i, np.nan) for i in range(4)}))
    res = pd.DataFrame(rows)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"dose_response_{encoder}.csv"
    res.to_csv(out, index=False)

    # --- 集計 ---
    print(f"\n=== 用量反応の単調性 [{encoder}] ===")
    print(f"評価できた (化合物×時点) の組: {len(res):,}  化合物 {res['compound'].nunique()}")
    for label, sub in [("全体", res),
                       ("所見あり", res[res["has_finding"] == 1]),
                       ("所見なし ★陰性対照", res[res["has_finding"] == 0])]:
        if len(sub) == 0:
            continue
        print(f"  {label:<18} n={len(sub):>5}  ρ 平均 {sub['rho'].mean():+.3f}  "
              f"中央値 {sub['rho'].median():+.3f}  ρ>0 {(sub['rho'] > 0).mean():.1%}  "
              f"ρ=1 {(sub['rho'] > 0.99).mean():.1%}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
