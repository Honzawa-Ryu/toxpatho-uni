#!/usr/bin/env python3
"""エンコーダ間で用量反応の単調性を比較する。

同じ (化合物×時点) の組に対する ρ を比べるので、**対応のある検定**を使う。
平均値を帰無分布の SD と比べるだけでは、群ごとのばらつきを無視して過小評価になる。
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="outputs/eval")
    ap.add_argument("--encoders", nargs="+", default=["uni", "uni2h", "vit_imagenet"])
    ap.add_argument("--prefix", default="dose_response",
                    help="読むファイル名の接頭辞 (例: dose_response_patch_top16)")
    args = ap.parse_args()

    tabs = {}
    for e in args.encoders:
        df = pd.read_csv(Path(args.eval_dir) / f"{args.prefix}_{e}.csv")
        tabs[e] = df.set_index(["compound", "period"])
    keys = set.intersection(*(set(t.index) for t in tabs.values()))
    keys = sorted(keys)
    print(f"共通の (化合物×時点) の組: {len(keys):,}\n")

    rho = pd.DataFrame({e: tabs[e].loc[keys, "rho"].values for e in args.encoders},
                       index=pd.MultiIndex.from_tuples(keys))
    meta = tabs[args.encoders[0]].loc[keys]

    strata = [("全体", np.ones(len(keys), bool)),
              ("所見あり", meta["has_finding"].values == 1),
              ("陰性対照(全時点で所見なし)", meta["compound_clean"].values == 1)]

    for label, mask in strata:
        print(f"=== {label}  (n={mask.sum()}) ===")
        for e in args.encoders:
            print(f"  {e:14s} ρ平均 {rho[e][mask].mean():+.4f}")
        for a, b in itertools.combinations(args.encoders, 2):
            d = (rho[a] - rho[b])[mask].values
            d = d[np.isfinite(d)]
            nz = d[d != 0]
            if len(nz) == 0:
                print(f"  {a} vs {b}: 差なし")
                continue
            stat, p = wilcoxon(nz)
            # 差の平均のブートストラップ 95% CI
            rng = np.random.default_rng(0)
            bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"  {a:>12s} - {b:<14s} 差 {d.mean():+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  Wilcoxon p={p:.2e}  "
                  f"({a}勝ち {(d > 0).mean():.1%} / 引分 {(d == 0).mean():.1%})")
        print()


if __name__ == "__main__":
    main()
