#!/usr/bin/env python3
"""化合物単位の交差検証 split を切る。

研究計画 §5.3 の通り、**分割は必ず化合物単位**。TG-GATEs は 1群5匹×複数時点なので、
動物単位やスライド単位で切ると同一 study の染色特徴を覚えるだけで probe が高得点を出す。

対照群は各 study が自前で持つため、化合物で切れば対照群も自動的に一緒に動く。
ただし GroupKFold は層化を保証せず、所見あり化合物が偏った fold ができる。
そのため StratifiedGroupKFold を使い、**化合物レベルの層**（その化合物が所見を出すか）
で層化する。

出力: data/splits.csv (slide_id, compound, fold)
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="data/splits.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["meta_ok"] == 1].reset_index(drop=True)

    # 層化のラベルはスライド単位（所見の有無）。StratifiedGroupKFold は
    # group を割らずに y の分布を fold 間で揃えようとする。
    y = (df["is_normal"] == 0).astype(int)
    groups = df["compound"]

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    df["fold"] = -1
    for fold, (_, test_idx) in enumerate(sgkf.split(df, y, groups)):
        df.loc[test_idx, "fold"] = fold
    assert (df["fold"] >= 0).all()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["slide_id", "compound", "fold"]].to_csv(out, index=False)

    print(f"wrote {out}  ({len(df):,} slides, {args.folds} folds)")
    print(f"{'fold':>4} {'slides':>8} {'化合物':>7} {'所見あり':>8} {'所見率':>7}")
    for f in range(args.folds):
        s = df[df["fold"] == f]
        pos = (s["is_normal"] == 0).sum()
        print(f"{f:>4} {len(s):>8,} {s['compound'].nunique():>7} {pos:>8,} {pos/len(s):>7.1%}")

    # 化合物が fold をまたいでいないことの確認（これが崩れると評価が無意味になる）
    leak = [c for c, g in df.groupby("compound")["fold"] if g.nunique() > 1]
    print(f"\n複数 fold にまたがる化合物: {len(leak)} 件"
          + (f" ★リーク {leak[:5]}" if leak else "  (OK)"))


if __name__ == "__main__":
    main()
