#!/usr/bin/env python3
"""所見タイプ別に、対照群からの距離の判別性能（AUROC）を見る。

grade 軸では slide_mean が top16 に勝ち、用量軸では逆だった。この矛盾は
**所見の空間分布**で説明できるはずである：

  びまん性（肝細胞肥大など、小葉全体に広がる）→ 平均が適する
  病巣性（壊死・肉芽腫など、局所に留まる）    → 上位k枚が適する

所見ありスライドの 42% が Hypertrophy（びまん性）なので、grade 軸全体では
平均が有利に出ていた、という筋書きを検証する。

陽性 = その所見を持つスライド / 陰性 = 所見が一切ないスライド。
他の所見を持つスライドは陰性に混ぜない（ラベルが濁るため）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 空間分布による分類。病理学的な判断なので明示しておく。
FOCAL = {           # 病巣性: 局所に留まる
    "f_necrosis": "Necrosis",
    "f_microgranuloma": "Microgranuloma",
    "f_single_cell_necrosis": "Single cell necrosis",
    "f_cellular_infiltration": "Cellular infiltration",
}
DIFFUSE = {         # びまん性: 小葉全体に広がる
    "f_hypertrophy": "Hypertrophy",
    "f_ground_glass_appearance": "Ground glass appearance",
    "f_vacuolization__cytoplasmic": "Vacuolization, cytoplasmic",
    "f_swelling": "Swelling",
}
ZONAL = {           # 中間: 帯状・散在性で、どちらとも言い切れない
    "f_change__eosinophilic": "Change, eosinophilic",
    "f_degeneration__granular__eosinophilic": "Degeneration, granular, eos.",
    "f_proliferation__bile_duct": "Proliferation, bile duct",
    "f_increased_mitosis": "Increased mitosis",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True,
                    metavar="NAME=PATH", help="例: uni/top16=outputs/scores/uni_top16.csv")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="outputs/eval/by_finding.csv")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    man = man[man["meta_ok"] == 1]
    man["slide_id"] = man["slide_id"].astype(str)

    runs = {}
    for spec in args.scores:
        name, path = spec.split("=", 1)
        s = pd.read_csv(path)
        s["slide_id"] = s["slide_id"].astype(str)
        runs[name] = s.set_index("slide_id")["score"]

    groups = [("病巣性", FOCAL), ("びまん性", DIFFUSE), ("中間", ZONAL)]
    rows = []
    for gname, mapping in groups:
        for col, label in mapping.items():
            for name, sc in runs.items():
                d = man.join(sc, on="slide_id")
                d = d[np.isfinite(d["score"])]
                pos = d[d[col] == 1]
                neg = d[d["is_normal"] == 1]
                if len(pos) < 20:
                    continue
                y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                s_ = np.r_[pos["score"].values, neg["score"].values]
                rows.append(dict(group=gname, finding=label, n_pos=len(pos),
                                 run=name, auroc=roc_auc_score(y, s_)))
    res = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    piv = res.pivot_table(index=["group", "finding", "n_pos"], columns="run",
                          values="auroc")
    runs_order = [c for c in runs if c in piv.columns]
    piv = piv[runs_order]

    print("所見タイプ別 AUROC（陽性=その所見あり / 陰性=所見なし 8,674 枚）\n")
    w = max(len(f) for _, f, _ in piv.index)
    hdr = " " * (w + 8) + "  ".join(f"{c:>14s}" for c in runs_order)
    print(hdr); print("-" * len(hdr))
    last = None
    for (g, f, n), row in piv.iterrows():
        if g != last:
            print(f"\n[{g}]"); last = g
        cells = "  ".join(f"{row[c]:>14.3f}" for c in runs_order)
        print(f"  {f:<{w}s} {n:>5d}  {cells}")

    # 集約の向きを群ごとに集計する
    print("\n\n=== 群ごとの平均 AUROC ===")
    m = res.groupby(["group", "run"])["auroc"].mean().unstack()[runs_order]
    print(m.round(3).to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
