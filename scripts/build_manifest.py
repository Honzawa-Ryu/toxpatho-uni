#!/usr/bin/env python3
"""TG-GATEs パッチ束と公開メタデータを突き合わせてスライド単位の manifest を作る。

パッチ実体: ``$TGGATES_PATCH_DIR/<slide_id>.npy``
  拡張子は .npy だが中身は **ヘッダなし raw uint8**。形状は (512, 256, 256, 3)。
  numpy.load ではなく numpy.fromfile + reshape で読むこと。

slide_id の構造: ``<EXP_ID><GROUP_ID:2桁><INDIVIDUAL_ID:1桁>``
  公開 CSV 側の EXP_ID は 4 桁ゼロ詰めなので、int() を通して突き合わせる。

出力: data/manifest.csv (1 行 = 1 スライド)
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
from pathlib import Path

# 公開 CSV は日本語由来で cp932 混じり。utf-8 では落ちる。
ENCODING = "cp932"
PATCHES_PER_SLIDE = 512
PATCH_HW = 256
GRADE_ORDER = ["minimal", "slight", "moderate", "severe"]
DOSE_ORDER = ["Control", "Low", "Middle", "High"]


def read_csv(path: Path):
    with open(path, encoding=ENCODING, errors="replace", newline="") as fh:
        yield from csv.DictReader(fh)


def slide_key(exp_id: str, group_id: str, individual_id: str) -> str:
    return f"{int(exp_id)}{int(group_id):02d}{int(individual_id)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-dir", default=os.environ.get(
        "TGGATES_PATCH_DIR", "/work/gd43/share/tggates/sample_patch_agg"))
    ap.add_argument("--meta-dir", default="data")
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--top-findings", type=int, default=12,
                    help="個別の二値ラベル列を立てる所見の数")
    args = ap.parse_args()

    patch_dir, meta_dir = Path(args.patch_dir), Path(args.meta_dir)
    slides = sorted(p.stem for p in patch_dir.glob("*.npy"))
    if not slides:
        raise SystemExit(f"no patch files under {patch_dir}")

    individual = {slide_key(r["EXP_ID"], r["GROUP_ID"], r["INDIVIDUAL_ID"]): r
                  for r in read_csv(meta_dir / "open_tggates_individual.csv")}

    image: dict[str, list[dict]] = collections.defaultdict(list)
    for r in read_csv(meta_dir / "open_tggates_pathological_image.csv"):
        image[slide_key(r["EXP_ID"], r["GROUP_ID"], r["INDIVIDUAL_ID"])].append(r)

    findings: dict[str, list[dict]] = collections.defaultdict(list)
    for r in read_csv(meta_dir / "open_tggates_pathology.csv"):
        if r["ORGAN"] == "Liver":
            findings[slide_key(r["EXP_ID"], r["GROUP_ID"], r["INDIVIDUAL_ID"])].append(r)

    # 二値ラベル列を立てる所見を、対象スライド内の頻度で選ぶ
    counts = collections.Counter(
        f["FINDING_TYPE"] for s in slides for f in findings.get(s, []))
    top = [f for f, _ in counts.most_common(args.top_findings)]

    def col(name: str) -> str:
        return "f_" + "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")

    fields = [
        "slide_id", "exp_id", "group_id", "individual_id",
        "compound", "dose_level", "dose_level_idx", "sacrifice_period",
        "sacrifice_day", "single_repeat", "organ", "n_captures",
        "patch_path", "n_patches", "patch_size",
        "is_normal", "n_findings", "max_grade", "max_grade_idx", "findings",
        "meta_ok",
    ] + [col(f) for f in top]

    rows = []
    for sid in slides:
        ind, imgs, fs = individual.get(sid), image.get(sid, []), findings.get(sid, [])
        grades = [g for g in (f["GRADE_TYPE"] for f in fs) if g in GRADE_ORDER]
        max_grade = max(grades, key=GRADE_ORDER.index) if grades else ""
        period = (ind or {}).get("SACRIFICE_PERIOD", "")
        dose = (ind or {}).get("DOSE_LEVEL", "")
        organ = imgs[0]["ORGAN"] if imgs else ""
        row = {
            "slide_id": sid,
            "exp_id": sid[:-3], "group_id": sid[-3:-1], "individual_id": sid[-1],
            "compound": (ind or imgs[0] if imgs else {}).get("COMPOUND_NAME", ""),
            "dose_level": dose,
            "dose_level_idx": DOSE_ORDER.index(dose) if dose in DOSE_ORDER else -1,
            "sacrifice_period": period,
            "sacrifice_day": period.split()[0] if period.endswith("day") else "",
            "single_repeat": (ind or {}).get("SINGLE_REPEAT_TYPE", ""),
            "organ": organ,
            "n_captures": len(imgs),
            "patch_path": str(patch_dir / f"{sid}.npy"),
            "n_patches": PATCHES_PER_SLIDE,
            "patch_size": PATCH_HW,
            "is_normal": int(not fs),
            "n_findings": len(fs),
            "max_grade": max_grade,
            "max_grade_idx": GRADE_ORDER.index(max_grade) if max_grade else -1,
            # FINDING|TOPOGRAPHY|GRADE を ";" 連結
            "findings": ";".join(
                f"{f['FINDING_TYPE']}|{f['TOPOGRAPHY_TYPE']}|{f['GRADE_TYPE']}" for f in fs),
            # 解析対象として使える行か（メタデータが揃い、肝で、反復投与）
            "meta_ok": int(bool(ind) and organ == "Liver"),
        }
        present = {f["FINDING_TYPE"] for f in fs}
        row.update({col(f): int(f in present) for f in top})
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["meta_ok"]]
    print(f"wrote {out}  ({len(rows)} slides, {len(ok)} usable)")
    print(f"  compounds        : {len({r['compound'] for r in ok})}")
    print(f"  normal / abnormal: {sum(r['is_normal'] for r in ok)} / "
          f"{sum(1 - r['is_normal'] for r in ok)}")
    print(f"  dose levels      : {dict(collections.Counter(r['dose_level'] for r in ok))}")
    print(f"  sacrifice periods: {dict(collections.Counter(r['sacrifice_period'] for r in ok))}")
    print(f"  dropped (no meta): {len(rows) - len(ok)}")
    print("  binary finding columns:")
    for f in top:
        print(f"    {sum(r[col(f)] for r in ok):5d}  {col(f):42s} {f}")


if __name__ == "__main__":
    main()
