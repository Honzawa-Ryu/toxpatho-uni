#!/usr/bin/env python3
"""TG-GATEs 全スライドを凍結エンコーダに通してパッチ特徴を書き出す。

パッチ実体は ``$TGGATES_PATCH_DIR/<slide_id>.npy``。拡張子に反して中身は
**ヘッダなし raw uint8** (512, 256, 256, 3) HWC なので np.fromfile で読む。

前処理は UNI 公式のモデルカードに合わせる:
    Resize(224) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet mean/std)
入力が 256x256 正方のため Resize(224) だけで 224x224 になり CenterCrop は no-op。
よって bicubic resize 一発と等価。倍率を保つ center-crop は将来のアブレーション。

出力 (HDF5, 1 ファイル / エンコーダ):
    patch_feat  (n_slides, 512, D) float16   パッチ特徴
    slide_mean  (n_slides, D)      float32   スライド平均（mean pooling ベースライン用）
    slide_id    (n_slides,)        str
    done        (n_slides,)        bool      再開用の完了フラグ
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import h5py
import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset

PATCHES_PER_SLIDE = 512
PATCH_HW = 256
INPUT_HW = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- models
def build_uni() -> tuple[torch.nn.Module, int]:
    m = timm.create_model("vit_large_patch16_224", img_size=INPUT_HW, patch_size=16,
                          init_values=1e-5, num_classes=0, dynamic_img_size=True)
    sd = torch.load(hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True)
    return m, 1024


def build_uni2h() -> tuple[torch.nn.Module, int]:
    kwargs = dict(img_size=INPUT_HW, patch_size=14, depth=24, num_heads=24,
                  init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
                  num_classes=0, no_embed_class=True,
                  mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                  reg_tokens=8, dynamic_img_size=True)
    m = timm.create_model("vit_giant_patch14_224", pretrained=False, **kwargs)
    sd = torch.load(hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True)
    return m, 1536


def build_vit_imagenet() -> tuple[torch.nn.Module, int]:
    """ドメイン適応の効果を測るための非病理ベースライン。"""
    m = timm.create_model("vit_large_patch16_224.augreg_in21k_ft_in1k",
                          pretrained=True, num_classes=0, dynamic_img_size=True)
    return m, 1024


def build_dino(ckpt_path: str) -> tuple[torch.nn.Module, int]:
    """toxpatho-ssl-comparison 側で事前学習した DINO ViT-B/16。

    保存されているのは `lib/sslmodel/models/dino.py: DINO` の state_dict で、
    student と teacher の両方が入っている(466キー)。特徴抽出に要るのは
    `student_backbone.*`(150キー) だけなので、それを剥がして timm の ViT-B/16 に
    載せる。DINOヘッド(out_dim 65536)と teacher は使わない。

    同じ学習ランの別 epoch を並べたいので、チェックポイントは引数で受ける
    (--dino-ckpt)。出力名は --name で分けること。
    """
    m = timm.create_model("vit_base_patch16_224", img_size=INPUT_HW,
                          num_classes=0, dynamic_img_size=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    prefix = "student_backbone."
    body = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not body:
        raise SystemExit(f"{ckpt_path} に {prefix}* が無い。DINO の state_dict か確認すること "
                         f"(先頭キー: {list(sd)[:3]})")
    m.load_state_dict(body, strict=True)
    return m, 768


ENCODERS = {"uni": build_uni, "uni2h": build_uni2h, "vit_imagenet": build_vit_imagenet}


# --------------------------------------------------------------------------- data
class SlideDataset(Dataset):
    """1 アイテム = 1 スライド分の raw uint8 (512, 256, 256, 3)。

    正規化と resize は GPU 側でやる。ここで float 化すると 1 スライド 308 MB に
    膨らみ、worker の prefetch が RAM を食い潰すため。
    """

    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[int, torch.Tensor]:
        a = np.fromfile(self.paths[i], dtype=np.uint8)
        expected = PATCHES_PER_SLIDE * PATCH_HW * PATCH_HW * 3
        if a.size != expected:
            raise ValueError(f"{self.paths[i]}: {a.size} bytes, expected {expected}")
        return i, torch.from_numpy(a.reshape(PATCHES_PER_SLIDE, PATCH_HW, PATCH_HW, 3))


def read_manifest(path: Path, limit: int | None) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["meta_ok"] == "1"]
    rows.sort(key=lambda r: r["slide_id"])
    return rows[:limit] if limit else rows


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True,
                    help=f"{' | '.join(sorted(ENCODERS))} | dino (--dino-ckpt が必要)")
    ap.add_argument("--dino-ckpt", default=None,
                    help="--encoder dino のときの DINO チェックポイント(.pt)")
    ap.add_argument("--name", default=None,
                    help="出力ファイル名/h5 attrs の encoder 名(既定は --encoder)。"
                         "同じ dino で epoch 違いを並べるときに dino_ep85 等と分ける")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out-dir", default=os.environ.get("FEATURE_DIR", "outputs/features"))
    ap.add_argument("--sub-batch", type=int, default=256,
                    help="1 スライド 512 パッチを何枚ずつ前向き計算するか")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="スモークテスト用")
    ap.add_argument("--no-patch-feat", action="store_true",
                    help="スライド平均だけ書く（容量節約）")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA が見えない。GPU ノードで実行すること")

    name = args.name or args.encoder

    rows = read_manifest(Path(args.manifest), args.limit)
    paths = [r["patch_path"] for r in rows]
    print(f"encoder={args.encoder} name={name}  slides={len(rows)}", flush=True)

    if args.encoder == "dino":
        if not args.dino_ckpt:
            raise SystemExit("--encoder dino には --dino-ckpt が要る")
        model, dim = build_dino(args.dino_ckpt)
    elif args.encoder in ENCODERS:
        if args.dino_ckpt:
            raise SystemExit("--dino-ckpt は --encoder dino のときだけ使える")
        model, dim = ENCODERS[args.encoder]()
    else:
        raise SystemExit(f"未知の encoder: {args.encoder} "
                         f"({' | '.join(sorted(ENCODERS))} | dino)")
    model = model.eval().cuda()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.h5"

    n = len(rows)
    with h5py.File(out_path, "a") as h5:
        if "slide_id" not in h5:
            h5.create_dataset("slide_id", data=np.array([r["slide_id"] for r in rows],
                                                        dtype=h5py.string_dtype()))
            h5.create_dataset("slide_mean", (n, dim), dtype="float32")
            h5.create_dataset("done", (n,), dtype="bool", data=np.zeros(n, dtype=bool))
            if not args.no_patch_feat:
                h5.create_dataset("patch_feat", (n, PATCHES_PER_SLIDE, dim),
                                  dtype="float16", chunks=(1, PATCHES_PER_SLIDE, dim))
            h5.attrs.update(encoder=name, dim=dim, input_hw=INPUT_HW,
                            preprocess="bicubic-resize-256to224+imagenet-norm")
            if args.dino_ckpt:
                h5.attrs.update(dino_ckpt=os.path.abspath(args.dino_ckpt))
        else:
            # 再開時: manifest と既存ファイルがずれていないか確認
            got = [s.decode() if isinstance(s, bytes) else s for s in h5["slide_id"][:]]
            if got != [r["slide_id"] for r in rows]:
                raise SystemExit(f"{out_path} の slide_id が manifest と一致しない。"
                                 "作り直すか別の --out-dir を使うこと")

        done = h5["done"][:]
        todo = [i for i in range(n) if not done[i]]

        # 共有ディレクトリの一部が -rw------- で読めない（提供元の設定漏れ、7件）。
        # 落とさずスキップする。h5 の行は残したまま done=False のままにするので、
        # 後で権限が直れば投げ直すだけで埋まる。下流は done をマスクとして使うこと。
        unreadable = [i for i in todo if not os.access(paths[i], os.R_OK)]
        if unreadable:
            todo = [i for i in todo if i not in set(unreadable)]
            print(f"skip {len(unreadable)} unreadable slides: "
                  f"{[rows[i]['slide_id'] for i in unreadable]}", flush=True)

        print(f"resume: {n - len(todo) - len(unreadable)} done, {len(todo)} to go, "
              f"{len(unreadable)} skipped", flush=True)
        if not todo:
            return

        loader = DataLoader(SlideDataset([paths[i] for i in todo]), batch_size=None,
                            num_workers=args.workers, pin_memory=True,
                            prefetch_factor=2 if args.workers else None)

        mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)

        t0 = time.time()
        for k, (j, raw) in enumerate(loader):
            idx = todo[j]
            raw = raw.cuda(non_blocking=True)
            feats = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for s in range(0, PATCHES_PER_SLIDE, args.sub_batch):
                    x = raw[s:s + args.sub_batch].permute(0, 3, 1, 2).float().div_(255.)
                    x = torch.nn.functional.interpolate(
                        x, size=INPUT_HW, mode="bicubic", align_corners=False,
                        antialias=True)
                    feats.append(model(x.sub_(mean).div_(std)).float())
            f = torch.cat(feats)
            if not args.no_patch_feat:
                h5["patch_feat"][idx] = f.half().cpu().numpy()
            h5["slide_mean"][idx] = f.mean(0).cpu().numpy()
            h5["done"][idx] = True

            if k == 0 or (k + 1) % 50 == 0 or k + 1 == len(todo):
                el = time.time() - t0
                rate = (k + 1) / el
                print(f"  {k + 1}/{len(todo)} slides  {rate:.2f} slide/s  "
                      f"{rate * PATCHES_PER_SLIDE:.0f} img/s  "
                      f"ETA {(len(todo) - k - 1) / rate / 60:.1f} min  "
                      f"VRAM peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
                      f"(reserved {torch.cuda.max_memory_reserved() / 2**30:.2f})",
                      flush=True)
                h5.flush()

    print(f"wrote {out_path}", flush=True)
    print(f"VRAM peak: allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB / "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB "
          f"(sub_batch={args.sub_batch})", flush=True)


if __name__ == "__main__":
    main()
