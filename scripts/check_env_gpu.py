#!/usr/bin/env python3
"""Phase 0 環境検証: GH200 (aarch64) 上で学習/推論に必要な要素が揃っているか確認する。

計算ノードで実行すること（ログインノードには GPU がない）。
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

SEP = "=" * 62


def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def check_device() -> torch.device:
    section("1. device")
    print("torch            :", torch.__version__, "| cuda", torch.version.cuda)
    print("cuda available   :", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - are we on a GPU node?")
    i = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(i)
    print("device           :", p.name)
    print("capability       :", f"sm_{p.major}{p.minor}")
    print("memory           :", f"{p.total_memory / 1024**3:.1f} GiB")
    print("bf16 supported   :", torch.cuda.is_bf16_supported())
    return torch.device("cuda")


def check_attention(device: torch.device) -> None:
    """timm の ViT は F.scaled_dot_product_attention を使うので、
    flash/mem-efficient バックエンドが効けば xformers / flash-attn は不要。"""
    section("2. attention backends (xformers/flash-attn の要否判定)")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(2, 16, 1024, 64, device=device, dtype=torch.bfloat16)
    for name, backend in [("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
                          ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
                          ("MATH", SDPBackend.MATH)]:
        try:
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
            print(f"  {name:22s} OK")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:22s} NG  ({type(exc).__name__}: {str(exc)[:70]})")
    for mod in ("xformers", "flash_attn"):
        try:
            __import__(mod)
            print(f"  {mod:22s} installed")
        except ImportError:
            print(f"  {mod:22s} not installed")


def bench_backbone(device: torch.device) -> None:
    section("3. backbone throughput (UNI / UNI2 相当のアーキで実測)")
    import timm

    # UNI = ViT-L/16 224, UNI2-h = ViT-H/14 224。重みなしで形だけ作る。
    for tag, kwargs in [
        ("UNI      (ViT-L/16, 224)", dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)),
        ("UNI2-h   (ViT-H/14, 224)", dict(patch_size=14, embed_dim=1536, depth=24, num_heads=24)),
    ]:
        model = timm.models.VisionTransformer(img_size=224, num_classes=0, **kwargs).to(device).eval()
        n_param = sum(p.numel() for p in model.parameters())
        for bs in (32, 128):
            x = torch.randn(bs, 3, 224, 224, device=device, dtype=torch.bfloat16)
            with torch.autocast("cuda", torch.bfloat16), torch.no_grad():
                for _ in range(3):
                    model(x)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    model(x)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / 10
            print(f"  {tag}  {n_param/1e6:6.1f}M  bs={bs:4d}  "
                  f"{bs/dt:8.1f} img/s  peak {torch.cuda.max_memory_allocated()/1024**3:5.2f} GiB")
            torch.cuda.reset_peak_memory_stats()
        del model
        torch.cuda.empty_cache()


def check_data() -> None:
    section("4. TG-GATEs パッチ読み出し (Lustre I/O)")
    patch_dir = os.environ.get("TGGATES_PATCH_DIR", "/work/gd43/share/tggates/sample_patch_agg")
    files = sorted(f for f in os.listdir(patch_dir) if f.endswith(".npy"))[:5]
    print("patch dir        :", patch_dir, f"({len(os.listdir(patch_dir))} files)")
    total, t0 = 0, time.perf_counter()
    for f in files:
        # ヘッダなし raw uint8。np.load ではなく fromfile で読む。
        a = np.fromfile(os.path.join(patch_dir, f), dtype=np.uint8).reshape(512, 256, 256, 3)
        total += a.nbytes
    dt = time.perf_counter() - t0
    print(f"read {len(files)} slides ({total/1024**3:.2f} GiB) in {dt:.2f}s "
          f"-> {total/1024**2/dt:.0f} MiB/s")
    print(f"1 slide = {a.shape} {a.dtype}, {a.nbytes/1024**2:.0f} MiB")
    print(f"全 11,294 スライドの 1 epoch 相当 = {11294*a.nbytes/1024**4:.2f} TiB の読み出し")


def check_uni() -> None:
    section("5. UNI / UNI2-h の重み取得 (gated repo)")
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        print("  HF_TOKEN 未設定 -> スキップ")
        print("  https://huggingface.co/MahmoodLab/UNI と .../UNI2-h で規約に同意し、")
        print("  read 権限のトークンを .env の HF_TOKEN に設定すること。")
        return
    from huggingface_hub import hf_hub_download
    for repo in ("MahmoodLab/UNI", "MahmoodLab/UNI2-h"):
        try:
            p = hf_hub_download(repo, "config.json", token=tok)
            print(f"  {repo:22s} OK ({p})")
        except Exception as exc:  # noqa: BLE001
            print(f"  {repo:22s} NG ({type(exc).__name__}: {str(exc)[:90]})")


if __name__ == "__main__":
    dev = check_device()
    check_attention(dev)
    bench_backbone(dev)
    check_data()
    check_uni()
    print("\ndone.")
