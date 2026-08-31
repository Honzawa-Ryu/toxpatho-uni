# Phase 0 進捗メモ

最終更新: 2026-08-31（HF_TOKEN 解消・特徴抽出スクリプト追加・データ選択基準の検証）
対象: `docs/research-plan.md` の Phase 0（環境検証 + データ + 評価パイプライン + 凍結ベースライン）

---

## 1. データ：TG-GATEs は既に Miyabi 上にある

`/work/gd43/share/tggates/sample_patch_agg/` に **11,294 ファイル / 計 1.1 TB**。
**WSI からのタイル抽出工程は不要**（研究計画 §7 の I/O 見積もりリスクは消滅）。

### ファイル形式（重要）

拡張子は `.npy` だが **中身はヘッダなしの raw uint8**。`np.load` は失敗する。

```python
a = np.fromfile(path, dtype=np.uint8).reshape(512, 256, 256, 3)   # HWC
```

- 全ファイル厳密に 100,663,296 bytes = 512 × 256 × 256 × 3
- 512 パッチ / スライド、256×256、RGB、約 20 倍相当
- 組織領域は抽出済み（空白パッチ 0 件、パッチ平均輝度 138–206）
- 配列レイアウトは隣接画素相関で検証済み（HWC が等方的、CHW は不整合）
- **UNI の入力は 224 なので、center-crop か resize が要る**

### 目視所見（2026-08-31、`outputs/patch_preview/`）

対照 / 壊死 severe / 肥大 high の 3 スライドから各 16 パッチを PNG に書き出して確認した。

- 想定通り **H&E のラット肝、20 倍相当**。肝細胞索と類洞が判別でき、核小体まで見える
- **染色のばらつきが大きい**（青紫寄り / 赤ピンク寄り / 淡い）。研究計画 §5.1 が言う
  「batch と biology の交絡」は染色差として実際に目視できる。染色正規化の有無で結論が変わりうる
- 高用量 thioacetamide では肝細胞の腫大と脂肪滴様の抜けが見える。一方 grade=severe の壊死スライドでも
  **ランダム 16 枚では壊死巣を引けない**。512 パッチ中どれが所見部位かは分からず、これが MIL を使う理由

**未解決**: パッチの元 WSI 上の座標が無い。512 枚の連番しか持たないため、
順序非依存の集約（ABMIL）は問題ないが、空間情報を使う手法は現状のデータでは組めない。
共有元に座標が残っているか確認する価値がある。

### slide_id とラベルの対応

ファイル名は Open TG-GATEs の実験ID体系そのもの：

```
<EXP_ID><GROUP_ID:2桁><INDIVIDUAL_ID:1桁>      例) 42011 → EXP 42 / GROUP 01 / 個体 1
```

公開 CSV 側の `EXP_ID` は 4 桁ゼロ詰めなので `int()` を通して join する。
末尾 3 桁が固定長なので分解は一意。これで **11,289 / 11,294 が公開メタデータと一致**した。

**この対応付けはファイル名だけを手がかりにした推定**なので、2026-08-31 に根拠を検証した：

- 一致率 11,289 / 11,294、連結キーの衝突は 0 件
- **144 実験のうち 140 実験で、ファイル側の (group, individual) 集合が CSV と完全一致**。
  偶然の数値一致では群構成や 1 群あたりの個体数までは揃わないので、これが最も強い証拠
- 対抗仮説「`FILE_LOCATION` の svs 画像ID」は棄却（一致 254/11,294 のみ。svs は全 52,879 件）
- 全ファイルが `SINGLE_REPEAT_TYPE=Repeat`、臓器は Liver に落ちる（意味的整合）

**限界**: これは一貫性の証明であって、`101011.npy` の中身が物理的に EXP 101 の肝である保証ではない。
決定的に確かめるには公開 svs を 1 枚だけ落として画素を突き合わせる（保留中）。

### 選択基準：ラット × 肝臓 × 反復投与 の全数

公開メタデータには 24,033 個体 / svs 画像 52,879 枚があるが、共有 dir はその一部。層別に見ると：

| 臓器 | 投与形式 | 種 | CSV の個体数 | 共有 dir にある | 被覆率 |
|---|---|---|---|---|---|
| Liver | **Repeat** | Rat | 11,276 | **11,276** | **100.0%** |
| Liver | Single | Rat | 12,401 | 0 | 0.0% |
| Kidney | Single | Rat | 12,164 | 0 | 0.0% |
| Kidney | Repeat | Rat | 11,050 | (同一個体の肝が当たるだけ) | — |

**肝・反復投与・ラットは 1 件も欠けていない。恣意的なサンプリングではない。**
逆に共有 dir 側で肝・反復に入らないのは 18 件のみ：

- 13 件（exp 42 / 315 / 569 の一部）は CSV 上「腎のみ」の動物。肝の行が公開メタに無い
- 5 件（`670161`–`670165`）は CSV に一切現れない。EXP 670 group 16（High）のメタ欠損
- いずれも `build_manifest.py` が `meta_ok=0` で除外済み

「svs 52,879 枚」との差は、単回投与の除外・腎の除外・1 個体あたり複数枚の重複による。
肝・反復に限れば svs は 11,299 枚しかなく、それが 11,276 ファイルに対応する
（21 個体が 2 枚、1 個体が 3 枚を持ち、そこは集約されたか片方のみ採用）。

**含意**: 肝の単回投与 12,401 個体が手つかずで残っている。継続 SSL のコーパスを約 2 倍にでき、
3–24 時間の急性期という別の形態変化もカバーできる。ただし共有 dir には無いので、
使うなら公開 svs からのタイル抽出工程が新たに必要（Phase 0 で不要と判断した工程が復活する）。
また用量反応の評価に単回と反復を混ぜると時間軸の性質が違って解釈が濁るため、
**学習には入れ、評価は反復投与に限る**という切り分けが素直。すぐやる話ではなく選択肢として記録。

### 取得したメタデータ

`scripts/fetch_tggates_metadata.sh` で `data/` に取得（dbarchive.biosciencedbc.jp）。
**CSV は cp932。utf-8 で読むと UnicodeDecodeError で落ちる。**

| ファイル | 内容 |
|---|---|
| `open_tggates_pathology.csv` | 所見 FINDING_TYPE / TOPOGRAPHY_TYPE / GRADE_TYPE |
| `open_tggates_individual.csv` | 化合物・用量レベル・屠殺時点・投与経路 |
| `open_tggates_pathological_image.csv` | 画像メタ（ORGAN, FILE_LOCATION, CAPTURE_NO） |
| `open_tggates_cel_file_attribute.csv` | BARCODE ↔ EXP_ID（トランスクリプトーム連携用） |
| `open_tggates_main.csv` | 化合物一覧 |

### マニフェスト

`scripts/build_manifest.py` → `data/manifest.csv`（1 行 = 1 スライド）。

```
11,294 スライド中 11,177 が使用可（残り 117 はメタ欠損 or 腎臓）
化合物         : 143
正常 / 所見あり : 8,658 / 2,519   ← 77.5% が正常。強い不均衡
用量レベル     : Control 2839 / Low 2820 / Middle 2815 / High 2703
屠殺時点       : 4day 2831 / 8day 2822 / 15day 2793 / 29day 2731
投与形式       : すべて Repeat（反復投与）。Single は含まれない
```

**用量 4 水準 × 時点 4 水準がほぼ完全に均衡**しており、研究計画 §5.2(1) の
「対照群からの Mahalanobis 距離の用量反応単調性」がそのまま実装できる。
対照群 2,839 枚は §5.2(4) の control 限定 eta² のプールにもなる。

主要所見の陽性スライド数（上位12、`data/manifest.csv` に二値列 `f_*` として展開済み）:

```
1057 Hypertrophy                          208 Ground glass appearance
 311 Microgranuloma                        202 Degeneration, granular, eosinophilic
 304 Necrosis                              175 Change, eosinophilic
 281 Cellular infiltration                 156 Proliferation, bile duct
 226 Vacuolization, cytoplasmic            130 Single cell necrosis
 207 Increased mitosis                     112 Swelling
```

grade は minimal / slight / moderate / severe の順序尺度（`max_grade_idx` 列）。
所見は全 58 種。

---

## 2. 環境検証：aarch64 の懸念は解消

`scripts/check_env_gpu.py` を `scripts/check_env_gpu.pbs` で debug-g に投入して確認（job 3236211, 全項目 pass）。

### 判明した事実

| 項目 | 結果 |
|---|---|
| GPU | **NVIDIA GH200 120GB**, sm_90, 95.0 GiB, bf16 対応 |
| ノード | 72 core / 212 GB RAM / 1 GPU |
| torch | **2.11.0+cu130 が aarch64 ネイティブで動作**（Apptainer 不要） |
| timm / peft | 1.0.29 / 0.19.1 |
| SDPA backend | **FLASH_ATTENTION / EFFICIENT_ATTENTION / MATH すべて OK** |
| xformers / flash-attn | 未インストール。**上記より Phase 0/1 では不要** |
| Lustre 読み出し | 4,500–5,200 MiB/s |

**研究計画 §7 で最大リスクとしていた「aarch64 で xformers / flash-attn が通るか」は、
timm の ViT が `F.scaled_dot_product_attention` を使い flash backend が有効なため、
推論と LoRA では回避できる。** DINOv2 公式実装で継続 SSL を回す場合のみ再検討が要る。

### スループット実測（bf16、重みなしの同型モデル）

| | パラメータ | bs=128 | 全パッチ 5,782,528 枚の特徴抽出 |
|---|---|---|---|
| UNI (ViT-L/16, 224) | 303.3M | ~3,000 img/s | **約 32 分 / 1 GPU** |
| UNI2-h (ViT-H/14, 224) | 681.3M | ~1,200 img/s | **約 80 分 / 1 GPU** |

全データの特徴抽出が 1 GPU 1〜2 時間で終わる。ベースライン確定は十分現実的。
1 epoch の読み出しは 1.03 TiB。

### PBS の書き方（Miyabi 固有、テンプレートの記述は誤り）

```bash
#PBS -q debug-g
#PBS -W group_list=gd43     # 必須。-P は Invalid Option
#PBS -l select=1            # gpus= は Unknown resource。select=1 でノードごと確保
#PBS -l walltime=00:20:00
```

`templates/headers/miyabi_gpu.txt` は `gpus=__GPU__` を含んでおり Miyabi では通らない。要修正。

### pyproject の変更

テンプレート既定は PubMed/NLP 向け（vllm, scispacy, elasticsearch, spacy, lightgbm,
xgboost, optuna, duckdb, trl, bitsandbytes 等）だったため、本プロジェクト用に差し替えた。
追加: `timm`, `peft`, `einops`, `opencv-python-headless`, `pillow`, `h5py`。

---

## 3. エンコーダと特徴抽出（2026-08-31）

### 重みのロードは確認済み

`HF_TOKEN` は `~/.bashrc` に export 済み（`.env` は空のまま）。fineGrained トークンだが
`MahmoodLab/UNI` / `UNI2-h` の gated repo にアクセスでき、実ファイルも取得できた。

| | 重み | timm へのロード | 実パッチでの前向き計算 |
|---|---|---|---|
| UNI | 1.21 GB | `strict=True` 完全一致 | 1024 次元 |
| UNI2-h | 2.73 GB | `strict=True` 完全一致 | 1536 次元 |

キャッシュは `$HOME/.cache/huggingface`（= /work 配下）なので `HF_HOME` の設定は不要。

**注意**: 既存シェルは `~/.bashrc` の変更を拾わない。バッチジョブや新規シェルでは
`source ~/.bashrc` するか、`#PBS -V` で環境を引き継ぐこと。

### `scripts/extract_features.py`（未実行）

manifest の `meta_ok=1` を順に処理し、HDF5 に `patch_feat` (n, 512, D) fp16 と
`slide_mean` (n, D) fp32 を書く。`done` フラグで中断再開可。

- エンコーダは `uni` / `uni2h` / `vit_imagenet`（`vit_large_patch16_224.augreg_in21k_ft_in1k`）
- 前処理は UNI 公式モデルカード準拠。入力が 256² 正方なので `Resize(224)+CenterCrop(224)` は
  **bicubic resize 一発と等価**。倍率を保つ center-crop はアブレーション扱い
- DataLoader は raw uint8 のまま渡し、resize / 正規化は GPU 側。worker で float 化すると
  1 スライド 308 MB に膨らんで prefetch が RAM を食うため
- 出力サイズ: UNI 約 11.7 GB / UNI2-h 約 17.8 GB

### VRAM 見積もり（解析値。実測はログの "VRAM peak" 行）

重みを fp32 のまま `autocast(bf16)` / `inference_mode` で推論した場合：

| sub_batch | UNI | UNI2-h |
|---|---|---|
| 64 | ~2.8 GB | ~5.4 GB |
| 128 | ~3.0 GB | ~6.1 GB |
| **256（既定）** | **~3.5 GB** | **~7.6 GB** |
| 512 | ~4.6 GB | ~10.7 GB |

- 支配項は重み（UNI 1.7 GB / UNI2-h 3.8 GB、fp32 + autocast の bf16 キャストキャッシュ）
- SDPA の flash backend が効くので N² の注意行列は出ない。効くのは MLP 中間層で、
  UNI は 4×1024=4096、UNI2-h は SwiGLUPacked の fc1 が 2×8192=**16384** まで膨らむ
- **16 GB の GPU なら既定のまま両モデルとも通る。** 12 GB なら `SUB_BATCH=64`。
  さらに削るなら `model.cuda()` を `.to(torch.bfloat16)` にすると重み分が浮く
- ホスト RAM は `num_workers=8` × prefetch 2 × 100 MB ≒ 1.6 GB の pinned メモリ

### 実行場所が未決

入力 1.137 TB は Miyabi の共有領域にあり、1 epoch の読み出しがそのまま 1.03 TiB。
出力は 3 モデル合わせても約 41 GB と小さい。**特徴抽出だけ Miyabi で回して h5 を持ち帰る**のが
転送量としては最小（1.1 TB 転送 vs 41 GB 転送）。以降の評価は h5 だけで CPU でも回る。

---

## 4. 残ブロッカー

### (A) HuggingFace トークン — 解消（2026-08-31）

`~/.bashrc` に `export HF_TOKEN` 済み。両 gated repo で規約同意済み、重み取得とロードまで確認。

### (B) 手持ちの毒性病理画像 — 所在のみ未確定

`6c08935`（別マシンでの計画確定）で規模は判明済み：**百スライド前後**、部位アノテーション付与予定。
ただし **Miyabi 上には見つかっていない**ので、実体の所在と転送手段が要確認。
施設数・スキャナ種も引き続き未確定（クロスソース転移の解釈に必要）。

### (C) ROI アノテーション — 解消

病理医アクセスがあり作成可能と確認済み（`6c08935`）。選択肢 A（教師あり LoRA）の並走が確定。
研究計画 §4 の通り、主軸は C（継続 SSL）のまま A を並走させる。

## 5. 次にやること

Phase 0 の 4 項目（研究計画 §6）のうち、環境検証とデータは完了。残りは評価パイプラインと
ベースライン確定の 2 つ。

1. **特徴抽出を実行**（実行場所の決定待ち）。`uni` / `uni2h` / `vit_imagenet` の 3 本。
   スモークテスト → 本番の順。GPU が要るのはここだけで、以降は h5 だけで足りる。
2. **化合物単位の split を切る**。研究計画 §5.3 の通り、動物・スライド単位では切らない。
   目視で染色のばらつきが大きいことを確認済みなので、この警告は現実的。
3. **評価パイプライン実装**（研究計画 §5.2）：
   - 用量反応の単調性 — 対照群セントロイドからの Mahalanobis 距離、化合物ごとに Spearman ρ。
     アノテーション不要で **これが本命**。用量 4 水準が均衡なのでそのまま実装できる
   - ABMIL（所見有無 / grade）+ ラベル効率カーブ（1/2/4/8/16 slides per class）。
     77.5% が正常の不均衡なので AUROC でなく **AUPRC 主軸**
   - control 限定 eta²。必ず用量反応と対で報告する
   - クロスソース転移 — **ブロッカー (B) 待ち**。他の 3 本は TG-GATEs だけで完結するので先に進める
4. **凍結 UNI / UNI2 / ImageNet ViT のベースライン確定。**
   研究計画 §6 の通り、この時点で「UNI2 vs UNI」「汎用 ViT からの改善幅」が出るので、
   LoRA をやらなくてもそれ自体が報告になる。

### 保留中の宿題

- 公開 svs を 1 枚落として、パッチとの画素突き合わせ（slide_id 対応付けの決定的検証）
- パッチの元 WSI 上の座標が共有元に残っているかの確認
- 肝・単回投与 12,401 個体を継続 SSL のコーパスに加えるか（要タイル抽出工程）
- `templates/headers/miyabi_gpu.txt` の `gpus=__GPU__` 修正
- `scripts/preview_patches.py` の整理（現状は使い捨てスクリプトで、出力のみ `outputs/patch_preview/`）
