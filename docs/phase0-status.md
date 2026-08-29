# Phase 0 進捗メモ

最終更新: 2026-08-29
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

### slide_id とラベルの対応

ファイル名は Open TG-GATEs の実験ID体系そのもの：

```
<EXP_ID><GROUP_ID:2桁><INDIVIDUAL_ID:1桁>      例) 42011 → EXP 42 / GROUP 01 / 個体 1
```

公開 CSV 側の `EXP_ID` は 4 桁ゼロ詰めなので `int()` を通して join する。
これで **11,289 / 11,294 が公開メタデータと一致**した。

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

## 3. 残ブロッカー

### (A) HuggingFace トークン ★これがないとベースラインが取れない

`MahmoodLab/UNI` と `MahmoodLab/UNI2-h` はいずれも **gated（auto 承認）**。
以下が必要：

1. HF アカウントで https://huggingface.co/MahmoodLab/UNI と
   https://huggingface.co/MahmoodLab/UNI2-h の規約に同意する
2. read 権限のアクセストークンを発行し、`.env` に `HF_TOKEN=hf_...` を書く
   （`.env` は .gitignore 対象なのでコミットされない）

`scripts/check_env_gpu.py` の section 5 が疎通確認を兼ねている。

### (B) 手持ちの毒性病理画像

Miyabi 上には見つからなかった。所在・枚数・施設数・スキャナ種が未確定。
研究計画 §5.2(2) のクロスソース転移（外部検証）に必要。

### (C) ROI アノテーションの可否

選択肢 A（教師あり LoRA、tile レベル）を開けるかの分岐。研究計画 §4 参照。

---

## 4. 次にやること

1. **HF_TOKEN 設定後**：UNI / UNI2-h をロードし、全 11,177 スライドの特徴を抽出
   （patch 特徴 + スライド平均）。約 2 時間 / 1 GPU。
2. 化合物単位の split を切る（研究計画 §5.3 の通り、動物・スライド単位では切らない）。
3. 評価パイプライン実装：
   - ABMIL（所見有無 / grade）
   - ラベル効率カーブ
   - 対照群からの Mahalanobis 距離の用量反応単調性（Spearman ρ）
   - control 限定 eta²
4. 凍結 UNI / UNI2 / ImageNet ViT のベースライン確定。
