# 作業ルール

## ⚠️ 実行環境の使い分け（最重要）

**Miyabi は従量課金。Miyabi 上では実行を極力避ける。**

| | やる場所 |
|---|---|
| **SSL 事前学習**（マルチノードGPU、日単位） | **Miyabi** |
| **評価・解析のすべて** | **andre01 = ユーザーの計算機** |

このリポジトリの中身は**ほぼ全部が「評価・解析」**にあたる。特徴抽出
（`scripts/extract_features.py`）、所見別 AUROC（`scripts/eval_by_finding.py`）、
用量反応（`scripts/eval_dose_response*.py`）、ABMIL（`scripts/train_abmil.py`）は
**すべて andre01 で回す**。

### エージェントの振る舞い

- **`qsub` を勝手に投げない。** 投入前に必ず確認を取る。
- コードは**書いてコミットするところまで**がここでの仕事。実行はユーザーが向こうで行う。
  「動かして確かめましょうか」と提案しない。
- ログインノードでの軽い検証（import 確認、単体テスト、CPU スモーク、チェックポイントの
  読み込み確認）は課金対象の計算ノードを使わないので行ってよい。
  判断基準は「計算ノードを使うか」。
- コミットに「実装済み・未実行」が含まれるのは想定どおりの受け渡し形態。

## リポジトリの関係

| リポジトリ | 役割 |
|---|---|
| `toxpatho-uni`（ここ） | 基盤モデル(UNI)比較と**下流評価基盤** |
| `toxpatho-ssl-comparison` | SSL 手法比較の本体。学習コードと表現比較パイプライン |

`toxpatho-ssl-comparison` で学習した DINO を評価する場合:

```
SSL=/work/gd43/d43000/toxpatho-ssl-comparison/outputs/0026_20260901_dino_clipgrad03
qsub -v ENCODER=dino,DINO_CKPT=$SSL/model_ep85.pt,NAME=dino_ep85 scripts/extract_features.pbs
```

## 評価基盤の要点

- `data/manifest.csv` — 11,295 スライド。所見12種の二値列 + `is_normal` + 用量・時点。
  `patch_path` は `/work/gd43/share/tggates/sample_patch_agg/<slide_id>.npy`
  （512パッチ/スライド, 256px, raw uint8 ヘッダなし, 計1.1TB）
- スコアは**対照群セントロイドからの Mahalanobis 距離**。学習を伴わないので
  val リークの心配が無く、エンコーダ間・チェックポイント間の比較が決定的になる
- プーリングは所見の空間分布で使い分ける。**びまん性 → slide_mean / 病巣性 → top16**
  （`scripts/eval_by_finding.py` の FOCAL / DIFFUSE / ZONAL 分類）
- 前処理は全エンコーダ共通で 256→224 bicubic + ImageNet 正規化。比較の公平性の前提
