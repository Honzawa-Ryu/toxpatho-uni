# 研究計画：毒性病理画像に対する病理基盤モデル（UNI / UNI2）のドメイン適応

作成日: 2026-08-29
確定日: 2026-08-29（§8 未確定事項を解消）
ステータス: 確定（実装未着手）

---

## 1. 目的

MahmoodLab の病理基盤モデル **UNI / UNI2** を毒性病理画像（ラット肝 H&E）にドメイン適応させ、
毒性病理タスクにおける表現性能を向上させる。計算資源はスパコン **Miyabi**（GH200 / aarch64）を使用。
まず小規模から開始し、フルファインチューニングではなく **LoRA** による parameter-efficient adaptation を採る。

## 2. データ

| | 内容 |
|---|---|
| 主データ | **Open TG-GATEs** ラット肝 H&E WSI |
| 追加データ | 手持ちの毒性病理画像（ラット肝 H&E） |
| ラベル粒度 | 基本 **スライドレベル**（所見名 + grade） |
| 付随メタデータ | 化合物 / 用量 / 投与時点 / vehicle control |

手持ちデータの規模：**百スライド前後**（少量）。**部位（病変）アノテーションを付与予定**であり、
外部検証（クロスソース転移評価、§5.2(2)）に加えて選択肢A（教師ありLoRA）の評価データとしても使える。
施設数・スキャナ種は引き続き未確定。

## 3. 先行研究とその位置づけ

### Ammeling et al., MELBA 2026
"Benchmarking Foundation Models for Mitotic Figure Classification"
https://www.melba-journal.org/papers/2026:003.html

- FM の **attention に LoRA** を適用、**教師あり分類**。
- linear probe より明確に優位。**訓練データ 10% で 100% 相当の性能**に到達。
- 最新 FM では未知の腫瘍ドメインに対する **OOD ギャップをほぼ解消**。

### Graf et al., arXiv 2602.02124 / Sci Rep 2026
"Toxicity Assessment in Preclinical Histopathology via Class-Aware Mahalanobis Distance for Known and Novel Anomalies"
Olga Graf, Dhrupal Patel, Peter Groß, Charlotte Lempp, Matthias Hein, Fabian Heinemann
https://arxiv.org/html/2602.02124v2 / https://www.nature.com/articles/s41598-026-56510-9

- DINOv2 + **LoRA を教師ありセグメンテーション**に適用。げっ歯類肝の pixel-wise アノテーション。
- 正常からの **Mahalanobis 距離 + クラス別閾値**で ID/OOD 所見を検出
  （apoptosis = near-OOD、染色アーチファクト = far-OOD）。

### 関連
- Robustifying pathology foundation models via fine-tuning (arXiv 2607.22861)
  https://arxiv.org/html/2607.22861v1 — PathoROB の robustness index、HEST / THUNDER / Patho-Bench で評価。

### ⚠ 重要な注意

**上記2件はいずれも「教師あり LoRA」であり、「LoRA による継続 SSL」を支持する報告ではない。**
両者が示すのは「下流タスクを持った教師あり LoRA 適応が効く」ことであって、
継続 SSL の根拠としては使えない。主張を組み立てる際にここを混同しないこと。

さらに、**「学習成果を証明するベンチマークがない」という問題は、継続 SSL を選んだことの必然的な帰結**である。
教師あり LoRA なら評価は自明（タスク性能）で先行研究とも整合する。
継続 SSL は新規性は高いがリスクと評価難度を同時に負う選択であることを自覚しておく。

## 4. アプローチの選択肢

| | 損失 | 評価のしやすさ | 先行整合 | リスク |
|---|---|---|---|---|
| **A. 教師あり LoRA（tile）** | ROI アノテーション少量 | 自明 | ◎ | 低 |
| **B. MIL + LoRA（slide）** | 手持ちの slide ラベル | 自明だがノイジー | ○ | 中（勾配が弱い・実装が重い） |
| **C. 継続 SSL + LoRA** | ラベル不要 | **困難** | × | 高 |

**確定：C を主軸とする。**

（検討の経緯）継続 SSL（C）にこだわる動機は「アノテーションコストの回避」であることを確認した。
一方で ROI アノテーションは病理医アクセスがあり少量なら作成可能、手持ちデータ（百スライド前後）にも
部位アノテーションを付与予定であることも判明し、C を選ぶ前提だった「アノテーションが作れない」という
制約自体は実は存在しないことが分かった。この場合は §8 の当初想定どおり A への切り替えが選択肢に上がるが、
検討の結果、**継続 SSL の新規性を優先し C を主軸のまま維持する**ことで確定した。

**方針：C を本命に置くので A を必ず並走させる。** A は「そのタスクでの到達上限」を与え、
C の結果を解釈する物差しになる。A 単独でも Ammeling のラベル効率カーブをラット毒性肝で
再現するだけで報告価値がある。ROI アノテーションは Ammeling の結果（10% で頭打ち）を踏まえれば少量で足りる。
このアノテーションは病理医アクセスがあるため作成可能（確認済み）。

## 5. 評価設計

### 5.1 当初案とその問題点

当初の主要評価案：**EffectiveRank の上昇** と **eta² によるバッチ間差の低減**。
いずれも診断指標としては有用だが、**主要評価項目にすると持たない。**

**(a) Effective rank は必要条件であって十分条件ではない**
実質 RankMe (Garrido et al., 2023) だが、あの論文の主張は「同一手法内のハイパラ比較で downstream と相関する」
であり、手法間で成り立つ保証はない。しかも LayerNorm スケール・温度・白色化を触るだけで水増しできる。
ノイズが増えただけのモデルでも rank は上がる。計算に使うサンプル数 N と中心化の有無に強く依存する。

**(b) 退化解が存在する**
rank ↑ と eta² ↓ は一見互いを牽制するが、「バッチとも病変とも無関係なノイズ方向」を増やせば両方改善する。

**(c) 最も深刻：毒性病理では batch と biology が交絡する**
TG-GATEs では **1 study ≒ 1 化合物**。study 効果の除去は、検出したい化合物起因の病変シグナルの除去と
区別がつかない。eta²(study) の最小化は目的そのものを壊しうる。

### 5.2 採用する評価軸

#### (1) 用量反応の単調性 ★アノテーション不要・本命
TG-GATEs は全 study に用量・時点・vehicle control が揃っている。これを使う：

- 各化合物について、**対照群セントロイドからの Mahalanobis 距離**（Graf et al. の手法をそのまま流用）が
  control < low < mid < high と単調増加するか。
- 化合物ごとに Spearman ρ を計算し、全化合物での分布を集計。投与時点についても同様。
- 既知の肝毒性化合物 / 非毒性化合物で層別すれば陰性対照が付く。

**「良い表現とは用量反応を線形に写す表現である」**という、effective rank より遥かに反論しにくい主張になる。

#### (2) クロスソース転移 ★本来やりたかった「バッチ頑健性」のタスク版
- TG-GATEs で学習した probe を手持ちデータで評価（およびその逆）。
- **手持ちデータは当初は学習に混ぜず、外部検証に取っておく。**
- 手持ちデータ（百スライド前後）には部位アノテーションを付与予定のため、
  スライドレベルの転移評価に加えてタイルレベル（選択肢A）の外部検証にも使える。
  ただし規模が小さいため統計的な頑健性には限界がある点は結果解釈時に明記する。

#### (3) 下流タスク性能（UNI の評価プロトコル踏襲、特徴量は凍結）
- **ABMIL（スライドレベル）** — 所見の有無 / grade。※スライドレベルラベルのみなので主軸はこれ
- **Linear probe / kNN（タイルレベル）** — ROI アノテーションを作れた場合
- **ラベル効率カーブ**（1/2/4/8/16 slides per class）— ドメイン適応の効果が最も出る領域。Ammeling の再現
- **Leave-one-study-out / leave-one-scanner-out**
- **退行チェック**：一般病理ベンチ（CRC-100K, PCam 等）で元の UNI から落ちていないこと。
  継続 SSL は汎用性能を壊しがちなので必須。

#### (4) バッチ指標（使う場合の条件）
- **vehicle control のタイルに限定して eta² を計算する。** TG-GATEs は全 study に対照群があるため、
  「生物学的にほぼ同一だが study / 染色ロットが異なる」大きなプールが取れる。
  ここでの残差分散は純粋にノイズなので「小さいほど良い」と言い切れる。
- **必ず (1)(2) と対で報告する**（scIB の batch correction / bio conservation 二軸と同じ構図）。
  片方だけの報告は必ず突かれる。

#### (5) Effective rank の正しい置き場所
ラット肝 H&E のみで継続 SSL を回すとタイルの多様性が極端に低い（どこも肝細胞索）ため
**DINOv2 が collapse しやすい**。ここでは effective rank は
**主要評価ではなく崩壊検知のトレーニングモニタとして有用**。step ごとに記録し、低下したら停止。
加えて **元の UNI 特徴への anchor 損失（KL / cosine）**を入れて保険とする。

### 5.3 データ分割の落とし穴（先に潰すこと）

1. **分割は必ず「化合物単位」。** TG-GATEs は 1群5匹×複数時点。動物単位・スライド単位で分けると
   同一 study の染色特徴を覚えるだけで probe が高得点を出す。汎化を主張するなら compound-level split 一択。
2. **極端な不均衡**（大半が所見なし）。AUROC より AUPRC / balanced metric を主軸に。

## 6. フェーズ計画

### Phase 0 — 学習なし（最優先）
- **Miyabi 環境検証**（下記 §7）。ここが通らないと全体が止まるので最初にやる。
- TG-GATEs WSI の取得・フォーマット確認・タイル抽出パイプライン。
- 評価パイプラインの完成：ABMIL / ラベル効率カーブ / 用量反応単調性 / eta²(control限定) / クロスソース転移。
- **凍結 UNI・UNI2 のベースライン確定。** ImageNet ViT との比較も取る。
- → この時点で「UNI2 vs UNI」「汎用 ViT からの改善幅」が出るので、それ自体が報告になる。

### Phase 1 — 小規模 LoRA
- 単一臓器（肝）・単施設・数百スライド。
- rank ∈ {8, 16, 32}、適用層 2〜3 条件（qkv のみ / qkv + MLP など）。
- Phase 0 のベンチで測定。effective rank と eta² は学習中モニタとして記録（崩壊検知用）。
- 選択肢 A（教師あり LoRA）を並走させ、到達上限を押さえる。

### Phase 2 — スケールアップ
- 勝ち筋が見えてから臓器・施設を広げて Miyabi の資源を投入。

### 保険
Phase 1 が空振り（LoRA で改善しない）だった場合に備え、
**「毒性病理では既存の汎用病理 FM で十分／不十分」という結論自体が報告価値を持つ形**に構成しておく。

## 7. Miyabi 側の実務リスク（Phase 0 で潰す）

- **aarch64（GH200）**：PyTorch + CUDA の arm64 wheel、**xformers / flash-attn のビルドが通るか**。
  DINOv2 の学習コードはこれらに依存する。**最初に確認すること。**
- **1ノード1GPU 構成**：DINOv2 の大バッチ（通常 global batch 1024+）には multi-node が必要。
  LoRA でメモリは浮いてもバッチサイズ要件は変わらない。
- **タイル抽出の I/O**：ラット肝だけで数万スライド規模になり得る。
  WSI フォーマット（NDPI 等）とストレージ・I/O 見積もりを先に。

## 8. 未確定事項（2026-08-29 解消済み）

1. **手持ちデータの規模** — 百スライド前後（少量）。部位アノテーションを付与予定。
   施設数・スキャナ種は引き続き未確定（Phase 0 で確認）。
2. **ROI アノテーションの余地** — 作成可能（病理医アクセスあり）。選択肢 A の並走が確定的に開いた。
3. **継続 SSL にこだわる理由** — アノテーションコストの回避。
   ただし上記 2 の通りアノテーション自体は作成可能と判明したため、この動機は前提を失っている。
   それでも検討の結果、**継続 SSL の新規性を優先し C を主軸のまま維持**することで確定（§4）。
   評価は引き続き用量反応単調性（§5.2(1)）を主軸に据える。

## 9. 参考文献

- Ammeling et al. "Benchmarking Foundation Models for Mitotic Figure Classification." MELBA 2026.
  https://www.melba-journal.org/papers/2026:003.html
- Graf O, Patel D, Groß P, Lempp C, Hein M, Heinemann F. "Toxicity Assessment in Preclinical Histopathology
  via Class-Aware Mahalanobis Distance for Known and Novel Anomalies." arXiv 2602.02124 / Sci Rep 2026.
  https://arxiv.org/html/2602.02124v2
- "Robustifying pathology foundation models via fine-tuning." arXiv 2607.22861.
  https://arxiv.org/html/2607.22861v1
- Garrido et al. "RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations
  by Their Rank." ICML 2023.
- Chen et al. "Towards a general-purpose foundation model for computational pathology (UNI)." Nat Med 2024.
- Luecken et al. "Benchmarking atlas-level data integration in single-cell genomics (scIB)." Nat Methods 2022.
  — batch correction / bio conservation の二軸評価の参照元
