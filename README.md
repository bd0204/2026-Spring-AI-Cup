# 2026 Spring AI CUP — 桌球戰術與勝負預測（Transformer 分支）

> 任務：給定一個 rally 的前 *k* 拍（prefix），預測**下一拍**的
> `actionId`（球種，19 類）、`pointId`（落點，10 類）、`serverGetPoint`（發球方是否得分，二元機率）。
> 官方評分：`Score = 0.4 × Macro-F1(action) + 0.4 × Macro-F1(point) + 0.2 × AUC(serverGetPoint)`

本 repo 是團隊方案的 **Transformer 分支**。最終提交由本分支與隊友的 **LGBM 分支**融合而成。

- **最終 Public LB：0.5618189（2/414）**
- LGBM 分支（隊友 ckt1022）：另一個 repo `2026-Spring-AI-Cup-ckt1022_lgbm`，產出 `proba_blend_lgbm.npz`
- 最終融合方式：`actionId / pointId` 採用 LGBM；`serverGetPoint` 採用 **Transformer ensemble × LGBM 的 rank-conditional 融合**

---

## 1. 團隊整體方案總覽

```
┌─ LGBM 分支（隊友 ckt1022 的獨立 repo）────────────────────────┐
│  train_unmasked.py (含選手ID)  ┐                              │
│  train_masked.py   (遮選手ID)  ┘─ blend.py (0.3 masked + 0.7  │
│                                   unmasked) → proba_blend_lgbm.npz
└───────────────────────────────────────────────────────────────┘
                                          │
┌─ Transformer 分支（本 repo）──────────┐ │
│  v38 ┐                                │ │
│  v40 ┼─ 各 5-fold → inference 存 proba│ │
│  v41 ┘   ↓ ensemble_proba.py          │ │
│       proba_ensemble_nn.npz           │ │
└───────────────────────────────────────┘ │
                    │                       │
                    ▼                       ▼
              hybrid_blend.py（最終融合）
   actionId/pointId = LGBM；serverGetPoint = NN×LGBM rank-conditional
                    │
                    ▼
              submission.csv（LB 0.5618）
```

---

## 2. 環境需求

```
Python 3.10
torch==2.7.1+cu118     # CUDA 11.8
numpy==1.26.4
pandas==2.3.3
scikit-learn==1.7.2
```
訓練於單張 NVIDIA Tesla V100-SXM2-16GB（PBS 排程，見 `train_job.sh` / `infer_job.sh`）。
詳見 `requirements.txt`。

---

## 3. 資料與外部檔案放置

放在專案根目錄：

| 檔案 | 來源 | 用途 |
|---|---|---|
| `train.csv` | 官方 | 訓練主資料 |
| `test_new.csv` | 官方 | 預測目標（inference） |
| `test.csv` | 官方 | 訓練時對齊用（IPW / fold split） |
| `processed_train_e.csv` | 官方額外資料集處理後 | 「全 unseen 選手」額外訓練資料 |
| `proba_blend_lgbm.npz` | **隊友 LGBM 分支產出** | 最終融合用（`hybrid_blend.py` 讀取） |

> `proba_blend_lgbm.npz` 由隊友 repo `2026-Spring-AI-Cup-ckt1022_lgbm` 的 `train/blend/blend.py` 產生，
> 內含 `action (N,19)` / `point (N,10)` / `winner (N,)` 三個機率陣列，以 `rally_uid` 對齊。

---

## 4. Transformer 模型架構（ShuttleNet）

`model.py` 的 `ShuttleNetModel`，受 ShuttleNet 啟發的雙視角結構：

- **雙視角 stroke embedding**：`e_t`（technique 技術視角）與 `e_a`（area 區域視角）
- **Match-level player fingerprint**：對每個 (match, player) 算「打法指紋」（action/落點/旋轉等直方圖，leave-one-out + Bayesian smoothing 防洩漏），投影後加到 stroke embedding，讓模型對 unseen 選手仍能讀取其打法
- **Rally encoder**（area 視角看整序列）＋ **Player A/B encoder**（technique 視角奇偶拆分）
- **TAA decoder**：產出 `z_t`（k=2 短窗，給 action）與 `z_a`（k=4 窗，給 point）
- **三個任務 head**（各含 *long-view* 全 rally masked-mean 路徑 + *short-prefix expert* 短拍 gated-MLP 修正）：
  - `action_head`、`point_head`、`winner_head`
- **訓練機制**：`GroupKFold(groups=match)` 鏡射 train→test 場次隔離；prefix-weighted validation（依 test prefix 長度分布加權）；class-aware re-sampling（稀有球種加權）；winner 加 pairwise AUC loss + label smoothing；`player_id_dropout=0.5`（隨機遮整 rally 的選手 ID，強迫學跨選手泛化）。

---

## 5. ⭐ v38 / v40 / v41 三版差異

三版**共用同一份 `model.py` 與 `train.py`**，只差 `config.py` 的幾個開關 —— 它們是「同一架構的漸進升級」，刻意保留架構差異以在最終 ensemble 取得 diversity（variance reduction）。

`config.py` 目前的預設值即 **v41**。要重現另外兩版，只需改下表的欄位：

| `config.py` 設定 | **v38** | **v40** | **v41（預設）** | 說明 |
|---|:---:|:---:|:---:|---|
| `use_action_long_view` | `True` | `True` | `True` | action 加全 rally 長視野（三版皆同）|
| `use_action_short_expert` | `True` | `True` | `True` | action 短拍專家（三版皆同）|
| `use_winner_long_view` | `True` | `True` | `True` | winner 長視野（三版皆同）|
| `use_winner_short_expert` | `True` | `True` | `True` | winner 短拍專家（三版皆同）|
| **`use_point_long_view`** | `False` | `True` | `True` | **v40 新增**：point 加全 rally 長視野 |
| **`use_point_short_expert`** | `False` | `True` | `True` | **v40 新增**：point 短拍專家 |
| **`use_fp_action_prior`** | `False` | `False` | `True` | **v41 新增**：fingerprint→action 直接注入 |
| `epochs` | `120` | `130` | `130` | |

其餘超參三版完全一致：`class_balance_power=0.5`、`class_balance_target="action"`、`player_id_dropout=0.5`、`dropout=0.2`、`max_seq_len=60`、`batch_size=128`、`lr=5e-4`、`weight_decay=1e-4`、`patience=15`。

**三版的設計演進：**

- **v38**：在 `action` 與 `winner` 兩個 head 各加「long-view（對整個 rally 做 masked-mean → Linear，補足 TAA 短窗看不到的長期 pattern）」＋「short-prefix expert（`seq_len≤2` 時用 gated MLP 修正，因 test 有 53% rally 只有 1–2 拍）」。此時 `point` head 仍是基本款。
- **v40**：把同一組 long-view + short-expert **鏡像到 `point` head**，並把 `epochs` 由 120 提到 130。
  （中間的 v39 曾同時改 `class_balance_power 0.4` + `target "both"` + `epochs 160`，造成過擬合；v40 是回退這些、只乾淨保留 point 結構升級的版本。）
- **v41**：新增 `fp_action_prior` —— 將「下一拍 striker 的 fingerprint action 直方圖」經 gated MLP **直接加到 action logits**，繞過 encoder 的稀釋，針對性補強「動作選擇＝選手戰術習慣」這種跨選手難泛化的訊號。

**各版純 Transformer 的 Public LB（單獨提交）：** v38 = 0.3773、v40 = 0.3725、v41 = 0.3758。三版單獨都在 0.37 量級，但因架構不同、預測互補（逐 rally 分歧 action≈38% / point≈53%），**ensemble 後達 0.3885**。

---

## 6. ⭐ 重現完整 pipeline

### Step 1 — 訓練三版 Transformer（各 5-fold）

對 **v41 / v40 / v38** 各做一次（依上表改 `config.py`，並把產出的 `./checkpoints/` 改名到對應資料夾）：

```bash
# === v41（config.py 預設值，直接訓練）===
qsub train_job.sh                 # 5-fold → ./checkpoints/
mv ./checkpoints ./ckpt_v41

# === v40（改 config.py：use_fp_action_prior=False）===
qsub train_job.sh
mv ./checkpoints ./ckpt_v40

# === v38（改 config.py：use_point_long_view=False,
#          use_point_short_expert=False, use_fp_action_prior=False, epochs=120）===
qsub train_job.sh
mv ./checkpoints ./ckpt_v38
```
> `train.py` 內建 fold-level resume：PBS 12hr walltime 中斷後，重新 `qsub` 會自動跳過已完成的 fold、接續未完成的。
> `known_player_ids.json` 會寫入 `./checkpoints/`（內容與版本無關），供 inference 遮蔽 unseen 選手用。

### Step 2 — 每版 inference，存 raw NN proba

每版需把 `config.py` 的**架構開關設回該版的值**（與訓練時一致，否則權重載入會 shape mismatch），並設定權重來源與輸出路徑：

```python
# config.py（v41 為例）
inference_dirs     = ["./ckpt_v41"]        # 載入該版 5-fold 權重做 ensemble
save_nn_proba_path = "./proba_v41_nn.npz"  # 存下 raw NN 機率（action/point/winner）
lgbm_proba_path    = ""                     # 純 NN 模式
```
```bash
qsub infer_job.sh    # → proba_v41_nn.npz
```
對 v40、v38 重複（`inference_dirs=["./ckpt_v40"]`、`save_nn_proba_path="./proba_v40_nn.npz"`，依此類推）。
完成後根目錄會有 `proba_v38_nn.npz`、`proba_v40_nn.npz`、`proba_v41_nn.npz`。

### Step 3 — 三版 Transformer ensemble

```bash
python ensemble_proba.py
```
- 對三版的 raw proba 做 **算術平均**（`FUSION_MODE="arithmetic"`；註：曾試 geometric 平均，0.3867 < 0.3885，因高方差任務算術平均較穩），再對 action/point 套 prior shift 取 argmax，winner 直接平均。
- 產出純 NN `submission.csv`（LB 0.3885）與 **`proba_ensemble_nn.npz`**（融合後機率，供下一步）。

### Step 4 — 與 LGBM 融合（最終提交）

把隊友的 `proba_blend_lgbm.npz` 放到根目錄，然後：

```bash
python hybrid_blend.py    # → submission.csv（最終，LB 0.5618）
```

---

## 7. ⭐ 與 LGBM 融合的細節（`hybrid_blend.py`）

最終 `submission.csv` 的三個欄位來源**不同**：

| 欄位 | 來源 | 做法 |
|---|---|---|
| `actionId` | **純 LGBM** | LGBM action 機率 argmax（NN 在 action 較弱，融合會稀釋 → 不採用 NN）|
| `pointId` | **純 LGBM** | LGBM point 機率 argmax（同上）|
| `serverGetPoint` | **Transformer × LGBM** | **rank-conditional 融合**（見下）|

**serverGetPoint 的 rank-conditional 融合**（`W_NN_WINNER=0.5`）：

```
nn_rank   = rank01(NN_ensemble_winner)     # 全體轉 [0,1] 排序
lgbm_rank = rank01(LGBM_winner)
seen rally（雙方選手都在訓練看過）：
    serverGetPoint = 0.5 × nn_rank + 0.5 × lgbm_rank
unseen rally：
    serverGetPoint = lgbm_rank              # 純 LGBM，零下檔風險
```

- 全程在 **rank space** 融合，避免兩模型機率刻度不一致（calibration mismatch）。
- 採 **conditional**（只在雙方選手都 seen 的 rally 才混入 NN）：NN winner 對 seen 選手的 OOF AUC≈0.86、對 unseen 僅≈0.73；且 NN 與 LGBM 的 winner rank 相關僅 0.45（diversity 高），故在 seen rally 融合能互補、在 unseen 維持純 LGBM 不冒險。
- `seen` 的判定（`build_seen()`）：rally 內所有 stroke 的 `gamePlayerId` 與 `gamePlayerOtherId` 都在 `known_player_ids.json` 內。

> 結果：相對純 LGBM（0.561），融合後 0.5618 —— Transformer 的 winner diversity 對 `serverGetPoint` 帶來正貢獻。

---

## 8. 檔案清單

**核心 Transformer（訓練／推論）**
| 檔案 | 說明 |
|---|---|
| `config.py` | 所有超參數與架構開關（預設＝v41）|
| `model.py` | ShuttleNet 雙視角模型 + 三任務 head + fp_action_prior |
| `dataset.py` | sliding-window 資料集、length-IPW + class-aware 取樣 |
| `utils.py` | player fingerprint、特徵工程、EMA、loss、scheduler 等 |
| `train.py` | 5-fold 訓練（GroupKFold + fold-level resume）|
| `inference.py` | 載入權重做 ensemble 推論，可輸出 submission 或 raw proba npz |
| `train_job.sh` / `infer_job.sh` | PBS 排程腳本 |

**融合（產出最終提交）**
| 檔案 | 說明 |
|---|---|
| `ensemble_proba.py` | 融合 v38/v40/v41 三版 raw proba → 純 NN submission + `proba_ensemble_nn.npz` |
| `hybrid_blend.py` | NN ensemble × LGBM 融合（action/point 純 LGBM；serverGetPoint rank-conditional）→ 最終 submission |

**LGBM 分支**：見隊友 repo `2026-Spring-AI-Cup-ckt1022_lgbm`（提供 `proba_blend_lgbm.npz`）。
