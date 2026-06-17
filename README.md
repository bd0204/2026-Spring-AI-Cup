# 2026 Spring AI CUP — Transformer Branch

This repository is the **Transformer branch**. The final submission
fuses this branch with our teammate's **LGBM branch**.

**Private Leaderboard: `0.3834716` (Rank 10).**

---

## 1. Team Solution Overview

```
┌─ LGBM branch (teammate ckt1022, separate repo) ───────────────┐
│  train_unmasked.py (with player ID)  ┐                        │
│  train_masked.py   (player ID masked)┘─ blend.py (0.3 masked + │
│                                  0.7 unmasked) → proba_blend_lgbm.npz
└────────────────────────────────────────────────────────────────┘
                                          │
┌─ Transformer branch (THIS repo) ──────┐ │
│  v38 ┐                                │ │
│  v40 ┼─ each 5-fold → inference proba │ │
│  v41 ┘   ↓ ensemble_proba.py          │ │
│       proba_ensemble_nn.npz           │ │
└───────────────────────────────────────┘ │
                    │                       │
                    ▼                       ▼
              hybrid_blend.py  (final fusion)
   actionId/pointId = LGBM ; serverGetPoint = transformer × LGBM (rank-conditional)
                    │
                    ▼
              submission.csv
```

---

## 2. Environment

```
Python 3.10
torch==2.7.1+cu118      # CUDA 11.8
numpy==1.26.4
pandas==2.3.3
scikit-learn==1.7.2
```
Trained on a single NVIDIA Tesla V100-SXM2-16GB via a PBS scheduler
(`train_job.sh` / `infer_job.sh`). See `requirements.txt`.

---

## 3. Data & External Files

Place the following in the project root:

| File | Source | Purpose |
|---|---|---|
| `train.csv` | official | main training data |
| `test_new.csv` | official | prediction target (inference) |
| `test.csv` | official | training-time alignment (IPW / fold split) |
| `processed_train_e.csv` | Kaggle | "all-unseen-player" extra training data |
| `proba_blend_lgbm.npz` | **teammate LGBM branch** | required for final fusion (read by `hybrid_blend.py`) |

> `proba_blend_lgbm.npz` is produced by the teammate repo `2026-Spring-AI-Cup-ckt1022_lgbm`
> (`train/blend/blend.py`). It contains `action (N,19)`, `point (N,10)`, `winner (N,)` probability
> arrays aligned by `rally_uid`.

---

## 4. Transformer Architecture (ShuttleNet)

`ShuttleNetModel` in `model.py` — a dual-view design inspired by ShuttleNet:

- **Dual-view stroke embedding:** `e_t` (technique view) and `e_a` (area view).
- **Match-level player fingerprint:** for each `(match, player)` we compute a "play-style
  fingerprint" (histograms of action / landing / spin, etc., via leave-one-out + Bayesian
  smoothing to avoid leakage), projected and added to the stroke embedding so the model can read
  a player's style even for unseen players.
- **Rally encoder** (area view, whole sequence) + **Player A/B encoder** (technique view, odd/even
  split).
- **TAA decoder:** produces `z_t` (k=2 short window, for action) and `z_a` (k=4 window, for point).
- **Three task heads** (each with a *long-view* whole-rally masked-mean path + a *short-prefix
  expert* gated-MLP correction): `action_head`, `point_head`, `winner_head`.
- **Training mechanisms:** `GroupKFold(groups=match)` mirrors the train→test match isolation;
  prefix-weighted validation (weighted by the test prefix-length distribution); class-aware
  re-sampling (up-weights rare shot types); winner head uses pairwise AUC loss + label smoothing;
  `player_id_dropout=0.5` (randomly masks a whole rally's player IDs to force cross-player
  generalization).

---

## 5. ⭐ Differences between v38 / v40 / v41

The three versions **share the same `model.py` and `train.py`**; they differ only in a few
`config.py` switches — they are progressive upgrades of one architecture, deliberately kept
different so the final ensemble gains diversity.

The current `config.py` defaults are **v41**. To reproduce the other two, change only the
fields below:

| `config.py` setting | **v38** | **v40** | **v41 (default)** | Note |
|---|:---:|:---:|:---:|---|
| `use_action_long_view` | `True` | `True` | `True` | action whole-rally long view (all three) |
| `use_action_short_expert` | `True` | `True` | `True` | action short-prefix expert (all three) |
| `use_winner_long_view` | `True` | `True` | `True` | winner long view (all three) |
| `use_winner_short_expert` | `True` | `True` | `True` | winner short-prefix expert (all three) |
| **`use_point_long_view`** | `False` | `True` | `True` | **added in v40**: point long view |
| **`use_point_short_expert`** | `False` | `True` | `True` | **added in v40**: point short expert |
| **`use_fp_action_prior`** | `False` | `False` | `True` | **added in v41**: fingerprint→action injection |
| `epochs` | `120` | `130` | `130` | |

All other hyper-parameters are identical across the three: `class_balance_power=0.5`,
`class_balance_target="action"`, `player_id_dropout=0.5`, `dropout=0.2`, `max_seq_len=60`,
`batch_size=128`, `lr=5e-4`, `weight_decay=1e-4`, `patience=15`.

---

## 6. ⭐ Reproduction Pipeline

Two ways to reproduce:

- **Option A — from scratch:** run Steps 1–4 below (train → inference → ensemble → fuse).
- **Option B — skip training, use released weights:** download `transformer_weights.zip`
  from this repo's **[Releases](../../releases)** and unzip it into the project root. You will get
  `ckpt_v38/`, `ckpt_v40/`, `ckpt_v41/` (5 `fold*.pt` each) and `checkpoints/known_player_ids.json`.
  Then **skip Step 1 and start from Step 2** (inference → ensemble → fuse).

### Step 1 — Train all three Transformer versions (5-fold each)

Run once for **v41 / v40 / v38** (edit `config.py` per the table in §5, then rename the produced
`./checkpoints/` to a per-version folder):

```bash
# === v41 (config.py defaults; train directly) ===
qsub train_job.sh                 # 5-fold → ./checkpoints/
mv ./checkpoints ./ckpt_v41

# === v40 (edit config.py: use_fp_action_prior=False) ===
qsub train_job.sh
mv ./checkpoints ./ckpt_v40

# === v38 (edit config.py: use_point_long_view=False,
#          use_point_short_expert=False, use_fp_action_prior=False, epochs=120) ===
qsub train_job.sh
mv ./checkpoints ./ckpt_v38
```

### Step 2 — Inference each version

For each version set `config.py`'s **architecture switches back to that version's values** (must
match training, otherwise the checkpoint won't load), and set the weight source / output path:

```python
# config.py (v41 example)
inference_dirs     = ["./ckpt_v41"]        # ensemble of that version's 5 folds
save_nn_proba_path = "./proba_v41_nn.npz"  # dump raw NN probabilities (action/point/winner)
lgbm_proba_path    = ""                     # pure-NN mode
```
```bash
qsub infer_job.sh    # → proba_v41_nn.npz
```
Repeat for v40 and v38 (`inference_dirs=["./ckpt_v40"]`, `save_nn_proba_path="./proba_v40_nn.npz"`,
and so on). You should end up with `proba_v38_nn.npz`, `proba_v40_nn.npz`, `proba_v41_nn.npz` in the root.

### Step 3 — Ensemble the three Transformer versions

```bash
python ensemble_proba.py
```
- Arithmetic mean of the three versions' raw proba (`FUSION_MODE="arithmetic"`), then
  a prior shift on action/point before argmax, and a plain average for winner.
- Produces the pure-transformer `submission.csv` and **`proba_ensemble_nn.npz`** (fused
  probabilities, used in the next step).

### Step 4 — Fuse with LGBM

Put the `proba_blend_lgbm.npz` from ckt1022_lgbm branch in the project root, then:

```bash
python hybrid_blend.py    # → submission.csv (final)
```

---

## 7. LGBM Fusion Details

The three columns of the final `submission.csv` come from **different sources**:

| Column | Source | How |
|---|---|---|
| `actionId` | **LGBM only** | argmax of LGBM action proba (Transformer is weaker on action) |
| `pointId` | **LGBM only** | argmax of LGBM point proba (same reason) |
| `serverGetPoint` | **Transformer × LGBM** | **rank-conditional fusion** (below) |

**Rank-conditional fusion for `serverGetPoint`** (`W_NN_WINNER=0.5`):

```
nn_rank   = rank01(NN_ensemble_winner)      # global [0,1] rank
lgbm_rank = rank01(LGBM_winner)
seen rally   (both players seen in training):
    serverGetPoint = 0.5 * nn_rank + 0.5 * lgbm_rank
unseen rally:
    serverGetPoint = lgbm_rank               # pure LGBM, no downside risk
```

- Fusion happens entirely in **rank space**, avoiding probability-scale (calibration) mismatch
  between the two models.
- It is **conditional** (transformer is mixed in only when both players were seen in training)
- `seen` is decided by `build_seen()`: every stroke's `gamePlayerId` and `gamePlayerOtherId` in the
  rally must appear in `known_player_ids.json`.

---

## 8. File List

**Core Transformer**
| File | Description |
|---|---|
| `config.py` | all hyper-parameters and architecture switches (defaults = v41) |
| `model.py` | ShuttleNet dual-view model + three task heads + fp_action_prior |
| `dataset.py` | sliding-window dataset, length-IPW + class-aware sampling |
| `utils.py` | player fingerprint, feature engineering, EMA, loss, scheduler |
| `train.py` | 5-fold training (GroupKFold + fold-level resume) |
| `inference.py` | ensemble inference; outputs a submission or raw proba npz |
| `train_job.sh` / `infer_job.sh` | PBS job scripts |

**Fusion**
| File | Description |
|---|---|
| `ensemble_proba.py` | fuse v38/v40/v41 raw proba → pure-transformer submission + `proba_ensemble_nn.npz` |
| `hybrid_blend.py` | transformer × LGBM fusion (action/point = pure LGBM; serverGetPoint = rank-conditional) → final submission |

**LGBM branch:** see the repo `2026-Spring-AI-Cup-ckt1022_lgbm`.
