"""
utils.py - 工具函數與輔助類別
包含：set_seed, EMA, CosineWarmupScheduler, FocalLoss, MultiTaskLoss, 特徵工程
"""

import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 隨機種子
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ============================================================
# 特徵工程
# ============================================================

def infer_dominant_hand(df):
    """
    從各場比賽(match)中每位選手的發球站位推斷慣用手，同時回傳推斷信心度。

    桌球規則下每人連續發球兩次才換邊，因此發球時站位相當穩定：
      - 發球(strikeId==1)時 positionId==1 多 → 右撇子(1)
      - 發球(strikeId==1)時 positionId==3 多 → 左撇子(2)
      - 若該場沒有任何有效站位 (positionId 皆為 0 或 2) → 預設右撇子(1), 信心度 0

    信心度 = |cnt1 - cnt3| / (cnt1 + cnt3)，範圍 [0, 1]；
    全部都落在 positionId==0 的選手信心度為 0，模型可據此忽略該 rally 的 hand 特徵。

    注意：gamePlayerId 只有 1 / 2，是場內相對編號，所以要以
    (match, gamePlayerId) 為 key，不能跨場合併。

    Returns:
        dict: {(match, gamePlayerId): (hand, confidence)}
    """
    serves = df[df["strikeId"] == 1]
    grouped = serves.groupby(["match", "gamePlayerId"])["positionId"]
    cnt1 = grouped.apply(lambda s: (s == 1).sum())
    cnt3 = grouped.apply(lambda s: (s == 3).sum())
    total = cnt1 + cnt3

    out = {}
    for key in cnt1.index:
        c1, c3, tot = int(cnt1[key]), int(cnt3[key]), int(total[key])
        if tot == 0:
            out[key] = (1, 0.0)
        else:
            hand = 2 if c3 > c1 else 1
            conf = abs(c1 - c3) / tot
            out[key] = (hand, conf)
    return out


# pointId 九宮格鏡射：對「接球方為左撇子」的 rally，把 pointId 換到
# 「假設接球方為右撇子」的 canonical frame (同一欄左右對換)。
_POINT_MIRROR = {0: 0, 1: 3, 2: 2, 3: 1, 4: 6, 5: 5, 6: 4, 7: 9, 8: 8, 9: 7}
# positionId 鏡射：對「擊球方為左撇子」的 row，把 positionId 換到
# 「假設擊球方為右撇子」的 canonical frame。
_POS_MIRROR = {0: 0, 1: 3, 2: 2, 3: 1}

# Phase C: actionId → action category 對映
#   0 = Zero/Other      {0}
#   1 = Attack          {1, 2, 3, 4, 5, 6, 7}  drive / counter / smash / twist / fast drive / fast push / flip
#   2 = Control         {8, 9, 10, 11}          pimple's long/fast push / long push / drop shot
#   3 = Defensive       {12, 13, 14}            chop / block / lob
#   4 = Serve           {15, 16, 17, 18}        traditional / hook / reverse / squat (rally 第一拍才出現)
ACTION_TO_CATEGORY = np.array(
    [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4],
    dtype=np.int64,
)


def _lookup_hand(df, hand_map, player_col, default=(1, 0.0)):
    """Return (hand_list, conf_list) aligned with df rows."""
    keys = list(zip(df["match"].tolist(), df[player_col].tolist()))
    pairs = [hand_map.get(k, default) for k in keys]
    hands = [p[0] for p in pairs]
    confs = [p[1] for p in pairs]
    return hands, confs


# ============================================================
# Match-level player fingerprint
# ------------------------------------------------------------
# 每個 (match, gamePlayerId) 計算該選手在該場比賽的「打法指紋」：
#   action / pointId_norm / positionId_norm / spinId / strengthId 五個直方圖
#   + log(1 + n_other) 樣本量信心度
# 共 19 + 10 + 4 + 6 + 4 + 1 = 44 維。
#
# 為什麼這個對 unseen player 特別有效？
#   train/test 完全沒共用 match (驗證為 0 重疊)。即使某選手 ID 在訓練集從沒
#   出現過，他在 test 的這場 match 裡仍然打了多個 rallies，可以從同 match 的
#   其他 rallies 算出他的指紋；模型訓練時已學會「讀指紋」，自然 transfer。
#
# 防 leakage 方法：leave-one-out + Bayesian smoothing
#   loo_sum   = total_sum  - this_rally_sum
#   loo_count = total_n    - this_rally_n
#   smoothed  = (loo_sum + α·global) / (loo_count + α)
#   loo_count=0 時自動退回 global prior（公式邊界自然處理）。
# ============================================================


class FingerprintTable:
    """Match-level player fingerprint lookup with LOO + Bayesian smoothing.

    Build from train.csv (for training) or test.csv (for inference) **separately**
    — train/test 沒共用 match，所以兩邊各算各的就是乾淨對稱的設計。
    """

    SPECS = (
        ("actionId", 19),
        ("pointId_norm", 10),
        ("positionId_norm", 4),
        ("spinId", 6),
        ("strengthId", 4),
    )

    @classmethod
    def total_dim(cls):
        """fp_dim = sum of histogram sizes + 1 (log_n_other)"""
        return sum(n for _, n in cls.SPECS) + 1  # 44

    def __init__(self, fp_loo, mp_sum, mp_count, global_prior, alpha, fp_inner):
        # fp_loo: dict[(match, player, rally_uid)] -> ndarray (fp_inner+1,)
        # mp_sum: dict[(match, player)]            -> ndarray (fp_inner,)
        # mp_count: dict[(match, player)]          -> int
        # global_prior: ndarray (fp_inner+1,)，最後一格 = 0 (log_n_other when n=0)
        self.fp_loo = fp_loo
        self.mp_sum = mp_sum
        self.mp_count = mp_count
        self.global_prior = global_prior
        self.alpha = float(alpha)
        self.fp_inner = int(fp_inner)
        self.fp_dim = int(fp_inner) + 1

    def lookup(self, match, player_id, rally_uid):
        """Return per-stroke fingerprint vector (fp_dim,) as np.float32.

        Lookup priority:
          1. (match, player, rally) → LOO fingerprint (most info)
          2. (match, player) without rally → no LOO subtraction (player struck
             in this match but not in this rally)
          3. global_prior (player never struck in this match)
        """
        m = int(match)
        p = int(player_id)
        u = int(rally_uid)
        v = self.fp_loo.get((m, p, u))
        if v is not None:
            return v
        S = self.mp_sum.get((m, p))
        if S is not None:
            N = self.mp_count[(m, p)]
            smoothed = (S + self.alpha * self.global_prior[:-1]) / (N + self.alpha)
            log_n = np.float32(np.log1p(N))
            return np.concatenate([smoothed, [log_n]]).astype(np.float32)
        return self.global_prior


def compute_match_player_fingerprints(df, alpha=10.0):
    """Build a FingerprintTable from df.

    Required columns: match, gamePlayerId, rally_uid + all SPECS columns
    (actionId, pointId_norm, positionId_norm, spinId, strengthId).

    Args:
        df: DataFrame, 需先呼叫過 add_engineered_features 以有 *_norm 欄位
        alpha: Bayesian smoothing 強度。α=10 等同 prior 強度為 10 個觀察值；
               n_other<10 時偏 global，>10 時偏 local。
    Returns:
        FingerprintTable
    """
    specs = FingerprintTable.SPECS
    n_rows = len(df)

    # 1. 把所有 categorical features 一次轉成 concat 後的 one-hot 矩陣
    onehot_blocks = []
    for col, n_cls in specs:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
        vals = vals.clip(lower=0, upper=n_cls - 1).astype(int).to_numpy()
        oh = np.zeros((n_rows, n_cls), dtype=np.float32)
        oh[np.arange(n_rows), vals] = 1.0
        onehot_blocks.append(oh)
    onehot = np.concatenate(onehot_blocks, axis=1)
    fp_inner = onehot.shape[1]

    # 2. Global prior = 整 df 的平均 one-hot
    global_prior_inner = onehot.mean(axis=0).astype(np.float32)
    global_prior = np.concatenate([global_prior_inner, [0.0]]).astype(np.float32)

    # 3. 用 pandas groupby 做聚合（O(n_rows)，遠快於 row-by-row）
    fp_cols = [f"_fp_{i}" for i in range(fp_inner)]
    aux = pd.DataFrame(onehot, columns=fp_cols)
    aux["_match"] = df["match"].to_numpy()
    aux["_p"] = df["gamePlayerId"].to_numpy()
    aux["_u"] = df["rally_uid"].to_numpy()

    grp_mp = aux.groupby(["_match", "_p"], sort=False)
    mp_sum_df = grp_mp[fp_cols].sum()
    mp_count_ser = grp_mp.size()

    grp_mpr = aux.groupby(["_match", "_p", "_u"], sort=False)
    mpr_sum_df = grp_mpr[fp_cols].sum()
    mpr_count_ser = grp_mpr.size()

    # 4. 把 pandas 結果 dump 到純 dict（避免 dataset 階段反覆做 .loc 慢操作）
    mp_sum = {
        (int(k[0]), int(k[1])): mp_sum_df.loc[k].to_numpy().astype(np.float32)
        for k in mp_sum_df.index
    }
    mp_count = {
        (int(k[0]), int(k[1])): int(mp_count_ser[k])
        for k in mp_count_ser.index
    }

    # 5. LOO fingerprint per (match, player, rally)
    fp_loo = {}
    for k3 in mpr_sum_df.index:
        m = int(k3[0]); p = int(k3[1]); u = int(k3[2])
        s_mpr = mpr_sum_df.loc[k3].to_numpy().astype(np.float32)
        n_mpr = int(mpr_count_ser[k3])
        S_mp = mp_sum[(m, p)]
        N_mp = mp_count[(m, p)]

        loo_sum = S_mp - s_mpr
        loo_count = N_mp - n_mpr
        # Bayesian smoothing — loo_count=0 時公式自動 → global_prior_inner
        smoothed = (loo_sum + alpha * global_prior_inner) / (loo_count + alpha)
        log_n = np.float32(np.log1p(loo_count))
        fp_loo[(m, p, u)] = np.concatenate(
            [smoothed.astype(np.float32), [log_n]]
        ).astype(np.float32)

    return FingerprintTable(
        fp_loo=fp_loo,
        mp_sum=mp_sum,
        mp_count=mp_count,
        global_prior=global_prior,
        alpha=alpha,
        fp_inner=fp_inner,
    )


def build_rally_fingerprints(rally_dict, fp_table):
    """For each rally_uid, build per-stroke fingerprint matrix.

    Output shape per rally: (seq_len, 2 * fp_dim) — first half = self_fp
    (lookup with gamePlayerId), second half = other_fp (lookup with
    gamePlayerOtherId).

    ⚠️ 注意：lookup 時必須用「未被 mask_unseen_player_ids 改寫的原始 ID」。
    呼叫順序應該是 build_rally_fingerprints → mask_unseen_player_ids，
    這樣 unseen player 的 fingerprint 仍可從同 match 的其他 test rallies 算出。

    Args:
        rally_dict: {rally_uid: rally_df}, rally_df 含 strikeNumber 排序前的
                    所有 strokes（含 match, gamePlayerId, gamePlayerOtherId,
                    rally_uid 欄位，且 ID 必須是 raw 未 masked）
        fp_table: FingerprintTable
    Returns:
        dict[rally_uid] -> np.ndarray (seq_len, 2 * fp_dim) float32
    """
    fp_dim = fp_table.fp_dim
    out = {}
    for uid, df in rally_dict.items():
        df_sorted = df.sort_values("strikeNumber")
        matches = df_sorted["match"].astype(int).to_numpy()
        pids = df_sorted["gamePlayerId"].astype(int).to_numpy()
        oids = df_sorted["gamePlayerOtherId"].astype(int).to_numpy()
        uids = df_sorted["rally_uid"].astype(int).to_numpy()
        seq_len = len(df_sorted)
        fp = np.zeros((seq_len, 2 * fp_dim), dtype=np.float32)
        for i in range(seq_len):
            fp[i, :fp_dim] = fp_table.lookup(matches[i], pids[i], uids[i])
            fp[i, fp_dim:] = fp_table.lookup(matches[i], oids[i], uids[i])
        out[uid] = fp
    return out


def mask_unseen_player_ids(df, known_player_ids):
    """
    把 df 裡沒在 train.csv 出現過的 gamePlayerId / gamePlayerOtherId 替換成 0
    （unknown，對應 embedding 的 padding_idx，吃零向量）。

    這是 unseen-player generalization 的關鍵步驟：
      - 訓練時 player_id_dropout 隨機把整 rally 的 player ID mask 成 0
      - 測試時 test.csv 約有 36.5% 選手是訓練資料完全沒看過的
      - 把這些未見過的 ID 全部當成 0，模型就會 fallback 到「沒有 player 資訊」的
        分支處理（正好對應訓練時的 dropout 分布）

    Args:
        df: 含 gamePlayerId / gamePlayerOtherId 欄位的 DataFrame
        known_player_ids: iterable of int，訓練時看過的 player ID
    Returns:
        新的 DataFrame (淺複製)；未見過的 ID 已被替換成 0
    """
    known = set(int(p) for p in known_player_ids)
    df = df.copy()
    for col in ("gamePlayerId", "gamePlayerOtherId"):
        if col in df.columns:
            df[col] = df[col].where(df[col].isin(known), 0).astype(int)
    return df


def add_engineered_features(df, hand_map=None):
    """
    在 DataFrame 上加入額外的工程特徵。

    Args:
        df: 原始 DataFrame
        hand_map: {(match, gamePlayerId): (hand, confidence)} 字典。
                  如果為 None，會從 df 自身推斷（僅限獨立使用時）。
    """
    df = df.copy()
    # v21: rallyProgress 改回 per-prefix（dataset.py 內 _extract_features 重算）。
    # 歷史：v18 用全 rally 分母，但這在 test 會 broken：
    #   - train sliding window: rally_max = 完整 rally 長度 → 每筆 last input 的
    #     rallyProgress = (n-1)/n ∈ [0.5, 0.95]，**永遠 < 1.0**
    #   - test: df 只含 prefix → rally_max = prefix 長度 → 每筆 last input 的
    #     rallyProgress = k/k = **1.0 永遠**
    # → 模型在 train 從沒看過 input rallyProgress = 1.0，但 test 全部是 1.0 → OOD。
    # v19 因為 mask 不學 class 0 此 bug silent；v20 學了 class 0 後爆掉（test 模型
    # 把「高 rallyProgress = rally 結束」捷徑套到 1.0 → 99.7% 預測 class 0）。
    # 修法：placeholder 0.0，dataset 構建樣本時 per-input 即時計算 strikeNumber/prefix_max
    # → train/test 兩邊每筆的 last input rallyProgress 都 = 1.0，OOD 消失。
    df["rallyProgress"] = 0.0

    # v18: rallyPhase (Tac-Simur, Wang et al. 2020 啟發的 4-phase tactic 結構)。
    # 1=serve, 2=receive, 3=stalemate-server (奇 ≥3), 4=stalemate-receiver (偶 ≥4),
    # 0 保留 padding。模型現有 strikeId 在 stroke ≥3 全等於 4，丟失 server/receiver
    # 視角；rallyPhase 把這個 server-side / receiver-side 拆開。純 strikeNumber 函數，
    # 不依賴 player ID → 跨選手 generalize。
    sn = df["strikeNumber"].astype(int).to_numpy()
    phase = np.where(sn == 1, 1,
            np.where(sn == 2, 2,
            np.where(sn % 2 == 1, 3, 4)))
    df["rallyPhase"] = phase.astype(np.int64)

    # v25: stepInTactic (同 paper "any 3 consecutive strokes form a tactic")。
    # rallyPhase 抓「誰在打」，stepInTactic 抓「tactic 進度」。
    # 1=serve (sN=1)：server 第 1 個 tactic 的 inducement
    # 2=receive (sN=2)：receiver 第 1 個 tactic 的 inducement
    # 3=completion：sN 3,4,7,8,11,12...（odd-tactic 的 completion 拍）
    # 4=new-inducement：sN 5,6,9,10,13,14...（odd-tactic 結束後新 tactic 的開始）
    # 對 sN ≥ 3：stepInTactic = 3 if ((sN-3)//2) % 2 == 0 else 4
    step = np.where(sn == 1, 1,
           np.where(sn == 2, 2,
           np.where(((sn - 3) // 2) % 2 == 0, 3, 4)))
    df["stepInTactic"] = step.astype(np.int64)

    if hand_map is None:
        hand_map = infer_dominant_hand(df)

    # 擊球方與接球方各自的慣用手 + 推斷信心度
    p_hand, p_conf = _lookup_hand(df, hand_map, "gamePlayerId")
    r_hand, r_conf = _lookup_hand(df, hand_map, "gamePlayerOtherId")
    df["playerHand"] = p_hand
    df["receiverHand"] = r_hand
    df["playerHandConf"] = p_conf
    df["receiverHandConf"] = r_conf

    # 慣用手配對 (依 pointId_and_positionId_definition 的圖片定義 pointId)：
    #   1: 右打右  2: 右打左  3: 左打右  4: 左打左
    df["handPair"] = (df["playerHand"] - 1) * 2 + df["receiverHand"]

    # ---- 防禦：sanitize pointId / positionId ----
    # 新版資料可能出現 NaN 或 out-of-range（_POINT_MIRROR 只 cover 0~9），
    # 兩者都會讓下面的 .map() / .astype(int) 爆 IntCastingNaNError。
    # 把 NaN + out-of-range 統一壓成 0 (padding/unknown)，並 print warning。
    def _sanitize(series, valid_max, name):
        raw = pd.to_numeric(series, errors="coerce")
        n_nan = int(raw.isna().sum())
        out = raw.fillna(0).astype(int)
        oor_mask = (out < 0) | (out > valid_max)
        n_oor = int(oor_mask.sum())
        if n_nan or n_oor:
            print(f"  [WARN] {name}: {n_nan} NaN + {n_oor} out-of-range "
                  f"(<0 or >{valid_max}) → coerced to 0")
        return out.where(~oor_mask, 0)

    df["pointId"] = _sanitize(df["pointId"], 9, "pointId")
    df["positionId"] = _sanitize(df["positionId"], 3, "positionId")

    # ---- canonical frame normalization ----
    # v28 bug fix: 官方 doc（黃底）確認 pointId 已是 receiver-perspective (canonical)，
    # positionId 也是 player-perspective。原本對左撇 receiver/striker 做的 1↔3 mirror
    # 是錯的 — 把已 canonical 的數據反向 → 對 ~35% 左撇相關 row 引入錯誤標籤。
    #
    # verify_pointid_frame.py 數據驗證：
    #   - canonical 假設 R/L 差距 13.0%
    #   - court-based 假設 R/L 差距 25.2%（差兩倍）
    #   - 兩邊都呈現「BH > FH」的攻擊偏好（桌球常識），符合 canonical
    #
    # 修法：直接 copy，不 mirror。pointId_norm / positionId_norm 變成 pointId /
    # positionId 的副本（保留欄位是為了不改 cfg.engineered_features 跟 model 維度）。
    df["pointId_norm"] = df["pointId"].astype(int)
    df["positionId_norm"] = df["positionId"].astype(int)

    # ---- Phase C: 4-class action category（粗粒度：Attack/Control/Defensive/Serve） ----
    # 模型輸入用：每一拍多帶一個 category embedding
    # Aux loss 用：target action 也對映到 category，提供額外監督訊號
    aid = pd.to_numeric(df["actionId"], errors="coerce").fillna(0).astype(int)
    aid = aid.clip(lower=0, upper=len(ACTION_TO_CATEGORY) - 1)
    df["actionCategory"] = ACTION_TO_CATEGORY[aid.to_numpy()]

    return df


# ============================================================
# EMA（指數移動平均）
# ============================================================

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.backup = {}

    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply_shadow(self, model: nn.Module):
        """將 EMA 參數套用到模型（驗證/推論時使用）"""
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]

    def restore(self, model: nn.Module):
        """還原模型的原始參數"""
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]
        self.backup = {}


# ============================================================
# 學習率 Scheduler
# ============================================================

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine Annealing with Linear Warmup."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = self.last_epoch / max(1, self.warmup_steps)
        else:
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        return [max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs]


# ============================================================
# 損失函數
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss 處理類別不平衡，支援 mask"""

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, mask=None):
        """
        Args:
            logits: (batch, n_classes)
            targets: (batch,)
            mask: (batch,) bool, True = 參與 loss 計算
        """
        ce = F.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if mask is not None and mask.any():
            # 只對 mask=True 的樣本計算 loss
            return focal[mask].mean()
        elif mask is not None and not mask.any():
            # 整個 batch 都被 mask 掉，回傳 0
            return focal.sum() * 0.0
        else:
            return focal.mean()


class MultiTaskLoss(nn.Module):
    """
    多任務加權損失
    TotalLoss = w_action * FocalLoss(action) + w_point * FocalLoss(point, masked)
              + w_winner * BCE(winner) + w_action_category * FocalLoss(category)

    v20 起 point_mask 永遠為 True：rally-end pointId=0 是 LB macro-F1 的合法 class，
    不能從 loss 排除（之前排除導致 model 從不預測 class 0）。point_mask 欄位保留
    以維持 collate 介面相容性。Action category aux loss 在 cat_logits / cat_target
    都有提供時才會加進來。
    """

    def __init__(self, cfg):
        super().__init__()
        self.w_action = cfg.w_action
        self.w_point = cfg.w_point
        self.w_winner = cfg.w_winner
        self.w_action_category = float(getattr(cfg, "w_action_category", 0.0))

        self.action_loss = FocalLoss(gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing)
        self.point_loss = FocalLoss(gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing)
        # v31.1: 改用 reduction='none' 以支援 winner_mask（test_k row 沒有真實 winner 標籤）
        self.winner_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.category_loss = FocalLoss(gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing)
        # v33 (Tier A2): winner BCE 的 label smoothing。target {0,1} → {ε, 1-ε}
        self.winner_label_smoothing = float(
            getattr(cfg, "winner_label_smoothing", 0.0)
        )
        self.winner_pairwise_weight = float(
            getattr(cfg, "winner_pairwise_weight", 0.0)
        )
        self.winner_pairwise_by_seq_bucket = bool(
            getattr(cfg, "winner_pairwise_by_seq_bucket", True)
        )
        self.winner_pairwise_max_pairs = int(
            getattr(cfg, "winner_pairwise_max_pairs", 4096)
        )

    def forward(self, a_logits, p_logits, w_logit, a_target, p_target, w_target,
                point_mask=None, winner_mask=None, cat_logits=None, cat_target=None,
                seq_lens=None):
        """
        Args:
            point_mask : (batch,) bool, True = 有效的 pointId 目標
                         None = 全部參與計算（向後相容）
            winner_mask: (batch,) bool, True = 有效的 serverGetPoint 目標
                         None = 全部參與計算（向後相容）。
                         test_k row 因 winner 未知而 mask=False，跳過 winner loss。
            cat_logits / cat_target: 可選的 4(+1)-class action category aux head
        """
        la = self.action_loss(a_logits, a_target)
        lp = self.point_loss(p_logits, p_target, mask=point_mask)
        # v33 (Tier A2): winner label smoothing — target 0/1 → eps/(1-eps)
        # 避免 winner head overconfident，改善 ranking quality (AUC)。
        if self.winner_label_smoothing > 0:
            eps = self.winner_label_smoothing
            w_target_smooth = w_target * (1.0 - 2.0 * eps) + eps
        else:
            w_target_smooth = w_target
        # v31.1: 對 winner 套 mask
        lw_per = self.winner_loss(w_logit, w_target_smooth)  # (batch,) 沒 reduce
        if winner_mask is not None:
            if winner_mask.any():
                lw = lw_per[winner_mask].mean()
            else:
                # 整 batch 都沒有有效 winner 標籤 → 0
                lw = w_logit.sum() * 0.0
        else:
            lw = lw_per.mean()
        total = self.w_action * la + self.w_point * lp + self.w_winner * lw

        details = {
            "action": la.item(),
            "point": lp.item(),
            "winner": lw.item(),
        }

        if self.winner_pairwise_weight > 0:
            lr = self._winner_pairwise_auc_loss(
                w_logit, w_target, winner_mask=winner_mask, seq_lens=seq_lens
            )
            total = total + self.w_winner * self.winner_pairwise_weight * lr
            details["winner_rank"] = lr.item()

        if cat_logits is not None and cat_target is not None and self.w_action_category > 0:
            lc = self.category_loss(cat_logits, cat_target)
            total = total + self.w_action_category * lc
            details["category"] = lc.item()

        details["total"] = total.item()
        return total, details

    def _winner_pairwise_auc_loss(self, logits, targets, winner_mask=None, seq_lens=None):
        """
        RankNet-style pairwise loss for serverGetPoint.

        LB uses AUC, so BCE calibration alone is not enough. This term optimizes
        ordering directly: positive samples should have larger logits than
        negative samples. When seq_lens is provided, pairs are built inside
        coarse prefix-length buckets so the model cannot win by only learning
        "long rally vs short rally" shortcuts.
        """
        valid = torch.isfinite(logits) & ((targets == 0) | (targets == 1))
        if winner_mask is not None:
            valid = valid & winner_mask.bool()

        logits = logits[valid]
        targets = targets[valid]
        if seq_lens is not None:
            seq_lens = seq_lens[valid]

        def _loss_for_subset(sub_logits, sub_targets):
            pos = sub_logits[sub_targets > 0.5]
            neg = sub_logits[sub_targets <= 0.5]
            if pos.numel() == 0 or neg.numel() == 0:
                return None
            diffs = neg.view(1, -1) - pos.view(-1, 1)
            if diffs.numel() > self.winner_pairwise_max_pairs:
                flat = diffs.reshape(-1)
                idx = torch.randint(
                    flat.numel(), (self.winner_pairwise_max_pairs,),
                    device=flat.device,
                )
                diffs = flat[idx]
            return F.softplus(diffs).mean()

        losses = []
        if self.winner_pairwise_by_seq_bucket and seq_lens is not None:
            # Buckets mirror the main test_new pain point: 1拍, 2拍, 3-4拍, 5+拍.
            buckets = torch.where(
                seq_lens <= 1, 0,
                torch.where(seq_lens == 2, 1, torch.where(seq_lens <= 4, 2, 3)),
            )
            for b in range(4):
                m = buckets == b
                if m.any():
                    loss_b = _loss_for_subset(logits[m], targets[m])
                    if loss_b is not None:
                        losses.append(loss_b)

        if not losses:
            loss_all = _loss_for_subset(logits, targets)
            if loss_all is not None:
                losses.append(loss_all)

        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()
