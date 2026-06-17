"""
train.py - 訓練主程式
包含：train_one_epoch, evaluate, K-Fold 交叉驗證, 模型儲存

用法：
    python train.py
"""

import copy
import json
import os
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

from config import Config
from model import MultiTaskTransformer, ShuttleNetModel
from dataset import RallyDataset, collate_fn
from utils import (
    set_seed, add_engineered_features, infer_dominant_hand,
    EMA, CosineWarmupScheduler, MultiTaskLoss,
    FingerprintTable, compute_match_player_fingerprints, build_rally_fingerprints,
)

warnings.filterwarnings("ignore")


# ============================================================
# 單 epoch 訓練
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, ema,
                    device, scaler=None, max_grad_norm=1.0):
    model.train()
    losses = defaultdict(float)
    n = 0

    for batch in loader:
        feat = batch["features"].to(device)
        sl = batch["seq_lens"].to(device)
        ta = batch["target_action"].to(device)
        tac = batch["target_action_category"].to(device)
        tp = batch["target_point"].to(device)
        tw = batch["target_winner"].to(device)
        pm = batch["point_mask"].to(device)
        wm = batch["winner_mask"].to(device)
        fp = batch["fingerprint"].to(device) if "fingerprint" in batch else None

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                al, pl, wl, cl = model(feat, sl, fingerprint=fp)
                loss, details = criterion(
                    al, pl, wl, ta, tp, tw,
                    point_mask=pm, winner_mask=wm,
                    cat_logits=cl, cat_target=tac,
                    seq_lens=sl,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            al, pl, wl, cl = model(feat, sl, fingerprint=fp)
            loss, details = criterion(
                al, pl, wl, ta, tp, tw,
                point_mask=pm, winner_mask=wm,
                cat_logits=cl, cat_target=tac,
                seq_lens=sl,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        ema.update(model)

        for k, v in details.items():
            losses[k] += v
        n += 1

    return {k: v / n for k, v in losses.items()}


# ============================================================
# 驗證
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device, metric_seq_len_counts=None):
    model.eval()
    losses = defaultdict(float)
    n = 0

    all_ap, all_at = [], []
    all_pp, all_pt = [], []
    all_pm = []
    all_wp, all_wt = [], []
    all_wm = []  # v31.1: winner_mask 收集，AUC 計算時 filter 掉 test_k row
    all_sl = []

    for batch in loader:
        feat = batch["features"].to(device)
        sl = batch["seq_lens"].to(device)
        ta = batch["target_action"].to(device)
        tac = batch["target_action_category"].to(device)
        tp = batch["target_point"].to(device)
        tw = batch["target_winner"].to(device)
        pm = batch["point_mask"].to(device)
        wm = batch["winner_mask"].to(device)
        fp = batch["fingerprint"].to(device) if "fingerprint" in batch else None

        al, pl, wl, cl = model(feat, sl, fingerprint=fp)
        _, details = criterion(
            al, pl, wl, ta, tp, tw,
            point_mask=pm, winner_mask=wm,
            cat_logits=cl, cat_target=tac,
            seq_lens=sl,
        )

        for k, v in details.items():
            losses[k] += v
        n += 1

        all_ap.append(al.argmax(1).cpu().numpy())
        all_at.append(ta.cpu().numpy())
        # 只評估 point_mask=True 的樣本的 pointId 指標
        pm_np = pm.cpu().numpy()
        all_pp.append(pl.argmax(1).cpu().numpy())
        all_pt.append(tp.cpu().numpy())
        all_pm.append(pm_np)
        all_wp.append(torch.sigmoid(wl).cpu().numpy())
        all_wt.append(tw.cpu().numpy())
        all_wm.append(wm.cpu().numpy())
        all_sl.append(sl.cpu().numpy())

    all_ap = np.concatenate(all_ap)
    all_at = np.concatenate(all_at)
    all_pp = np.concatenate(all_pp)
    all_pt = np.concatenate(all_pt)
    all_pm = np.concatenate(all_pm)
    all_wp = np.concatenate(all_wp)
    all_wt = np.concatenate(all_wt)
    all_wm = np.concatenate(all_wm).astype(bool)
    all_sl = np.concatenate(all_sl)

    sample_weight = None
    if metric_seq_len_counts is not None:
        sample_weight = _prefix_metric_weights(all_sl, metric_seq_len_counts)

    action_f1 = f1_score(all_at, all_ap, average="macro", sample_weight=sample_weight)

    # pointId F1: v20 起 point_mask 全 True，filter 變 no-op；保留以防之後再啟用。
    # 包含 class 0 (rally-end) → 與 LB macro-F1 評分對齊（v19 之前不含 0 才有 0.08 gap）。
    if all_pm.any():
        point_weight = sample_weight[all_pm] if sample_weight is not None else None
        point_f1 = f1_score(
            all_pt[all_pm], all_pp[all_pm], average="macro",
            sample_weight=point_weight,
        )
    else:
        point_f1 = 0.0
    # v31.1: winner AUC 只算 winner_mask=True 的 row（test_k 的 winner=-1 sentinel 要 filter）
    try:
        if all_wm.any():
            winner_weight = sample_weight[all_wm] if sample_weight is not None else None
            winner_auc = roc_auc_score(
                all_wt[all_wm], all_wp[all_wm],
                sample_weight=winner_weight,
            )
        else:
            winner_auc = 0.5
    except ValueError:
        winner_auc = 0.5

    overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * winner_auc

    return {
        "action_f1": action_f1,
        "point_f1": point_f1,
        "winner_auc": winner_auc,
        "overall": overall,
        **{k: v / n for k, v in losses.items()},
    }


def _prefix_metric_weights(seq_lens, target_counts):
    """Weight validation prefixes to match the real test prefix-length distribution."""
    seq_lens = np.asarray(seq_lens, dtype=np.int64)
    val_counts = Counter(seq_lens.tolist())
    total_val = max(sum(val_counts.values()), 1)
    total_target = max(sum(target_counts.values()), 1)
    out = np.ones(len(seq_lens), dtype=np.float64)
    for i, k in enumerate(seq_lens):
        p_target = target_counts.get(int(k), 0) / total_target
        p_val = val_counts.get(int(k), 0) / total_val
        out[i] = (p_target / p_val) if (p_target > 0 and p_val > 0) else 1e-6
    return out


# ============================================================
# 主程式
# ============================================================

def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.model_dir, exist_ok=True)
    print(f"Device: {device}")

    # ---- 載入資料 ----
    print("\n=== Loading Data ===")
    train_df = pd.read_csv(cfg.train_path)
    test_df_for_hand = pd.read_csv(cfg.test_path)
    prefix_path = getattr(cfg, "inference_test_path", None) or cfg.test_path
    test_df_for_prefix = pd.read_csv(prefix_path)
    print(f"  Base train: {len(train_df)} rows, "
          f"{train_df['rally_uid'].nunique()} rallies, "
          f"{train_df['match'].nunique()} matches")

    # 合併額外訓練資料 (e.g. processed_train_e.csv)。
    # 這些檔案的 match / rally_uid 會與 train.csv 衝突，所以先加 offset 再合併。
    # v23: gamePlayerId / gamePlayerOtherId 直接全部設成 0（"unseen" 標記），
    #   不做 ID offset。理由：
    #   (1) processed_train_e 的 player ID 是獨立 ID 系統 (sex agreement 47% = random)，
    #       即使 offset 了也仍是「全新但未對齊任何 fingerprint 行為」的 noise IDs
    #   (2) 把它們當 unseen → 訓練「player_id=0 fallback 分支」更多樣本，模擬 test
    #       的 unseen player rows
    #   (3) match-level fingerprint 仍正常運作（per-match aggregate over all id=0 rows）
    for extra_path in getattr(cfg, "extra_train_paths", []) or []:
        extra_df = pd.read_csv(extra_path)
        match_off = int(max(train_df["match"].max(),
                            test_df_for_hand["match"].max())) + 1
        uid_off = int(max(train_df["rally_uid"].max(),
                          test_df_for_hand["rally_uid"].max())) + 1
        extra_df["match"] = extra_df["match"] + match_off
        extra_df["rally_uid"] = extra_df["rally_uid"] + uid_off
        # v23: 強制 player IDs = 0 → 全部當 unseen
        extra_df["gamePlayerId"] = 0
        extra_df["gamePlayerOtherId"] = 0
        print(f"  + Extra: {extra_path}  ({len(extra_df)} rows, "
              f"{extra_df['rally_uid'].nunique()} rallies, "
              f"match+={match_off}, uid+={uid_off}, "
              f"player IDs forced to 0 [unseen])")
        train_df = pd.concat([train_df, extra_df], ignore_index=True)

    # 從 train+test 合併推斷每場比賽中每位選手的慣用手
    hand_map = infer_dominant_hand(pd.concat([train_df, test_df_for_hand], ignore_index=True))
    print(f"  Inferred dominant hand for {len(hand_map)} (match, player) pairs")

    train_df = add_engineered_features(train_df, hand_map=hand_map)
    print(f"Train: {train_df.shape}, {train_df['rally_uid'].nunique()} rallies")

    # ---- 紀錄訓練時看過的 player ID（給 inference.py 用來 mask unseen 選手） ----
    # test.csv 約 36.5% 選手是 train.csv 沒看過的；訓練時 player_id_dropout
    # 隨機把整 rally 的 player ID mask 成 0，inference 時把 unseen 也設為 0
    # 即可匹配訓練時看過的分布。
    known_pids = sorted(
        set(int(p) for p in train_df["gamePlayerId"].unique() if int(p) > 0) |
        set(int(p) for p in train_df["gamePlayerOtherId"].unique() if int(p) > 0)
    )
    os.makedirs(os.path.dirname(cfg.known_player_ids_path), exist_ok=True)
    with open(cfg.known_player_ids_path, "w") as f:
        json.dump({"known_player_ids": known_pids}, f)
    print(f"  Saved {len(known_pids)} known player IDs → {cfg.known_player_ids_path}")

    # 計算真實推論測試集的 prefix 長度分佈（訓練 sampler + validation metric 用）
    test_seq_len_counts = Counter(
        len(g) for _, g in test_df_for_prefix.groupby("rally_uid")
    )
    print(f"  Prefix distribution source: {prefix_path} "
          f"({sum(test_seq_len_counts.values())} rows, "
          f"{len(test_seq_len_counts)} distinct lengths)")

    # 按 rally 分組
    train_rally_dict = {uid: g for uid, g in train_df.groupby("rally_uid")}
    rally_uids = list(train_rally_dict.keys())

    # ---- Phase B: Match-Level Player Fingerprint ----
    # 計算每個 (match, gamePlayerId) 的「打法指紋」並對每個 rally 用 LOO 扣除自己的貢獻。
    # 必須在 K-Fold split 之前算（fp_table 用整 train_df 算 global prior 與 sums），
    # GroupKFold(groups=match) 保證 val 的 match 不會出現在 train，鏡射 test 場景。
    fp_dict = None
    if getattr(cfg, "use_match_fingerprint", False):
        print("\n=== Match-Level Player Fingerprint ===")
        expected_fp_dim = FingerprintTable.total_dim()
        if int(cfg.fingerprint_dim) != expected_fp_dim:
            raise ValueError(
                f"cfg.fingerprint_dim ({cfg.fingerprint_dim}) != "
                f"FingerprintTable.total_dim() ({expected_fp_dim}). "
                f"Update Config to match SPECS."
            )
        fp_table = compute_match_player_fingerprints(
            train_df, alpha=cfg.fingerprint_alpha
        )
        fp_dict = build_rally_fingerprints(train_rally_dict, fp_table)
        print(f"  fp_dim per side = {expected_fp_dim} (× 2 sides = {2*expected_fp_dim})")
        print(f"  alpha = {cfg.fingerprint_alpha}, "
              f"built fingerprints for {len(fp_dict)} rallies")

    # ---- K-Fold 交叉驗證 ----
    # train.csv 的 match 與 test.csv 完全不重疊 (驗證為 0)，所以用 GroupKFold(groups=match)
    # 來鏡射 test 場景：val fold 的選手對訓練模型來說是 unseen，但 fingerprint 仍可從
    # 同 match 內其他 rallies 算出，模擬 inference 時面對 unseen player 的情況。
    use_winner_sup = getattr(cfg, "use_winner_supervision", True)
    rally_match = {
        uid: int(train_rally_dict[uid].iloc[0]["match"]) for uid in rally_uids
    }
    groups = np.array([rally_match[uid] for uid in rally_uids])
    if getattr(cfg, "use_match_fingerprint", False):
        kf = GroupKFold(n_splits=cfg.n_folds)
        kf_split = kf.split(rally_uids, groups=groups)
        print("  K-Fold: GroupKFold(groups=match) — mirrors train→test match isolation")
    elif use_winner_sup:
        rally_labels = [
            int(train_rally_dict[uid].iloc[0]["serverGetPoint"]) for uid in rally_uids
        ]
        kf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        kf_split = kf.split(rally_uids, rally_labels)
        print("  K-Fold: StratifiedKFold (stratified on serverGetPoint)")
    else:
        kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
        kf_split = kf.split(rally_uids)
        print("  K-Fold: plain KFold (winner supervision disabled per official guidance)")
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf_split):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{cfg.n_folds}")
        print(f"{'='*60}")

        # v38: 續訓支援 — 若 checkpoint 已存在則跳過該 fold
        # （PBS queue walltime 12hr 限制，v38 多訓練要分多次跑完）。
        # 想完全重訓 = 把 ./checkpoints/ 內舊檔刪掉即可。
        save_path = os.path.join(cfg.model_dir, f"fold{fold}.pt")
        if os.path.exists(save_path):
            print(f"  [resume] {save_path} 已存在，跳過本 fold")
            fold_scores.append(None)
            continue

        tr_dict = {rally_uids[i]: train_rally_dict[rally_uids[i]] for i in train_idx}
        va_dict = {rally_uids[i]: train_rally_dict[rally_uids[i]] for i in val_idx}

        tr_ds = RallyDataset(
            tr_dict, cfg, mode="train",
            use_sliding_window=cfg.use_sliding_window,
            fingerprint_dict=fp_dict,
        )

        # v36: validation 全部改用 sliding-window prefix，並在 evaluate() 內按
        # test_new prefix 長度分佈加權。舊的 nosw 近完整回合驗證會高估 winner AUC，
        # 因為 test_new 有大量 1~2 拍 prefix。
        va_ds_sw = RallyDataset(
            va_dict, cfg, mode="train", use_sliding_window=True,
            fingerprint_dict=fp_dict,
        )
        print(f"  Train samples: {len(tr_ds)}, Val(prefix-sw): {len(va_ds_sw)}")

        # WeightedRandomSampler：讓每個 epoch 的輸入長度分佈貼近測試集
        # v37 任務 1：可選 class-aware re-sampling 疊加在 length-IPW 之上。
        cb_power_cfg = (
            float(getattr(cfg, "class_balance_power", 0.0))
            if bool(getattr(cfg, "use_class_aware_sampling", False)) else 0.0
        )
        cb_target = str(getattr(cfg, "class_balance_target", "action"))
        sample_weights = tr_ds.get_sample_weights(
            test_seq_len_counts,
            class_balance_power=cb_power_cfg,
            class_balance_target=cb_target,
        )
        if fold == 0 and cb_power_cfg > 0.0:
            # diagnostic：印 top-5 被加重的 class（看是否真的有把稀有類拉高）
            from collections import defaultdict
            target_key = (
                "target_point" if cb_target == "point" else "target_action"
            )
            cls_w_acc = defaultdict(list)
            for s, w in zip(tr_ds.samples, sample_weights):
                cls_w_acc[s[target_key]].append(w)
            cls_mean = sorted(
                ((c, float(np.mean(ws))) for c, ws in cls_w_acc.items()),
                key=lambda kv: -kv[1],
            )[:5]
            print(
                f"  Class-aware sampling: power={cb_power_cfg}, target={cb_target}, "
                f"top-5 boosted (cls, mean_w): {cls_mean}"
            )
        tr_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(tr_ds),
            replacement=True,
        )
        tr_dl = DataLoader(
            tr_ds, batch_size=cfg.batch_size, sampler=tr_sampler,
            collate_fn=collate_fn, num_workers=0, pin_memory=True, drop_last=True,
        )
        va_dl_sw = DataLoader(
            va_ds_sw, batch_size=cfg.batch_size * 2, shuffle=False,
            collate_fn=collate_fn, num_workers=0, pin_memory=True,
        )

        # 模型 & 優化器
        model_type = getattr(cfg, "model_type", "transformer")
        if model_type == "shuttlenet":
            model = ShuttleNetModel(cfg).to(device)
        else:
            model = MultiTaskTransformer(cfg).to(device)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Model: {model_type}  |  Parameters: {n_params:,}")

        criterion = MultiTaskLoss(cfg)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

        total_steps = cfg.epochs * len(tr_dl)
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

        ema = EMA(model, decay=cfg.ema_decay)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        best_score = -1
        best_state = None
        no_improve = 0

        for epoch in range(cfg.epochs):
            train_loss = train_one_epoch(
                model, tr_dl, criterion, optimizer, scheduler, ema,
                device, scaler, cfg.max_grad_norm,
            )

            # 用 EMA 參數做驗證
            ema.apply_shadow(model)
            val_sw = evaluate(
                model, va_dl_sw, criterion, device,
                metric_seq_len_counts=(
                    test_seq_len_counts
                    if getattr(cfg, "use_prefix_weighted_validation", True)
                    else None
                ),
            )
            ema.restore(model)

            # v36: action / point / winner 都來自同一個 test-prefix-weighted 世界。
            action_f1 = val_sw["action_f1"]
            point_f1 = val_sw["point_f1"]
            winner_auc = val_sw["winner_auc"]
            # 若停掉 winner 監督，overall 只看 action+point（避免被未訓練的 winner_auc 誤導）
            if use_winner_sup:
                overall = 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * winner_auc
            else:
                overall = 0.5 * action_f1 + 0.5 * point_f1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  Ep {epoch+1:3d} | "
                    f"TrLoss {train_loss['total']:.4f} | "
                    f"VaLoss {val_sw['total']:.4f} | "
                    f"ActF1 {action_f1:.4f} | "
                    f"PtF1 {point_f1:.4f} | "
                    f"WinAUC {winner_auc:.4f} | "
                    f"Score {overall:.4f}"
                )

            if overall > best_score:
                best_score = overall
                ema.apply_shadow(model)
                best_state = copy.deepcopy(model.state_dict())
                ema.restore(model)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= cfg.patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

        # 儲存最佳模型（save_path 在 fold 開頭已定義）
        torch.save(best_state, save_path)
        print(f"  Best Score: {best_score:.4f} -> saved to {save_path}")
        fold_scores.append(best_score)

    # ---- 摘要 ----
    # v38 續訓：fold_scores 內 None 表示該 fold 是 resume 跳過的
    print(f"\n{'='*60}")
    print("CV Summary")
    print(f"{'='*60}")
    for i, s in enumerate(fold_scores):
        if s is None:
            print(f"  Fold {i+1}: (resume 跳過，使用既有 checkpoint)")
        else:
            print(f"  Fold {i+1}: {s:.4f}")
    new_scores = [s for s in fold_scores if s is not None]
    if new_scores:
        print(f"  Mean (本次新訓練 {len(new_scores)} fold): "
              f"{np.mean(new_scores):.4f} ± {np.std(new_scores):.4f}")
    else:
        print("  本次未新訓練任何 fold（全部 resume）")
    print(f"\nModels saved in {cfg.model_dir}/")
    print("Run `python inference.py` to generate submission.")


if __name__ == "__main__":
    main()
