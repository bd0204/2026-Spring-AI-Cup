"""
inference.py - 推論與提交檔案生成
載入所有 fold 的模型，做 ensemble 預測，輸出 submission.csv

用法：
    python inference.py
"""

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from model import MultiTaskTransformer, ShuttleNetModel
from dataset import RallyDataset, collate_fn
from utils import (
    add_engineered_features, infer_dominant_hand, mask_unseen_player_ids,
    FingerprintTable, compute_match_player_fingerprints, build_rally_fingerprints,
)

warnings.filterwarnings("ignore")


@torch.no_grad()
def predict(model, loader, device):
    """對整個 DataLoader 進行推論"""
    model.eval()
    uids = []
    a_logits_all, p_logits_all, w_probs_all = [], [], []

    for batch in loader:
        feat = batch["features"].to(device)
        sl = batch["seq_lens"].to(device)
        fp = batch["fingerprint"].to(device) if "fingerprint" in batch else None

        al, pl, wl, _ = model(feat, sl, fingerprint=fp)

        uids.extend(batch["rally_uids"])
        a_logits_all.append(al.cpu().numpy())
        p_logits_all.append(pl.cpu().numpy())
        w_probs_all.append(torch.sigmoid(wl).cpu().numpy())

    return (
        uids,
        np.concatenate(a_logits_all),
        np.concatenate(p_logits_all),
        np.concatenate(w_probs_all),
    )


def ensemble_predict(cfg, test_dl, device):
    """
    載入所有 fold 模型，做 ensemble 預測（logits/probs 平均）。
    v24 fix (H1): 支援多目錄 ensemble — 用 cfg.extra_model_dirs 增加更多 fold 來源
    （例如 ./pt/v23_pt + ./pt/v24_pt 一起 = 10 models），平掉 fold split variance。
    """
    ensemble_a, ensemble_p, ensemble_w = None, None, None
    uids = None
    n_models = 0

    # v25: inference 權重來源優先序：
    #   (1) 如果 cfg.inference_dirs 非空 → 完全 override，只用這個 list
    #       （適合「v23-only 提交」或「v23+v25 ensemble」這種要明確指定的情況）
    #   (2) 否則 fallback 到 cfg.model_dir + cfg.extra_model_dirs
    inf_dirs = list(getattr(cfg, "inference_dirs", []) or [])
    if inf_dirs:
        model_dirs = inf_dirs
    else:
        model_dirs = [cfg.model_dir] + list(getattr(cfg, "extra_model_dirs", []) or [])

    for model_dir in model_dirs:
        for fold in range(cfg.n_folds):
            model_path = os.path.join(model_dir, f"fold{fold}.pt")
            if not os.path.exists(model_path):
                print(f"  [WARN] {model_path} not found, skipping.")
                continue

            model_type = getattr(cfg, "model_type", "transformer")
            if model_type == "shuttlenet":
                model = ShuttleNetModel(cfg).to(device)
            else:
                model = MultiTaskTransformer(cfg).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"  Loaded fold {fold} from {model_path}")

            uids, a_logits, p_logits, w_probs = predict(model, test_dl, device)

            if ensemble_a is None:
                ensemble_a = a_logits
                ensemble_p = p_logits
                ensemble_w = w_probs
            else:
                ensemble_a += a_logits
                ensemble_p += p_logits
                ensemble_w += w_probs

            n_models += 1

    if n_models == 0:
        raise FileNotFoundError(
            f"No model checkpoints found in {model_dirs}. Run `python train.py` first."
        )

    # 平均
    ensemble_a /= n_models
    ensemble_p /= n_models
    ensemble_w /= n_models

    print(f"  Ensemble of {n_models} models from {len(model_dirs)} dir(s)")
    return uids, ensemble_a, ensemble_p, ensemble_w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nn-only",
        action="store_true",
        help="Disable LGBM/stacking/rank-conditional blending and export pure NN predictions.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override cfg.output_path, e.g. submission_v36_nnonly.csv.",
    )
    parser.add_argument(
        "--lgbm-proba",
        default=None,
        help="Override the default LGBM proba .npz path for NN+LGBM ensemble inference.",
    )
    parser.add_argument(
        "--no-rank-conditional",
        action="store_true",
        help="Disable winner rank-conditional blending for this run.",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Disable winner stacking meta-learner for this run.",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.output:
        cfg.output_path = args.output
    if args.nn_only:
        cfg.lgbm_proba_path = ""
        cfg.winner_meta_path = ""
        cfg.winner_rank_conditional = False
    elif args.lgbm_proba is not None:
        cfg.lgbm_proba_path = args.lgbm_proba
    if args.no_rank_conditional:
        cfg.winner_rank_conditional = False
    if args.no_meta:
        cfg.winner_meta_path = ""

    device = torch.device(cfg.device)
    print(f"Device: {device}")

    # ---- 載入測試資料 ----
    # v25: 用 cfg.inference_test_path（如果有設）作為預測目標，否則 fallback 到
    # cfg.test_path。這樣 train.py 跟 inference.py 可以用不同 test 檔（train 對齊
    # ./test.csv 走 v23 best alignment，inference 預測 ./test_new.csv）。
    print("\n=== Loading Test Data ===")
    train_df = pd.read_csv(cfg.train_path)
    test_path = getattr(cfg, "inference_test_path", None) or cfg.test_path
    test_df = pd.read_csv(test_path)
    print(f"  test file: {test_path}")

    # 從 train+test 推斷慣用手（與 train.py 一致）
    hand_map = infer_dominant_hand(pd.concat([train_df, test_df], ignore_index=True))
    print(f"  Inferred dominant hand for {len(hand_map)} (match, player) pairs")

    test_df = add_engineered_features(test_df, hand_map=hand_map)
    print(f"Test: {test_df.shape}, {test_df['rally_uid'].nunique()} rallies")

    # ---- Phase B: Match-level player fingerprint（必須在 mask_unseen_player_ids 之前算） ----
    # 即使選手 ID 在 train 沒看過，他在 test 同 match 內仍打了多個 rallies，可從那邊
    # 算出「打法指紋」。fp_table 與 fp_dict 都在 raw（未 mask 的）test_df 上計算。
    fp_dict = None
    if getattr(cfg, "use_match_fingerprint", False):
        print("\n=== Match-Level Player Fingerprint ===")
        expected_fp_dim = FingerprintTable.total_dim()
        if int(cfg.fingerprint_dim) != expected_fp_dim:
            raise ValueError(
                f"cfg.fingerprint_dim ({cfg.fingerprint_dim}) != "
                f"FingerprintTable.total_dim() ({expected_fp_dim})."
            )
        fp_table_test = compute_match_player_fingerprints(
            test_df, alpha=cfg.fingerprint_alpha
        )
        # 重要：用 raw IDs 建 rally_dict 給 fingerprint lookup 用
        test_rally_dict_raw = {uid: g for uid, g in test_df.groupby("rally_uid")}
        fp_dict = build_rally_fingerprints(test_rally_dict_raw, fp_table_test)
        print(f"  fp_dim per side = {expected_fp_dim} (× 2 sides = {2*expected_fp_dim})")
        print(f"  alpha = {cfg.fingerprint_alpha}, "
              f"built fingerprints for {len(fp_dict)} rallies")

    # ---- 把 test.csv 中沒在 train.csv 出現過的選手 ID 替換成 0 (unknown) ----
    # 對應訓練時 player_id_dropout 看過的「沒有 player ID」分布，避免模型對
    # unseen 選手 ID 隨機產生 embedding 噪訊。
    if os.path.exists(cfg.known_player_ids_path):
        with open(cfg.known_player_ids_path) as f:
            known = json.load(f)["known_player_ids"]
        n_self_unseen = (~test_df["gamePlayerId"].isin(known) &
                         (test_df["gamePlayerId"] != 0)).sum()
        n_other_unseen = (~test_df["gamePlayerOtherId"].isin(known) &
                          (test_df["gamePlayerOtherId"] != 0)).sum()
        test_df = mask_unseen_player_ids(test_df, known)
        print(f"  Masked unseen player IDs: gamePlayerId={int(n_self_unseen)}, "
              f"gamePlayerOtherId={int(n_other_unseen)} rows → 0")
    else:
        print(f"  [WARN] {cfg.known_player_ids_path} not found — "
              f"unseen player IDs will go through random embedding lookup. "
              f"Run train.py first.")

    test_rally_dict = {uid: g for uid, g in test_df.groupby("rally_uid")}

    test_ds = RallyDataset(test_rally_dict, cfg, mode="test", fingerprint_dict=fp_dict)
    test_dl = DataLoader(
        test_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ---- NN Ensemble 預測 ----
    print("\n=== NN Ensemble Prediction ===")
    uids, a_logits, p_logits, w_probs = ensemble_predict(cfg, test_dl, device)

    # ---- 轉成 probabilities ----
    def _softmax_rows(x):
        x_max = x.max(axis=1, keepdims=True)
        e = np.exp(x - x_max)
        return e / e.sum(axis=1, keepdims=True)

    a_probs_nn = _softmax_rows(a_logits)   # (N, 19)
    p_probs_nn = _softmax_rows(p_logits)   # (N, 10)
    # w_probs 已經是 sigmoid 過的 (ensemble_predict 內處理)
    # v35 stacking: 保留原始 NN winner 機率（下面 hybrid blend 會覆寫 w_probs）
    w_probs_nn = w_probs.copy()

    # v41+: 存 raw NN proba（softmax 前的 prior-shift 都還沒套）供 ensemble_proba.py
    # 做多版本 probability-level 融合（variance reduction）。設 cfg.save_nn_proba_path
    # 才會存；不影響正常 submission 流程。每個版本（v38/v40/v41…）各跑一次存一個 npz。
    _save_proba = getattr(cfg, "save_nn_proba_path", "")
    if _save_proba:
        np.savez(
            _save_proba,
            rally_uid=np.asarray(uids),
            action=a_probs_nn,   # (N, 19) raw softmax
            point=p_probs_nn,    # (N, 10) raw softmax
            winner=w_probs_nn,   # (N,)    sigmoid 機率
        )
        print(f"  [ensemble] raw NN proba saved → {_save_proba}  "
              f"(action {a_probs_nn.shape}, point {p_probs_nn.shape}, winner {w_probs_nn.shape})")

    # 每個 test rally 的已知拍數（= 模型 condition 的序列長度），給條件式 meta-learner。
    # 與 nn_oof.py 的 n_strokes 定義一致（皆為「模型輸入序列長度」）。
    test_n_strokes = np.array(
        [len(test_rally_dict[int(u)]) for u in uids], dtype=np.int64
    )
    # v35 條件式 stacking：每個 test rally「雙方選手是否都在訓練看過」。
    # test_df 已過 mask_unseen_player_ids → unseen 選手 ID 已被設成 0。
    # NN winner 對「看過的選手」OOF AUC 0.86（真材實料），對 unseen（masked）只
    # ~0.73 → 只對「雙方都看過」的 rally 套 meta-learner，其餘退回純 LGBM。
    test_rally_seen = np.array([
        bool((test_rally_dict[int(u)]["gamePlayerId"] != 0).all() and
             (test_rally_dict[int(u)]["gamePlayerOtherId"] != 0).all())
        for u in uids
    ])

    # ---- 算 train prior for pointId（給 prior shift 用） ----
    # v20: 用 non-first-stroke prior (strikeNumber != 1) — 對齊 sliding-window target
    target_rows = train_df[train_df["strikeNumber"] != 1]
    train_pid_counts = target_rows["pointId"].value_counts().sort_index()
    train_prior = np.zeros(cfg.n_point_classes, dtype=np.float64)
    for c, n in train_pid_counts.items():
        if 0 <= int(c) < cfg.n_point_classes:
            train_prior[int(c)] = float(n)
    train_prior = train_prior / max(train_prior.sum(), 1.0)

    # ---- Helper: 機率空間的 prior shift ----
    # 公式（在 prob 空間）：adjusted_prob[c] ∝ prob[c] · (p_train[c] / p_pred[c])^alpha
    # 等同於 logit 空間的 logit + alpha·log(p_train/p_pred)，然後 softmax 重新 normalize
    pt_alpha = float(getattr(cfg, "pointid_prior_shift_alpha", 1.0))
    eps = 1e-9
    def _apply_prior_shift(p, alpha):
        if alpha <= 0:
            return p
        pred_prior = p.mean(axis=0)
        factor = ((train_prior + eps) / (pred_prior + eps)) ** alpha
        p_adj = p * factor
        p_adj /= p_adj.sum(axis=1, keepdims=True)
        return p_adj

    def _rank01(x):
        # 把 1-D 分數轉成 [0,1] 的 rank（0-indexed rank / (N-1)）。
        # AUC 只看排序，rank-average 直接在 AUC 目標空間做 ensemble，
        # 不受兩個模型 calibration scale 不一致影響。stable sort 確保可重現。
        order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
        n = len(x)
        return order.astype(np.float64) / max(n - 1, 1)

    def _logit(p, eps=1e-6):
        # winner 機率 → logit space（給 stacking meta-learner 用）。
        # 必須與 stack_winner.py 的 logit() 完全一致。
        p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    # ---- LGBM ensemble (跨 paradigm: Shuttlenet NN + LightGBM GBDT) ----
    lgbm_path = getattr(cfg, "lgbm_proba_path", None) or ""
    ensemble_active = bool(lgbm_path) and os.path.exists(lgbm_path)

    if ensemble_active:
        print(f"\n=== LGBM Ensemble Active ===")
        print(f"  LGBM proba file: {lgbm_path}")
        lgbm = np.load(lgbm_path)
        lgbm_uids = lgbm["rally_uid"].astype(int)
        lgbm_uid_to_idx = {int(u): i for i, u in enumerate(lgbm_uids)}

        # Align rally_uid order: 把 lgbm 的 row 重排成跟 uids 一致
        try:
            align_idx = np.array([lgbm_uid_to_idx[int(u)] for u in uids])
        except KeyError as e:
            raise ValueError(
                f"LGBM proba 缺少 rally_uid {e}（在 NN 預測裡有但 LGBM npz 沒有）"
            )
        a_probs_lgbm = lgbm["action"][align_idx]
        p_probs_lgbm = lgbm["point"][align_idx]
        w_probs_lgbm = lgbm["winner"][align_idx]
        print(f"  Aligned {len(align_idx)} rallies by rally_uid")

        # v29: hybrid_swap mode — 對每個任務各自選最強 model
        hybrid = bool(getattr(cfg, "hybrid_swap", False))

        if hybrid:
            # v29: per-task NN 權重。每個任務獨立加權平均 NN 與 LGBM 機率。
            # v29 (winner only) LB 0.4349 +0.0088 over LGBM alone — 確認平均對
            # uncorrelated errors 顯著有效。v29.2 試 action 也 50/50。
            w_a = float(getattr(cfg, "hybrid_action_w_nn", 0.0))
            w_p = float(getattr(cfg, "hybrid_point_w_nn",  0.0))
            w_w = float(getattr(cfg, "hybrid_winner_w_nn", 0.5))
            rank_avg = bool(getattr(cfg, "hybrid_winner_rank_avg", False))
            # v35 ablation: action/point 也支援 conditional 模式
            #   False → uniform 50/50（全 rally 都 blend）
            #   True  → conditional（seen → blend；unseen → 純 LGBM，跟 winner 同邏輯）
            #   注意：action/point 是 multi-class（19/10 類機率向量），兩邊已經是同刻度
            #   的 softmax 機率，不需要 rank-space 轉換 → 條件式 prob-blend 即可。
            a_cond = bool(getattr(cfg, "hybrid_action_conditional", False))
            p_cond = bool(getattr(cfg, "hybrid_point_conditional", False))
            print(f"  Mode: HYBRID SWAP (per-task NN weight)")
            # actionId
            a_blend = w_a * a_probs_nn + (1.0 - w_a) * a_probs_lgbm
            if a_cond:
                a_probs = np.where(test_rally_seen[:, None], a_blend, a_probs_lgbm)
                print(f"    actionId       : CONDITIONAL (seen → NN={w_a:.2f}+LGBM={1-w_a:.2f}; "
                      f"unseen → 純 LGBM)")
            else:
                a_probs = a_blend
                print(f"    actionId       : NN={w_a:.2f} + LGBM={1-w_a:.2f}")
            # pointId
            p_blend = w_p * p_probs_nn + (1.0 - w_p) * p_probs_lgbm
            if p_cond:
                p_probs = np.where(test_rally_seen[:, None], p_blend, p_probs_lgbm)
                print(f"    pointId        : CONDITIONAL (seen → NN={w_p:.2f}+LGBM={1-w_p:.2f}; "
                      f"unseen → 純 LGBM)")
            else:
                p_probs = p_blend
                print(f"    pointId        : NN={w_p:.2f} + LGBM={1-w_p:.2f}")
            # winner blend 優先序：
            #   1. rank-conditional（winner_rank_conditional=True）— v35 最終方案
            #   2. stacking meta-learner（winner_meta_path 的 .pkl 存在）
            #   3. rank-average（hybrid_winner_rank_avg=True）
            #   4. prob-average（傳統加權平均）
            rank_cond = bool(getattr(cfg, "winner_rank_conditional", False))
            meta_path = getattr(cfg, "winner_meta_path", "") or ""
            if rank_cond:
                # 全程在 rank space（兩邊都轉 [0,1] rank，刻度一致 → 無 scale mismatch）：
                #   seen rally  → w_w·rank(NN) + (1-w_w)·rank(LGBM)
                #   unseen rally→ 純 rank(LGBM)（維持朋友 0.5212 的排序，那組零下檔）
                nn_r   = _rank01(w_probs_nn)
                lgbm_r = _rank01(w_probs_lgbm)
                blended = w_w * nn_r + (1.0 - w_w) * lgbm_r
                w_probs = np.where(test_rally_seen, blended, lgbm_r)
                n_seen = int(test_rally_seen.sum())
                print(f"    serverGetPoint : RANK-CONDITIONAL — "
                      f"seen {n_seen}/{len(uids)} → {w_w:.2f}·rank(NN)+{1-w_w:.2f}·rank(LGBM)；"
                      f"unseen {len(uids)-n_seen} → 純 rank(LGBM)")
            elif meta_path and os.path.exists(meta_path):
                bundle = joblib.load(meta_path)
                meta_model, use_ctx = bundle["model"], bundle["use_context"]
                m_cols = [_logit(w_probs_nn), _logit(w_probs_lgbm)]
                if use_ctx:
                    ns = np.log1p(test_n_strokes.astype(np.float64))
                    m_cols.append(ns)
                    m_cols.append(_logit(w_probs_nn) * ns)
                meta_pred = meta_model.predict_proba(np.column_stack(m_cols))[:, 1]
                # 條件式套用：只有「雙方選手都看過」的 rally 才信 meta-learner
                # （NN 主導，對看過的選手強）；有 unseen 選手的 rally 退回純 LGBM。
                w_probs = np.where(test_rally_seen, meta_pred, w_probs_lgbm)
                n_seen = int(test_rally_seen.sum())
                print(f"    serverGetPoint : STACKING meta-learner（條件式）"
                      f"context={'on' if use_ctx else 'off'}, {meta_path}")
                print(f"      {n_seen}/{len(uids)} rallies 雙方選手皆 seen → meta；"
                      f"{len(uids)-n_seen} → 純 LGBM")
            elif rank_avg:
                w_probs = w_w * _rank01(w_probs_nn) + (1.0 - w_w) * _rank01(w_probs_lgbm)
                print(f"    serverGetPoint : RANK-AVG (NN={w_w:.2f} + LGBM={1-w_w:.2f})")
            else:
                w_probs = w_w * w_probs_nn + (1.0 - w_w) * w_probs_lgbm
                print(f"    serverGetPoint : PROB-AVG (NN={w_w:.2f} + LGBM={1-w_w:.2f})")

            # Diagnostic: 顯示 LGBM 跟 NN 的 argmax 差異（純資訊性）
            nn_act = a_probs_nn.argmax(axis=1)
            nn_pt = _apply_prior_shift(p_probs_nn, pt_alpha).argmax(axis=1)
            lgbm_act = a_probs_lgbm.argmax(axis=1)
            lgbm_pt = p_probs_lgbm.argmax(axis=1)
            print(f"  Diff (LGBM argmax vs NN argmax-with-shift):")
            print(f"    actionId: {(nn_act != lgbm_act).sum()} / {len(uids)} ({(nn_act != lgbm_act).mean()*100:.1f}%)")
            print(f"    pointId : {(nn_pt != lgbm_pt).sum()} / {len(uids)} ({(nn_pt != lgbm_pt).mean()*100:.1f}%)")
        else:
            # v26 averaging mode — 雙邊都 shift + 單一 w_nn
            p_probs_nn_adj = _apply_prior_shift(p_probs_nn, pt_alpha)
            p_probs_lgbm_adj = _apply_prior_shift(p_probs_lgbm, pt_alpha)

            w_nn = float(getattr(cfg, "ensemble_nn_weight", 0.4))
            print(f"  Mode: AVERAGING (NN={w_nn:.2f}, LGBM={1-w_nn:.2f})")

            a_probs = w_nn * a_probs_nn       + (1.0 - w_nn) * a_probs_lgbm
            p_probs = w_nn * p_probs_nn_adj   + (1.0 - w_nn) * p_probs_lgbm_adj
            w_probs = w_nn * w_probs          + (1.0 - w_nn) * w_probs_lgbm

            nn_act = a_probs_nn.argmax(axis=1)
            nn_pt = p_probs_nn_adj.argmax(axis=1)
            lgbm_act = a_probs_lgbm.argmax(axis=1)
            lgbm_pt = p_probs_lgbm_adj.argmax(axis=1)
            en_act = a_probs.argmax(axis=1)
            en_pt = p_probs.argmax(axis=1)
            print(f"  argmax flipped from NN-only → ensemble:")
            print(f"    actionId: {(nn_act != en_act).sum()} / {len(uids)} ({(nn_act != en_act).mean()*100:.1f}%)")
            print(f"    pointId : {(nn_pt != en_pt).sum()} / {len(uids)} ({(nn_pt != en_pt).mean()*100:.1f}%)")
            print(f"  argmax flipped from LGBM-only → ensemble:")
            print(f"    actionId: {(lgbm_act != en_act).sum()} / {len(uids)} ({(lgbm_act != en_act).mean()*100:.1f}%)")
            print(f"    pointId : {(lgbm_pt != en_pt).sum()} / {len(uids)} ({(lgbm_pt != en_pt).mean()*100:.1f}%)")
    else:
        if lgbm_path:
            print(f"\n[WARN] LGBM proba file not found: {lgbm_path} — running NN-only.")
        # NN only — 對 pointId 套 prior shift
        a_probs = a_probs_nn
        p_probs = _apply_prior_shift(p_probs_nn, pt_alpha)

    # ---- pointId prior shift diagnostic table ----
    if pt_alpha > 0:
        from collections import Counter
        pt_argmax_final = p_probs.argmax(axis=1)
        cnt_final = Counter(pt_argmax_final.tolist())
        pred_prior_final = p_probs.mean(axis=0)
        print(f"\n=== Final pointId Distribution (alpha={pt_alpha}) ===")
        print(f"  {'class':>6} {'train_prior':>12} {'final_prior':>12} {'argmax':>8}")
        for c in range(cfg.n_point_classes):
            print(f"  {c:>6} {train_prior[c]:>12.4f} {pred_prior_final[c]:>12.4f} "
                  f"{cnt_final.get(c, 0):>8}")

    final_point = p_probs.argmax(axis=1)

    # v37 任務 3：actionId prior shift（沿用 pointId 同公式）。
    # 注意：原本 final_action = a_probs.argmax(axis=1) 直接取 hybrid blend 後的
    # a_probs；為了不破壞 v35 hybrid ensemble 的 +0.0050 增益，prior shift 套用在
    # 已 blend 的 a_probs 上（prob 空間 multiplicative shift，等價於 logit 空間
    # additive shift 後 re-softmax）。alpha = 0 退回原本「直接 a_probs.argmax」。
    # train prior 排除 strikeNumber == 1（serve 拍），對齊「預測下一拍 action」的
    # target 分布。
    at_alpha = float(getattr(cfg, "action_prior_shift_alpha", 0.0))
    if at_alpha > 0:
        from collections import Counter as _CntA
        target_rows_a = train_df[train_df["strikeNumber"] != 1]
        train_act_counts = target_rows_a["actionId"].value_counts().sort_index()
        train_prior_a = np.zeros(cfg.n_action_classes, dtype=np.float64)
        for c, n in train_act_counts.items():
            if 0 <= int(c) < cfg.n_action_classes:
                train_prior_a[int(c)] = float(n)
        train_prior_a = train_prior_a / max(train_prior_a.sum(), 1.0)

        eps_a = 1e-9
        pred_prior_a = a_probs.mean(axis=0)
        factor_a = ((train_prior_a + eps_a) / (pred_prior_a + eps_a)) ** at_alpha
        a_probs_adj = a_probs * factor_a
        a_probs_adj /= a_probs_adj.sum(axis=1, keepdims=True)
        log_shift_a = at_alpha * (
            np.log(train_prior_a + eps_a) - np.log(pred_prior_a + eps_a)
        )

        cnt_orig_a = _CntA(a_probs.argmax(axis=1).tolist())
        final_action = a_probs_adj.argmax(axis=1)
        cnt_adj_a = _CntA(final_action.tolist())

        print(f"\n=== Final actionId Distribution (alpha={at_alpha}) ===")
        print(
            f"  {'cls':>4} {'train_prior':>12} {'pred_prior':>12} "
            f"{'log_shift':>10} {'argmax(orig)':>14} {'argmax(adj)':>14}"
        )
        for c in range(cfg.n_action_classes):
            print(
                f"  {c:>4} {train_prior_a[c]:>12.4f} {pred_prior_a[c]:>12.4f} "
                f"{log_shift_a[c]:>10.4f} {cnt_orig_a.get(c, 0):>14} "
                f"{cnt_adj_a.get(c, 0):>14}"
            )
    else:
        final_action = a_probs.argmax(axis=1)
    # ⚠️ serverGetPoint 改成輸出 probability（float）而不是 0/1。LB 評分是 WinAUC，
    # AUC 用 probability 才能正確算 ranking；之前 v23 輸出 0/1 等同 binary AUC = accuracy
    # 而非 probability AUC，可能損失些分數。submission_0509.csv 已驗證 float 格式被接受。
    final_winner = w_probs.astype(np.float64)

    # ---- 生成提交檔案 ----
    submission = pd.DataFrame({
        "rally_uid": uids,
        "actionId": final_action,
        "pointId": final_point,
        "serverGetPoint": final_winner,
    })
    submission = submission.sort_values("rally_uid").reset_index(drop=True)
    submission.to_csv(cfg.output_path, index=False)

    print(f"\n=== Submission Saved ===")
    print(f"Path: {cfg.output_path}")
    print(f"Shape: {submission.shape}")
    print(f"\nPreview:")
    print(submission.head(10))

    # 預測分布統計
    print(f"\n=== Prediction Distribution ===")
    print(f"  actionId: {dict(submission['actionId'].value_counts().sort_index())}")
    print(f"  pointId:  {dict(submission['pointId'].value_counts().sort_index())}")
    print(f"  serverGetPoint: {dict(submission['serverGetPoint'].value_counts().sort_index())}")


if __name__ == "__main__":
    main()
