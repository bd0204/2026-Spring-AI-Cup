"""
config.py - 所有超參數與設定
"""

import torch


class Config:
    # ---- 路徑 ----
    train_path = "./train.csv"
    # 額外的訓練資料 (會與 train_path 合併；match / rally_uid 會自動加 offset
    # 以免與 train.csv / test.csv 的 ID 衝突)。留空 list 代表不使用。
    # ⚠️ 啟用 extra_train_paths 時記得在 train.py 的 offset 區塊也加上
    #    gamePlayerId / gamePlayerOtherId 的 offset；processed_train_e.csv
    #    的 player ID 是獨立 ID 系統 (sex agreement 47% = random)，沒 offset
    #    會被誤判成同一個選手。
    # v23: 啟用 processed_train_e.csv 當「全 unseen」訓練資料，配合 player_id_dropout
    # 從 0.5 降到 0.3 → 整體 masked rate 維持 ~50%，但加上 ~15k 額外 rallies 的訓練量。
    #
    # 失敗紀錄 v31.1: 加 train_chatgpt + test_k_processed (+53% rallies)。
    #   CV Mean 0.4727 vs v28 0.4269（虛漲）但 LB NN-only 0.2930 大跌（CV-LB gap
    #   0.18，比 v28 的 0.06 大 3 倍）。合成資料在 val 上「假象漂亮」（Ep 5 WinAUC
    #   已 0.92）但對真實 test_new 是噪音。退回 v23 起的單一 extra 設定。
    extra_train_paths = ["./processed_train_e.csv"]
    # v25 revert G2: test_path 退回 ./test.csv 給 train.py 用（IPW、hand_map、
    # match/rally_uid offset 都對齊 v23 best 設定）。v24 實證 cfg.test_path = test_new
    # 反而 LB -0.008（CV 4/5 fold 退步），主因是 match ID offset 從 +323 → +364
    # 改變 GroupKFold tie-breaker 排序、訓練 fold split 變動。
    test_path = "./test.csv"
    # v25: inference 時用的 test 檔（預測目標）。設成不同 path 來分離「訓練對齊
    # 的 test」與「inference 預測的 test」。設 None 或留空 → fallback 到 cfg.test_path。
    inference_test_path = "./test_new.csv"
    output_path = "./submission.csv"
    model_dir = "./checkpoints"
    # v25 onwards: inference 權重來源管理（互不影響 train.py，train.py 永遠寫
    # cfg.model_dir = ./checkpoints）：
    #
    # (A) inference_dirs 非空 → 完全 override，只用這個 list 做 ensemble。
    #     例：inference_dirs = ["./ckpt_v23"]                  → 純 v23 inference
    #         inference_dirs = ["./ckpt_v23", "./checkpoints"] → v23+v25 ensemble
    # (B) inference_dirs 留空 → fallback 用 cfg.model_dir + extra_model_dirs。
    #     例：extra_model_dirs = []                         → 用 ./checkpoints (最新訓練)
    #
    # 預設 (A) 路線：v23 是已驗證 0.3575 的 best baseline。每次新訓練後先評估再決定
    # 是否切到新權重 — 預設提交永遠用 v23-only 確保不會退步。
    extra_model_dirs = []
    # v29 hybrid swap: NN side 用 v28 (./checkpoints/，剛訓練)，只取它的 serverGetPoint
    # 機率輸出（actionId/pointId 在 hybrid_swap 模式會直接被 LGBM proba 取代）。
    # v35 stacking 路線：固定權重 blend（prob-avg / rank-avg）全失敗，改走
    # stacking meta-learner。NN side 用 v33 weights（winner head 比 v28 更強，
    # w_winner=0.4）。nn_oof.py 也要用同一個 ./ckpt_v33 抽 OOF，兩邊 NN 必須一致。
    # v36: 新的 prefix-weighted + short-prefix-expert 訓練會輸出到 ./checkpoints。
    # 若要回放舊 v33，請把 use_winner_short_expert 關掉或用當時的 config。
    inference_dirs = []                # 留空 → fallback 到 ./checkpoints（train.py 訓練輸出的 v41 權重）
    # 訓練時觀察到的 player IDs；inference 時 test.csv 出現的新 ID 都會被
    # 替換成 0（unknown），以匹配 player_id_dropout 在訓練時看過的分布。
    known_player_ids_path = "./checkpoints/known_player_ids.json"
    # v41+: 設非空路徑 → inference 會把 raw NN proba（action/point/winner）存成 npz，
    # 供 ensemble_proba.py 做多版本 probability-level 融合（variance reduction）。
    # 每個版本各跑一次 inference（用各自架構 flag + weights dir）存一個 npz。空 = 不存。
    save_nn_proba_path = ""          # 設非空路徑時 inference 會把 raw NN proba 存成 npz（給 ensemble 融合用）

    # ---- 模型架構 ----
    # model_type 切換：
    #   "transformer" = 原始 MultiTaskTransformer（單 encoder, last-position pooling）
    #   "shuttlenet"  = ShuttleNet-inspired（雙視角 + Rally + Player A/B + Gated Fusion）
    model_type = "shuttlenet"

    # ShuttleNet Phase 2 + 3 開關（僅 model_type="shuttlenet" 時有效）：
    #   True  → 用 TAADecoder（Type-Area Attention + 任務專屬 sliding window）
    #             - q_t (action 視角): hard window k=2 拍（player A/B 各看 1 拍）
    #             - q_a (point  視角): hard window k=4 拍（rally encoder）
    #   False → 用 Phase 1 的 last-position pooling + GatedFusion
    use_taa_decoder = True

    # v37 任務 2：Winner head 獨立長視野路徑。
    #   TAA decoder 的 z_t / z_a 是 hard sliding window（k=2 / k=4）產生，視野
    #   太短不利於「整 rally 勝負」這種長期累積指標。開啟此 flag 後，winner head
    #   改吃 masked-mean(h_rally) 經一層 Linear 投影 → 看到整 rally 而非 window。
    #   False = 維持 0.5*(z_t+z_a)（v36 行為）。
    use_winner_long_view = True

    # v38 任務 A：對 action head 鏡像 winner long-view —— action_head 除了原本
    # 的 z_t（TAA decoder 的 k=2 hard window）之外，再加一條從整 rally 來的
    # masked-mean h_rally → action_pool_proj，然後跟 z_t 用 action_fusion (Linear)
    # 合併。預期：短窗看「下一拍最近 context」、長窗看「rally 全局 pattern」，
    # 兩者互補。False 時 forward 完全 fallback 到 v37 行為。
    use_action_long_view = True

    # v41：fingerprint → action 直接路徑（conditional-shift domain adaptation）。
    # 診斷 action 的 CV-LB gap 0.25 = conditional shift（test 選手動作習慣 train 沒有），
    # importance weighting 治不了（且 prefix 長度 covariate shift 已被 length-IPW 對齊）。
    # 唯一能在 test 取得選手習慣的是 fingerprint；但現在 NN 把它加到 stroke embedding
    # 經 encoder 稀釋，LGBM 卻直接拿 action 直方圖當特徵 → 這是 LGBM action 贏的關鍵。
    # 開此 flag → action head 直接吃「下一拍 striker 的 fingerprint action 直方圖」當
    # gated logit bias，繞過稀釋。需 use_match_fingerprint=True 才生效。
    # False 時 forward 完全 fallback（fp_action_bias=None），與 v40 等價。
    use_fp_action_prior = True

    # v39 任務 A1：對 point head 也鏡像 long-view 路徑（point_pool_proj +
    # point_fusion）。z_a 來自 TAA decoder 的 k=4 hard window，對 pointId 預測
    # 雖然比 action 寬鬆但仍有限。加 masked-mean(h_rally) 補上 rally 全局 pattern
    # （特別是 pointId 的 spatial transition 跟 cumulative count 需要看整 rally）。
    # False 時 forward 完全 fallback 到 v38 行為（point_logits = point_head(z_a)）。
    use_point_long_view = True     # v40：重新開啟結構升級（point head 長視野路徑）。
                                   # v39 overfit 主因不是這 +76k 參數，而是 class-aware
                                   # 強度被砍（power 0.5→0.4、target action→both）。
                                   # v40 保留結構升級、回退 class-aware 強度到 v38 水位。

    d_model = 128
    n_heads = 4
    n_layers = 3
    dropout = 0.2
    max_seq_len = 60
    embed_dim = 16  # 每個類別特徵的 embedding 維度

    # ---- 訓練 ----
    n_folds = 5
    epochs = 130        # v40: 160 → 130。v39 的 160 配合 power=0.4 讓 train loss
                        # 掉到 0.67（v38 是 0.74），val loss 反升 → overfit。回退
                        # power=0.5 後 train loss 會回到健康水位，130 給結構升級
                        # 的較大模型一點空間（v38 用 120，多 fold 末端還在升）。
                        # 仍配合 fold-level resume 應對 PBS 12hr walltime。
    batch_size = 128
    lr = 5e-4
    weight_decay = 1e-4   # v41 退回 v40 值（乾淨歸因，只測 fp_action_prior）
    label_smoothing = 0.05
    # v33 (Tier A2): winner BCE 也加 label smoothing。target 從 {0, 1} → {ε, 1-ε}
    # 避免 winner head overconfident → 改善 calibration → AUC ranking 更平滑。
    # 設成 0 退回原本硬 0/1 target。
    winner_label_smoothing = 0.05
    # v36: winner 的 CV-LB gap 主要來自驗證場景不匹配。test_new 有大量 1~2 拍
    # prefix，因此 validation / OOF 改用 sliding-window prefix，並按 test prefix
    # 長度分佈加權；winner loss 也加 pairwise AUC loss，直接對齊 LB 的 AUC 目標。
    use_prefix_weighted_validation = True
    winner_pairwise_weight = 0.15
    winner_pairwise_by_seq_bucket = True
    winner_pairwise_max_pairs = 4096
    # v36: 短 prefix 專家。只在 seq_len <= 2 時讓一個 winner-only MLP 以 gated
    # residual 方式修正 base winner logit，避免 Transformer 在 1~2 拍退化成弱訊號。
    use_winner_short_expert = True
    winner_short_prefix_len = 2
    # v38 任務 B：對 action 鏡像 winner short-prefix expert。test 53% rally
    # 是 1-2 拍 prefix，這段 NN 的 sequence signal 弱，由獨立的小 MLP 修正
    # action base_logits。短拍 (seq_len <= action_short_prefix_len) 才會被 gate
    # 啟動，長拍時 gate ≈ 0 → 不影響 base。False 時整段邏輯與舊版完全一致。
    use_action_short_expert = True
    action_short_prefix_len = 2
    # v39 任務 A2：對 point 也鏡像 short-prefix expert。test_new 53% 是 1-2 拍
    # prefix，pointId 在這段同樣信號弱（NN PtF1 整體只 ~0.30，短拍更低）。讓
    # PointShortPrefixExpert 在 seq_len <= action_short_prefix_len 時 gate 啟動
    # 修正 base_logits；長拍 gate≈0 退化成原 point_head。
    use_point_short_expert = True     # v40：重新開啟（短 prefix point 專家）。同上，
                                      # 結構升級保留，overfit 由 class-aware 強度回退處理。
    point_short_prefix_len = 2
    # v37 任務 1：Class-aware re-sampling。在原 length-IPW 之上疊加一個
    #   class_weight = (1 / freq(target_action))**power，讓稀有 action class
    #   被多抽到（LB 的 Macro-F1 對 19 類少數類敏感，serve 系尤其稀有）。
    #   power=0.0 = 完全不調整（退回 v20 length-only 行為）
    #   power=0.5 = 1/sqrt(freq)（推薦起點，溫和提升）
    #   power=1.0 = 完全反比 1/freq（會過度膨脹超稀有類，不建議）
    #   class_balance_target: "action" / "point" / "both"（目前只實作 action）
    use_class_aware_sampling = True
    # v39 任務 A3（已回退）：曾把 power 0.5→0.4、target action→"both"。
    #   實證這是 v39 overfit 的主因：top-5 稀有類 boost 從 v38 的 68/45/21x 暴跌到
    #   v40 對照的 8.5/6.4/4.3x（強度剩 ~12.5%）。少了集中 oversample，train loss
    #   掉太深（0.74→0.67），少數類在 test 上崩潰 → CV-LB gap 0.075→0.123，LB 退步。
    # v40：回退到 v38 已驗證的設定（power=0.5、target=action），保留結構升級。
    class_balance_power = 0.5
    class_balance_target = "action"
    focal_gamma = 1.5
    patience = 15        # v40: 20 → 15（回退 v38 已驗證值；20 配合 epochs=160
                         # 鼓勵訓練跑太遠加深 overfit）
    ema_decay = 0.999
    warmup_ratio = 0.05
    max_grad_norm = 1.0

    # ---- Player ID Dropout (Phase A: unseen player generalization) ----
    # 訓練時隨機把整個 rally 的 gamePlayerId / gamePlayerOtherId 改成 0（unknown），
    # 強迫模型學會「沒有 player ID 也能 fallback 到一般化特徵」。test.csv 中
    # 36.5% 選手是訓練資料完全沒看過的，這個 dropout 是核心一般化機制。
    # 0.0 = 永遠用 player ID，1.0 = 永遠 mask。
    #
    # v19: 0.3 → 0.5。v18 fold 分析（analyze_folds.py）發現 fold score 與 unseen
    # player 比例強烈反相關：fold 1 (42% unseen) Score 0.3867 vs fold 4 (34% unseen)
    # Score 0.4188，差 0.032。這代表模型過度依賴 player ID embedding，遇到 unseen
    # 就 fallback 不力。提高 dropout 強迫模型在更多訓練步驟中沒看到 player ID，
    # 才會把權重轉移到 fingerprint + structural 特徵 (rallyPhase/strikeId/score…)。
    # 副作用：seen-player 分支可能略弱（dropout 提高 → seen player 看到次數變少），
    # 但 test 36.5% unseen 與 high-unseen fold 高度對應，net effect 預期正向。
    #
    # v23: 0.5 → 0.3，因為加 processed_train_e.csv 當 extra 訓練資料時，所有 extra
    # rallies 的 player_id 強制設成 0（永遠 masked）。如果 dropout 維持 0.5，總體
    # masked rate 會是 ~75%（train.csv 50% mask + extra 100% mask），偏離 test 的
    # 25% 過遠。降到 0.3 後 masked rate 約 ~50%（train.csv 30% + extra 100%），
    # 跟 v22 同水位，乾淨對照「2x 資料量」的效果。
    #
    # v33 (Tier A3)：0.3 → 0.5。目的是攻擊 NN winner overfit (CV 0.87 → LB 0.748,
    # gap 0.12)。提高 dropout → winner head 在更多 train steps 看不到 player ID
    # → 強迫學跨選手通用 pattern。會推高整體 mask rate 到 ~75% > test 31%，但
    # hybrid mode 下我們只在乎 winner head 的 generalization，不在乎 action/point
    # CV 是否退步。
    # v41：維持 v40 的 0.5（不動）。v41 改走「fp_action_prior 架構直接路徑」攻
    # action 的 conditional shift，為了乾淨歸因，v41 vs v40 只差 fp_action_prior
    # 這一個變因；player_id_dropout / weight_decay 都退回 v40 值。
    # （備案：若 fp_action_prior 有效但想加碼，下一版可再提 dropout 0.8 協同逼用
    #  fingerprint。）
    player_id_dropout = 0.5

    # ---- Match-level Player Fingerprint (Phase B: unseen player generalization) ----
    # 對每個 (match, gamePlayerId) 算「打法指紋」：action / pointId_norm /
    # positionId_norm / spinId / strengthId 五個直方圖 + log_n_other 信心度，
    # 共 19+10+4+6+4+1 = 44 維。每個 stroke 同時輸入 self_fp + other_fp = 88 維。
    #
    # 為什麼有效：train/test 完全不共用 match (驗證 0 重疊)；即使選手 ID 從沒出現過，
    # 他在 test 同 match 內仍打了多個 rallies，可從那邊算出他的指紋。模型訓練時
    # 已透過 GroupKFold(groups=match) 學會「讀指紋」，自然 transfer 到 unseen player。
    #
    # 防 leakage：leave-one-out + Bayesian smoothing α
    #   loo_sum = total_sum - this_rally_sum；smoothed = (loo_sum + α·global)/(loo_n + α)
    use_match_fingerprint = True
    fingerprint_alpha = 10.0
    # 19 (actionId) + 10 (pointId_norm) + 4 (positionId_norm) + 6 (spinId)
    # + 4 (strengthId) + 1 (log_n_other) = 44。請與 utils.FingerprintTable.SPECS 同步。
    fingerprint_dim = 44

    # ---- 損失權重 ----
    # 註：官方 2026/04/17 公告建議「不要把 serverGetPoint 當輸入特徵」以避免過度
    # 擬合 Public LB。我們的 raw_features / categorical_features 從來就不包含
    # serverGetPoint（它只是訓練 target 與 submission 欄位），所以本來就符合。
    #
    # 之前嘗試把 w_winner 降為 0（完全停止監督 winner_head）→ Public LB 反而從
    # 0.28 退到 0.26，原因是 LB 公式 0.4·ActF1 + 0.4·PtF1 + 0.2·WinAUC 中
    # WinAUC 本來就有 ~0.13 的貢獻，停掉監督後 winner_head 變隨機，反倒抵消了
    # action/point 的提升。
    #
    # v33 (Tier A1)：因為 hybrid mode 下 action/point 100% 用 LGBM，NN 的這兩個
    # head 完全沒用 → 把 gradient 重新分配給 winner。winner 從 0.2 提到 0.4，
    # action/point 各從 0.4 降到 0.3。
    # v34：更激進 — 移除 player ID 特徵後 NN 完全專攻 winner。w_winner 拉到 0.8，
    # action/point 各降到 0.1（保留少量讓 encoder 不完全偏離 action/point 結構，
    # 但 gradient 主要由 winner driver）。
    # v34 結果失敗（LB 0.5107 < LGBM-alone 0.513）：原因 (a) w_winner 0.8 太重，
    # train.out 顯示 WinAUC Ep5 0.93→Ep30 0.80 持續劣化；(b) 拔掉 player IDs
    # 後 NN 跟 LGBM 看同一份特徵，diversity 消失，ensemble 反而放大 NN 的過信。
    # v35：恢復 player IDs（diversity），均衡三任務 (0.35/0.35/0.30)，
    # player_id_dropout=0.5 維持（控制過擬合）。
    use_winner_supervision = True
    w_action = 0.35
    w_point = 0.35
    w_winner = 0.30

    # ---- Action Category Aux Loss (Phase C — DISABLED, kept for record) ----
    # actionId 19 類可粗分成 5 大類：0=Zero, 1=Attack(1-7), 2=Control(8-11),
    # 3=Defensive(12-14), 4=Serve(15-18)。
    #
    # 失敗紀錄（三輪驗證均 LB 退步）：
    # v13 (aux+input+patience=18) LB 0.3161 vs v12 baseline 0.3277 (-0.012)
    # v14 (aux+patience=18, 拿掉 input)        LB 0.3081 (-0.020)
    # v15 (aux+patience=12)                    LB 0.3210 (-0.007)  ← cleanest test
    #
    # 結論：category 是 actionId 的 deterministic 映射，aux loss 不是獨立信號，
    # 而是把 action supervision 的 effective weight 從 0.4 拉到 ~0.45+，破壞
    # 原本 0.4/0.4/0.2 平衡，並讓訓練曲線變平、stop epoch 普遍延長 → over-fit
    # train domain。CV 升 +0.004 但 LB 降 -0.007 是同樣的 train/test domain
    # gap pattern。設 use_action_category=False 退回 v12 baseline。
    use_action_category = False
    w_action_category = 0.1

    # ---- 類別數 (從資料分析得知) ----
    n_action_classes = 19   # actionId: 0~18
    n_point_classes = 10    # pointId: 0~9
    n_action_category_classes = 5   # 0=Zero, 1=Attack, 2=Control, 3=Defensive, 4=Serve

    # ---- v19/v20: Inference-time pointId prior shift (logit adjustment) ----
    # 模型一直沒預測過 pointId=0（rally-end marker）。inference 時用 logit adjustment
    # 把 model 隱含 prior 對齊到 train 真實 prior：
    #   adjusted_logit[c] = logit[c] + alpha · log(p_train[c] / p_pred[c])
    # alpha = 0 → 不修正；alpha = 1.0 → 完全對齊。
    # v19: 用全 stroke prior（含 serve placement 雜訊）。v20 改用 non-first-stroke
    # prior（strikeNumber != 1），對齊 sliding-window target；class 0 boost 從
    # log(0.18/0.01) 提到 log(0.219/0.01) — 但 v20 的 train 已經會學 class 0，
    # p_pred(0) 應從 0.01 升到 ~0.22，shift 會自動變 near-no-op，alpha=1 安全。
    #
    # v22: 模型 pred_prior(0) = 0.20 ≈ train_prior 0.22，shift 把 class 0 從 252
    # 推到 305 (+53)。三點 α tuning 結果（同 v22 weights）：
    #   α=0   → 252 class 0, LB 0.3201
    #   α=1.0 → 305 class 0, LB 0.3262 ← optimum
    #   α=1.5 → 334 class 0, LB 0.3251 (over-shift)
    # 確認 α=1.0 是 sweet spot；test 真實 class 0 比例 ≈ 24~25%。固定 α=1.0。
    pointid_prior_shift_alpha = 1.0

    # v37 任務 3：actionId 的 logit prior shift（與 pointid_prior_shift 同公式）：
    #   adj[c] = logit[c] + alpha · log(p_train(c) / p_pred(c))
    # action 是 19-class macro F1 評分，理應做跟 pointId 同樣的 prior alignment。
    # train prior 用「sliding-window target」分布更貼，即排除 strikeNumber == 1
    # （serve 拍）的 actionId 分布。alpha=0 → 不調整（退回舊版行為）。
    action_prior_shift_alpha = 1.0   # v40 純 NN 測試：開回 1.0（純 NN 模式 prior
                                     # shift 對齊 train 分布有效；切 hybrid 時記得改回 0.0）

    # ---- v30 hybrid: cross-paradigm with LightGBM (train_0513_2) ----
    # 朋友的 LGBM 持續進步：0509 (0.3702) → 0510 (0.4261) → 0513_2 (0.4580)。
    # 0513_2 改善：CV ActF1 0.365→0.388, PtF1 0.333→0.370, WinAUC 0.690→0.727。
    # NN winner AUC 仍略強（test_new 約 0.748 vs LGBM 0.78），但差距縮小。
    # 維持 v29 策略：action/point 純 LGBM，winner 50/50 averaging。
    # 設 lgbm_proba_path = "" 或 None → 退回 NN-only inference。
    # v33 + LGBM 0516：朋友 LGBM 從 0513 (LB 0.458) 跳到 0516 (LB 0.500)，CV
    # Action 0.388→0.528, Point 0.370→0.511, Winner AUC 0.727→0.804 全面進步。
    # 我們的 NN (v33 with Tier A) 只在 winner 50/50 ensemble 貢獻。
    # v34 + LGBM 0517：朋友 LGBM 再進步到 LB 0.513。使用訓練好的 v34 NN
    # 做 serverGetPoint 50/50 averaging，actionId/pointId 純 LGBM。
    # v33 + LGBM 0518：朋友 LGBM 又進步到 LB 0.521。先用 v33 weights 跟它做
    # hybrid 對照，再 retrain v35。
    # v35 stacking 路線：v28/v33/v34 跟 LGBM 0518 做任何固定權重 blend（prob-avg
    # / rank-avg）都退步 → 改走條件式 stacking。
    # actionId/pointId：用朋友原本的 proba_0518.npz（LB 0.5212 級）。多 seed 版
    #   lgbm_test.npz 經實測 LB 只有 0.5170（MATCH_ZERO_PROB=0 + GPU 淨退步）→ 不用。
    # winner：由 inference.py 的條件式 meta-learner 處理（看過選手→meta，其餘→LGBM）。
    lgbm_proba_path = ""             # v40 純 NN 測試：關掉 hybrid（退回 NN-only）。
                                     # 確認純 NN LB 後再切回 "./model0528/proba_0528.npz"
                                     # 做 hybrid（記得同時把 action_prior_shift_alpha 改 0.0）
    # v35 stacking: winner meta-learner（stack_winner.py 產出）。
    # ablation 期間設成 ""，避免 prob-avg cell 被 meta 分支搶走。
    winner_meta_path = ""
    # v35 最終方案：rank-conditional winner blend。meta-learner hard-split 把
    # meta 刻度跟 LGBM 刻度硬混進同一個 AUC 欄位 → scale mismatch → winner AUC
    # 掉 0.02。改成「全程 rank space」：兩邊都轉 [0,1] rank（刻度一致），
    #   seen rally  → hybrid_winner_w_nn·rank(NN) + (1-w)·rank(LGBM)
    #   unseen rally→ 純 rank(LGBM)（維持朋友 0.5212 排序，零下檔）
    # True 時優先序高於 meta-learner / rank-avg / prob-avg。
    winner_rank_conditional = True     # v37 best combo: rank-cond 50% (v35 已驗證最佳)
    # v29: hybrid swap mode。True → 直接 task-level 替換，跳過 prior shift 跟
    # 機率平均。False → v26 averaging behavior (with prior shift on both)。
    hybrid_swap = True
    # v29: hybrid mode 下 per-task NN 權重（每個任務獨立可調）。
    # 公式: final_probs = w_nn * NN_probs + (1 - w_nn) * LGBM_probs，逐 class 加權後 argmax。
    #
    #   hybrid_winner_w_nn:
    #     1.0 = pure NN (v28_0510, LB 0.4254)
    #     0.5 = 50/50 (v29, LB 0.4349 ← 已驗證)
    #     0.0 = pure LGBM (朋友 alone, LB 0.4261)
    #     拟合曲線最佳 w* ≈ 0.49，0.5 已最佳，無 grid 空間。
    #
    #   hybrid_action_w_nn:
    #     CV F1 接近但 NN 在 test_new 有 CV-LB overfit。
    #     試驗結果：w_nn=0.5 LB 0.4262 (-0.0087), w_nn=0.4 LB 0.4262 (vs pure 0.4349)
    #     → NN action 無 ensemble benefit，pure LGBM 最佳。
    #
    #   hybrid_point_w_nn:
    #     NN PtF1 ~0.24 顯著低於 LGBM 0.370 → 平均會被 NN 拖下來，
    #     保持 0.0 = pure LGBM。
    #
    #   hybrid_winner_w_nn:
    #     0510 case: NN AUC 0.748 vs LGBM 0.752 → 50/50 averaged AUC 0.80
    #     (LB +0.044, v29 0.4349)。0513_2 case: LGBM 升到 0.78，預期 +0.02~+0.04
    #     AUC，LB +0.004~+0.008。
    # v35 ablation grid 結果（baseline = 朋友純 LGBM 0.5309）：
    #                            50/50 uniform    rank-conditional
    #   actionId                 0.5356 (+0.0047) ?
    #   pointId                  0.5275 (−0.0034) ?
    #   serverGetPoint           ?                0.5310 (+0.00006)
    # 補完 6 格後挑最佳組合（每次測試只啟用一個 task；其他 w_nn=0、cond=False）。
    # Cell C 結果：winner 50/50 prob-avg = 0.5299 (−0.0010)，<rank-cond 0.5310
    # → 確認 winner 最佳模式是 rank-cond。
    # v37 best combo（v35 ablation 找到 + v37 NN 升級到 LB 0.3613）：
    #   action: rank-conditional 50/50（v35 已驗證 +0.0050；v37 NN 更強，預期更高）
    #   point : 完全關掉（NN PtF1 0.20 vs LGBM 0.52 差太多，兩種模式都退步）
    #   winner: rank-conditional 50/50（v35 +0.00006；v37 winner long-view 預期再 +）
    hybrid_winner_w_nn = 0.5
    hybrid_action_w_nn = 0.0
    hybrid_point_w_nn  = 0.0
    # v35 ablation: action/point 也支援 conditional 模式（seen→blend, unseen→純LGBM）。
    # 對應 ablation table 的「Rank-Conditional」欄。multi-class 機率本身就同刻度，
    # 不需 rank 轉換，conditional prob-blending 就是正確的 multi-class 類比。
    hybrid_action_conditional = False     # v37 best combo: action conditional 開
    hybrid_point_conditional  = False
    # winner blend 模式：
    #   False = prob-average（傳統加權平均機率）— v28/v33/v34 + LGBM 0518 全失敗，
    #           因 NN 與 LGBM winner AUC 接近但 calibration scale 不一致，算術
    #           平均互相打散訊號（LB -0.0013~-0.0015）。
    #   True  = rank-average（先轉 [0,1] rank 再加權平均）— AUC 只看排序，
    #           rank-avg 直接在 AUC 目標空間 blend，免疫 calibration mismatch。
    hybrid_winner_rank_avg = False    # Cell C: 走 prob-avg → rank-avg 也要關

    # ---- v26 ensemble: single weight ----
    # NN (shuttlenet) 在 ensemble 中的權重；LGBM 拿 (1 - ensemble_nn_weight)。
    # 因為 LGBM LB > v23 LB，給 LGBM 較多權重 → ensemble_nn_weight=0.4 (LGBM 0.6)。
    #
    # 失敗紀錄 v27（per-task weights + 拿掉 LGBM prior shift）：
    #   試 action=0.3 / point=0.4 / winner=0.7 + LGBM 不 shift → LB 0.3699 (-0.0004)
    #   原因：拿掉 LGBM shift 把 class 0 從 454 拉到 325（test_new 真實 ~22%≈406），
    #   LGBM 的 13% native prior 並非 well-calibrated。維持雙邊都 shift + 單一權重。
    ensemble_nn_weight = 0.4

    # ---- v20+: Sliding-window last-stroke target sub-sampling ----
    # 取消 point_mask 後，每個 rally 的 last-stroke target (pointId=0) 全進 loss。
    # 1.0 → 全部納入（class 0 占 sliding window target ~22%）
    # 0.5 → 半數 rally 納入 last-stroke (class 0 約占 ~12%)
    # 0.0 → 完全不納入 last-stroke target（退回 v19 行為，但 point_mask 仍 True）
    #
    # v20 用 0.5 是 hedge — 怕 class 0 過量訓練 → test collapse。但事後驗證 v20 的
    # collapse 根源是 rallyProgress OOD（v21 已修），不是樣本量。v21 (p=0.5) 看到
    # pred_prior(0) = 0.16，無 over-predict 跡象，模型 calibrated。
    # v22 改 1.0：讓 model 看到自然 class 0 比例 (~22%)，把 class 0 學得更精準
    # → F1_0 從 ~0.40 升到 ~0.55（推估）→ macro PtF1 +0.015 → LB +0.006。
    last_stroke_target_prob = 1.0

    # ---- 其他 ----
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_sliding_window = True

    # ---- 特徵定義 ----
    # 模型輸入的原始特徵
    # v18: 完整退回 v12 baseline (raw_features 不含 numberGame)。v17 加的
    # numberGame + rallyProgress per-prefix fix 一起 LB 0.3277 → 0.3128 (-0.015)，
    # 不確定是哪一個害的或交互害的，整組退回後從 v12 baseline 單獨疊加 rallyPhase。
    #
    # v34: 移除 gamePlayerId / gamePlayerOtherId。鏡像朋友 LGBM 0516 的關鍵改動
    # （Action F1 0.388→0.528, Point F1 0.370→0.511, Winner AUC 0.727→0.804 全面飛躍）。
    # 完全切斷 NN encoder 對 player ID 的記憶，強迫只從 score + action sequence
    # + fingerprint 學跨選手 pattern。
    # 注意：fingerprint 仍保留（不同性質，是 style summary 不是 player identifier）。
    # v35: 恢復 gamePlayerId / gamePlayerOtherId。v34 拔掉這兩個之後 NN 跟 LGBM 0517
    # 都不看 player ID，diversity 消失 → hybrid 反而退步 (LB 0.5107 < LGBM 單獨 0.513)。
    # 配合 player_id_dropout=0.5（rally-level 隨機 mask）控制過擬合：50% 樣本仍會
    # 看到 player ID（保留 NN 的「player-aware」訊號 → 對 LGBM 互補），
    # 另外 50% 看不到（強迫模型學跨選手 pattern → 不會純依賴 player lookup）。
    raw_features = [
        "sex", "handId", "strengthId", "spinId",
        "pointId", "actionId", "positionId",
        "strikeId", "strikeNumber",
        "scoreSelf", "scoreOther",
        "gamePlayerId", "gamePlayerOtherId",
    ]

    # 類別特徵 -> (feat_name: num_classes)
    categorical_features = {
        "sex":              3,   # 1,2 + padding 0
        "handId":           3,   # 0,1,2
        "strengthId":       4,   # 0,1,2,3
        "spinId":           6,   # 0~5
        "actionId":        19,   # 0~18
        "pointId":         10,   # 0~9
        "positionId":       4,   # 0~3
        "strikeId":         5,   # 1,2,4 -> max+1=5
        "playerHand":       3,   # 1=右, 2=左 (+ 0 for padding)
        "receiverHand":     3,   # 1=右, 2=左 (+ 0 for padding)
        "handPair":         5,   # 1=右右,2=右左,3=左右,4=左左 (+ 0 for padding)
        "pointId_norm":    10,   # canonical frame 的 pointId (0~9)
        "positionId_norm":  4,   # canonical frame 的 positionId (0~3)
        # v34: 移除 gamePlayerId / gamePlayerOtherId embedding（鏡像 LGBM 0516
        # 的關鍵改動）。完全切斷 NN encoder 對 player ID 的記憶。
        # v35: 恢復 player ID embedding。max raw ID ≈ 196 (train max), 用 200
        # 留 buffer (model.py 會 +2 → 實際 embedding [202, 16]，與 v33 checkpoint 對齊)。
        # 用 player_id_dropout=0.5 控制過擬合，同時保留 NN-LGBM diversity。
        "gamePlayerId":      200,
        "gamePlayerOtherId": 200,
        # v18: rally phase: 0=padding, 1=serve, 2=receive,
        # 3=stalemate-server (奇 ≥3), 4=stalemate-receiver (偶 ≥4)
        "rallyPhase":          5,
        # v25 stepInTactic 已移除（LB -0.015，太 redundant）
        # v32 LGBM-style features 已移除（LB -0.022，NN 不需要 explicit tactical features）
    }

    # 連續特徵
    continuous_features = ["strikeNumber", "scoreSelf", "scoreOther"]

    # 工程特徵
    # 失敗紀錄 v16: (scoreSelf, scoreOther) → (scoreDiff, scoreSum) reparam，LB -0.008。
    # 失敗紀錄 v17: + numberGame + rallyProgress per-prefix fix，LB -0.015。
    # v18: 加 rallyPhase（4-class tactic phase，受 Tac-Simur, Wang et al. 2020 啟發）。
    #   1=serve (sN=1), 2=receive (sN=2), 3=stalemate-server (sN 奇 ≥3),
    #   4=stalemate-receiver (sN 偶 ≥4), 0=padding。論文核心發現是這 4 phase 的
    #   stroke 轉移分佈完全不同；現有 strikeId 把 stalemate 全混為 4，丟失
    #   server/receiver 視角。純 strikeNumber 函數 → 跨選手 generalize。
    # 失敗紀錄 v25: 加 stepInTactic 後 LB 0.3575 → 0.3428 (-0.015)，
    #   結構性 deterministic 跟 rallyPhase + strikeNumber 過度 redundant，
    #   反而成 spurious shortcut。revert 掉，model 維度回到 v23 (n_features=22)。
    # 失敗紀錄 v32: 移植 LGBM 0513 的 Tier 1+2 features (pt_x/y, cross_court,
    #   zone_shift, opp_disp, is_clutch...) 共 +12 個。CV +0.006 (微漲) 但
    #   NN-only LB 0.3350 (vs v23 0.3575, -0.022)。確認 NN 對 deterministic
    #   feature 反應跟 LGBM 完全不同 — self-attention 已隱式學會這些模式，
    #   explicit feature 反而增加 capacity 造成 train 過擬合。revert 回 v23 設定。
    engineered_features = [
        "rallyProgress", "rallyPhase",
        "playerHand", "receiverHand", "handPair",
        "pointId_norm", "positionId_norm",
        "playerHandConf", "receiverHandConf",  # 連續值 ∈ [0,1]
    ]

    @property
    def all_features(self):
        return self.raw_features + self.engineered_features

    @property
    def n_features(self):
        return len(self.all_features)
