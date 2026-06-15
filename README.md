# 2026-Spring-AI-Cup
Temporal Sequence-Based Prediction of Table  Tennis Tactics and Outcomes

This is LGBM training branch from chen kai ting

## Environment
- 作業系統:Ubuntu 24.04.1 LTS(via WSL2 on Windows)
- Linux-kernel:6.6.87.2-microsoft-standard-WSL2
- 語言:Python 3.12.3
- 額外資料集:kaggle(https://www.kaggle.com/competitions/introduction-to-data-secience-competition-ttmatch/data)
- 使用套件
```
joblib==1.5.3
lightgbm==4.6.0
numpy==2.4.4
pandas==3.0.2
scikit-learn==1.8.0
```

## Algorithm and Model Framework

本方案以 **LightGBM（LGBM）** 作為核心模型。LightGBM 是基於梯度提升決策樹（GBDT）的演算法，每一輪迭代都訓練一棵新樹來修正前一輪的殘差，最終將所有弱決策樹的預測結果加總輸出。選用 LGBM 的原因：資料為數值編碼的結構化表格，樣本數約 15 萬筆，LGBM 能在短時間內收斂且不易過擬合；此外，預測下一拍戰術的流程本質上是一個分類問題，恰好符合 LGBM 的強項。

**模型架構：三個子模型**

1. **Action 模型**（球種預測）：19 類多分類，`num_leaves=127`，`n_estimators=5000`，指標為 Macro-F1。
2. **Point 模型**（落點預測）：10 類多分類，`num_leaves=127`，`n_estimators=5000`，指標為 Macro-F1。
3. **Server 模型**（得分方預測）：二元分類，`num_leaves=63`，`n_estimators=3000`，指標為 AUC。

三個模型共用基礎超參數：`learning_rate=0.05`、`subsample=0.85`、`colsample_bytree=0.8`、`reg_lambda=1`，並配合 Early Stopping（patience=200）防止過擬合。評分函數為 `0.4×F1_action + 0.4×F1_point + 0.2×AUC`。

**雙模型混合策略**：分別訓練「含選手 ID（unmasked）」與「不含選手 ID（masked）」兩套模型，以 70（masked）：30（unmasked）的比例加權融合機率矩陣，使模型在 test 中對未知選手仍具備泛化能力，同時保留已知選手的個人戰術資訊。
## Innovation

**不加入 match 特徵**

test 與 train 的比賽場次完全不重疊，若將 match ID 納入模型，模型只會記憶特定場次的統計，對 test 毫無意義且引入雜訊。因此 match 欄位從未進入特徵清單。

**落點座標分解（pointId → 空間座標）**

pointId 1–9 本是接球方視角下的九宮格格位，直接以類別值輸入模型會失去空間語意。本方案將其拆解為兩個獨立維度：`pt_x`（左右，1=正手側 2=中路 3=反手側）與 `pt_y`（深淺，1=短球 2=半出台 3=長球）。如此模型能學習「往反手側打」與「打短球」各自的邊際效應，而非只看九個不相關的類別。

**斜線／直線指標**

結合擊球方慣用手（`handId`）與落點方向（`pt_x`），推導出本拍是斜線（cross-court）或直線（down-the-line）。正手打到對方反手側屬斜線，正手打到對方正手側屬直線，反手方向相反。這個複合指標捕捉了桌球戰術中「斜線拉正手、直線攻反手」的對角線調度模式。

**時序特徵：Lag + 三拍 n-gram + 連續性**

分類器本質上不擅長學習序列，因此以工程方式注入時序資訊：
- Lag 1–4 拍的原始欄位（action、point、hand、spin 等）直接拼入特徵
- 三拍 n-gram 將 lag2→lag1→cur 的 action / point / spin 序列編碼為單一整數，讓模型辨識戰術三段組合
- 連續性旗標（`same_action_as_prev`、`same_point_as_prev` 等）與最近五拍的落點變化次數（`num_point_changes_last5`），讓模型感知對打節奏與選手是否在重複同一戰術

**Rally 累計落點分佈**

累計統計 rally 內至當前拍為止的短球、長球、正手區、反手區出現次數（`count_short`、`count_long`、`count_forehand_zone`⋯⋯）。這讓模型能感知當前對打的落點偏好，例如「本 rally 已連打多顆反手長球，接下來的落點分佈應與全局統計不同」。

**對手被調動幅度**

計算當前落點與 lag2 落點（同側選手的上一拍）之間的歐式距離（`opp_disp_dist`）以及 x、y 方向位移，量化對手在這兩拍之間被調動了多遠。被大幅調動的對手往往無法維持攻擊路線，下一拍的球種與落點分佈也會隨之改變。

**發球站位廣播作為慣用手依據**

桌球選手為維持正手主導的攻擊，右撇子習慣站在左側（正手位）發球，左撇子相反。將 rally 第 1 拍的 `positionId`（發球站位）廣播至整個 rally，即可讓模型在每一拍都能參考此信號，作為推斷慣用手的輔助依據，即使 `handId` 本身存在缺值或噪音也能補強。

**以 test 分佈對齊訓練樣本權重**

分析 test_new.csv 的 strikeNumber 分佈後，發現待預測的第 2 拍（strikeNumber=1）占 27.5%、第 3 拍（strikeNumber=2）占 25.7%，明顯高於 train 中的比例。因此對訓練樣本加入短拍加權（strikeNumber=1 乘以 1.5、strikeNumber=2 乘以 1.3），使訓練集的難度分佈與 test 對齊，避免模型對長 rally 過度最佳化。

**雙模型 ID 遮蔽策略差異 + 機率矩陣融合**

觀察到 test 中雙方都在 train 出現的場次佔 40%、單方出現占 44%、雙方都未出現只佔 16%，因此訓練兩套模型：
- **unmasked 模型**：從 test 中取出出現過的選手 ID 集合，train 中只保留這些 ID（其餘歸零），使模型精確學習「test 會遇到的人」的個人習慣
- **masked 模型**：train 選手 ID 以 30% 機率隨機歸零，強迫模型在無 ID 資訊下只依賴動作序列泛化預測

最終以 70%（masked）: 30%（unmasked）的比例加權融合兩套模型的機率矩陣，泛化模型為主、習慣模型為輔，使整體預測在已知與未知選手上均能穩定。

## Data Process

### 前處理

**資料來源合併**

本方案共使用四份資料：`train.csv`（主要）、`processed_train_e.csv`（額外處理版）、`train_e.csv`、`train_k.csv`。

1. 四份資料的 rally_uid / match / gamePlayerId 均從 1 開始計數，直接合併會造成同一 ID 指向不同對象的衝突，因此對每份額外資料加上偏移量使 ID 空間完全不重疊：

移除train_k中的serveID,

| 資料集 | rally_uid 偏移 | match 偏移 |
|---|---|---|
| processed_train_e | +20000 | +500 |
| train_e | +60000 | +1500 |
| train_k | +80000 | +2000 |

2.  train_k與train_e均移除rally_id, let, serverGetPoint, serveId, serveNumber這四個欄位
3. `train_k` 在合併前另需去重（移除與 train_e 完全相同的 rows）並過濾掉只有 1 拍的 rally。
4. 除了train之外另外三份資料均將playerID歸0，因為這些是外部資料，與test中的id無關。
5. train_e與train_k中的serverGetPoint移除，不加入訓練。


**選手 ID 遮蔽策略（兩個模型差異所在）**

- **masked 模型**：`train.csv` 中的 `gamePlayerId` / `gamePlayerOtherId` 以 30% 機率隨機歸零（`PLAYER_ZERO_PROB=0.30`），迫使模型學習「不依賴選手 ID 也能預測」的泛化模式。額外資料集因無對應選手 ID，全部設為 0。
- **unmasked 模型**：先從 `test_new.csv` 取出雙方所有出現過的選手 ID 集合（`test_player_ids`），然後在 `train.csv` 中將**未在 test 出現**的選手 ID 歸零，只保留 test 中真正會遇到的選手，讓模型學習這些人的個人習慣。

這個設計來自觀察：test 中雙方都在 train 出現的場次佔 40%、單方出現佔 44%、雙方都未出現只佔 16%，因此 unmasked 模型保留已知選手的習慣，masked 模型負責泛化。

**類別權重調整**

訓練資料存在嚴重的類別不平衡，透過 `make_weights()` 做兩層加權：

1. **反比例基礎權重**：`w = total_samples / (num_classes × class_count)`，稀有類別自動獲得更高權重，再以 `np.clip(w, 1.0, max_w)` 限制上限（action: 12、point: 8），避免稀有類別主導訓練。
2. **個別覆蓋（overrides）**：對特定類別手動指定權重—action 模型中拱球×30、磕球×20、放高球×20、殺球×10、極稀有發球類別壓回×1；point 模型中反手位短球×40、正手位短球×20、出界壓低為×0.25。
3. **短拍加權**：`strikeNumber=1` 的樣本額外乘以 1.5、`strikeNumber=2` 乘以 1.3，對齊 test 資料中第 1、2 拍分別佔 27.5% 與 25.7% 的分佈比例。

---

### 特徵建立
特徵建立有兩個部分，第一是整個rally都相同的比賽資訊，如比分、id、性別等等
第二部分是lag的特徵，給予前幾拍的資訊與連續的變化，讓模型學習分類每種rally情況。

#### 全局比賽資訊
- `score_diff`、`total_score`：分差與總分）
- `is_clutch`：雙方至少一方達 9 分且分差 ≤ 2 的關鍵球
- `is_deuce`：雙方均達 10 分的平局
- `is_server_turn`：依 strikeNumber 奇偶數判斷當前是否輪到發球方出手

#### Lag特徵
**Lag 特徵（前 1–4 拍歷史）**
對 `["actionId", "pointId", "handId", "strengthId", "spinId", "positionId", "strikeId"]` 七個欄位各產生 lag1 ~ lag4，以 `groupby("rally_uid").shift(k).fillna(0)` 計算，rally 邊界前的缺值補 0。

**對手限制**
結合擊球方 `handId` 與落點方向 `pt_x`，判斷本拍是否能限制對手，無法判斷時為 -1：
- 正手（handId=1）打到反手側（pt_x=3）→ 對手回球容易受限
- 正手（handId=1）打到正手側（pt_x=1）→ 對手有較多餘裕處理
- 反 -> 反 對手受限
- 反 -> 正 對手易處理

**3-gram 序列特徵**
桌球的戰術執行常常需要看三拍，包含進攻、對手回應、自己的下一步，因此以三拍為單位切分，視為一個戰術單位，
連續的三拍都給予編碼，模型分類時能分出不同的序列組合。
- `*_transition`：3-gram，`lag2_actionId × 10000 + lag1_actionId × 100 + actionId`

**連續性特徵**
大量給予連續性特徵，加強模型時序的能力
`same_action_as_prev`、`same_point_as_prev`、`same_spin_as_prev`、`same_depth_as_prev`、`same_zone_as_prev`：各判斷 lag1 與 lag2 的對應值是否相同，lag2 為 0（資料不足）則視為不同。

**Rally 內累計落點分佈**
對 rally 內截至當前拍之前（不含當前拍）的歷史落點做 cumsum，得到 6 個累計計數特徵：`count_short`、`count_half_long`、`count_long`、`count_forehand_zone`、`count_middle_zone`、`count_backhand_zone`，讓模型了解本局對打的落點趨勢。

**調動幅度特徵**
- `zone_shift_x/y`：當前落點與 lag1 落點的 x、y 位移（描述本拍對對手的調動方向）
- `zone_shift2_x/y`：lag1 與 lag2 落點的位移（描述對方同側選手的前一輪調動）
- `opp_disp_x/y/dist`：當前落點與 lag2 落點（同側選手的前一拍）的歐式距離，量化對手被調動的幅度

**發球站位廣播**

將每個 rally 第 1 拍的 `positionId`（發球站位）和第 2 拍的 `positionId`（接球站位）以 `rally_uid` 為 key 廣播到整個 rally 的所有拍次，形成 `rally_serve_pos`、`rally_receive_pos` 及複合欄位 `serve_receive_pos_combo`。第 1 拍預測第 2 拍時，接球站位尚未發生，`rally_receive_pos` 在此設為 0 避免資料洩漏。

---

### 建立訓練資料

**目標變數設定**

以 `groupby("rally_uid")["actionId"].shift(-1)` 將「下一拍的 actionId / pointId」對齊至當前拍，使每一行成為一個訓練樣本，預測目標是下一拍的球種、落點與得分方。最後一拍（沒有下一拍）dropna 移除，不納入訓練。

**以基本訊息 + 前四拍的資訊為訓練資料，預測下一拍**

每一筆樣本的輸入特徵包含：當前拍的所有資訊（`CURRENT_FEATS`）+ lag1~lag3 的歷史記錄（`LAG1_FEATS` + `LAG23_FEATS`）+ 比賽戰況（`CONTEXT_FEATS`）。訓練資料為「第 n 拍預測第 n+1 拍」，涵蓋 rally 中每一拍（第 1 拍到倒數第 2 拍）。

**前四拍資料不足時補 0（n=1, 2, 3, 4 的情形）**

Lag 特徵以 `groupby("rally_uid").shift(k).fillna(0)` 計算，rally 開頭資料不足時自動補 0：
- `strikeNumber=1`（預測第 2 拍）：lag1~lag4 全為 0
- `strikeNumber=2`（預測第 3 拍）：lag1 有真實值，lag2~lag4 為 0
- `strikeNumber=3`（預測第 4 拍）：lag1、lag2 有值，lag3~lag4 為 0
- `strikeNumber=4`（預測第 5 拍）：lag1~lag3 有值，lag4 為 0

為明確告知模型哪些 lag 是真實歷史、哪些是補 0 的假值，另設 `lag1_available`、`lag2_available`、`lag3_available` 旗標（strikeNumber ≥ 2/3/4 時分別為 1）。

**測試集推論方式**

測試集每個 rally 取最後一拍（`groupby("rally_uid").tail(1)`）作為特徵輸入，預測該 rally 的下一拍戰術與結果。

**模型訓練資料**
- action與point模型使用train、train_process、train_k、train_e訓練。
- sever模型只使用train、train_process訓練。

## Approach

### LGBM的訓練方式為以下的流程。
#### 1. 訓練資料處理
詳細參考Data Process
針對test_mew.csv也建立特徵

#### 2. 切分資料集
我們以match為單位進行訓練集、驗證集的切分，訓練集80%。驗證集20%，這樣切分的目的是若隨機切分訓練資料，
可能會造成同場比賽的特徵出現在訓練集與驗證集兩邊，造成過擬合與分數不準確的情況

#### 3. 訓練模型
利用5 fold的訓練，每個fold都訓練三個模型，輸出Loss方便調整，並利用驗證集進行預測，產生三項分數，這個階段的分數只會有單個fold的結果進行預測，相較整個模型會有因為切分時切到長rally與短rally的區別而起伏較大。

當五個fold都訓練完後，會以整個模型重新預測一次驗證集，產生全局的OOF做為參考，並產生每個類別預測的F1-score與特徵的重要度，讓我們判斷是要改進的有那些類別的預測與增減特徵依據。

#### 4.儲存權重與預測
利用模型權重預測test_new.csv並生成submission檔案。 輸出預測結果的統計，利用每個類別的分布判斷這次的訓練效果，輸出機率矩陣，後續使用。

### 雙LGBM
我們觀察到test_new中，一場比賽中playerID雙方都出現在train中的比例為40%，一邊有出現，另一邊沒出現的佔44%，雙方都沒出現只佔16%，因此我們訓練兩個LGBM模型。

unmasked模型在建立特徵時，會將playerID在test與train都出現的保留，其餘都設為0，這個模型代表會學習到這些人的習慣，而都未出現就當作是0，代表沒出現過的人都當成0，以一大群的資料來預測。

masked模型則不把id放入訓練，模型只會從動作的模式，比賽的戰況來預測下一拍，沒有帶入任何習慣，處理泛化的預測。

### 融合LGBM矩陣
將兩個LGBM模型輸出的機率矩陣依照70為unmasked，30%masked的方式融合機率，這樣的理由是，泛化的模型要非常確定才有可能扭轉結果，否則會以帶有習慣的unmasked模型為主。接著產生一個機率具矩陣。



## Analysis and Conclusion
內容規定：分析所使用的模型及其成效，簡述未來可能改進的方向。分析必須附圖，可將幾個成功的和失敗的例子附上並說明之。

**改動成功案例:**
1. 關於playerID與match的使用 
起初有train、proc_train兩份各80000筆的資料，對於playerID與match的策略都是隨機遮蔽30%，成效為LB=0.37
再來我們發現只有train的資料id與test有關聯，因此將proc_train中的id、match均改為0，成效為OOF=0.4172、LB=0.426(+0.05)

中間有經過其他改動，但與id、match無關，不討論，分數基準變為OOF=0.4484,LB=0.456
後來再將match、id直接移除不加入訓練，OOF=5766、LB=0.503(+0.045)
中間有經過其他改動，但與id、match無關，不討論，分數基準變為OOF=0.5841,LB=0.52
之後想到將訓練資料對齊test只遮蔽沒見過的id，OOF=0.6008,LB=0.539(+0.01)

2. 加入n-gram的特徵
從LB=0.41 -> LB=0.45
**改動失敗案例**
1. 將action改看2拍，小降0.01

改進方向:
1. 我們利用生成式AI給予特徵的方向，因此都是一個區塊一個區塊的加入特徵，例如連續性特徵、戰術、落點位置、落點動作統計等等。
因為繳交次數有限，並沒有一個特徵一個特徵的拆開來單獨作為變因重新訓練，只看一整組是否有在LB、自己的OOF收到效果，未來可以將特徵分群進一步縮小，進而過濾掉包含在大群特徵中的雜訊特徵。

2. 目前只有針對測試集只有做統計，並沒有分析資料集中是否有不完整比賽等雜訊，且多數桌球比賽約在5-6拍便會結束，可以嘗試過濾掉大於10拍的rally，這些可能會影響模型的分類判斷。

3. 只有要預測第二拍與第三拍的訓練資料，因為資訊量太少，導致預測時actin與point的accurancy明顯相較預測第5到第20拍的來的低，且test中要預測第二與第三拍的資料佔了45%，導致在LB上分數不高，但因為預測短拍本來就會比較困難，未來可能調整將預測n=2,3,4的資料另外獨立訓練一個模型，減少因為受到中長拍資料影響的決策。

Per-strikeNumber 答對率（預測第 strikeNumber+1 拍）:
    sn      樣本數  action_acc  point_acc
     1   30,382      0.6967     0.5954
     2   26,488      0.7721     0.6256
     3   21,180      0.7941     0.6329
     4   15,115      0.7995     0.6510
     5   11,135      0.7976     0.6506
     6    7,926      0.8121     0.6665
     7    5,907      0.8163     0.6562
     8    4,303      0.8206     0.6788
     9    3,262      0.8375     0.6934

4. 目前獨立訓練三個模型對應三項任務，但action的結果與point的預測高度相關，攻擊與防守會出現的落點會有所不同，未來會嘗試針對每一筆訓練資料中，point的特徵新增一欄下一拍的action，預測test時也先預測該筆的action，將這個結果加入point特徵進行預測。保留預測拍中action與point的關聯性

