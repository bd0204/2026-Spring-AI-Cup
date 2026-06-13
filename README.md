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
LGBM是甚麼、特性、怎麼利用特性
模型參數選擇、leaves
輸出三個子模型與三個權重
## Innovation
不加入match特徵
id的遮蔽策略差異
在分類型的模型加入lag特徵
連續三拍的特徵讓分類器學習到戰術模式
發球站位加入到全資料中，作為左右撇子的依據


## Data Process
資料的類別權重調整
id與match的編號偏移，不與原始train相撞
由test的比例調整id遮蔽策略

## Approach
以match作為fold切分單位
以基本訊息 + 前四拍的資訊為訓練資料，預測下一拍
訓練兩個不同的模型，一個有加入id，一個沒有加入，訓練兩種不同能力的模型
以70 30合成機率矩陣，代表完全沒有依據的一方需要非常確定才能扭轉結果

## Analysis and Conclusion
內容規定：分析所使用的模型及其成效，簡述未來可能改進的方向。分析必須附圖，可將幾個成功的和失敗的例子附上並說明之。
