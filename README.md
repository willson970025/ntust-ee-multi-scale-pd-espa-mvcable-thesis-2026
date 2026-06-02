# 應用高效金字塔擠壓注意力機制於中壓電纜局部放電圖譜之多尺度辨識

**Multi-Scale Recognition of Partial Discharge Patterns in Medium-Voltage Cables Using an Efficient Pyramid Squeeze Attention Mechanism**

> 國立臺灣科技大學　電機工程系碩士論文（2026）

---

## 研究摘要

本研究針對中壓電纜局部放電（Partial Discharge, PD）的相位解析放電圖譜（PRPD），提出基於 **EPSANet-Large**（Efficient Pyramid Squeeze Attention Network）的多尺度辨識方法。透過 PSA 模組之多分組金字塔卷積與通道注意力，結合 Sobel 多通道特徵增強，於七類 PD 圖譜（空洞 AH、碳痕 CT、接頭異常 HT、不規則邊緣 TD、典型電暈 CD、典型內部 ID、典型表面 SD）達到極高辨識準確率。

## 模型架構

- **Backbone**：EPSANet-Large（from scratch）
- **通道配置**：[128, 256, 512, 1024]，PSA Groups：[32, 32, 32, 32]
- **輸入**：180 × 80 × 3
  - 通道 1：翻轉後二值化影像（PRPD 訊號點從黑色翻為白色，讓模型學到正確的訊號分佈）
  - 通道 2：x 方向 Sobel 梯度響應 |G<sub>x</sub>|
  - 通道 3：y 方向 Sobel 梯度響應 |G<sub>y</sub>|
- **可訓練參數**：55,472,839
- **正則化**：Dropout 0.2、L2 Weight Decay 1e-5、Label Smoothing 0.05
- **優化器**：SGD（momentum 0.9, nesterov）+ Warmup + Cosine Decay
- **資料擴增**：14× 循環水平位移 + 隨機底部噪聲（噪聲類型 1–8）

## 資料使用邏輯（重要）

為避免「擴增驗證集」造成 val_accuracy 失真，本研究在主模型與所有對照模型上均採用以下邏輯：

| 階段 | 是否擴增 | 用途 |
|---|---|---|
| 訓練集（Train） | 擴增 14× | 模型參數更新（每 epoch 隨機相位平移 + 隨機底部噪聲） |
| 驗證集（Val） | **未擴增** | `validation_data` / `ModelCheckpoint(monitor='val_accuracy')` 之最佳權重選取唯一依據 |
| 原始測試集（Test） | 未擴增 | **主要效能依據** |
| 擴增測試集 | 擴增 14× | 僅作相位平移與雜訊擾動下的「穩健性觀察」，非主要效能 |

`val_accuracy` 反映模型於原始相位分佈上的表現，而非擴增資料；最佳權重亦依未擴增驗證集挑選。對照模型（ResNet18、ResNet50）沿用相同邏輯以確保公平比較。

## 環境需求

| 套件 | 版本 |
|---|---|
| Python | 3.10+ |
| TensorFlow | 2.20.0 |
| Keras | 隨 TF 內建 |
| scikit-learn | 1.3+ |
| OpenCV | 4.8+ |
| matplotlib / seaborn / pandas / numpy | latest |

建議使用 NVIDIA GPU + CUDA 12.x（Mixed Precision Training 已啟用）。

安裝：
```bash
pip install tensorflow==2.20.0 scikit-learn opencv-python matplotlib seaborn pandas numpy
```

## 如何重現訓練

### 主模型（EPSANet-Large）
```bash
cd "all_7class_6(G)_add_noise_mix_train"
python epsanet50_all_in_one.py
```

### 5-Fold 交叉驗證
```bash
cd all_7class_6_add_noise_mix_train_kfold
python epsanet50_all_in_one_kfold.py
```

### 對照模型（第 4.5 節 baseline 比較）
```bash
cd "all_7class_6(G)_add_noise_mix_train/Comparison of Different Models"
python resnet18_all_in_one.py     # ResNet18 baseline
python resnet50_all_in_one.py     # ResNet50 baseline
```
所有對照模型沿用 EPSANet-Large 主模型相同之資料分割、三通道輸入特徵、訓練集擴增策略、未擴增驗證集、最佳權重選取方式與原始測試集評估流程，僅替換模型骨幹。

### 推論 / 批次預測
```bash
cd "all_7class_6(G)_add_noise_mix_train"
python batch_predict_and_classify_performance_0224.py
```

> 注意：訓練所需的 PRPD 圖資料集未隨 repo 釋出，請洽論文作者。模型權重（`.keras`，主模型約 637 MB、各 fold 與對照模型約 141–425 MB）亦未上傳，可依上述指令重新訓練。

## 訓練成果摘要

### 主模型（單次訓練，EPSANet-Large）
| 指標 | 數值 |
|---|---|
| Best Val Accuracy（未擴增驗證集） | **0.9978**（Epoch 67） |
| 原始測試集 Accuracy（主要效能） | **0.9995** |
| 平均 F1 | **0.9996** |
| 擴增測試集 Accuracy（25,522 張，穩健性觀察） | 0.9981 |
| 訓練時長 | 2h 58m |

### 5-Fold 交叉驗證
| Fold | Val Accuracy |
|---|---|
| 1 | 0.9981 |
| 2 | 0.9976 |
| 3 | 0.9961 |
| 4 | 0.9985 |
| 5 | 0.9990 |
| **平均** | **0.9979 ± 0.0010** |

測試集（Fold 5 模型）：原始 0.9984 / 擴增 0.9980。

### 各類別表現（原始測試集）
| 類別 | F1 (單次) | F1 (5-Fold) | Support |
|---|---|---|---|
| AH 空洞 | 0.9987 | 0.9961 | 386 |
| CT 碳痕 | 0.9987 | 0.9960 | 372 |
| HT 接頭異常 | 1.0000 | 1.0000 | 237 |
| TD 不規則邊緣 | 1.0000 | 1.0000 | 288 |
| CD 典型電暈 | 1.0000 | 1.0000 | 270 |
| ID 典型內部 | 1.0000 | 1.0000 | 135 |
| SD 典型表面 | 1.0000 | 1.0000 | 135 |

### 不同模型骨幹比較（第 4.5 節）

固定相同資料分割、三通道輸入、未擴增驗證集與最佳權重選取流程，僅替換 backbone：

| 模型 | 參數量 | Best Epoch | Val Acc | 原始 Test Acc | Macro F1 | 擴增 Test Acc | 訓練時間 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **EPSANet-Large（本研究）** | 55,472,839 | 67 | **0.9978** | **0.9995** | **0.9996** | 0.9981 | 2h 58m |
| ResNet50 | 26,137,671 | 62 | 0.9983 | 0.9989 | 0.9992 | 0.9982 | 1h 0m |
| ResNet18 | 12,233,287 | 18 | 0.9978 | 0.9973 | 0.9981 | 0.9968 | 0h 44m |

EPSANet-Large 在原始測試集 Accuracy 與 Macro F1 上同時取得最高，驗證 PSA 多分組金字塔注意力於 PRPD 多尺度特徵之有效性。

## Repo 結構

```
.
├── all_7class_6(G)_add_noise_mix_train/         # 主模型訓練
│   ├── epsanet50_all_in_one.py                  # 主訓練程式
│   ├── batch_predict_and_classify*.py           # 推論 / 批次分類
│   ├── visualize_model_architecture.py          # 架構視覺化
│   ├── demo_preprocessing.py / demo_preprocessing/
│   ├── model_architecture_output/               # 架構輸出
│   ├── tsne/                                    # t-SNE 視覺化
│   ├── training_log.txt
│   ├── *.png                                    # confusion matrix / training curves
│   └── Comparison of Different Models/          # 不同模型骨幹比較（第 4.5 節）
│       ├── resnet18_all_in_one.py               # ResNet18 baseline 訓練程式
│       ├── resnet50_all_in_one.py               # ResNet50 baseline 訓練程式
│       ├── model_compare_resnet18/              # ResNet18 結果（權重 .keras 已排除）
│       │   ├── training_log.txt
│       │   ├── model_result_summary.csv
│       │   └── *.png                            # confusion matrix / training curves / t-SNE
│       └── model_compare_resnet50/              # ResNet50 結果（權重 .keras 已排除）
│           ├── training_log.txt
│           ├── model_result_summary.csv
│           └── *.png
└── all_7class_6_add_noise_mix_train_kfold/      # 5-Fold 交叉驗證
    ├── epsanet50_all_in_one_kfold.py
    ├── fold_{1..5}/                             # 各 fold 結果 PNG（權重 .keras 已排除）
    ├── kfold_report.txt
    └── *.png
```

## 引用

```bibtex
@mastersthesis{multi_scale_pd_epsa_2026,
  title  = {應用高效金字塔擠壓注意力機制於中壓電纜局部放電圖譜之多尺度辨識},
  author = {(作者姓名待補)},
  school = {國立臺灣科技大學 電機工程系},
  year   = {2026},
  type   = {碩士論文}
}
```

## 授權

本 repo 為論文研究用途，正式授權方式待論文完成後確認。

---
*Repo 目前為 Public；論文發表後將視情況調整為 Private。*
