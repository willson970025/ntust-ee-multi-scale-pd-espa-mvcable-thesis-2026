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
cd "all_7class_7(G)_add_noise_mix_train_kfold"
python epsanet50_all_in_one_kfold.py
```

### 對照模型（第 4.5 節 baseline 比較）
```bash
cd "all_7class_6(G)_add_noise_mix_train/Comparison of Different Models"
python resnet18_all_in_one.py     # ResNet18 baseline
python resnet50_all_in_one.py     # ResNet50 baseline
```
所有對照模型沿用 EPSANet-Large 主模型相同之資料分割、三通道輸入特徵、訓練集擴增策略、未擴增驗證集、最佳權重選取方式與原始測試集評估流程，僅替換模型骨幹。

### 消融實驗（第 4.6 節）
```bash
cd "all_7class_6(G)_add_noise_mix_train"
python epsanet50_ablation_all_in_one.py
```
依序執行注意力消融、輸入通道消融、雜訊穩健性三組共 6 個變體；產出之 `ablation_results/ablation_summary.csv` 與 `ablation_results/ablation_summary_for_thesis.md` 即為論文第 4.6 節表 4-5 / 4-6 / 4-7 數據來源。

### 消融實驗 Grad-CAM 同圖跨模型比較（第 4.6 節 輔助可解釋性）
```bash
cd "all_7class_6(G)_add_noise_mix_train"
python epsanet50_ablation_gradcam_compare.py --auto_select
```
本程式不重新訓練模型，僅讀取消融實驗已訓練好的 `best_model_*.keras`，對同一張原始（未擴增）測試圖譜產生不同消融設定下的 Grad-CAM，輸出至 `ablation_results/gradcam_compare/`。需指定圖片時可改用 `--image_path <path> --true_label <CT|AH|...>`，或以 `--target_mode true` 將 Grad-CAM 目標類別改為真實類別。

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
| Fold | Best Epoch | Val Accuracy | Macro F1 |
|---|---:|---:|---:|
| 1 | 53 | 0.9990 | 0.9993 |
| 2 | 15 | 0.9976 | 0.9982 |
| 3 | 15 | 0.9976 | 0.9976 |
| 4 | 58 | 0.9985 | 0.9990 |
| 5 | 46 | 0.9976 | 0.9983 |
| **平均** |  | **0.9981 ± 0.0006** | **0.9985 ± 0.0006** |

**整體 Out-of-Fold (OOF) 結果（10,296 張）**：Accuracy **0.9981**、Macro F1 **0.9985**。

**固定測試集（1,823 張，五個 Fold 模型平均 ± 標準差）**：
| 指標 | 原始測試集 | 擴增測試集 |
|---|---:|---:|
| Accuracy | **0.9981 ± 0.0008** | 0.9979 ± 0.0004 |
| Macro F1 | **0.9987 ± 0.0006** | 0.9986 ± 0.0003 |

> 各 fold 之原始測試集 Accuracy：1=0.9973、2=0.9984、3=0.9984、4=0.9995、5=0.9973。

### 各類別表現（原始測試集）
| 類別 | F1 (單次主模型) | F1 (5-Fold OOF) | Support |
|---|---|---|---|
| AH 空洞 | 0.9987 | 0.9961 | 386 |
| CT 碳痕 | 0.9987 | 0.9960 | 372 |
| HT 接頭異常 | 1.0000 | 1.0000 | 237 |
| TD 不規則邊緣 | 1.0000 | 0.9994 | 288 |
| CD 典型電暈 | 1.0000 | 0.9993 | 270 |
| ID 典型內部 | 1.0000 | 1.0000 | 135 |
| SD 典型表面 | 1.0000 | 0.9987 | 135 |

> 5-Fold OOF F1 來自 `all_7class_7(G)_add_noise_mix_train_kfold/overall_oof_classification_report.csv`，涵蓋 10,296 張驗證折樣本。

### 不同模型骨幹比較（第 4.5 節）

固定相同資料分割、三通道輸入、未擴增驗證集與最佳權重選取流程，僅替換 backbone：

| 模型 | 參數量 | Best Epoch | Val Acc | 原始 Test Acc | Macro F1 | 擴增 Test Acc | 訓練時間 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **EPSANet-Large（本研究）** | 55,472,839 | 67 | **0.9978** | **0.9995** | **0.9996** | 0.9981 | 2h 58m |
| ResNet50 | 26,137,671 | 62 | 0.9983 | 0.9989 | 0.9992 | 0.9982 | 1h 0m |
| ResNet18 | 12,233,287 | 18 | 0.9978 | 0.9973 | 0.9981 | 0.9968 | 0h 44m |

EPSANet-Large 在原始測試集 Accuracy 與 Macro F1 上同時取得最高，驗證 PSA 多分組金字塔注意力於 PRPD 多尺度特徵之有效性。

### 消融實驗（第 4.6 節）

> **前處理一致性聲明**：為使 1 / 2 / 3 通道輸入消融具備公平比較基礎，本消融實驗統一採用通道數無關之前處理方式 `generic_preprocess`，即 `(x − 127.5) / 127.5`。故本節完整三通道模型之數值主要作為消融實驗內部相對比較基準，**不取代第 4.3 節主要模型效能結果**（4.3 節使用 ResNet preprocess）。隨機種子固定為 42。

#### 表 4-5　注意力機制消融

| 模型配置 | 原始 Test Acc | Macro F1 | 參數量 |
|---|---:|---:|---:|
| **EPSANet-Large（含 EPSA 機制）** | **99.78%** | **99.85%** | 55.47 M |
| 移除跨尺度注意力（保留四尺度卷積） | 99.73% | 99.81% | 55.43 M |

#### 表 4-6　多通道輸入特徵消融

| 輸入配置 | 通道數 | 原始 Test Acc | Macro F1 |
|---|:---:|---:|---:|
| 僅翻轉後二值化 | 1 | 99.89% | 99.92% |
| 翻轉後二值化 + \|G<sub>x</sub>\| | 2 | 99.78% | 99.85% |
| 翻轉後二值化 + \|G<sub>y</sub>\| | 2 | 99.67% | 99.77% |
| 翻轉後二值化 + \|G<sub>x</sub>\| + \|G<sub>y</sub>\| | 3 | 99.78% | 99.85% |

> 註：本表為輸入特徵組合之消融。為避免不同通道數造成前處理不一致，所有配置均採相同之通道數無關前處理；故重點為 Sobel 方向性通道加入前後之相對變化。

#### 表 4-7　雜訊條件下穩健性

| 測試條件 | 準確率 | Macro F1 |
|---|---:|---:|
| 原始測試集 | 99.78% | 99.85% |
| 擴增測試集（含相位平移與雜訊） | 99.61% | 99.71% |

> 穩健性差異 robustness_gap（擴增 − 原始 準確率）= **−0.17%**

> 完整 6 個變體（baseline_epsa_3ch、no_attention_3ch、epsa_inv_only、epsa_inv_gx、epsa_inv_gy、epsa_inv_gx_gy）之逐項指標、confusion matrix、training curves 與 per-sample predictions 均收錄於 `ablation_results/<variant>/`。其中 `epsa_inv_gx_gy` 與 `baseline_epsa_3ch` 為同一實驗（`reused_from = baseline_epsa_3ch`），為論文表格對應之完整性同時保留。

#### 同圖跨模型 Grad-CAM 比較（輔助可解釋性）

針對同一張原始（未擴增）測試圖譜（範例選用 AH 空洞類）疊加各消融模型之 Grad-CAM 熱力圖，輸出收錄於 `ablation_results/gradcam_compare/`，作為論文第 4.6 節之輔助可解釋性觀察：

| 模型 | 輸入模式 | Pred | Confidence |
|---|---|---|---:|
| baseline_epsa_3ch | inv + \|G<sub>x</sub>\| + \|G<sub>y</sub>\| | AH | 99.09% |
| no_attention_3ch | inv + \|G<sub>x</sub>\| + \|G<sub>y</sub>\| | AH | 97.22% |
| epsa_inv_only | inv | AH | 95.82% |
| epsa_inv_gx | inv + \|G<sub>x</sub>\| | AH | 95.88% |
| epsa_inv_gy | inv + \|G<sub>y</sub>\| | AH | 96.44% |

> Grad-CAM 為定性視覺化工具：各模型 heatmap 各自正規化至 0–1，顏色僅代表「該模型內部相對關注區域」，不可跨模型作絕對強度比較，亦不可單獨作為模型優劣之證據。

## Repo 結構

```
.
├── all_7class_6(G)_add_noise_mix_train/         # 主模型訓練
│   ├── epsanet50_all_in_one.py                  # 主訓練程式
│   ├── epsanet50_ablation_all_in_one.py         # 消融訓練程式（六變體一鍵跑完）
│   ├── epsanet50_ablation_gradcam_compare.py    # 同圖 Grad-CAM 跨模型比較（inference only）
│   ├── batch_predict_and_classify*.py           # 推論 / 批次分類
│   ├── visualize_model_architecture.py          # 架構視覺化
│   ├── demo_preprocessing.py / demo_preprocessing/
│   ├── model_architecture_output/               # 架構輸出
│   ├── tsne/                                    # t-SNE 視覺化
│   ├── training_log.txt
│   ├── *.png                                    # confusion matrix / training curves
│   ├── Comparison of Different Models/          # 不同模型骨幹比較（第 4.5 節）
│   │   ├── resnet18_all_in_one.py               # ResNet18 baseline 訓練程式
│   │   ├── resnet50_all_in_one.py               # ResNet50 baseline 訓練程式
│   │   ├── model_compare_resnet18/              # ResNet18 結果（權重 .keras 已排除）
│   │   │   ├── training_log.txt
│   │   │   ├── model_result_summary.csv
│   │   │   └── *.png                            # confusion matrix / training curves / t-SNE
│   │   └── model_compare_resnet50/              # ResNet50 結果（權重 .keras 已排除）
│   │       ├── training_log.txt
│   │       ├── model_result_summary.csv
│   │       └── *.png
│   └── ablation_results/                        # 消融實驗（第 4.6 節，權重 .keras 已排除）
│       ├── epsanet50_ablation_all_in_one.py     # 消融訓練程式副本（同父目錄；保留供獨立執行）
│       ├── ablation_summary.csv                 # 六變體彙整指標
│       ├── ablation_summary_for_thesis.md       # 表 4-5 / 4-6 / 4-7 直接複製版
│       ├── baseline_epsa_3ch/                   # 含 EPSA + 三通道（4-5 主對照）
│       ├── no_attention_3ch/                    # 移除跨尺度注意力（4-5）
│       ├── epsa_inv_only/                       # 僅翻轉影像，1 通道（4-6）
│       ├── epsa_inv_gx/                         # 翻轉 + |G_x|，2 通道（4-6）
│       ├── epsa_inv_gy/                         # 翻轉 + |G_y|，2 通道（4-6）
│       ├── epsa_inv_gx_gy/                      # 翻轉 + |G_x| + |G_y|（reused_from baseline_epsa_3ch）
│       └── gradcam_compare/                     # 同圖跨模型 Grad-CAM 比較輸出（PNG + metadata）
└── all_7class_7(G)_add_noise_mix_train_kfold/   # 5-Fold 交叉驗證
    ├── epsanet50_all_in_one_kfold.py
    ├── environment_info.txt / .json             # 實驗環境資訊（論文表 4-1）
    ├── experiment_config.json                   # K-Fold / 訓練 / 資料邏輯設定快照
    ├── fold_class_distribution.csv              # 各 fold 各類別樣本數
    ├── fold_{1..5}/                             # 各 fold 結果（權重 .keras 已排除）
    │   ├── classification_report_fold{n}.csv
    │   ├── confusion_matrix_fold{n}.csv / .png
    │   ├── training_curves_fold{n}.png
    │   └── tsne_*.png                           # GAP / FC1 / FC2 / comparison
    ├── kfold_summary.csv                        # 各 fold val 指標彙整
    ├── kfold_report.txt
    ├── kfold_accuracy_curves.png / kfold_loss_curves.png / kfold_comparison.png
    ├── overall_confusion_matrix.png             # OOF 混淆矩陣
    ├── overall_oof_classification_report.csv    # OOF 各類別指標
    ├── overall_oof_confusion_matrix.csv
    ├── overall_oof_metrics.json                 # OOF 總體指標
    ├── test_metrics_by_fold.csv                 # 五 fold 模型於固定測試集之指標
    └── test_confusion_matrix_original.png / _augmented.png
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
