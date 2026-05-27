# 應用高效金字塔擠壓注意力機制於中壓電纜局部放電圖譜之多尺度辨識

**Multi-Scale Recognition of Partial Discharge Patterns in Medium-Voltage Cables Using an Efficient Pyramid Squeeze Attention Mechanism**

> 國立臺灣科技大學　電機工程系碩士論文（2026）

---

## 研究摘要

本研究針對中壓電纜局部放電（Partial Discharge, PD）的相位解析放電圖譜（PRPD），提出基於 **EPSANet-Large**（Efficient Pyramid Squeeze Attention Network）的多尺度辨識方法。透過 PSA 模組之多分組金字塔卷積與通道注意力，結合 Sobel 多通道特徵增強，於七類 PD 圖譜（空洞 AH、碳痕 CT、接頭異常 HT、不規則邊緣 TD、典型電暈 CD、典型內部 ID、典型表面 SD）達到極高辨識準確率。

## 模型架構

- **Backbone**：EPSANet-Large（from scratch）
- **通道配置**：[128, 256, 512, 1024]，PSA Groups：[32, 32, 32, 32]
- **輸入**：180 × 80 × 3（原始圖 + Sobel X + Sobel Y）
- **可訓練參數**：55,472,839
- **正則化**：Dropout 0.2、L2 Weight Decay 1e-5、Label Smoothing 0.05
- **優化器**：SGD（momentum 0.9, nesterov）+ Warmup + Cosine Decay
- **資料擴增**：14× 循環水平位移 + 隨機底部噪聲

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

### 單次訓練（主模型）
```bash
cd "all_7class_6(G)_add_noise_mix_train"
python epsanet50_all_in_one.py
```

### 5-Fold 交叉驗證
```bash
cd all_7class_6_add_noise_mix_train_kfold
python epsanet50_all_in_one_kfold.py
```

### 推論 / 批次預測
```bash
cd "all_7class_6(G)_add_noise_mix_train"
python batch_predict_and_classify_performance_0224.py
```

> 注意：訓練所需的 PRPD 圖資料集未隨 repo 釋出，請洽論文作者。模型權重（`.keras`，每個約 425–637 MB）亦未上傳，可依上述指令重新訓練。

## 訓練成果摘要

### 主模型（單次訓練）
| 指標 | 數值 |
|---|---|
| Best Val Accuracy | **0.9978**（Epoch 67） |
| 原始測試集 Accuracy | **0.9995** |
| 平均 F1 | **0.9996** |
| 擴增測試集 Accuracy（25,522 張） | 0.9981 |
| 訓練時長 | 3h 20m |

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
│   └── *.png                                    # confusion matrix / training curves
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
