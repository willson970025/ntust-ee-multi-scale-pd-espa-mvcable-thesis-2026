# -*- coding: utf-8 -*-
"""
================================================================================
第 4.6 節「消融實驗」專用程式 (EPSANet-Large Ablation Study)
================================================================================

本檔案說明（重要）：
- 此檔為論文「第 4.6 節 消融實驗」專用，由 `epsanet50_all_in_one.py` 複製改寫。
- 原始主程式 `epsanet50_all_in_one.py` 未被修改，本檔僅「複製並參數化」其既有流程，
  並未重寫訓練邏輯。
- 公平性與論文一致性原則（與主程式相同）：
    * 原始（未擴增）測試集 test_ds 為「主要效能依據」。
    * 擴增測試集 test_ds_augmented 僅供 4.6.3「穩健性觀察」，不得作為主要 accuracy。
    * validation_data 一律使用「未擴增」驗證集 val_ds。
    * ModelCheckpoint 監看「未擴增驗證集」的 val_accuracy 來挑選最佳權重，
      使最佳權重反映原始相位分佈下的驗證表現。
    * 不改變 train/val/test 分割、不改變資料夾讀取邏輯、test 不混入 train/val。
    * 每個消融實驗都「重新建立並初始化模型權重」，不沿用前一個實驗的權重。
    * 除被消融的因素（注意力機制 / 輸入通道）外，其餘訓練條件完全一致。

本檔支援兩個消融維度：
  A. 注意力機制：attention_mode ∈ {"epsa", "no_attention"}
       - "epsa"        ：原始 PSAModule（四尺度金字塔卷積 + SE + 跨尺度 softmax 注意力）
       - "no_attention"：PSAModuleNoAttention（保留四尺度金字塔卷積與 grouped conv，
                         但移除 SEWeightModule 與跨尺度 softmax 注意力，直接 concat 四分支）
  B. 多通道輸入：input_mode ∈ {"inv_only", "inv_gx", "inv_gy", "inv_gx_gy"}
       - "inv_only"  ：[I_inv]                 -> (80,180,1)
       - "inv_gx"    ：[I_inv, |G_x|]          -> (80,180,2)
       - "inv_gy"    ：[I_inv, |G_y|]          -> (80,180,2)
       - "inv_gx_gy" ：[I_inv, |G_x|, |G_y|]   -> (80,180,3)  （主模型 baseline）

執行方式：
  python3 epsanet50_ablation_all_in_one.py --list_experiments       # 只列出實驗
  python3 epsanet50_ablation_all_in_one.py --exp baseline_epsa_3ch  # 跑單一實驗
  python3 epsanet50_ablation_all_in_one.py --run_all                # 跑所有實驗

================================================================================
"""

# ============================================================================
# 抑制 TensorFlow/XLA 編譯器警告訊息（與主程式一致）
# ============================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


import glob
import json
import math
import shutil
import argparse
import random
import numpy as np

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# [優化] 啟用 Mixed Precision Training（與主程式一致）
tf.keras.mixed_precision.set_global_policy('mixed_float16')

import keras
from keras import layers, models, Model
from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
from keras.callbacks import ModelCheckpoint
from keras.metrics import Precision, Recall
# 注意：本檔「刻意不」import ResNet50 的 preprocess_input。
#       原因見 generic_preprocess() 之說明（需支援 1/2/3 通道，故改用通用前處理）。

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import time
from datetime import datetime, timedelta


# ============================================================================
# 中文字型（與主程式一致）
# ============================================================================
def setup_chinese_font():
    font_candidates = [
        'Microsoft YaHei', 'Noto Sans CJK TC', 'Noto Sans CJK SC',
        'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans',
    ]
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            print(f"使用中文字型: {font}")
            return font
    print("警告：找不到中文字型")
    return None

setup_chinese_font()
plt.rcParams['axes.unicode_minus'] = False

# 啟用動態記憶體增長（與主程式一致）
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# ============================================================================
# 全域隨機種子 / Determinism（規範第十條：固定為 42）
# ============================================================================
RANDOM_STATE = 42

# 嘗試啟用 op-level determinism；若失敗（某些 GPU/XLA 組合不支援）則於 log 註記。
DETERMINISM_NOTE = ""


if gpus:
    DETERMINISM_NOTE += (
    "本消融實驗固定 RANDOM_STATE=42，並設定 random / numpy / tensorflow 種子；"
    "但未啟用 tf.config.experimental.enable_op_determinism()，"
    "以避免 GPU MaxPool gradient deterministic kernel 與訓練流程衝突。"
    "因此 GPU 訓練結果可能仍存在極小浮點差異。"
    )
else:
    DETERMINISM_NOTE += " 目前未偵測到 GPU。"


def set_global_seeds(seed=RANDOM_STATE):
    """在每個實驗開始前重設所有隨機種子，確保各實驗的權重初始化條件一致、可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)  # 一次設定 python / numpy / tensorflow 種子


# ============================================================================
# 全域常數（與主程式一致，不得任意更動）
# ============================================================================
WEIGHT_DECAY = 1e-5
INPUT_HEIGHT = 80
INPUT_WIDTH = 180
DROPOUT_RATE = 0.2
USE_MULTICHANNEL = True   # 本檔以 input_mode 控制實際通道，此旗標僅保留語意
SOBEL_KSIZE = 3

# 訓練超參數（規範第二條，全部沿用主程式）
batch_size = 128
epochs = 100
target_lr = 0.002
min_lr = 1e-5
LABEL_SMOOTHING = 0.05
AUGMENT_FACTOR = 14
NOISE_COUNT = 8

# 類別定義（與主程式一致，不得更動）
class_names = ['AH', 'CT', 'HT', 'TD', 'CD', 'ID', 'SD']
num_classes = len(class_names)
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
chinese_labels = ['空洞', '碳痕', '接頭異常', '不規則邊緣', '典型電暈放電', '典型內部放電', '典型表面放電']

# 資料路徑（沿用主程式 data_dir，不得更動）
data_dir = '/home/cckuo/m11307u09/GG/1126/data_original G'
# 主程式輸出資料夾（保持不動，避免覆蓋主程式結果）
main_output_dir = '/home/cckuo/m11307u09/GG/1126/epsanet50/all in one/all_7class_7(G)_add_noise_mix_train/ablation'
# [消融] 所有消融結果輸出根目錄（與主程式輸出同層的獨立資料夾，避免互相覆蓋）
output_root = os.path.join(os.path.dirname(main_output_dir), 'ablation_results')

# [一致性說明] 前處理公平性註記（於程式註解、training_log、論文 md 共用同一段文字）
PREPROCESS_FAIRNESS_NOTE = (
    "為使 1/2/3 通道輸入消融具備公平比較基礎，本消融實驗統一採用通道數無關之前處理方式 "
    "generic_preprocess，即 (x - 127.5) / 127.5。故本節完整三通道模型之數值主要作為消融實驗"
    "內部相對比較基準，不取代 4.3 節主要模型效能結果。"
)

# input_mode -> 通道數
INPUT_MODE_CHANNELS = {
    'inv_only': 1,
    'inv_gx': 2,
    'inv_gy': 2,
    'inv_gx_gy': 3,
}
INPUT_MODE_DESC = {
    'inv_only': '僅翻轉後二值化 I_inv',
    'inv_gx': '翻轉後二值化 + |G_x|',
    'inv_gy': '翻轉後二值化 + |G_y|',
    'inv_gx_gy': '翻轉後二值化 + |G_x| + |G_y|（三通道）',
}


# ============================================================================
# EPSANet-Large 模型定義（與主程式一致；新增 No-Attention 變體）
# ============================================================================
@keras.saving.register_keras_serializable()
class SEWeightModule(layers.Layer):
    """Squeeze-and-Excitation Weight Module（與主程式完全一致）"""
    def __init__(self, channels, reduction=16, weight_decay=WEIGHT_DECAY, **kwargs):
        super(SEWeightModule, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.weight_decay = weight_decay

    def build(self, input_shape):
        reduced_channels = max(self.channels // self.reduction, 1)
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = layers.Conv2D(
            reduced_channels, kernel_size=1, padding='same', use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.weight_decay)
        )
        self.relu = layers.ReLU()
        self.fc2 = layers.Conv2D(
            self.channels, kernel_size=1, padding='same', use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.weight_decay)
        )
        self.sigmoid = layers.Activation('sigmoid')
        super().build(input_shape)

    def call(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        return weight

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'reduction': self.reduction,
            'weight_decay': self.weight_decay
        })
        return config


def _get_valid_groups(in_ch, out_ch, desired_groups):
    """動態降低 grouped-conv 的 group 數以整除通道（與主程式 PSAModule 內邏輯一致）。"""
    valid = desired_groups
    while valid > 1:
        if in_ch % valid == 0 and out_ch % valid == 0:
            return valid
        valid -= 1
    return 1


@keras.saving.register_keras_serializable()
class PSAModule(layers.Layer):
    """Pyramid Squeeze Attention Module（baseline，與主程式完全一致）"""
    def __init__(self, in_channels, out_channels, stride=1,
                 conv_kernels=[3, 5, 7, 9], conv_groups=[32, 32, 32, 32],
                 weight_decay=WEIGHT_DECAY, **kwargs):
        super(PSAModule, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv_kernels = conv_kernels
        self.conv_groups = conv_groups
        self.weight_decay = weight_decay
        self.split_channel = out_channels // 4

    def build(self, input_shape):
        g0 = _get_valid_groups(self.in_channels, self.split_channel, self.conv_groups[0])
        g1 = _get_valid_groups(self.in_channels, self.split_channel, self.conv_groups[1])
        g2 = _get_valid_groups(self.in_channels, self.split_channel, self.conv_groups[2])
        g3 = _get_valid_groups(self.in_channels, self.split_channel, self.conv_groups[3])

        self.conv_1 = layers.Conv2D(
            self.split_channel, kernel_size=self.conv_kernels[0],
            padding='same', strides=self.stride, groups=g0, use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.weight_decay)
        )
        self.conv_2 = layers.Conv2D(
            self.split_channel, kernel_size=self.conv_kernels[1],
            padding='same', strides=self.stride, groups=g1, use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.weight_decay)
        )
        self.conv_3 = layers.Conv2D(
            self.split_channel, kernel_size=self.conv_kernels[2],
            padding='same', strides=self.stride, groups=g2, use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.weight_decay)
        )
        self.conv_4 = layers.Conv2D(
            self.split_channel, kernel_size=self.conv_kernels[3],
            padding='same', strides=self.stride, groups=g3, use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.weight_decay)
        )
        self.se = SEWeightModule(self.split_channel, weight_decay=self.weight_decay)
        super().build(input_shape)

    def call(self, x):
        x1, x2, x3, x4 = self.conv_1(x), self.conv_2(x), self.conv_3(x), self.conv_4(x)
        x1_se, x2_se, x3_se, x4_se = self.se(x1), self.se(x2), self.se(x3), self.se(x4)

        x_se = tf.concat([x1_se, x2_se, x3_se, x4_se], axis=-1)
        x_se = tf.reshape(x_se, [-1, 4, self.split_channel])
        attention_vectors = tf.nn.softmax(x_se, axis=1)

        att1 = tf.reshape(attention_vectors[:, 0:1, :], [-1, 1, 1, self.split_channel])
        att2 = tf.reshape(attention_vectors[:, 1:2, :], [-1, 1, 1, self.split_channel])
        att3 = tf.reshape(attention_vectors[:, 2:3, :], [-1, 1, 1, self.split_channel])
        att4 = tf.reshape(attention_vectors[:, 3:4, :], [-1, 1, 1, self.split_channel])

        out = tf.concat([x1 * att1, x2 * att2, x3 * att3, x4 * att4], axis=-1)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels, 'out_channels': self.out_channels,
            'stride': self.stride, 'conv_kernels': self.conv_kernels,
            'conv_groups': self.conv_groups, 'weight_decay': self.weight_decay
        })
        return config


@keras.saving.register_keras_serializable()
class PSAModuleNoAttention(layers.Layer):
    """
    [消融 A] 無跨尺度注意力版本的 PSA。

    與 PSAModule 的唯一差異：
      - 仍保留四個並行分支卷積 conv_1~conv_4（四尺度金字塔卷積）。
      - 仍保留各分支的 grouped convolution 設定邏輯（_get_valid_groups）。
      - 移除 SEWeightModule 與跨尺度 softmax 注意力加權。
      - call() 直接 concat([x1, x2, x3, x4], axis=-1)。
    目的：只拿掉「跨尺度注意力加權」，不拿掉「多尺度卷積」，
          以單純比較注意力機制本身的貢獻。
    """
    def __init__(self, in_channels, out_channels, stride=1,
                 conv_kernels=[3, 5, 7, 9], conv_groups=[32, 32, 32, 32],
                 weight_decay=WEIGHT_DECAY, **kwargs):
        super(PSAModuleNoAttention, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv_kernels = conv_kernels
        self.conv_groups = conv_groups
        self.weight_decay = weight_decay
        self.split_channel = out_channels // 4

    def build(self, input_shape):
        g0 = _get_valid_groups(self.in_channels, self.split_channel, self.conv_groups[0])
        g1 = _get_valid_groups(self.in_channels, self.split_channel, self.conv_groups[1])
        g2 = _get_valid_groups(self.in_channels, self.split_channel, self.conv_groups[2])
        g3 = _get_valid_groups(self.in_channels, self.split_channel, self.conv_groups[3])

        self.conv_1 = layers.Conv2D(
            self.split_channel, kernel_size=self.conv_kernels[0],
            padding='same', strides=self.stride, groups=g0, use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.weight_decay)
        )
        self.conv_2 = layers.Conv2D(
            self.split_channel, kernel_size=self.conv_kernels[1],
            padding='same', strides=self.stride, groups=g1, use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.weight_decay)
        )
        self.conv_3 = layers.Conv2D(
            self.split_channel, kernel_size=self.conv_kernels[2],
            padding='same', strides=self.stride, groups=g2, use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.weight_decay)
        )
        self.conv_4 = layers.Conv2D(
            self.split_channel, kernel_size=self.conv_kernels[3],
            padding='same', strides=self.stride, groups=g3, use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.weight_decay)
        )
        # [消融] 不建立 SEWeightModule
        super().build(input_shape)

    def call(self, x):
        x1, x2, x3, x4 = self.conv_1(x), self.conv_2(x), self.conv_3(x), self.conv_4(x)
        # [消融] 不做注意力加權，直接 concat 四個分支
        out = tf.concat([x1, x2, x3, x4], axis=-1)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels, 'out_channels': self.out_channels,
            'stride': self.stride, 'conv_kernels': self.conv_kernels,
            'conv_groups': self.conv_groups, 'weight_decay': self.weight_decay
        })
        return config


@keras.saving.register_keras_serializable()
class EPSABlock(layers.Layer):
    """
    Efficient Pyramid Squeeze Attention Block。

    [消融 A] 新增 attention_mode：
      - "epsa"        -> 使用 PSAModule（含跨尺度注意力）
      - "no_attention"-> 使用 PSAModuleNoAttention（移除注意力，保留四尺度卷積）
    其餘結構與主程式完全一致。
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, use_downsample=False,
                 conv_kernels=[3, 5, 7, 9], conv_groups=[32, 32, 32, 32],
                 weight_decay=WEIGHT_DECAY, attention_mode='epsa', **kwargs):
        super(EPSABlock, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv_kernels = conv_kernels
        self.conv_groups = conv_groups
        self.use_downsample = use_downsample
        self.weight_decay = weight_decay
        self.attention_mode = attention_mode

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(
            self.out_channels, kernel_size=1, use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.weight_decay)
        )
        self.bn1 = layers.BatchNormalization()

        # [消融 A] 依 attention_mode 選用 PSA 模組
        if self.attention_mode == 'no_attention':
            self.psa = PSAModuleNoAttention(
                self.out_channels, self.out_channels, stride=self.stride,
                conv_kernels=self.conv_kernels, conv_groups=self.conv_groups,
                weight_decay=self.weight_decay
            )
        else:  # 'epsa'
            self.psa = PSAModule(
                self.out_channels, self.out_channels, stride=self.stride,
                conv_kernels=self.conv_kernels, conv_groups=self.conv_groups,
                weight_decay=self.weight_decay
            )

        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(
            self.out_channels * self.expansion, kernel_size=1, use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.weight_decay)
        )
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()

        if self.use_downsample:
            self.downsample_conv = layers.Conv2D(
                self.out_channels * self.expansion,
                kernel_size=1, strides=self.stride, use_bias=False,
                kernel_regularizer=keras.regularizers.l2(self.weight_decay)
            )
            self.downsample_bn = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, x, training=None):
        identity = x
        out = self.relu(self.bn1(self.conv1(x), training=training))
        out = self.relu(self.bn2(self.psa(out), training=training))
        out = self.bn3(self.conv3(out), training=training)

        if self.use_downsample:
            identity = self.downsample_bn(self.downsample_conv(x), training=training)

        return self.relu(out + identity)

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels, 'out_channels': self.out_channels,
            'stride': self.stride, 'conv_kernels': self.conv_kernels,
            'conv_groups': self.conv_groups, 'use_downsample': self.use_downsample,
            'weight_decay': self.weight_decay, 'attention_mode': self.attention_mode
        })
        return config


def build_epsanet_large(input_shape=(80, 180, 3), num_classes=7, dropout_rate=0.2,
                        weight_decay=WEIGHT_DECAY, attention_mode='epsa'):
    """
    建立 EPSANet-Large 模型。

    [消融 A] attention_mode 控制各 EPSABlock 是否使用跨尺度注意力：
      - "epsa"        ：原始 PSAModule
      - "no_attention"：PSAModuleNoAttention

    其餘（層命名、kernel 配置、分類頭）與主程式完全一致。
    各層 conv_kernels 配置：
      Layer1 [3,5,7,9] / Layer2 [3,5,7,9] / Layer3 [3,3,5,5] / Layer4 [3,3,3,3]
    """
    conv_groups = [32, 32, 32, 32]
    layers_config = [3, 4, 6, 3]
    channels = [128, 256, 512, 1024]  # EPSANet-Large 通道配置

    layer_kernels = [
        [3, 5, 7, 9],  # Layer1
        [3, 5, 7, 9],  # Layer2
        [3, 3, 5, 5],  # Layer3
        [3, 3, 3, 3],  # Layer4
    ]

    inputs = layers.Input(shape=input_shape)

    # Stem（命名與主程式一致）
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False,
                      kernel_initializer='he_normal', name='stem_conv',
                      kernel_regularizer=keras.regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(name='stem_relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='stem_pool')(x)

    # Layer 1
    in_ch = 64
    for i in range(layers_config[0]):
        use_ds = (i == 0) and (in_ch != channels[0] * 4)
        x = EPSABlock(in_ch if i == 0 else channels[0] * 4, channels[0], stride=1,
                      use_downsample=use_ds, conv_kernels=layer_kernels[0],
                      conv_groups=conv_groups, weight_decay=weight_decay,
                      attention_mode=attention_mode, name=f'layer1_block{i}')(x)

    # Layer 2
    prev_ch = channels[0] * 4
    for i in range(layers_config[1]):
        stride = 2 if i == 0 else 1
        use_ds = (i == 0)
        x = EPSABlock(prev_ch if i == 0 else channels[1] * 4, channels[1], stride=stride,
                      use_downsample=use_ds, conv_kernels=layer_kernels[1],
                      conv_groups=conv_groups, weight_decay=weight_decay,
                      attention_mode=attention_mode, name=f'layer2_block{i}')(x)

    # Layer 3
    prev_ch = channels[1] * 4
    for i in range(layers_config[2]):
        stride = 2 if i == 0 else 1
        use_ds = (i == 0)
        x = EPSABlock(prev_ch if i == 0 else channels[2] * 4, channels[2], stride=stride,
                      use_downsample=use_ds, conv_kernels=layer_kernels[2],
                      conv_groups=conv_groups, weight_decay=weight_decay,
                      attention_mode=attention_mode, name=f'layer3_block{i}')(x)

    # Layer 4
    prev_ch = channels[2] * 4
    for i in range(layers_config[3]):
        stride = 2 if i == 0 else 1
        use_ds = (i == 0)
        x = EPSABlock(prev_ch if i == 0 else channels[3] * 4, channels[3], stride=stride,
                      use_downsample=use_ds, conv_kernels=layer_kernels[3],
                      conv_groups=conv_groups, weight_decay=weight_decay,
                      attention_mode=attention_mode, name=f'layer4_block{i}')(x)

    # Grad-CAM 專用層（恆等函數，與主程式一致）
    x = layers.Activation('linear', name='gradcam_target')(x)

    # 分類頭（命名與主程式一致）
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(dropout_rate, name='head_dropout')(x)
    x = layers.Dense(1024, kernel_regularizer=keras.regularizers.l2(weight_decay), name='head_fc1')(x)
    x = layers.BatchNormalization(name='head_bn1')(x)
    x = layers.ReLU(name='head_relu1')(x)
    x = layers.Dense(512, kernel_regularizer=keras.regularizers.l2(weight_decay), name='head_fc2')(x)
    x = layers.BatchNormalization(name='head_bn2')(x)
    x = layers.ReLU(name='head_relu2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=outputs, name='EPSANet_Large')
    return model


# 自訂預熱學習率調度器（與主程式完全一致）
@keras.saving.register_keras_serializable()
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, target_lr, total_steps, min_lr=0.0):
        super(WarmupCosineDecay, self).__init__()
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.target_lr * step / warmup_steps
        cosine_lr = self.min_lr + (self.target_lr - self.min_lr) * 0.5 * (
            1.0 + tf.cos(tf.constant(np.pi) * (step - warmup_steps) / (total_steps - warmup_steps))
        )
        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            'warmup_steps': self.warmup_steps,
            'target_lr': self.target_lr,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr
        }


# ============================================================================
# 資料載入（與主程式一致，不得更動讀取邏輯）
# ============================================================================
def load_data_from_split_dir(base_dir, class_to_idx):
    """從 train/val/test 目錄載入圖片路徑和標籤（與主程式完全一致）。"""
    image_paths = []
    labels = []
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for subdir in subdirs:
        class_prefix = subdir[:2]
        if class_prefix in class_to_idx:
            class_idx = class_to_idx[class_prefix]
            class_dir = os.path.join(base_dir, subdir)
            paths = glob.glob(os.path.join(class_dir, '*.png'))
            image_paths.extend(paths)
            labels.extend([class_idx] * len(paths))
    return np.array(image_paths), np.array(labels)


# Sobel 卷積核（與主程式完全一致）
_SOBEL_X_KERNEL = tf.reshape(tf.constant(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=tf.float32), [3, 3, 1, 1])
_SOBEL_Y_KERNEL = tf.reshape(tf.constant(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=tf.float32), [3, 3, 1, 1])


def _sobel_abs(gray, kernel):
    """計算 |Sobel| 並正規化到 0~255（正規化方式與主程式完全一致）。"""
    gray_4d = tf.reshape(gray, [1, INPUT_HEIGHT, INPUT_WIDTH, 1])
    s = tf.nn.conv2d(gray_4d, kernel, strides=[1, 1, 1, 1], padding='SAME')
    s = tf.squeeze(s, axis=[0, 3])
    s = tf.abs(s)
    s_max = tf.reduce_max(s)
    s = tf.cond(s_max > 0, lambda: s / s_max * 255.0, lambda: s)
    return s


def generic_preprocess(image):
    """
    通用前處理（取代 ResNet50 的 preprocess_input）。

    為何不用 ResNet50 的 preprocess_input：
      ResNet50 的 preprocess_input（caffe 模式）會做 RGB->BGR 並對「3 個固定通道」
      各自減去 ImageNet 均值，前提是輸入必須是 3 通道的 RGB 影像。
      本消融實驗需要支援 1 / 2 / 3 通道（且通道語意是 I_inv 與 Sobel 梯度，
      並非 RGB），直接套用會不適用且不一致。

    解法：對「所有 input_mode 一致」採用同一個與通道數無關的線性縮放：
      x' = (x - 127.5) / 127.5，將 0~255 的輸入映射到約 [-1, 1]。
      此縮放對 1/2/3 通道完全相同，確保各消融設定之間只有「通道數」差異，
      不引入前處理上的不公平。
    註：因此本消融 baseline（generic_preprocess）與主程式（ResNet caffe 前處理）
        的前處理不同；但消融研究內部所有實驗皆一致，故比較仍公平。

    一致性聲明（與 training_log / ablation_summary_for_thesis.md 同步，見 PREPROCESS_FAIRNESS_NOTE）：
        為使 1/2/3 通道輸入消融具備公平比較基礎，本消融實驗統一採用通道數無關之前處理方式
        generic_preprocess，即 (x - 127.5) / 127.5。故本節完整三通道模型之數值主要作為消融實驗
        內部相對比較基準，不取代 4.3 節主要模型效能結果。
    """
    return (tf.cast(image, tf.float32) - 127.5) / 127.5


def make_load_image(input_mode):
    """
    [消融 B] 依 input_mode 產生對應通道組合的 load_image 函式。

    黑白翻轉、Sobel 計算與正規化方式皆沿用主程式；
    僅最後堆疊的通道組合不同，且不把 1 通道複製成 3 通道。
    """
    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)

        gray = tf.image.rgb_to_grayscale(image)
        gray = tf.squeeze(gray, axis=-1)
        gray = tf.cast(gray, tf.float32)
        gray = 255.0 - gray  # 黑白翻轉（與主程式一致）

        if input_mode == 'inv_only':
            chans = [gray]
        elif input_mode == 'inv_gx':
            chans = [gray, _sobel_abs(gray, _SOBEL_X_KERNEL)]
        elif input_mode == 'inv_gy':
            chans = [gray, _sobel_abs(gray, _SOBEL_Y_KERNEL)]
        elif input_mode == 'inv_gx_gy':
            chans = [gray, _sobel_abs(gray, _SOBEL_X_KERNEL), _sobel_abs(gray, _SOBEL_Y_KERNEL)]
        else:
            raise ValueError(f"未知 input_mode: {input_mode}")

        image = tf.stack(chans, axis=-1)       # (H, W, C)，C = 1/2/3
        image = generic_preprocess(image)      # 一致前處理
        label = tf.one_hot(label, num_classes)
        return image, label

    return load_image


# ============================================================================
# 資料擴增（與主程式一致；改為與「通道數無關」以支援 1/2/3 通道）
# ============================================================================
def fill_rows_vectorized(image, rows_to_fill):
    """向量化塗白多行（與主程式一致；本身即與通道數無關）。"""
    height = tf.shape(image)[0]
    fill_value = tf.reduce_max(image)

    valid_mask = rows_to_fill >= 0
    valid_rows = tf.boolean_mask(rows_to_fill, valid_mask)
    valid_rows = tf.clip_by_value(valid_rows, 0, height - 1)
    num_valid = tf.shape(valid_rows)[0]

    def do_fill():
        row_indices = tf.expand_dims(valid_rows, 1)
        row_mask = tf.scatter_nd(row_indices, tf.ones([num_valid], dtype=tf.bool), [height])
        row_mask_expanded = tf.reshape(row_mask, [height, 1, 1])
        row_mask_broadcast = tf.broadcast_to(row_mask_expanded, tf.shape(image))
        return tf.where(row_mask_broadcast, fill_value, image)

    return tf.cond(num_valid > 0, do_fill, lambda: image)


def add_bottom_noise(image, noise_type):
    """底部噪聲增強 type 1-4（與主程式一致；與通道數無關）。"""
    gray = image[:, :, 0]
    height = tf.shape(image)[0]

    row_has_signal = tf.reduce_any(gray > 0, axis=1)
    row_has_signal_reversed = tf.reverse(row_has_signal, axis=[0])
    signal_indices_reversed = tf.where(row_has_signal_reversed)
    signal_indices_reversed = tf.squeeze(signal_indices_reversed, axis=1)
    signal_rows = (height - 1) - tf.cast(signal_indices_reversed, tf.int32)
    num_signal_rows = tf.shape(signal_rows)[0]

    def get_rows_type_1():
        return tf.cond(num_signal_rows >= 1, lambda: signal_rows[0:1], lambda: tf.constant([-1], tf.int32))

    def get_rows_type_2():
        return tf.cond(num_signal_rows >= 2, lambda: signal_rows[1:2], lambda: tf.constant([-1], tf.int32))

    def get_rows_type_3():
        return tf.cond(num_signal_rows >= 3, lambda: signal_rows[2:3], lambda: tf.constant([-1], tf.int32))

    def get_rows_type_4():
        def get_three_rows():
            start_row = signal_rows[0]
            rows = start_row - tf.range(3, dtype=tf.int32)
            return tf.maximum(rows, -1)
        return tf.cond(num_signal_rows >= 1, get_three_rows, lambda: tf.constant([-1], tf.int32))

    rows_to_fill = tf.case([
        (tf.equal(noise_type, 1), get_rows_type_1),
        (tf.equal(noise_type, 2), get_rows_type_2),
        (tf.equal(noise_type, 3), get_rows_type_3),
        (tf.equal(noise_type, 4), get_rows_type_4),
    ], default=lambda: tf.constant([-1], tf.int32))

    return fill_rows_vectorized(image, rows_to_fill)


def add_random_pixel_noise(image, noise_type):
    """
    隨機像素點噪聲增強 type 5-8（與主程式語意一致）。

    [改寫說明] 主程式原以 c0/c1/c2 三通道分別 scatter 更新；
    本檔改為「以 (H,W) 遮罩廣播到任意通道數」更新，邏輯等價但與通道數無關，
    使 1/2/3 通道皆可使用。被選中的像素一律設為當前最大值（白）。
    """
    gray = image[:, :, 0]
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    fill_value = tf.reduce_max(image)

    row_has_signal = tf.reduce_any(gray > 0, axis=1)
    row_has_signal_reversed = tf.reverse(row_has_signal, axis=[0])
    signal_indices_reversed = tf.where(row_has_signal_reversed)
    signal_indices_reversed = tf.squeeze(signal_indices_reversed, axis=1)
    signal_rows = (height - 1) - tf.cast(signal_indices_reversed, tf.int32)
    num_signal_rows = tf.shape(signal_rows)[0]

    def add_pixels_to_row(img, row_idx, num_pixels):
        valid = tf.logical_and(row_idx >= 0, row_idx < height)

        def do_add():
            actual_pixels = tf.minimum(num_pixels, width)
            x_coords = tf.random.shuffle(tf.range(width))[:actual_pixels]
            y_coords = tf.fill([actual_pixels], row_idx)
            indices = tf.stack([y_coords, x_coords], axis=1)  # (P, 2)
            mask = tf.scatter_nd(indices, tf.ones([actual_pixels], dtype=tf.bool), [height, width])
            mask = tf.broadcast_to(tf.expand_dims(mask, -1), tf.shape(img))
            return tf.where(mask, fill_value, img)

        return tf.cond(valid, do_add, lambda: img)

    def get_target_row_type_5():
        return tf.cond(num_signal_rows >= 1, lambda: signal_rows[0], lambda: tf.constant(-1, tf.int32))

    def get_target_row_type_6():
        return tf.cond(num_signal_rows >= 2, lambda: signal_rows[1], lambda: tf.constant(-1, tf.int32))

    def get_target_row_type_7():
        return tf.cond(num_signal_rows >= 3, lambda: signal_rows[2], lambda: tf.constant(-1, tf.int32))

    def process_single_row(target_row):
        above_row = target_row - 1
        below_row = target_row + 1
        num_above = tf.random.uniform([], 20, 31, dtype=tf.int32)
        num_below = tf.random.uniform([], 4, 11, dtype=tf.int32)
        result = add_pixels_to_row(image, above_row, num_above)
        result = add_pixels_to_row(result, below_row, num_below)
        return result

    def process_block():
        def do_process():
            bottom_row = signal_rows[0]
            top_row = bottom_row - 2
            above_row = top_row - 1
            below_row = bottom_row + 1
            num_above = tf.random.uniform([], 20, 31, dtype=tf.int32)
            num_below = tf.random.uniform([], 4, 11, dtype=tf.int32)
            result = add_pixels_to_row(image, above_row, num_above)
            result = add_pixels_to_row(result, below_row, num_below)
            return result
        return tf.cond(num_signal_rows >= 1, do_process, lambda: image)

    result = tf.case([
        (tf.equal(noise_type, 5), lambda: process_single_row(get_target_row_type_5())),
        (tf.equal(noise_type, 6), lambda: process_single_row(get_target_row_type_6())),
        (tf.equal(noise_type, 7), lambda: process_single_row(get_target_row_type_7())),
        (tf.equal(noise_type, 8), process_block),
    ], default=lambda: image)

    return result


def make_augment_with_circular_shift(num_channels):
    """
    [消融 B] 依通道數產生對應的擴增函式（與主程式邏輯一致）。

    主程式中 fn_output_signature 寫死 3 通道；本檔以 num_channels 參數化，
    使 1/2/3 通道皆可用。擴增策略（循環位移 + type1-8 噪聲、AUGMENT_FACTOR、
    NOISE_COUNT）皆與主程式完全一致。
    """
    @tf.function
    def augment_with_circular_shift(image, label):
        shifts = tf.concat([
            [0],
            tf.random.uniform([AUGMENT_FACTOR - 1], 0, INPUT_WIDTH, dtype=tf.int32)
        ], axis=0)

        def shift_image(shift):
            return tf.roll(image, shift=shift, axis=1)

        images = tf.map_fn(shift_image, shifts, fn_output_signature=tf.TensorSpec(
            shape=[INPUT_HEIGHT, INPUT_WIDTH, num_channels], dtype=image.dtype
        ))

        all_indices = tf.random.shuffle(tf.range(AUGMENT_FACTOR))
        noise_indices = all_indices[:NOISE_COUNT]
        noise_types = tf.random.uniform([NOISE_COUNT], 1, 9, dtype=tf.int32)

        noisy_images = tf.gather(images, noise_indices)

        def apply_noise_single(args):
            img, n_type = args
            return tf.cond(
                n_type <= 4,
                lambda: add_bottom_noise(img, n_type),
                lambda: add_random_pixel_noise(img, n_type)
            )

        noisy_images_processed = tf.map_fn(
            apply_noise_single,
            (noisy_images, noise_types),
            fn_output_signature=tf.TensorSpec(
                shape=[INPUT_HEIGHT, INPUT_WIDTH, num_channels], dtype=image.dtype
            )
        )

        images = tf.tensor_scatter_nd_update(
            images, tf.expand_dims(noise_indices, 1), noisy_images_processed
        )

        labels = tf.tile(tf.expand_dims(label, 0), [AUGMENT_FACTOR, 1])
        return images, labels

    return augment_with_circular_shift


# ============================================================================
# 實驗清單（規範第五條）
# ============================================================================
# 註：baseline_epsa_3ch 與 epsa_inv_gx_gy 設定完全相同。
#     為避免重複訓練，epsa_inv_gx_gy 設定 same_as="baseline_epsa_3ch"，
#     在 --run_all 時若 baseline 已完成，則直接複用其結果（複製輸出檔），不重新訓練。
EXPERIMENTS = [
    {  # Ablation 0：baseline
        'exp_name': 'baseline_epsa_3ch',
        'attention_mode': 'epsa',
        'input_mode': 'inv_gx_gy',
    },
    {  # Ablation 1：移除跨尺度注意力
        'exp_name': 'no_attention_3ch',
        'attention_mode': 'no_attention',
        'input_mode': 'inv_gx_gy',
    },
    {  # Ablation 2：僅翻轉後二值化
        'exp_name': 'epsa_inv_only',
        'attention_mode': 'epsa',
        'input_mode': 'inv_only',
    },
    {  # Ablation 3：翻轉後二值化 + x 方向 Sobel
        'exp_name': 'epsa_inv_gx',
        'attention_mode': 'epsa',
        'input_mode': 'inv_gx',
    },
    {  # Ablation 4：翻轉後二值化 + y 方向 Sobel
        'exp_name': 'epsa_inv_gy',
        'attention_mode': 'epsa',
        'input_mode': 'inv_gy',
    },
    {  # Ablation 5：三通道完整輸入（與 baseline 相同 -> 複用）
        'exp_name': 'epsa_inv_gx_gy',
        'attention_mode': 'epsa',
        'input_mode': 'inv_gx_gy',
        'same_as': 'baseline_epsa_3ch',
    },
]
EXP_BY_NAME = {e['exp_name']: e for e in EXPERIMENTS}


def exp_output_dir(exp_name):
    return os.path.join(output_root, exp_name)


def result_json_path(exp_name):
    return os.path.join(exp_output_dir(exp_name), f'result_{exp_name}.json')


# ============================================================================
# 資料載入（延遲到實際執行時才載入，使 --list_experiments 無需資料即可運作）
# ============================================================================
_DATA_CACHE = {}


def load_all_data():
    """載入 train/val/test 路徑與標籤、計算 class_weight（全部實驗共用，只載入一次）。"""
    if _DATA_CACHE:
        return _DATA_CACHE

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"找不到資料夾 data_dir: {data_dir}\n"
            f"（與主程式相同，data_dir 目前指向訓練機器的絕對路徑，請在該機器上執行，"
            f"或自行調整 data_dir。本程式刻意沿用主程式的 data_dir 與資料夾讀取邏輯。）"
        )

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    print("\n載入訓練集...")
    train_paths, train_labels = load_data_from_split_dir(train_dir, class_to_idx)
    print("載入驗證集...")
    val_paths, val_labels = load_data_from_split_dir(val_dir, class_to_idx)
    print("載入測試集...")
    test_paths, test_labels = load_data_from_split_dir(test_dir, class_to_idx)

    train_size = len(train_paths)
    val_size = len(val_paths)
    test_size = len(test_paths)
    total_images = train_size + val_size + test_size

    print(f"\n資料集統計:")
    print(f"  總圖片數: {total_images} 張")
    print(f"  訓練集: {train_size} 張 ({train_size/total_images*100:.1f}%)")
    print(f"  驗證集: {val_size} 張 ({val_size/total_images*100:.1f}%)")
    print(f"  測試集: {test_size} 張 ({test_size/total_images*100:.1f}%)")

    unique_classes = np.unique(train_labels)
    cw = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
    class_weights = dict(zip(unique_classes, cw))

    _DATA_CACHE.update(dict(
        train_paths=train_paths, train_labels=train_labels,
        val_paths=val_paths, val_labels=val_labels,
        test_paths=test_paths, test_labels=test_labels,
        train_size=train_size, val_size=val_size, test_size=test_size,
        total_images=total_images, class_weights=class_weights,
    ))
    return _DATA_CACHE


def build_datasets(input_mode):
    """依 input_mode 建立 train/val/test/test_augmented 四個 dataset（資料分割不變）。"""
    data = load_all_data()
    num_ch = INPUT_MODE_CHANNELS[input_mode]
    load_image = make_load_image(input_mode)
    augment = make_augment_with_circular_shift(num_ch)

    # 訓練集：擴增（與主程式一致）
    train_ds = tf.data.Dataset.from_tensor_slices((data['train_paths'], data['train_labels']))
    train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.unbatch()
    # [可重現性] shuffle 明確帶 seed，確保各實驗在相同條件下打散資料
    train_ds = train_ds.shuffle(
        data['train_size'] * AUGMENT_FACTOR,
        seed=RANDOM_STATE,
        reshuffle_each_iteration=True
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 驗證集：未擴增（validation_data / checkpoint 的唯一依據）
    val_ds = tf.data.Dataset.from_tensor_slices((data['val_paths'], data['val_labels']))
    val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache()
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 測試集：原始（主要效能依據，不擴增）
    test_ds = tf.data.Dataset.from_tensor_slices((data['test_paths'], data['test_labels']))
    test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 測試集：擴增（僅作穩健性觀察）
    test_ds_aug = tf.data.Dataset.from_tensor_slices((data['test_paths'], data['test_labels']))
    test_ds_aug = test_ds_aug.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds_aug = test_ds_aug.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds_aug = test_ds_aug.unbatch()
    test_ds_aug = test_ds_aug.cache()
    test_ds_aug = test_ds_aug.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_labels_aug = np.repeat(data['test_labels'], AUGMENT_FACTOR)

    return train_ds, val_ds, test_ds, test_ds_aug, test_labels_aug


# ============================================================================
# 指標計算與輸出
# ============================================================================
def compute_split_metrics(true_labels, pred_labels):
    """計算 macro/weighted 與 per-class 指標、混淆矩陣、accuracy。"""
    p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(
        true_labels, pred_labels, labels=list(range(num_classes)), average=None, zero_division=0)
    macro = precision_recall_fscore_support(true_labels, pred_labels, average='macro', zero_division=0)
    weighted = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))
    acc = float(np.sum(pred_labels == true_labels) / len(true_labels))
    return {
        'accuracy': acc,
        'macro_precision': float(macro[0]), 'macro_recall': float(macro[1]), 'macro_f1': float(macro[2]),
        'weighted_precision': float(weighted[0]), 'weighted_recall': float(weighted[1]), 'weighted_f1': float(weighted[2]),
        'per_class_precision': [float(v) for v in p_c],
        'per_class_recall': [float(v) for v in r_c],
        'per_class_f1': [float(v) for v in f1_c],
        'per_class_support': [int(v) for v in sup_c],
        'confusion_matrix': cm.tolist(),
    }


def save_metrics_csv(path, m):
    """將 per-class + macro/weighted + accuracy 寫成 CSV。"""
    rows = []
    for i, cls in enumerate(class_names):
        rows.append({
            'class': cls, 'chinese': chinese_labels[i],
            'precision': m['per_class_precision'][i],
            'recall': m['per_class_recall'][i],
            'f1': m['per_class_f1'][i],
            'support': m['per_class_support'][i],
        })
    rows.append({'class': 'macro_avg', 'chinese': '宏觀平均',
                 'precision': m['macro_precision'], 'recall': m['macro_recall'],
                 'f1': m['macro_f1'], 'support': sum(m['per_class_support'])})
    rows.append({'class': 'weighted_avg', 'chinese': '加權平均',
                 'precision': m['weighted_precision'], 'recall': m['weighted_recall'],
                 'f1': m['weighted_f1'], 'support': sum(m['per_class_support'])})
    rows.append({'class': 'accuracy', 'chinese': '準確率',
                 'precision': '', 'recall': '', 'f1': m['accuracy'],
                 'support': sum(m['per_class_support'])})
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def save_predictions_csv(path, true_labels, preds):
    """儲存每張圖的真實/預測標籤與各類別機率。"""
    pred_labels = np.argmax(preds, axis=1)
    data = {
        'index': np.arange(len(true_labels)),
        'true_idx': true_labels,
        'true_class': [class_names[i] for i in true_labels],
        'pred_idx': pred_labels,
        'pred_class': [class_names[i] for i in pred_labels],
        'correct': (pred_labels == true_labels).astype(int),
    }
    for c, cls in enumerate(class_names):
        data[f'prob_{cls}'] = preds[:, c]
    pd.DataFrame(data).to_csv(path, index=False, encoding='utf-8-sig')


def save_confusion_matrix_png(path, cm, acc, title, cmap):
    cm = np.array(cm)
    plt.figure(figsize=(10, 8))
    with np.errstate(invalid='ignore', divide='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm, annot=False, fmt='d', cmap=cmap, xticklabels=chinese_labels, yticklabels=chinese_labels)
    for j in range(cm.shape[0]):
        for k in range(cm.shape[1]):
            plt.text(k + 0.5, j + 0.5, f"{cm[j, k]}\n({cm_norm[j, k]:.1f}%)",
                     ha='center', va='center', color='black', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.ylabel('True Label', fontsize=13)
    plt.title(f'{title} (Acc: {acc:.4f})', fontsize=15, fontweight='bold')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def save_training_curves_png(path, history, exp_name):
    hist = history.history
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(hist['loss'], label='Train Loss', linewidth=2)
    plt.plot(hist['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'Loss Curve ({exp_name})', fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(hist['accuracy'], label='Train Acc', linewidth=2)
    plt.plot(hist['val_accuracy'], label='Val Acc', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title(f'Accuracy Curve ({exp_name})', fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    if 'precision' in hist:
        plt.plot(hist['precision'], label='Train Precision', linewidth=2)
        plt.plot(hist['val_precision'], label='Val Precision', linewidth=2)
        plt.plot(hist['recall'], label='Train Recall', linewidth=2, linestyle='--')
        plt.plot(hist['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
    plt.xlabel('Epoch'); plt.ylabel('Score')
    plt.title('Precision & Recall', fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    acc_gap = np.array(hist['val_accuracy']) - np.array(hist['accuracy'])
    loss_gap = np.array(hist['val_loss']) - np.array(hist['loss'])
    plt.plot(loss_gap, label='Val-Train Loss Gap', linewidth=2)
    plt.plot(acc_gap, label='Val-Train Acc Gap', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Gap (Val - Train)')
    plt.title('Overfitting Analysis', fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def write_training_log(path, exp, result, history, durations):
    hist = history.history
    actual_epochs = len(hist['loss'])
    om = result['original']
    am = result['augmented']
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"EPSANet-Large 消融實驗訓練日誌 — {exp['exp_name']}\n")
        f.write("=" * 80 + "\n\n")

        f.write("【消融設定】\n")
        f.write(f"exp_name        : {exp['exp_name']}\n")
        f.write(f"attention_mode  : {exp['attention_mode']}\n")
        f.write(f"input_mode      : {exp['input_mode']} ({INPUT_MODE_DESC[exp['input_mode']]})\n")
        f.write(f"input_channels  : {INPUT_MODE_CHANNELS[exp['input_mode']]}\n")
        f.write(f"input_shape     : ({INPUT_HEIGHT}, {INPUT_WIDTH}, {INPUT_MODE_CHANNELS[exp['input_mode']]})\n\n")

        f.write("【可重現性 / Determinism】\n")
        f.write(f"random_seed     : {RANDOM_STATE} (random / numpy / tensorflow)\n")
        f.write(f"determinism_note: {DETERMINISM_NOTE}\n\n")

        f.write("【基本資訊】\n")
        f.write(f"開始時間: {durations['start'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"結束時間: {durations['end'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"訓練時長: {str(timedelta(seconds=int(durations['seconds'])))}\n")
        f.write(f"資料集  : {os.path.basename(data_dir)}\n\n")

        f.write("【固定訓練配置（所有實驗一致）】\n")
        f.write(f"Batch: {batch_size}, Epochs: {epochs} (實際: {actual_epochs})\n")
        f.write(f"優化器: SGD (momentum=0.9, nesterov=True)\n")
        f.write(f"學習率: 0->{target_lr}->{min_lr} (WarmupCosineDecay)\n")
        f.write(f"正則化: Dropout={DROPOUT_RATE}, L2={WEIGHT_DECAY}, LabelSmoothing={LABEL_SMOOTHING}\n")
        f.write(f"class_weight: balanced, mixed_precision: on, jit_compile: True\n")
        f.write(f"擴增: {AUGMENT_FACTOR}x循環位移 + {NOISE_COUNT}張噪聲/組\n")
        f.write(f"前處理: generic_preprocess (x-127.5)/127.5（對所有 input_mode 一致）\n\n")

        f.write("【前處理一致性聲明】\n")
        f.write(PREPROCESS_FAIRNESS_NOTE + "\n\n")

        f.write("【資料使用邏輯】\n")
        f.write("train_augment = true\n")
        f.write("validation_augment = false\n")
        f.write("test_augment_for_main_eval = false\n")
        f.write("augmented_test_for_robustness_only = true\n")
        f.write('checkpoint_monitor = "val_accuracy on unaugmented validation set"\n\n')

        f.write(f"可訓練參數量: {result['trainable_params']:,}\n")
        f.write(f"最佳Epoch: {result['best_epoch']} (Val Acc: {result['best_val_accuracy']:.4f})\n")
        f.write(f"最終 Train Loss/Acc: {result['final_train_loss']:.4f} / {result['final_train_accuracy']:.4f}\n")
        f.write(f"最終 Val   Loss/Acc: {result['final_val_loss']:.4f} / {result['final_val_accuracy']:.4f}\n\n")

        # 訓練過程
        f.write("=" * 80 + "\n訓練過程\n" + "=" * 80 + "\n")
        f.write(f"{'Epoch':>5} {'T_Loss':>8} {'T_Acc':>7} {'V_Loss':>8} {'V_Acc':>7} {'Gap':>7} {'Note':<10}\n")
        f.write("-" * 60 + "\n")
        best_epoch = result['best_epoch']
        for epoch in range(actual_epochs):
            ep = epoch + 1
            show = (ep <= 20) or (ep % 5 == 0) or (ep == best_epoch) or (ep == actual_epochs)
            if show:
                t_loss, t_acc = hist['loss'][epoch], hist['accuracy'][epoch]
                v_loss, v_acc = hist['val_loss'][epoch], hist['val_accuracy'][epoch]
                gap = t_acc - v_acc
                note = "*Best" if ep == best_epoch else ("[!]過擬合" if gap > 0.05 else "")
                f.write(f"{ep:>5} {t_loss:>8.4f} {t_acc:>7.4f} {v_loss:>8.4f} {v_acc:>7.4f} {gap:>+7.3f} {note:<10}\n")
        f.write("\n")

        # 原始測試集
        f.write("=" * 80 + "\n測試結果\n" + "=" * 80 + "\n\n")
        f.write("【原始測試集（主要效能依據）】\n")
        f.write(f"Accuracy: {om['accuracy']:.4f}, Loss: {result['original_loss']:.4f}\n")
        f.write(f"Macro    P/R/F1: {om['macro_precision']:.4f} / {om['macro_recall']:.4f} / {om['macro_f1']:.4f}\n")
        f.write(f"Weighted P/R/F1: {om['weighted_precision']:.4f} / {om['weighted_recall']:.4f} / {om['weighted_f1']:.4f}\n\n")
        f.write(f"{'Class':<4} {'中文':<8} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Support':>8}\n")
        f.write("-" * 50 + "\n")
        for i, cls in enumerate(class_names):
            f.write(f"{cls:<4} {chinese_labels[i]:<8} "
                    f"{om['per_class_precision'][i]:>6.4f} {om['per_class_recall'][i]:>7.4f} "
                    f"{om['per_class_f1'][i]:>6.4f} {om['per_class_support'][i]:>8}\n")
        f.write("\n")

        # 擴增測試集
        f.write("【擴增測試集（僅作穩健性觀察，非主要效能依據）】\n")
        f.write(f"Accuracy: {am['accuracy']:.4f}\n")
        f.write(f"Macro    P/R/F1: {am['macro_precision']:.4f} / {am['macro_recall']:.4f} / {am['macro_f1']:.4f}\n")
        f.write(f"Weighted P/R/F1: {am['weighted_precision']:.4f} / {am['weighted_recall']:.4f} / {am['weighted_f1']:.4f}\n")
        f.write(f"robustness_gap (擴增 - 原始 accuracy): {result['robustness_gap']*100:+.2f}%\n\n")
        f.write("=" * 80 + "\n")


# ============================================================================
# 單一實驗執行
# ============================================================================
def run_experiment(exp):
    """訓練並評估單一消融實驗，輸出所有檔案，回傳 result dict。"""
    exp_name = exp['exp_name']
    attention_mode = exp['attention_mode']
    input_mode = exp['input_mode']
    num_ch = INPUT_MODE_CHANNELS[input_mode]
    input_shape = (INPUT_HEIGHT, INPUT_WIDTH, num_ch)

    out_dir = exp_output_dir(exp_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"[實驗] {exp_name}  | attention={attention_mode} | input={input_mode} ({num_ch}ch)")
    print("=" * 80)

    # 每個實驗都「重新初始化」：清除 session + 重設種子
    keras.backend.clear_session()
    set_global_seeds(RANDOM_STATE)

    data = load_all_data()
    train_ds, val_ds, test_ds, test_ds_aug, test_labels_aug = build_datasets(input_mode)

    augmented_train_size = data['train_size'] * AUGMENT_FACTOR
    steps_per_epoch = math.ceil(augmented_train_size / batch_size)
    warmup_steps = int(0.1 * epochs * steps_per_epoch)

    # 建立模型（依 attention_mode；新權重）
    model = build_epsanet_large(
        input_shape=input_shape, num_classes=num_classes,
        dropout_rate=DROPOUT_RATE, weight_decay=WEIGHT_DECAY,
        attention_mode=attention_mode
    )

    lr_schedule = WarmupCosineDecay(
        warmup_steps=warmup_steps, target_lr=target_lr,
        total_steps=epochs * steps_per_epoch, min_lr=min_lr
    )
    model.compile(
        optimizer=SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True),
        loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')],
        jit_compile=True
    )

    trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print(f"可訓練參數量: {trainable_params:,}")

    model_path = os.path.join(out_dir, f'best_model_{exp_name}.keras')
    checkpoint = ModelCheckpoint(
        model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    start_dt = datetime.now()
    start_t = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        class_weight=data['class_weights'], callbacks=[checkpoint], verbose=2
    )
    end_t = time.time()
    end_dt = datetime.now()
    durations = {'start': start_dt, 'end': end_dt, 'seconds': end_t - start_t}

    hist = history.history
    best_epoch = int(np.argmax(hist['val_accuracy']) + 1)
    best_val_acc = float(np.max(hist['val_accuracy']))

    best_model = models.load_model(model_path)

    # ---- 原始測試集（主要效能依據）----
    test_loss, test_acc, test_prec, test_rec = best_model.evaluate(test_ds, verbose=0)
    preds = best_model.predict(test_ds, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    om = compute_split_metrics(data['test_labels'], pred_labels)

    # ---- 擴增測試集（僅穩健性觀察）----
    preds_aug = best_model.predict(test_ds_aug, verbose=0)
    pred_labels_aug = np.argmax(preds_aug, axis=1)
    am = compute_split_metrics(test_labels_aug, pred_labels_aug)

    robustness_gap = am['accuracy'] - om['accuracy']

    result = {
        'exp_name': exp_name,
        'attention_mode': attention_mode,
        'input_mode': input_mode,
        'input_channels': num_ch,
        'trainable_params': trainable_params,
        'best_epoch': best_epoch,
        'best_val_accuracy': best_val_acc,
        'final_train_loss': float(hist['loss'][-1]),
        'final_train_accuracy': float(hist['accuracy'][-1]),
        'final_val_loss': float(hist['val_loss'][-1]),
        'final_val_accuracy': float(hist['val_accuracy'][-1]),
        'original_loss': float(test_loss),
        'original': om,
        'augmented': am,
        'robustness_gap': float(robustness_gap),
        'output_dir': out_dir,
        'model_path': model_path,
        'reused_from': None,
        'determinism_note': DETERMINISM_NOTE,
    }

    # ---- 輸出檔案 ----
    save_training_curves_png(os.path.join(out_dir, f'training_curves_{exp_name}.png'), history, exp_name)
    save_confusion_matrix_png(
        os.path.join(out_dir, f'confusion_matrix_original_{exp_name}.png'),
        om['confusion_matrix'], om['accuracy'], f'原始測試集混淆矩陣（主要效能）- {exp_name}', 'Blues')
    save_confusion_matrix_png(
        os.path.join(out_dir, f'confusion_matrix_augmented_{exp_name}.png'),
        am['confusion_matrix'], am['accuracy'], f'擴增測試集混淆矩陣（穩健性觀察）- {exp_name}', 'Greens')
    save_metrics_csv(os.path.join(out_dir, f'metrics_original_{exp_name}.csv'), om)
    save_metrics_csv(os.path.join(out_dir, f'metrics_augmented_{exp_name}.csv'), am)
    save_predictions_csv(os.path.join(out_dir, f'predictions_original_{exp_name}.csv'), data['test_labels'], preds)
    save_predictions_csv(os.path.join(out_dir, f'predictions_augmented_{exp_name}.csv'), test_labels_aug, preds_aug)
    write_training_log(os.path.join(out_dir, f'training_log_{exp_name}.txt'), exp, result, history, durations)

    with open(result_json_path(exp_name), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] {exp_name}")
    print(f"  原始測試集 Accuracy（主要）: {om['accuracy']:.4f} | Macro-F1: {om['macro_f1']:.4f}")
    print(f"  擴增測試集 Accuracy（穩健性）: {am['accuracy']:.4f} | robustness_gap: {robustness_gap*100:+.2f}%")
    print(f"  輸出資料夾: {out_dir}")
    return result


def reuse_experiment(exp, source_name):
    """
    [複用] 對與既有實驗設定完全相同者（如 epsa_inv_gx_gy == baseline_epsa_3ch），
    不重新訓練，直接複製來源實驗的輸出檔（檔名改為本實驗名），並標記 reused_from。
    """
    src_dir = exp_output_dir(source_name)
    dst_dir = exp_output_dir(exp['exp_name'])
    os.makedirs(dst_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"[複用] {exp['exp_name']} 設定與 {source_name} 完全相同，直接複用其結果（不重新訓練）")
    print("=" * 80)

    dst_name = exp['exp_name']
    for fname in os.listdir(src_dir):
        new_name = fname.replace(source_name, dst_name)
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, new_name))

    # 1) 正確更新 result_json 的 exp_name / output_dir / model_path / reused_from
    with open(result_json_path(source_name), 'r', encoding='utf-8') as f:
        result = json.load(f)
    result['exp_name'] = dst_name
    result['reused_from'] = source_name
    result['output_dir'] = dst_dir
    result['model_path'] = os.path.join(dst_dir, f"best_model_{dst_name}.keras")
    with open(result_json_path(dst_name), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 2)+3) 修正 training_log：把內文中的來源實驗名改為本實驗名，並於開頭加入複用聲明，
    #        避免讀者誤以為 epsa_inv_gx_gy 為獨立重跑。
    reuse_banner = (
        "=" * 80 + "\n"
        f"[複用聲明] 本實驗（{dst_name}）設定與 {source_name} 完全相同，"
        f"結果由 {source_name} 複用，未重新訓練。\n"
        f"本日誌內容（含訓練過程、指標、混淆矩陣等）均來自 {source_name} 的訓練結果，"
        f"僅作為 {dst_name} 在消融總表中的對應紀錄。\n"
        + "=" * 80 + "\n\n"
    )
    dst_log = os.path.join(dst_dir, f'training_log_{dst_name}.txt')
    if os.path.exists(dst_log):
        with open(dst_log, 'r', encoding='utf-8') as f:
            log_content = f.read()
        log_content = log_content.replace(source_name, dst_name)
        with open(dst_log, 'w', encoding='utf-8') as f:
            f.write(reuse_banner + log_content)

    print(f"  已複用並輸出至: {dst_dir}")
    return result


def maybe_run(exp):
    """執行單一實驗：若設定 same_as 且來源已完成，則複用；否則訓練。"""
    source = exp.get('same_as')
    if source and os.path.exists(result_json_path(source)):
        return reuse_experiment(exp, source)
    return run_experiment(exp)


# ============================================================================
# 總表輸出（規範第九條）
# ============================================================================
def load_available_results():
    """讀取 ablation_results/ 下所有已完成實驗的 result json（依 EXPERIMENTS 順序）。"""
    results = {}
    for exp in EXPERIMENTS:
        p = result_json_path(exp['exp_name'])
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                results[exp['exp_name']] = json.load(f)
    return results


def write_summary():
    os.makedirs(output_root, exist_ok=True)
    results = load_available_results()
    if not results:
        print("\n[總表] 尚無任何已完成實驗結果，略過總表輸出。")
        return

    # ---- ablation_summary.csv ----
    rows = []
    for exp in EXPERIMENTS:
        r = results.get(exp['exp_name'])
        if not r:
            continue
        rows.append({
            'exp_name': r['exp_name'],
            'attention_mode': r['attention_mode'],
            'input_mode': r['input_mode'],
            'input_channels': r['input_channels'],
            'trainable_params': r['trainable_params'],
            'best_epoch': r['best_epoch'],
            'best_val_accuracy': round(r['best_val_accuracy'], 4),
            'original_accuracy': round(r['original']['accuracy'], 4),
            'original_macro_f1': round(r['original']['macro_f1'], 4),
            'original_weighted_f1': round(r['original']['weighted_f1'], 4),
            'augmented_accuracy': round(r['augmented']['accuracy'], 4),
            'augmented_macro_f1': round(r['augmented']['macro_f1'], 4),
            'augmented_weighted_f1': round(r['augmented']['weighted_f1'], 4),
            'robustness_gap': round(r['robustness_gap'], 4),
            'reused_from': r.get('reused_from'),
            'output_dir': r['output_dir'],
        })
    csv_path = os.path.join(output_root, 'ablation_summary.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[總表] 已輸出: {csv_path}")

    # ---- ablation_summary_for_thesis.md ----
    def pm(m):  # params in M
        return f"{m / 1e6:.2f}"

    def acc(r):
        return f"{r['original']['accuracy']*100:.2f}%"

    def mf1(r):
        return f"{r['original']['macro_f1']*100:.2f}%"

    lines = []
    lines.append("# 第 4.6 節 消融實驗結果（可直接複製到論文表格）\n")
    lines.append(f"> 註：原始（未擴增）測試集為主要效能依據；擴增測試集僅供穩健性觀察。\n")
    lines.append(f"> 隨機種子固定為 {RANDOM_STATE}。{DETERMINISM_NOTE}\n")
    lines.append(f"\n> **前處理一致性聲明**：{PREPROCESS_FAIRNESS_NOTE}\n")

    # 表 4-5 注意力機制
    lines.append("\n## 表 4-5 注意力機制消融實驗結果\n")
    lines.append("| 模型配置 | 準確率 | 宏觀平均 F1 | 參數量（M） |")
    lines.append("| --- | --- | --- | --- |")
    base = results.get('baseline_epsa_3ch')
    noatt = results.get('no_attention_3ch')
    if base:
        lines.append(f"| EPSANet-Large（含 EPSA 機制） | {acc(base)} | {mf1(base)} | {pm(base['trainable_params'])} |")
    if noatt:
        lines.append(f"| 移除跨尺度注意力（保留四尺度卷積） | {acc(noatt)} | {mf1(noatt)} | {pm(noatt['trainable_params'])} |")

    # 表 4-6 多通道輸入
    lines.append("\n## 表 4-6 多通道輸入特徵消融實驗結果\n")
    lines.append("| 輸入配置 | 通道數 | 準確率 | 宏觀平均 F1 |")
    lines.append("| --- | --- | --- | --- |")
    input_rows = [
        ('epsa_inv_only', '僅翻轉後二值化'),
        ('epsa_inv_gx', '翻轉後二值化 + |G_x|'),
        ('epsa_inv_gy', '翻轉後二值化 + |G_y|'),
        ('epsa_inv_gx_gy', '三通道（翻轉後二值化 + |G_x| + |G_y|）'),
    ]
    for name, desc in input_rows:
        r = results.get(name)
        # 三通道若僅有 baseline（複用對象）也可採用
        if not r and name == 'epsa_inv_gx_gy':
            r = results.get('baseline_epsa_3ch')
        if r:
            lines.append(f"| {desc} | {r['input_channels']} | {acc(r)} | {mf1(r)} |")
    lines.append(
        "\n> 註：本表為輸入特徵組合之消融實驗。為避免不同通道數造成前處理不一致，"
        "所有輸入配置均採相同之通道數無關前處理；因此本表重點在於比較 Sobel 方向性通道"
        "加入前後之相對變化，而非取代主要模型效能。\n")

    # 表 4-7 雜訊條件下之辨識結果（以 baseline 主模型為例）
    lines.append("\n## 表 4-7 雜訊條件下之辨識結果\n")
    lines.append("| 測試條件 | 準確率 | 宏觀平均 F1 |")
    lines.append("| --- | --- | --- |")
    ref = base if base else results.get('epsa_inv_gx_gy')
    if ref:
        lines.append(f"| 原始測試集 | {ref['original']['accuracy']*100:.2f}% | {ref['original']['macro_f1']*100:.2f}% |")
        lines.append(f"| 擴增測試集（含相位平移與雜訊） | {ref['augmented']['accuracy']*100:.2f}% | {ref['augmented']['macro_f1']*100:.2f}% |")
        lines.append(f"\n> 穩健性差異 robustness_gap（擴增 − 原始 準確率）= {ref['robustness_gap']*100:+.2f}%\n")

    md_path = os.path.join(output_root, 'ablation_summary_for_thesis.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[總表] 已輸出: {md_path}")


# ============================================================================
# CLI
# ============================================================================
def print_experiments():
    print("\n可用消融實驗 (EXPERIMENTS):")
    print("-" * 80)
    print(f"{'exp_name':<22} {'attention_mode':<14} {'input_mode':<12} {'channels':<9} {'note'}")
    print("-" * 80)
    for exp in EXPERIMENTS:
        note = f"== {exp['same_as']}（複用）" if exp.get('same_as') else ""
        print(f"{exp['exp_name']:<22} {exp['attention_mode']:<14} {exp['input_mode']:<12} "
              f"{INPUT_MODE_CHANNELS[exp['input_mode']]:<9} {note}")
    print("-" * 80)
    print(f"輸出根目錄: {output_root}\n")


def main():
    parser = argparse.ArgumentParser(
        description="EPSANet-Large 消融實驗（第 4.6 節）。原始測試集為主要效能；擴增測試集僅供穩健性觀察。")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--exp', type=str, help='執行單一實驗（exp_name）')
    g.add_argument('--run_all', action='store_true', help='依序執行所有實驗')
    g.add_argument('--list_experiments', action='store_true', help='只列出可用實驗')
    args = parser.parse_args()

    if args.list_experiments:
        print_experiments()
        return

    os.makedirs(output_root, exist_ok=True)

    if args.run_all:
        for exp in EXPERIMENTS:
            maybe_run(exp)
        write_summary()
        return

    if args.exp:
        if args.exp not in EXP_BY_NAME:
            print(f"\n[錯誤] 找不到實驗 '{args.exp}'。")
            print_experiments()
            raise SystemExit(1)
        maybe_run(EXP_BY_NAME[args.exp])
        write_summary()  # 每次單一實驗後增量更新總表
        return


if __name__ == '__main__':
    main()
