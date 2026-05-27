# -*- coding: utf-8 -*-
"""
批次預測與自動分類程式 (EPSANet-Large 版本 + Grad-CAM)

專為 epsanet50_all_in_one.py 訓練出的 .keras 模型設計
預處理流程與訓練時完全一致，確保推論結果正確

功能：使用 EPSANet-Large 7 類別模型對多個子資料夾的圖片進行預測並分類
特點：
  - 使用 EPSANet-Large 模型（Pyramid Squeeze Attention）
  - 輸入尺寸: 180x80 (寬x高)
  - 預處理（與訓練一致）:
    * 二值化處理（RGB > 254 為白，其餘為黑）
    * 黑白翻轉 (255 - gray)
    * Sobel 多通道特徵增強（使用與訓練相同的手動 kernel）
  - 為每張圖片生成 Grad-CAM 熱力圖，顯示模型關注的區域
  - 效能優化：XLA 編譯
"""

# ============================================================================
# 環境設定（必須在 import tensorflow 之前）
# ============================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制 TensorFlow 警告
# XLA JIT 編譯：可能導致第一次推論延遲，設為 0 可關閉
# 執行時可用環境變數覆蓋: DISABLE_XLA=1 python batch_predict_and_classify.py
if os.environ.get('DISABLE_XLA', '0') != '1':
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
else:
    print("[INFO] XLA JIT 已關閉")

import glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import tensorflow as tf

# 推論時使用 float32 以避免 dtype 不一致問題
tf.keras.mixed_precision.set_global_policy('float32')

import keras
from keras import layers, Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from PIL import Image
import cv2

# 抑制 TensorFlow 警告
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

# 啟用動態記憶體增長
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ============================================================================
# Weight Decay 設定
# ============================================================================
WEIGHT_DECAY = 1e-5


# ============================================================================
# EPSANet-Large 自訂層定義（載入模型需要）
# 注意：定義順序必須是 SEWeightModule -> PSAModule -> EPSABlock
# ============================================================================

@keras.saving.register_keras_serializable()
class SEWeightModule(layers.Layer):
    """Squeeze-and-Excitation Weight Module"""
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


@keras.saving.register_keras_serializable()
class PSAModule(layers.Layer):
    """Pyramid Squeeze Attention Module"""
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
        def get_valid_groups(in_ch, out_ch, desired_groups):
            valid = desired_groups
            while valid > 1:
                if in_ch % valid == 0 and out_ch % valid == 0:
                    return valid
                valid -= 1
            return 1

        g0 = get_valid_groups(self.in_channels, self.split_channel, self.conv_groups[0])
        g1 = get_valid_groups(self.in_channels, self.split_channel, self.conv_groups[1])
        g2 = get_valid_groups(self.in_channels, self.split_channel, self.conv_groups[2])
        g3 = get_valid_groups(self.in_channels, self.split_channel, self.conv_groups[3])

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

        out = tf.concat([x1*att1, x2*att2, x3*att3, x4*att4], axis=-1)
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
    """Efficient Pyramid Squeeze Attention Block"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, use_downsample=False,
                 conv_kernels=[3, 5, 7, 9], conv_groups=[32, 32, 32, 32],
                 weight_decay=WEIGHT_DECAY, **kwargs):
        super(EPSABlock, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv_kernels = conv_kernels
        self.conv_groups = conv_groups
        self.use_downsample = use_downsample
        self.weight_decay = weight_decay

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(
            self.out_channels, kernel_size=1, use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.weight_decay)
        )
        self.bn1 = layers.BatchNormalization()
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

        if out.dtype != identity.dtype:
            identity = tf.cast(identity, out.dtype)

        return self.relu(out + identity)

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels, 'out_channels': self.out_channels,
            'stride': self.stride, 'conv_kernels': self.conv_kernels,
            'conv_groups': self.conv_groups, 'use_downsample': self.use_downsample,
            'weight_decay': self.weight_decay
        })
        return config


# ============================================================================
# EPSANet-Large 模型建構函數（載入模型時可能需要）
# ============================================================================
def build_epsanet_large(input_shape=(80, 180, 3), num_classes=7, dropout_rate=0.2,
                        weight_decay=WEIGHT_DECAY):
    """
    建立 EPSANet-Large 模型（適用於 PRPD 分類）
    """
    conv_groups = [32, 32, 32, 32]
    layers_config = [3, 4, 6, 3]
    channels = [128, 256, 512, 1024]

    layer_kernels = [
        [3, 5, 7, 9],
        [3, 5, 7, 9],
        [3, 3, 5, 5],
        [3, 3, 3, 3],
    ]

    inputs = layers.Input(shape=input_shape)

    # Stem
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
                      name=f'layer1_block{i}')(x)

    # Layer 2
    prev_ch = channels[0] * 4
    for i in range(layers_config[1]):
        stride = 2 if i == 0 else 1
        use_ds = (i == 0)
        x = EPSABlock(prev_ch if i == 0 else channels[1] * 4, channels[1], stride=stride,
                      use_downsample=use_ds, conv_kernels=layer_kernels[1],
                      conv_groups=conv_groups, weight_decay=weight_decay,
                      name=f'layer2_block{i}')(x)

    # Layer 3
    prev_ch = channels[1] * 4
    for i in range(layers_config[2]):
        stride = 2 if i == 0 else 1
        use_ds = (i == 0)
        x = EPSABlock(prev_ch if i == 0 else channels[2] * 4, channels[2], stride=stride,
                      use_downsample=use_ds, conv_kernels=layer_kernels[2],
                      conv_groups=conv_groups, weight_decay=weight_decay,
                      name=f'layer3_block{i}')(x)

    # Layer 4
    prev_ch = channels[2] * 4
    for i in range(layers_config[3]):
        stride = 2 if i == 0 else 1
        use_ds = (i == 0)
        x = EPSABlock(prev_ch if i == 0 else channels[3] * 4, channels[3], stride=stride,
                      use_downsample=use_ds, conv_kernels=layer_kernels[3],
                      conv_groups=conv_groups, weight_decay=weight_decay,
                      name=f'layer4_block{i}')(x)

    # Grad-CAM target layer
    x = layers.Activation('linear', name='gradcam_target')(x)

    # Classification head
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


# ============================================================================
# 自訂學習率調度器（載入模型需要）
# ============================================================================
@keras.saving.register_keras_serializable()
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
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
# 配置參數
# ============================================================================
# EPSANet-Large 輸入尺寸
INPUT_HEIGHT = 80
INPUT_WIDTH = 180
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, 3)

# 多通道特徵增強設定
USE_MULTICHANNEL = True

# 模型路徑（請修改為您的 EPSANet-Large 模型路徑）
MODEL_PATH = r'/home/cckuo/m11307u09/GG/1126/epsanet50/all in one/all_7class_6(G)_add_noise_mix_train/best_model_epsanet_large_dropout_wd_180x80.keras'

# 測試圖片的根目錄列表（可以有多個根目錄）
INPUT_ROOT_DIRS = [
    r'/home/cckuo/m11307u09/GG/1126/data_GG_discharge_filter/risk_discharge',
    r'/home/cckuo/m11307u09/GG/1126/data_GG_discharge_filter/81955271'
]

# 輸出根目錄的基礎路徑
OUTPUT_BASE_DIR = r'/home/cckuo/m11307u09/GG/1126/epsanet50/all in one/all_7class_6(G)_add_noise_mix_train/test'

# 7 個類別名稱
CLASS_NAMES = ['AH', 'CT', 'HT', 'TD', 'CD', 'ID', 'SD']
CHINESE_LABELS = ['空洞', '碳痕', '接頭異常', '不規則邊緣', '典型電暈放電', '典型內部放電', '典型表面放電']

# 批次大小（EPSANet 效能優化，可使用較大 batch size）
BATCH_SIZE = 64  # 較小的 batch 啟用流水線並行（512→64 提升 41%）

print("="*80)
print("批次預測與自動分類程式 (EPSANet-Large 版本 + Grad-CAM)")
print("="*80)
print(f"模型路徑: {MODEL_PATH}")
print(f"輸入尺寸: {INPUT_WIDTH}x{INPUT_HEIGHT} (寬x高)")
print(f"多通道特徵增強: {'啟用' if USE_MULTICHANNEL else '停用'}")
if USE_MULTICHANNEL:
    print(f"  通道 1: 原始灰度圖（黑白翻轉後）")
    print(f"  通道 2: Sobel X（垂直邊緣）")
    print(f"  通道 3: Sobel Y（水平邊緣）")
print(f"要處理的根目錄數: {len(INPUT_ROOT_DIRS)}")
for idx, dir_path in enumerate(INPUT_ROOT_DIRS, 1):
    print(f"  {idx}. {dir_path}")
print(f"輸出基礎目錄: {OUTPUT_BASE_DIR}")
print(f"類別: {CLASS_NAMES}")
print(f"效能優化: XLA 編譯")
print(f"CAM 方法: Grad-CAM（使用類別梯度加權，class-specific）")
print("="*80)

# ============================================================================
# 載入模型
# ============================================================================
print("\n 載入 EPSANet-Large 模型...")

model = None

# 方法 1: 直接載入完整模型
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(" 方法 1 成功：直接載入完整模型")
except Exception as e1:
    print(f" 方法 1 失敗（直接載入）: {e1}")

    # 方法 2: 先建構模型結構，再載入權重
    print(" 嘗試方法 2：先建構模型結構，再載入權重...")
    try:
        model = build_epsanet_large(
            input_shape=INPUT_SHAPE,
            num_classes=len(CLASS_NAMES),
            dropout_rate=0.2,
            weight_decay=WEIGHT_DECAY
        )
        model.load_weights(MODEL_PATH)
        print(" 方法 2 成功：建構模型 + 載入權重")
    except Exception as e2:
        print(f" 方法 2 失敗（載入權重）: {e2}")

if model is None:
    print("\n 模型載入失敗！")
    print("\n可能的解決方案：")
    print("  1. 確認 MODEL_PATH 指向正確的 .keras 檔案")
    print("  2. 檢查模型檔案是否損壞（檔案大小是否正常）")
    print("  3. 檢查 TensorFlow/Keras 版本是否與訓練時一致")
    print(f"     當前版本: TensorFlow {tf.__version__}, Keras {keras.__version__}")
    exit(1)

print(" 使用 float32 進行推論")

# ============================================================================
# Grad-CAM 相關函數（EPSANet-Large 版本）
# ============================================================================

# 合併推論+Grad-CAM 的快取
_combined_model = None
_combined_compute_fn = None


def list_available_layers():
    """
    列出模型中所有可用的層（有 4D 輸出的卷積層）
    """
    print("\n  【可用的目標層列表】")
    print("  " + "-"*70)
    print(f"  {'層名稱':<40} {'輸出形狀':<20} {'空間位置數'}")
    print("  " + "-"*70)

    available_layers = []
    for layer in model.layers:
        try:
            output_shape = layer.output.shape
            # 只顯示 4D 輸出的層（卷積層）
            if len(output_shape) == 4:
                h = output_shape[1] if output_shape[1] is not None else '?'
                w = output_shape[2] if output_shape[2] is not None else '?'
                c = output_shape[3] if output_shape[3] is not None else '?'

                if isinstance(h, int) and isinstance(w, int):
                    spatial_count = h * w
                    spatial_str = str(spatial_count)
                else:
                    spatial_str = '?'

                shape_str = f"({h}, {w}, {c})"
                print(f"  {layer.name:<40} {shape_str:<20} {spatial_str}")
                available_layers.append({
                    'name': layer.name,
                    'layer': layer,
                    'shape': output_shape,
                    'spatial': spatial_count if isinstance(h, int) and isinstance(w, int) else 0
                })
        except Exception:
            pass

    print("  " + "-"*70)
    return available_layers


# 目標層選擇設定
# 可選值: 'auto', 'gradcam_target', 'layer3', 'layer2', 或具體層名稱
# 'auto' = 自動選擇空間解析度最佳的層（推薦）
GRADCAM_TARGET_LAYER = 'layer1'


def get_epsanet_target_layer():
    """
    獲取 EPSANet-Large 的目標層用於 Grad-CAM

    選擇策略：
    - 'auto': 自動選擇空間解析度較高且深度適中的層
    - 'gradcam_target': 使用 v2 模型的專用層（可能空間解析度太低）
    - 'layer3', 'layer2': 使用較淺的層（空間解析度更高）
    - 具體層名稱: 直接指定
    """
    # 列出所有可用層
    available_layers = list_available_layers()

    target_layer = None
    target_layer_name = None

    if GRADCAM_TARGET_LAYER == 'auto':
        # 自動選擇策略：優先選擇空間位置數 >= 100 且名稱含 layer 的最深層
        print(f"\n  自動選擇模式：尋找空間解析度較高的層...")

        # 候選層優先順序：layer3 > layer2 > layer1（空間解析度由高到低）
        # 但也要考慮特徵的語義強度（較深的層特徵更有意義）
        best_candidates = []

        for info in available_layers:
            # 篩選空間位置數 >= 50 的層
            if info['spatial'] >= 50:
                # 優先考慮 layer3, layer2 的 block
                priority = 0
                if 'layer3' in info['name']:
                    priority = 3
                elif 'layer2' in info['name']:
                    priority = 2
                elif 'layer1' in info['name']:
                    priority = 1
                elif 'layer4' in info['name']:
                    priority = 0  # layer4 通常太小

                if priority > 0:
                    best_candidates.append((priority, info['spatial'], info))

        if best_candidates:
            # 按 priority 降序排列，同 priority 按 spatial 升序（選較小但足夠的）
            best_candidates.sort(key=lambda x: (-x[0], x[1]))
            best_info = best_candidates[0][2]
            target_layer = best_info['layer']
            target_layer_name = best_info['name']
            print(f"  [V] 自動選擇: {target_layer_name} (空間位置: {best_info['spatial']})")
        else:
            # 沒有找到合適的，嘗試找任何空間位置數最大的層
            available_layers.sort(key=lambda x: x['spatial'], reverse=True)
            if available_layers:
                best_info = available_layers[0]
                target_layer = best_info['layer']
                target_layer_name = best_info['name']
                print(f"  [!] 自動選擇（備選）: {target_layer_name} (空間位置: {best_info['spatial']})")

    elif GRADCAM_TARGET_LAYER == 'gradcam_target':
        # 使用原本的 gradcam_target 層
        for layer in model.layers:
            if layer.name == 'gradcam_target':
                target_layer = layer
                target_layer_name = layer.name
                break

    elif GRADCAM_TARGET_LAYER.startswith('layer'):
        # 使用指定的 layer 級別（如 'layer3', 'layer2'）
        layer_prefix = GRADCAM_TARGET_LAYER
        candidates = []
        for info in available_layers:
            if layer_prefix in info['name'] and 'block' in info['name']:
                candidates.append(info)
        if candidates:
            # 選該 layer 的最後一個 block
            candidates.sort(key=lambda x: x['name'])
            best_info = candidates[-1]
            target_layer = best_info['layer']
            target_layer_name = best_info['name']
            print(f"  使用指定層級: {target_layer_name}")
    else:
        # 直接使用指定的層名稱
        for layer in model.layers:
            if layer.name == GRADCAM_TARGET_LAYER:
                target_layer = layer
                target_layer_name = layer.name
                break

    # 回退邏輯
    if target_layer is None:
        print(f"  [!] 指定的層 '{GRADCAM_TARGET_LAYER}' 未找到，嘗試回退...")

        # 嘗試 gradcam_target
        for layer in model.layers:
            if layer.name == 'gradcam_target':
                target_layer = layer
                target_layer_name = layer.name
                print(f"  回退到 gradcam_target 層")
                break

        # 嘗試 layer4_block
        if target_layer is None:
            for layer in model.layers:
                if 'layer4_block' in layer.name:
                    target_layer = layer
                    target_layer_name = layer.name
            if target_layer is not None:
                print(f"  回退到 {target_layer_name} 層")

    if target_layer is None:
        raise RuntimeError("找不到任何適合的目標層，無法生成 Grad-CAM")

    # 輸出最終選擇的層資訊
    print(f"\n  【最終選擇的 Grad-CAM 目標層】")
    print(f"  層名稱: {target_layer_name}")
    print(f"  類型: {type(target_layer).__name__}")
    try:
        shape = target_layer.output.shape
        h, w, c = shape[1], shape[2], shape[3]
        spatial = h * w if h and w else '?'
        print(f"  輸出形狀: ({h}, {w}, {c})")
        print(f"  空間位置數: {spatial}")
        if isinstance(spatial, int) and spatial < 50:
            print(f"  [!] 警告：空間位置數較少，熱力圖可能缺乏細節")
    except Exception:
        pass

    return target_layer


def apply_cam_to_image(img_array, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    將 CAM 熱力圖疊加到原圖上

    參數:
        img_array: 圖片 (H, W, 3)，0-255 範圍
        heatmap: CAM 熱力圖 (H', W')
        alpha: 熱力圖透明度 (0-1)
        colormap: OpenCV色彩映射

    返回:
        superimposed_img: 疊加後的圖片 (numpy array, uint8)
    """
    heatmap = np.array(heatmap, dtype=np.float32)

    if heatmap.ndim != 2:
        heatmap = heatmap.squeeze()

    # 調整熱力圖大小（使用 CUBIC 插值）
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)

    # 正規化到 [0, 1]
    heatmap_min = heatmap_resized.min()
    heatmap_max = heatmap_resized.max()
    if heatmap_max > heatmap_min:
        heatmap_resized = (heatmap_resized - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap_resized = np.full_like(heatmap_resized, 0.3)

    # 對比度增強
    gamma = 0.7
    heatmap_resized = np.power(heatmap_resized, gamma)

    # 轉換為 0-255 範圍
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # 應用色彩映射
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 疊加
    img_array = np.array(img_array, dtype=np.float32)
    superimposed_img = heatmap_colored * alpha + img_array * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img


# 初始化 Grad-CAM
print("\n 初始化 Grad-CAM...")
epsanet_target_layer = get_epsanet_target_layer()
if epsanet_target_layer is not None:
    print(f" Grad-CAM 初始化成功！使用層: {epsanet_target_layer.name}")
else:
    print(" Grad-CAM 初始化失敗")
    exit(1)


# ============================================================================
# TensorFlow 原生預處理函數（GPU 加速版）
# ============================================================================

# 預先定義 Sobel kernel 為常數（避免重複建立）
SOBEL_X_KERNEL = tf.constant([
    [-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]
], dtype=tf.float32)
SOBEL_X_KERNEL_4D = tf.reshape(SOBEL_X_KERNEL, [3, 3, 1, 1])

SOBEL_Y_KERNEL = tf.constant([
    [-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]
], dtype=tf.float32)
SOBEL_Y_KERNEL_4D = tf.reshape(SOBEL_Y_KERNEL, [3, 3, 1, 1])


# ============================================================================
# 推論 Dataset 流水線（CPU/GPU 並行優化）
# ============================================================================
def create_inference_dataset(image_paths, batch_size=512):
    """
    建立 tf.data.Dataset 推論流水線

    優勢：
    - CPU 預處理與 GPU 推論並行執行
    - 使用 prefetch 預先準備下一批次
    - Sobel 計算在 GPU 上執行
    - 自動利用多核心並行讀取

    Args:
        image_paths: 圖片路徑列表
        batch_size: 批次大小

    Returns:
        tf.data.Dataset: 預處理完成的 Dataset
    """
    # 建立路徑 Dataset
    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)

    # 並行預處理（使用 GPU 加速的 tf_preprocess_image）
    dataset = paths_ds.map(
        tf_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True  # 保持順序以對應原始路徑
    )

    # 批次化
    dataset = dataset.batch(batch_size)

    # 預取下一批次（CPU/GPU 流水線核心）
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def tf_preprocess_image(file_path):
    """
    TensorFlow 原生預處理（與 preprocess_image 邏輯完全一致）

    優勢：
    - tf.io.read_file 支援並行讀取
    - Sobel 計算使用 GPU
    - 自動與 tf.data Pipeline 整合

    注意：使用 set_shape() 設定靜態形狀以支援 tf.data.Dataset 流水線
    """
    # 讀取圖片
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    # 設定靜態形狀（Dataset 流水線需要）
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32)

    # 二值化：RGB 三通道都 > 254 為白，否則為黑
    white_mask = tf.reduce_all(image > 254, axis=-1)
    gray = tf.where(white_mask, 255.0, 0.0)

    # 黑白翻轉
    gray = 255.0 - gray

    # Resize 到模型輸入尺寸
    gray = tf.image.resize(
        tf.expand_dims(gray, -1),
        [INPUT_HEIGHT, INPUT_WIDTH],
        method='nearest'
    )
    gray = tf.squeeze(gray, -1)

    if USE_MULTICHANNEL:
        channel_1 = gray

        # Sobel X（GPU 加速）
        gray_4d = tf.reshape(gray, [1, INPUT_HEIGHT, INPUT_WIDTH, 1])

        sobel_x = tf.nn.conv2d(gray_4d, SOBEL_X_KERNEL_4D, strides=[1, 1, 1, 1], padding='SAME')
        sobel_x = tf.squeeze(sobel_x, axis=[0, 3])
        sobel_x = tf.abs(sobel_x)
        sobel_x_max = tf.reduce_max(sobel_x)
        channel_2 = tf.cond(sobel_x_max > 0, lambda: sobel_x / sobel_x_max * 255.0, lambda: sobel_x)

        # Sobel Y（GPU 加速）
        sobel_y = tf.nn.conv2d(gray_4d, SOBEL_Y_KERNEL_4D, strides=[1, 1, 1, 1], padding='SAME')
        sobel_y = tf.squeeze(sobel_y, axis=[0, 3])
        sobel_y = tf.abs(sobel_y)
        sobel_y_max = tf.reduce_max(sobel_y)
        channel_3 = tf.cond(sobel_y_max > 0, lambda: sobel_y / sobel_y_max * 255.0, lambda: sobel_y)

        image_out = tf.stack([channel_1, channel_2, channel_3], axis=-1)
    else:
        image_out = tf.stack([gray, gray, gray], axis=-1)

    # ResNet 預處理
    image_out = preprocess_input(image_out)

    return image_out




# ============================================================================
# 原圖快取類別（並行載入）
# ============================================================================
class OriginalImageCache:
    """並行載入原圖快取（供保存時使用，避免重複讀取）"""

    def __init__(self, max_workers=24):
        self.max_workers = max_workers
        self.cache = {}
        self.lock = threading.Lock()

    def _load_single(self, path):
        """載入單張原圖"""
        try:
            img = np.array(Image.open(path).convert('RGB'))
            with self.lock:
                self.cache[path] = img
            return path, True
        except Exception:
            return path, False

    def load_batch(self, image_paths):
        """並行載入一批原圖"""
        # 過濾已快取的
        paths_to_load = [p for p in image_paths if p not in self.cache]

        if not paths_to_load:
            return

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._load_single, p) for p in paths_to_load]
            for future in tqdm(as_completed(futures), total=len(futures), desc="  快取原圖"):
                future.result()

    def get(self, path):
        """取得原圖（快取未命中時會讀取）"""
        if path not in self.cache:
            self._load_single(path)
        return self.cache.get(path)

    def clear(self):
        """清除快取釋放記憶體"""
        self.cache.clear()



def predict_and_gradcam_combined(images_or_dataset, target_layer, batch_size=512, n_samples=None):
    """
    合併推論和 Grad-CAM 計算（一次 forward pass）

    支援兩種輸入：
    1. numpy array：傳統方式，內部分批處理
    2. tf.data.Dataset：流水線方式，直接迭代（prefetch 預先準備下一批次）

    Args:
        images_or_dataset: numpy array 或 tf.data.Dataset
        target_layer: Grad-CAM 目標層
        batch_size: 批次大小（僅 numpy 輸入時使用）
        n_samples: 樣本總數（Dataset 輸入時需指定，供進度條使用）

    Returns:
        (predictions, heatmaps): 預測結果和熱力圖
    """
    global _combined_model, _combined_compute_fn

    import time
    start_time = time.time()

    # 判斷輸入類型
    is_dataset = isinstance(images_or_dataset, tf.data.Dataset)

    if is_dataset:
        if n_samples is None:
            # 嘗試計算 Dataset 元素數（可能較慢）
            n_samples = sum(1 for _ in images_or_dataset) * batch_size
            images_or_dataset = images_or_dataset  # 重新建立 iterator
        n_batches = (n_samples + batch_size - 1) // batch_size
    else:
        n_samples = images_or_dataset.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

    # 建立合併模型（只建立一次）
    if _combined_model is None:
        print("  [合併計算] 建立模型...")
        _combined_model = Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output]
        )

        @tf.function(reduce_retracing=True)
        def _compute_combined(imgs):
            """一次 forward pass 得到特徵圖和預測"""
            with tf.GradientTape() as tape:
                conv_outputs, predictions = _combined_model(imgs, training=False)
                tape.watch(conv_outputs)

                # 使用預測的 argmax 作為類別
                pred_classes = tf.argmax(predictions, axis=1, output_type=tf.int32)
                batch_indices = tf.range(tf.shape(pred_classes)[0])
                indices = tf.stack([batch_indices, pred_classes], axis=1)
                class_outputs = tf.gather_nd(predictions, indices)

            grads = tape.gradient(class_outputs, conv_outputs)

            # 計算 Grad-CAM
            weights = tf.reduce_mean(grads, axis=[1, 2])
            weights_expanded = tf.reshape(weights, [-1, 1, 1, tf.shape(weights)[1]])
            cam = tf.reduce_sum(conv_outputs * weights_expanded, axis=-1)
            cam = tf.nn.relu(cam)

            return predictions, cam

        _combined_compute_fn = _compute_combined
        print("  [合併計算] 模型建立完成")

    # 分批處理
    all_predictions = []
    all_cams = []

    # 詳細計時
    preprocess_time = 0
    inference_time = 0

    if is_dataset:
        # Dataset 流水線模式
        for batch in tqdm(images_or_dataset, desc="  推論+GradCAM", total=n_batches):
            t0 = time.time()
            # 強制執行預處理（Dataset 是 lazy 的）
            batch_ready = batch
            preprocess_time += time.time() - t0

            t1 = time.time()
            preds, cams = _combined_compute_fn(batch_ready)
            all_predictions.append(preds.numpy())
            all_cams.append(cams.numpy())
            inference_time += time.time() - t1
    else:
        # numpy array 傳統模式
        for i in tqdm(range(0, n_samples, batch_size), desc="  推論+GradCAM", total=n_batches):
            batch = images_or_dataset[i:i+batch_size]
            batch_tensor = tf.constant(batch, dtype=tf.float32)
            t1 = time.time()
            preds, cams = _combined_compute_fn(batch_tensor)
            all_predictions.append(preds.numpy())
            all_cams.append(cams.numpy())
            inference_time += time.time() - t1

    print(f"    - 預處理耗時: {preprocess_time:.2f} 秒")
    print(f"    - 推論+CAM耗時: {inference_time:.2f} 秒")

    concat_start = time.time()
    predictions = np.concatenate(all_predictions, axis=0)
    cams = np.concatenate(all_cams, axis=0)
    print(f"    - concat耗時: {time.time() - concat_start:.2f} 秒")

    # 上採樣和正規化熱力圖（批次化處理）
    resize_start = time.time()

    # 使用 TensorFlow 批次化 resize（GPU 加速）
    cams_4d = tf.constant(cams[:, :, :, np.newaxis], dtype=tf.float32)
    cams_resized = tf.image.resize(cams_4d, [INPUT_HEIGHT, INPUT_WIDTH], method='bilinear')
    cams_resized = tf.squeeze(cams_resized, axis=-1).numpy()

    # 批次正規化
    cam_min = cams_resized.min(axis=(1, 2), keepdims=True)
    cam_max = cams_resized.max(axis=(1, 2), keepdims=True)
    denom = cam_max - cam_min
    denom = np.where(denom > 0, denom, 1.0)  # 避免除以零
    heatmaps = (cams_resized - cam_min) / denom

    print(f"    - heatmap resize+norm耗時: {time.time() - resize_start:.2f} 秒")

    print(f"  [合併計算] 推論+Grad-CAM 完成: {time.time() - start_time:.2f} 秒")

    return predictions, heatmaps


# ============================================================================
# 並行保存函數
# ============================================================================
def save_single_result(args):
    """保存單張結果圖片（供 ThreadPoolExecutor 使用）"""
    (img_path, pred_class_name, pred_class_idx, heatmap, pred_prob, all_class_probs,
     subfolder_output_dir, original_cache) = args

    try:
        # 從快取取得原圖
        orig_array = original_cache.get(img_path)
        if orig_array is None:
            return None

        # 二值化處理
        white_mask = np.all(orig_array > 254, axis=-1)
        binary_gray = np.where(white_mask, 255, 0).astype(np.uint8)
        binary_gray = cv2.resize(binary_gray, (INPUT_WIDTH, INPUT_HEIGHT),
                                  interpolation=cv2.INTER_NEAREST)
        img_to_save = cv2.cvtColor(binary_gray, cv2.COLOR_GRAY2RGB)

        # 建立輸出目錄
        class_output_dir = os.path.join(subfolder_output_dir, pred_class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # 處理檔名
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(class_output_dir, base_filename + '.png')

        counter = 1
        final_base_name = base_filename
        while os.path.exists(output_path):
            final_base_name = f"{base_filename}_{counter}"
            output_path = os.path.join(class_output_dir, final_base_name + '.png')
            counter += 1

        # 保存原圖
        Image.fromarray(img_to_save).save(output_path)

        # 保存 CAM 圖
        cam_img = apply_cam_to_image(img_to_save, heatmap, alpha=0.4)
        cam_output_path = os.path.join(class_output_dir, final_base_name + '_gradcam.png')
        Image.fromarray(cam_img).save(cam_output_path)

        # 計算信心度資訊
        sorted_probs = np.sort(all_class_probs)[::-1]
        second_highest_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        confidence_gap = pred_prob - second_highest_prob

        return {
            'path': img_path,
            'output_path': output_path,
            'class': pred_class_name,
            'class_idx': pred_class_idx,
            'final_name': final_base_name,
            'pred_prob': pred_prob,
            'second_prob': second_highest_prob,
            'confidence_gap': confidence_gap,
            'all_probs': all_class_probs
        }
    except Exception as e:
        print(f"    保存失敗 ({img_path}): {e}")
        return None


def save_results_parallel(save_tasks, max_workers=12):
    """並行保存所有結果"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(save_single_result, task) for task in save_tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="  並行保存"):
            result = future.result()
            if result:
                results.append(result)
    return results


# ============================================================================
# 主處理流程
# ============================================================================
def generate_performance_report(output_dir, timing_stats, class_counts,
                                 confidence_list, confidence_by_class,
                                 start_time, end_time, model_path):
    """
    生成效能與統計報告 .txt 檔案

    參數:
        output_dir: 輸出目錄
        timing_stats: 計時統計字典 {'preprocess_time', 'inference_time', 'save_time', 'total_images'}
        class_counts: 各類別數量字典
        confidence_list: 所有信心度列表
        confidence_by_class: 各類別信心度字典
        start_time: 開始時間 (datetime)
        end_time: 結束時間 (datetime)
        model_path: 模型檔案路徑
    """
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f'performance_report_{timestamp}.txt')

    total_duration = end_time - start_time
    total_seconds = total_duration.total_seconds()
    total_images = timing_stats['total_images']

    # 計算每張圖平均耗時和處理速度
    avg_time = total_seconds / total_images if total_images > 0 else 0
    speed = total_images / total_seconds if total_seconds > 0 else 0

    # 格式化總時間
    minutes, seconds = divmod(int(total_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        duration_str = f"{hours} 時 {minutes} 分 {seconds} 秒"
    elif minutes > 0:
        duration_str = f"{minutes} 分 {seconds} 秒"
    else:
        duration_str = f"{total_seconds:.2f} 秒"

    # 取得 GPU 資訊
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        gpu_info = []
        for gpu in gpu_devices:
            try:
                details = tf.config.experimental.get_device_details(gpu)
                gpu_name = details.get('device_name', gpu.name)
                gpu_info.append(gpu_name)
            except Exception:
                gpu_info.append(gpu.name)
        gpu_str = ', '.join(gpu_info)
    else:
        gpu_str = "CPU 模式 (無 GPU)"

    with open(report_path, 'w', encoding='utf-8') as f:
        # 標題
        f.write("=" * 60 + "\n")
        f.write("批次預測效能與統計報告\n")
        f.write("=" * 60 + "\n\n")

        # 效能統計
        f.write("【效能統計】\n")
        f.write(f"預處理時間: {timing_stats['preprocess_time']:.2f} 秒\n")
        f.write(f"推論+GradCAM: {timing_stats['inference_time']:.2f} 秒\n")
        f.write(f"保存時間: {timing_stats['save_time']:.2f} 秒\n")
        f.write(f"總時間: {duration_str}\n")
        f.write(f"每張圖平均: {avg_time:.3f} 秒\n")
        f.write(f"處理速度: {speed:.1f} 張/秒\n\n")

        # 系統設定
        f.write("【系統設定】\n")
        f.write(f"模型: {os.path.basename(model_path)}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"GPU: {gpu_str}\n")
        f.write(f"CAM 方法: Grad-CAM\n\n")

        # 辨識結果統計
        f.write("【辨識結果統計】\n")
        f.write(f"{'類別':<18}{'數量':>8}{'百分比':>10}{'平均信心度':>12}\n")
        f.write("-" * 48 + "\n")

        total_count = sum(class_counts.values())
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            chinese_label = CHINESE_LABELS[cls_idx]
            count = class_counts.get(cls_name, 0)
            pct = (count / total_count * 100) if total_count > 0 else 0
            # 計算該類別平均信心度
            cls_confidences = confidence_by_class.get(cls_name, [])
            avg_conf = np.mean(cls_confidences) if cls_confidences else 0
            f.write(f"{cls_name} ({chinese_label}){' ' * (10 - len(chinese_label))}{count:>8}{pct:>9.1f}%{avg_conf:>11.4f}\n")

        f.write("-" * 48 + "\n")
        overall_avg_conf = np.mean(confidence_list) if confidence_list else 0
        f.write(f"{'總計':<18}{total_count:>8}{'100.0%':>10}{overall_avg_conf:>11.4f}\n\n")

        # 信心度統計
        if confidence_list:
            conf_array = np.array(confidence_list)
            f.write("【信心度統計】\n")
            f.write(f"平均信心度: {np.mean(conf_array):.4f}\n")
            f.write(f"最高信心度: {np.max(conf_array):.4f}\n")
            f.write(f"最低信心度: {np.min(conf_array):.4f}\n")
            f.write(f"標準差: {np.std(conf_array):.4f}\n")

    print(f"效能報告已保存: {report_path}")
    return report_path


def process_all_subfolders():
    """處理多個根目錄下的所有子資料夾"""
    global_start_time = datetime.now()

    for root_idx, INPUT_ROOT_DIR in enumerate(INPUT_ROOT_DIRS, 1):
        print("\n" + "="*80)
        print(f"處理根目錄 {root_idx}/{len(INPUT_ROOT_DIRS)}")
        print("="*80)
        print(f"路徑: {INPUT_ROOT_DIR}")

        if not os.path.exists(INPUT_ROOT_DIR):
            print(f"根目錄不存在，跳過...")
            continue

        root_dir_name = os.path.basename(INPUT_ROOT_DIR)
        OUTPUT_ROOT_DIR = os.path.join(OUTPUT_BASE_DIR, root_dir_name)

        os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
        print(f"輸出目錄: {OUTPUT_ROOT_DIR}")

        root_results = []
        root_start_time = datetime.now()

        # 效能報告用的計時統計
        timing_stats = {
            'preprocess_time': 0.0,
            'inference_time': 0.0,
            'save_time': 0.0,
            'total_images': 0
        }
        confidence_list = []  # 收集所有信心度
        confidence_by_class = {cls: [] for cls in CLASS_NAMES}  # 各類別信心度
        total_class_counts = {cls: 0 for cls in CLASS_NAMES}  # 各類別總計

        subfolders = [f for f in os.listdir(INPUT_ROOT_DIR)
                      if os.path.isdir(os.path.join(INPUT_ROOT_DIR, f))]

        if len(subfolders) == 0:
            print(f"在此根目錄中未找到任何子資料夾，跳過...")
            continue

        print(f"\n找到 {len(subfolders)} 個子資料夾")
        print(f"子資料夾列表: {subfolders}\n")

        for subfolder_idx, subfolder_name in enumerate(subfolders, 1):
            print("="*80)
            print(f"處理子資料夾 {subfolder_idx}/{len(subfolders)}: {subfolder_name}")
            print(f"   (根目錄 {root_idx}/{len(INPUT_ROOT_DIRS)}: {root_dir_name})")
            print("="*80)

            subfolder_path = os.path.join(INPUT_ROOT_DIR, subfolder_name)

            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(subfolder_path, '**', ext), recursive=True))

            if len(image_paths) == 0:
                print(f"  子資料夾 '{subfolder_name}' 中未找到任何圖片，跳過...\n")
                continue

            print(f"找到 {len(image_paths)} 張圖片")

            # ================================================================
            # 流水線預處理 + 推論（CPU/GPU 並行）
            # ================================================================
            import time
            original_cache = OriginalImageCache(max_workers=12)
            n_images = len(image_paths)

            # 背景載入原圖（與 GPU 預處理/推論並行）
            print(f"  [階段1] 啟動背景原圖載入...")
            cache_thread = threading.Thread(
                target=original_cache.load_batch,
                args=(image_paths,)
            )
            cache_thread.start()

            subfolder_output_dir = os.path.join(OUTPUT_ROOT_DIR, subfolder_name)
            os.makedirs(subfolder_output_dir, exist_ok=True)

            class_counts = {cls: 0 for cls in CLASS_NAMES}

            # ================================================================
            # 合併計算：預處理 + 推論 + Grad-CAM
            # ================================================================
            print(f"\n  正在執行計算...")

            # === 階段 A: 預處理（分離計時）===
            preprocess_start = time.time()
            dataset = create_inference_dataset(image_paths, BATCH_SIZE)
            # 強制執行預處理，收集為 numpy
            batches = []
            for batch in tqdm(dataset, desc="  預處理", total=(n_images + BATCH_SIZE - 1) // BATCH_SIZE):
                batches.append(batch.numpy())
            preprocessed_images = np.concatenate(batches, axis=0)
            preprocess_elapsed = time.time() - preprocess_start
            print(f"  [階段A] 預處理完成: {preprocess_elapsed:.2f} 秒")
            timing_stats['preprocess_time'] += preprocess_elapsed

            # === 階段 B: 推論 + Grad-CAM ===
            inference_start = time.time()
            predictions, heatmaps = predict_and_gradcam_combined(
                preprocessed_images, epsanet_target_layer, batch_size=BATCH_SIZE
            )
            pred_classes = np.argmax(predictions, axis=1)
            pred_probs = np.max(predictions, axis=1)
            inference_elapsed = time.time() - inference_start
            print(f"  [階段B] 推論+GradCAM 完成: {inference_elapsed:.2f} 秒")
            timing_stats['inference_time'] += inference_elapsed

            print(f"  熱力圖形狀: {heatmaps.shape}, 數值範圍: [{heatmaps.min():.4f}, {heatmaps.max():.4f}]")

            # 等待原圖快取完成
            print(f"  [等待原圖快取]...")
            cache_wait_start = time.time()
            cache_thread.join()
            print(f"  [原圖快取] 完成: {time.time() - cache_wait_start:.2f} 秒")

            # 所有圖片都視為有效（流水線模式）
            valid_indices = list(range(n_images))

            # ================================================================
            # 使用並行保存（12 執行緒）
            # ================================================================
            print(f"\n  正在並行保存圖片...")
            save_start = time.time()

            # 建立保存任務列表
            save_tasks = []
            for i in range(len(valid_indices)):
                valid_idx = valid_indices[i]
                img_path = image_paths[valid_idx]
                pred_class_idx = pred_classes[i]
                pred_class_name = CLASS_NAMES[pred_class_idx]
                pred_prob = pred_probs[i]
                all_class_probs = predictions[i]
                heatmap = heatmaps[i]

                task = (img_path, pred_class_name, pred_class_idx, heatmap, pred_prob,
                        all_class_probs, subfolder_output_dir, original_cache)
                save_tasks.append(task)

            # 執行並行保存
            save_results = save_results_parallel(save_tasks, max_workers=12)
            save_elapsed = time.time() - save_start
            print(f"  並行保存完成，耗時: {save_elapsed:.2f} 秒")
            timing_stats['save_time'] += save_elapsed
            timing_stats['total_images'] += len(valid_indices)

            # 收集信心度（整體和按類別）
            confidence_list.extend(pred_probs.tolist())
            for i, (cls_idx, prob) in enumerate(zip(pred_classes, pred_probs)):
                cls_name = CLASS_NAMES[cls_idx]
                confidence_by_class[cls_name].append(float(prob))

            # 統計分類結果並建立報告
            for result in save_results:
                if result:
                    pred_class_name = result['class']
                    class_counts[pred_class_name] += 1
                    total_class_counts[pred_class_name] += 1

                    # 建立結果字典
                    result_dict = {
                        '根目錄': root_dir_name,
                        '子資料夾': subfolder_name,
                        '原始路徑': result['path'],
                        '檔名': result['final_name'] + '.png',
                        '預測類別': pred_class_name,
                        '中文標籤': CHINESE_LABELS[result['class_idx']],
                        '最高信心度': f"{result['pred_prob']:.4f}",
                        '第二高信心度': f"{result['second_prob']:.4f}",
                        '與第二高差異': f"{result['confidence_gap']:.4f}",
                    }

                    for cls_idx, cls_name in enumerate(CLASS_NAMES):
                        chinese_label = CHINESE_LABELS[cls_idx]
                        result_dict[f'{cls_name}({chinese_label})_信心度'] = f"{result['all_probs'][cls_idx]:.4f}"

                    result_dict['輸出路徑'] = result['output_path']
                    root_results.append(result_dict)

            # 清除原圖快取釋放記憶體
            original_cache.clear()

            print(f"\n  子資料夾 '{subfolder_name}' 處理完成！")
            print(f"  分類結果統計:")
            for cls_name in CLASS_NAMES:
                if class_counts[cls_name] > 0:
                    chinese_label = CHINESE_LABELS[CLASS_NAMES.index(cls_name)]
                    print(f"    {cls_name} ({chinese_label}): {class_counts[cls_name]} 張")
            print(f"  輸出位置: {subfolder_output_dir}\n")

        # 生成報告
        root_end_time = datetime.now()
        root_duration = root_end_time - root_start_time

        if len(root_results) > 0:
            df = pd.DataFrame(root_results)
            csv_path = os.path.join(OUTPUT_ROOT_DIR, f'prediction_report_gradcam_{root_start_time.strftime("%Y%m%d_%H%M%S")}.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"CSV 報告已保存: {csv_path}")

            # 生成效能與統計報告
            generate_performance_report(
                OUTPUT_ROOT_DIR, timing_stats, total_class_counts,
                confidence_list, confidence_by_class,
                root_start_time, root_end_time, MODEL_PATH
            )

            print(f"\n{'='*80}")
            print(f"根目錄 '{root_dir_name}' 處理完成！")
            print(f"{'='*80}")
            print(f"處理時間: {root_duration}")
            print(f"成功預測的圖片總數: {len(root_results)}")

    global_end_time = datetime.now()
    global_duration = global_end_time - global_start_time

    print(f"\n{'='*80}")
    print("所有根目錄處理完成！")
    print(f"{'='*80}")
    print(f"總處理時間: {global_duration}")
    print(f"輸出基礎目錄: {OUTPUT_BASE_DIR}")


# ============================================================================
# 執行主程式
# ============================================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"模型檔案不存在: {MODEL_PATH}")
        exit(1)

    missing_dirs = [d for d in INPUT_ROOT_DIRS if not os.path.exists(d)]
    if missing_dirs:
        print(f"以下根目錄不存在:")
        for d in missing_dirs:
            print(f"  - {d}")
        print("\n程式將跳過不存在的目錄繼續執行...")

    process_all_subfolders()
