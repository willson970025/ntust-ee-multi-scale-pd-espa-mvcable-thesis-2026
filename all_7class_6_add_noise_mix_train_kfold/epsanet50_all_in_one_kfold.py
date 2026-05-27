# -*- coding: utf-8 -*-
"""
PRPD 局部放電分類模型訓練程式 - K-Fold 交叉驗證版本
使用 EPSANet-Large 進行七類別分類
TensorFlow 2.20.0 版本

特點：
- 5-Fold 分層交叉驗證 (StratifiedKFold)
- 合併 train + val 進行 k-fold，保留 test 做最終評估
- 每折輸出模型、混淆矩陣、訓練曲線
- 最終彙整平均準確率與標準差
- 使用最佳 fold 模型評估測試集
"""
# ============================================================================
# 抑制 TensorFlow/XLA 編譯器警告訊息
# ============================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import glob
import shutil
import numpy as np

# 在 import tensorflow 之前設定，抑制 ptxas 警告
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# [優化] 啟用 Mixed Precision Training 加速訓練 (約 30-50%)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

import math

# TensorFlow 2.16+ 使用獨立的 keras 套件
import keras
from keras import layers, models, Model
from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
from keras.callbacks import ModelCheckpoint
from keras.metrics import Precision, Recall
from keras.applications.resnet50 import preprocess_input

# Grad-CAM 所需
import cv2

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold  # K-Fold 分層抽樣
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import sys
import time
from datetime import datetime, timedelta

# 設定 Matplotlib 字型以支援中文
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
# K-Fold 交叉驗證設定
# ============================================================================
N_FOLDS = 5          # 折數
RANDOM_STATE = 42    # 隨機種子（確保可重複性）


# ============================================================================
# EPSANet-Large 模型定義 (已加入 Weight Decay)
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
        # [修改] 為 SE 模組的 Conv2D 加入 L2 正則化
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
        
        # [修改] 為 PSA 模組的所有 Conv2D 加入 L2 正則化
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
        # [修改] SE 模組也傳入 weight_decay
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
        # [修改] 為 EPSABlock 的所有 Conv2D 加入 L2 正則化
        self.conv1 = layers.Conv2D(
            self.out_channels, kernel_size=1, use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.weight_decay)
        )
        self.bn1 = layers.BatchNormalization()
        # [修改] PSA 模組傳入 weight_decay
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
            # [修改] Downsample Conv2D 也加入 L2 正則化
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
            'weight_decay': self.weight_decay
        })
        return config


def build_epsanet_large(input_shape=(80, 180, 3), num_classes=7, dropout_rate=0.2, 
                        weight_decay=WEIGHT_DECAY):
    """
    建立 EPSANet-Large 模型（適用於 PRPD 分類）
    
    重要：層命名與 2_epsanet_large.py 一致，方便遷移學習時複製權重
    
    Args:
        input_shape: 輸入張量形狀 (H, W, C)
        num_classes: 分類類別數
        dropout_rate: Dropout 比率
        weight_decay: L2 正則化係數 (新增參數)
        
    Returns:
        Keras Model
    """
    conv_groups = [32, 32, 32, 32]
    layers_config = [3, 4, 6, 3]
    channels = [128, 256, 512, 1024]  # EPSANet-Large 通道配置

    # 各層的 kernel 配置（根據特徵圖尺寸調整）
    # 特徵圖尺寸變化: 輸入 80x180 -> Stem 40x90 -> MaxPool 20x45
    # Layer1: 20x45 (stride=1), Layer2: 10x23 (stride=2)
    # Layer3: 5x12 (stride=2), Layer4: 3x6 (stride=2)
    # 過大的 kernel 會產生大量 padding，降低有效感受野
    layer_kernels = [
        [3, 5, 7, 9],  # Layer1: 20x45，維持原設計
        [3, 5, 7, 9],  # Layer2: 10x23，維持原設計
        [3, 3, 5, 5],  # Layer3: 5x12，縮小避免超過高度 5
        [3, 3, 3, 3],  # Layer4: 3x6，全部用 3x3 配合 3x6 特徵圖
    ]

    inputs = layers.Input(shape=input_shape)

    # Stem（命名與 2_epsanet_large.py 一致）
    # [修改] stem_conv 加入 L2 正則化
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

    # [新增] Grad-CAM 專用層：確保梯度能正確追蹤
    # 使用 linear activation（恆等函數），不影響模型計算結果
    x = layers.Activation('linear', name='gradcam_target')(x)

    # 分類頭（命名與遷移學習時一致）
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
# 輸入尺寸設定
# ============================================================================
INPUT_HEIGHT = 80
INPUT_WIDTH = 180
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, 3)

# ============================================================================
# Dropout 設定
# ============================================================================
DROPOUT_RATE = 0.2

# ============================================================================
# 多通道特徵增強設定
# ============================================================================
USE_MULTICHANNEL = True
SOBEL_KSIZE = 3

print(f"輸入尺寸設定: {INPUT_WIDTH}x{INPUT_HEIGHT} (寬x高)")
print(f"TensorFlow 輸入格式: {INPUT_SHAPE}")
print(f"Dropout 設定: {DROPOUT_RATE}")
print(f"Weight Decay (L2 正則化): {WEIGHT_DECAY}")  # [新增] 顯示 Weight Decay 設定
print(f"多通道特徵增強: {'啟用' if USE_MULTICHANNEL else '停用'}")
if USE_MULTICHANNEL:
    print(f"  通道 1: 原始圖像")
    print(f"  通道 2: Sobel X (垂直邊緣)")
    print(f"  通道 3: Sobel Y (水平邊緣)")


# 自訂預熱學習率調度器
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


# 類別定義
class_names = ['AH', 'CT', 'HT', 'TD', 'CD', 'ID', 'SD']
num_classes = len(class_names)
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
chinese_labels = ['空洞', '碳痕', '接頭異常', '不規則邊緣', '典型電暈放電', '典型內部放電', '典型表面放電']

# 資料路徑和輸出資料夾
data_dir = '/home/cckuo/m11307u09/GG/1126/data_original'
output_dir = '/home/cckuo/m11307u09/GG/1126/epsanet50/all in one/all_7class_6_add_noise_mix_train_kfold'

print("=" * 80)
print(f"EPSANet-Large {N_FOLDS}-Fold 交叉驗證訓練")
print("=" * 80)

os.makedirs(output_dir, exist_ok=True)

training_start_time = time.time()
training_start_datetime = datetime.now()

print(f"\n{'='*80}")
print(f"使用已分割的資料集: {os.path.basename(data_dir)}")
print(f"{'='*80}")
print(f"開始時間: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")


def load_data_from_split_dir(base_dir, class_to_idx):
    """從 train/val/test 目錄載入圖片路徑和標籤"""
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


train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

print("\n載入訓練集...")
train_paths, train_labels = load_data_from_split_dir(train_dir, class_to_idx)
print("載入驗證集...")
val_paths, val_labels = load_data_from_split_dir(val_dir, class_to_idx)
print("載入測試集...")
test_paths, test_labels = load_data_from_split_dir(test_dir, class_to_idx)

# 合併 train + val 進行 K-Fold
all_paths = np.concatenate([train_paths, val_paths])
all_labels = np.concatenate([train_labels, val_labels])

train_size = len(train_paths)
val_size = len(val_paths)
test_size = len(test_paths)
kfold_size = len(all_paths)
total_images = train_size + val_size + test_size

print(f"\n資料集統計:")
print(f"  總圖片數: {total_images} 張")
print(f"  原始訓練集: {train_size} 張")
print(f"  原始驗證集: {val_size} 張")
print(f"  K-Fold 資料集: {kfold_size} 張 (train + val 合併)")
print(f"  測試集: {test_size} 張 (保留做最終評估)")


def load_image(path, label):
    """載入並預處理圖片"""
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    
    gray = tf.image.rgb_to_grayscale(image)
    gray = tf.squeeze(gray, axis=-1)
    gray = tf.cast(gray, tf.float32)
    
    # [新增] 黑白翻轉：將黑色訊號點(0)變為白色(255)，白色背景(255)變為黑色(0)
    # 這樣模型可以學習到訊號點為高亮度值，更符合傳統影像處理的邏輯
    gray = 255.0 - gray
    
    if USE_MULTICHANNEL:
        channel_1 = gray
        
        sobel_x_kernel = tf.constant([
            [-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]
        ], dtype=tf.float32)
        sobel_x_kernel = tf.reshape(sobel_x_kernel, [3, 3, 1, 1])
        
        sobel_y_kernel = tf.constant([
            [-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]
        ], dtype=tf.float32)
        sobel_y_kernel = tf.reshape(sobel_y_kernel, [3, 3, 1, 1])
        
        gray_4d = tf.reshape(gray, [1, INPUT_HEIGHT, INPUT_WIDTH, 1])
        
        sobel_x = tf.nn.conv2d(gray_4d, sobel_x_kernel, strides=[1, 1, 1, 1], padding='SAME')
        sobel_x = tf.squeeze(sobel_x, axis=[0, 3])
        sobel_x = tf.abs(sobel_x)
        
        sobel_y = tf.nn.conv2d(gray_4d, sobel_y_kernel, strides=[1, 1, 1, 1], padding='SAME')
        sobel_y = tf.squeeze(sobel_y, axis=[0, 3])
        sobel_y = tf.abs(sobel_y)
        
        sobel_x_max = tf.reduce_max(sobel_x)
        sobel_x_normalized = tf.cond(sobel_x_max > 0, lambda: sobel_x / sobel_x_max * 255.0, lambda: sobel_x)
        
        sobel_y_max = tf.reduce_max(sobel_y)
        sobel_y_normalized = tf.cond(sobel_y_max > 0, lambda: sobel_y / sobel_y_max * 255.0, lambda: sobel_y)
        
        channel_2 = sobel_x_normalized
        channel_3 = sobel_y_normalized
        
        image = tf.stack([channel_1, channel_2, channel_3], axis=-1)
    else:
        image = tf.stack([gray, gray, gray], axis=-1)
    
    image = preprocess_input(image)
    label = tf.one_hot(label, num_classes)
    
    return image, label


# ============================================================================
# 循環水平位移資料擴增設定
# ============================================================================
AUGMENT_FACTOR = 14  # 1 張原始圖 + 13 張隨機位移圖 = 14 張
NOISE_COUNT = 8      # 14 張圖中隨機 8 張加噪聲

# [優化] 向量化版本的噪聲函數
def fill_rows_vectorized(image, rows_to_fill):
    """
    [優化] 向量化塗白多行，使用 scatter_nd 取代 tf.tile

    Args:
        image: (H, W, C)
        rows_to_fill: (K,) 要塗白的行號，-1 表示無效

    Returns:
        filled_image: (H, W, C)
    """
    height = tf.shape(image)[0]
    fill_value = tf.reduce_max(image)

    # 過濾掉無效行號 (-1)
    valid_mask = rows_to_fill >= 0
    valid_rows = tf.boolean_mask(rows_to_fill, valid_mask)
    valid_rows = tf.clip_by_value(valid_rows, 0, height - 1)

    # 若無有效行，直接返回
    num_valid = tf.shape(valid_rows)[0]

    def do_fill():
        # 使用 scatter_nd 建立稀疏遮罩（避免 tf.tile）
        row_indices = tf.expand_dims(valid_rows, 1)  # (K, 1)
        row_mask = tf.scatter_nd(
            row_indices,
            tf.ones([num_valid], dtype=tf.bool),
            [height]
        )  # (H,)

        # 廣播到完整形狀
        row_mask_expanded = tf.reshape(row_mask, [height, 1, 1])
        row_mask_broadcast = tf.broadcast_to(row_mask_expanded, tf.shape(image))

        return tf.where(row_mask_broadcast, fill_value, image)

    return tf.cond(num_valid > 0, do_fill, lambda: image)


def add_bottom_noise(image, noise_type):
    """
    [優化] 底部噪聲增強 - 向量化版本

    Args:
        image: 圖像 (H, W, C)，訊號點為白色（高值）
        noise_type: 噪聲類型 (1, 2, 3, 4)

    Returns:
        加噪聲後的圖像
    """
    gray = image[:, :, 0]  # (H, W)
    height = tf.shape(image)[0]

    # 判斷每行是否有訊號
    row_has_signal = tf.reduce_any(gray > 0, axis=1)  # (H,)

    # 由下往上找有訊號的列索引
    row_has_signal_reversed = tf.reverse(row_has_signal, axis=[0])
    signal_indices_reversed = tf.where(row_has_signal_reversed)
    signal_indices_reversed = tf.squeeze(signal_indices_reversed, axis=1)
    signal_rows = (height - 1) - tf.cast(signal_indices_reversed, tf.int32)

    num_signal_rows = tf.shape(signal_rows)[0]

    # 根據噪聲類型取得要塗白的行
    def get_rows_type_1():
        row = tf.cond(num_signal_rows >= 1, lambda: signal_rows[0:1], lambda: tf.constant([-1], tf.int32))
        return row

    def get_rows_type_2():
        row = tf.cond(num_signal_rows >= 2, lambda: signal_rows[1:2], lambda: tf.constant([-1], tf.int32))
        return row

    def get_rows_type_3():
        row = tf.cond(num_signal_rows >= 3, lambda: signal_rows[2:3], lambda: tf.constant([-1], tf.int32))
        return row

    def get_rows_type_4():
        def get_three_rows():
            start_row = signal_rows[0]
            rows = start_row - tf.range(3, dtype=tf.int32)  # [start, start-1, start-2]
            return tf.maximum(rows, -1)  # 負值設為 -1（無效）
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
    隨機像素點噪聲增強 (Type 5-8)

    選中列不塗白，而是：
    - 選中列的上一列：隨機 20-30 個像素點轉白
    - 選中列的下一列：隨機 4-10 個像素點轉白

    Args:
        image: 圖像 (H, W, C)
        noise_type: 噪聲類型 (5, 6, 7, 8)

    Returns:
        加噪聲後的圖像
    """
    gray = image[:, :, 0]
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    fill_value = tf.reduce_max(image)

    # 找出有訊號的列（由下往上）
    row_has_signal = tf.reduce_any(gray > 0, axis=1)
    row_has_signal_reversed = tf.reverse(row_has_signal, axis=[0])
    signal_indices_reversed = tf.where(row_has_signal_reversed)
    signal_indices_reversed = tf.squeeze(signal_indices_reversed, axis=1)
    signal_rows = (height - 1) - tf.cast(signal_indices_reversed, tf.int32)
    num_signal_rows = tf.shape(signal_rows)[0]

    def add_pixels_to_row(img, row_idx, num_pixels):
        """在指定列隨機加入像素點"""
        valid = tf.logical_and(row_idx >= 0, row_idx < height)

        def do_add():
            # 確保 num_pixels 不超過 width
            actual_pixels = tf.minimum(num_pixels, width)
            # 隨機選擇 x 座標
            x_coords = tf.random.shuffle(tf.range(width))[:actual_pixels]
            y_coords = tf.fill([actual_pixels], row_idx)

            # 建立更新索引
            indices = tf.stack([y_coords, x_coords], axis=1)

            # 對每個通道更新
            c0 = tf.tensor_scatter_nd_update(img[:, :, 0], indices, tf.fill([actual_pixels], fill_value))
            c1 = tf.tensor_scatter_nd_update(img[:, :, 1], indices, tf.fill([actual_pixels], fill_value))
            c2 = tf.tensor_scatter_nd_update(img[:, :, 2], indices, tf.fill([actual_pixels], fill_value))
            return tf.stack([c0, c1, c2], axis=-1)

        return tf.cond(valid, do_add, lambda: img)

    # 根據噪聲類型決定目標列
    def get_target_row_type_5():
        return tf.cond(num_signal_rows >= 1,
                       lambda: signal_rows[0],
                       lambda: tf.constant(-1, tf.int32))

    def get_target_row_type_6():
        return tf.cond(num_signal_rows >= 2,
                       lambda: signal_rows[1],
                       lambda: tf.constant(-1, tf.int32))

    def get_target_row_type_7():
        return tf.cond(num_signal_rows >= 3,
                       lambda: signal_rows[2],
                       lambda: tf.constant(-1, tf.int32))

    # Type 5, 6, 7: 單列處理
    def process_single_row(target_row):
        above_row = target_row - 1
        below_row = target_row + 1
        num_above = tf.random.uniform([], 20, 31, dtype=tf.int32)
        num_below = tf.random.uniform([], 4, 11, dtype=tf.int32)

        result = add_pixels_to_row(image, above_row, num_above)
        result = add_pixels_to_row(result, below_row, num_below)
        return result

    # Type 8: 區塊邊界處理（第 1 個有訊號列 + 上方 2 列的區塊）
    def process_block():
        def do_process():
            bottom_row = signal_rows[0]  # 最底部
            top_row = bottom_row - 2     # 上方 2 列
            above_row = top_row - 1      # 區塊上方
            below_row = bottom_row + 1   # 區塊下方
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


@tf.function  # [優化] 圖形編譯加速
def augment_with_circular_shift(image, label):
    """
    [優化] 循環水平位移資料擴增 + 底部噪聲 - 向量化版本
    - 使用 tf.map_fn 取代 Python for 迴圈
    - 使用 gather/scatter 取代 TensorArray

    Args:
        image: 預處理後的圖像 (H, W, C)
        label: one-hot 編碼的標籤

    Returns:
        images: 擴增後的圖像 (AUGMENT_FACTOR, H, W, C)
        labels: 複製的標籤 (AUGMENT_FACTOR, num_classes)
    """
    # [優化] 一次產生所有位移量
    shifts = tf.concat([
        [0],  # 第一張是原始圖（位移量 0）
        tf.random.uniform([AUGMENT_FACTOR - 1], 0, INPUT_WIDTH, dtype=tf.int32)
    ], axis=0)  # (AUGMENT_FACTOR,)

    # [優化] 使用 tf.map_fn 批次處理位移
    def shift_image(shift):
        return tf.roll(image, shift=shift, axis=1)

    images = tf.map_fn(shift_image, shifts, fn_output_signature=tf.TensorSpec(
        shape=[INPUT_HEIGHT, INPUT_WIDTH, 3], dtype=image.dtype
    ))  # (AUGMENT_FACTOR, H, W, C)

    # 隨機選擇 NOISE_COUNT 張圖加入噪聲
    all_indices = tf.random.shuffle(tf.range(AUGMENT_FACTOR))
    noise_indices = all_indices[:NOISE_COUNT]  # (NOISE_COUNT,)
    noise_types = tf.random.uniform([NOISE_COUNT], 1, 9, dtype=tf.int32)

    # [優化] 使用 gather/scatter 取代 TensorArray 迴圈
    # 取出要加噪聲的圖像
    noisy_images = tf.gather(images, noise_indices)  # (NOISE_COUNT, H, W, C)

    # 批次應用噪聲
    # Type 1-4: 整列塗白
    # Type 5-8: 隨機像素點
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
            shape=[INPUT_HEIGHT, INPUT_WIDTH, 3], dtype=image.dtype
        )
    )  # (NOISE_COUNT, H, W, C)

    # 將處理後的圖像散回原位置
    images = tf.tensor_scatter_nd_update(
        images,
        tf.expand_dims(noise_indices, 1),  # (NOISE_COUNT, 1)
        noisy_images_processed
    )

    # 標籤複製 AUGMENT_FACTOR 次
    labels = tf.tile(tf.expand_dims(label, 0), [AUGMENT_FACTOR, 1])

    return images, labels


# ============================================================================
# 訓練參數
# ============================================================================
batch_size = 128
epochs = 100

# 建立 Dataset 輔助函數
def create_dataset(paths, labels, augment=True, shuffle=True):
    """建立 tf.data.Dataset"""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.cache()
        ds = ds.map(augment_with_circular_shift, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.unbatch()
        if shuffle:
            ds = ds.shuffle(len(paths) * AUGMENT_FACTOR)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# 測試集（原始，不擴增）
test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 測試集（擴增版）
test_ds_augmented = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_ds_augmented = test_ds_augmented.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds_augmented = test_ds_augmented.map(augment_with_circular_shift, num_parallel_calls=tf.data.AUTOTUNE)
test_ds_augmented = test_ds_augmented.unbatch()
test_ds_augmented = test_ds_augmented.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_labels_augmented = np.repeat(test_labels, AUGMENT_FACTOR)
augmented_test_size = test_size * AUGMENT_FACTOR

print(f"\n訓練配置:")
print(f"  輸入尺寸: {INPUT_WIDTH}x{INPUT_HEIGHT}")
print(f"  Batch Size: {batch_size}")
print(f"  Epochs: {epochs}")
print(f"  K-Fold: {N_FOLDS} 折")
print(f"  Dropout: {DROPOUT_RATE}")
print(f"  Weight Decay: {WEIGHT_DECAY}")
print(f"  多通道特徵增強: {'啟用' if USE_MULTICHANNEL else '停用'}")
print(f"\n資料擴增配置:")
print(f"  循環水平位移: 啟用")
print(f"  擴增倍數: {AUGMENT_FACTOR}x (1 原始圖 + {AUGMENT_FACTOR-1} 隨機位移圖)")
print(f"  位移範圍: 0-{INPUT_WIDTH} 像素 (對應 0-360 度相位)")
print(f"  底部噪聲: 啟用 (每 {AUGMENT_FACTOR} 張隨機 {NOISE_COUNT} 張加噪聲)")
print(f"    噪聲類型 1: 由下往上第 1 個有訊號列塗白")
print(f"    噪聲類型 2: 由下往上第 2 個有訊號列塗白")
print(f"    噪聲類型 3: 由下往上第 3 個有訊號列塗白")
print(f"    噪聲類型 4: 由下往上第 1 個有訊號列 + 上方 2 列塗白")
print(f"    噪聲類型 5: 第 1 個有訊號列上下加隨機點")
print(f"    噪聲類型 6: 第 2 個有訊號列上下加隨機點")
print(f"    噪聲類型 7: 第 3 個有訊號列上下加隨機點")
print(f"    噪聲類型 8: 連續 3 列區塊上下邊界加隨機點")

# ============================================================================
# 模型架構資訊（先建立一個模型來取得參數量）
# ============================================================================
temp_model = build_epsanet_large(
    input_shape=INPUT_SHAPE,
    num_classes=num_classes,
    dropout_rate=DROPOUT_RATE,
    weight_decay=WEIGHT_DECAY
)
trainable_params = sum([tf.reduce_prod(w.shape).numpy() for w in temp_model.trainable_weights])
del temp_model

print(f"\n模型架構 (EPSANet-Large + Dropout + Weight Decay + 多通道特徵增強):")
print(f"  輸入尺寸: {INPUT_SHAPE} (HxWxC)")
print(f"  Base Model: EPSANet-Large")
print(f"  通道配置: [128, 256, 512, 1024] (Large 版本)")
print(f"  PSA Groups: [32, 32, 32, 32]")
print(f"  架構: EPSANet-Large -> GAP -> Dropout({DROPOUT_RATE}) -> Dense(1024) -> BN -> ReLU")
print(f"         -> Dense(512) -> BN -> ReLU -> Dense({num_classes})")
print(f"  L2 Regularization (Weight Decay): {WEIGHT_DECAY}")
print(f"  Loss: CategoricalCrossentropy (label_smoothing=0.05)")
print(f"  Optimizer: SGD (momentum=0.9, nesterov=True)")
print(f"  可訓練參數量: {trainable_params:,}")

# ============================================================================
# t-SNE 特徵視覺化函數
# ============================================================================
def generate_tsne_visualization(model, test_ds, test_labels, output_dir, test_size, title_prefix=""):
    """
    從模型的三個層提取特徵並生成 t-SNE 視覺化圖

    特徵提取層：
    - global_avg_pool: GAP 層輸出 (4096 維)
    - head_fc1: 第一個全連接層輸出 (1024 維)
    - head_fc2: 第二個全連接層輸出 (512 維)

    Args:
        model: 訓練好的 Keras 模型
        test_ds: 測試集 tf.data.Dataset
        test_labels: 測試集標籤 (numpy array)
        output_dir: 輸出目錄
        test_size: 測試集樣本數
        title_prefix: 圖片標題前綴（用於 K-Fold 區分）
    """
    print(f"\n{'='*80}")
    print(f"{title_prefix}生成 t-SNE 特徵視覺化圖...")
    print("="*80)

    # 建立三個特徵提取模型
    print("\n建立特徵提取模型...")
    feature_layers = ['global_avg_pool', 'head_fc1', 'head_fc2']
    feature_models = {}

    for layer_name in feature_layers:
        try:
            layer_output = model.get_layer(layer_name).output
            feature_models[layer_name] = Model(
                inputs=model.input,
                outputs=layer_output
            )
            print(f"  {layer_name}: {layer_output.shape}")
        except ValueError as e:
            print(f"  警告：找不到層 {layer_name}: {e}")

    if not feature_models:
        print("  錯誤：無法建立任何特徵提取模型，跳過 t-SNE 視覺化")
        return

    # 提取特徵
    print("\n提取特徵...")
    features_dict = {name: [] for name in feature_models.keys()}

    for batch_images, _ in test_ds:
        for layer_name, feat_model in feature_models.items():
            batch_features = feat_model.predict(batch_images, verbose=0)
            features_dict[layer_name].append(batch_features)

    # 合併批次
    for layer_name in features_dict:
        features_dict[layer_name] = np.concatenate(features_dict[layer_name], axis=0)
        print(f"  {layer_name}: {features_dict[layer_name].shape}")

    # t-SNE 降維
    print("\n執行 t-SNE 降維...")
    tsne_results = {}
    perplexity = 30

    for layer_name, features in features_dict.items():
        print(f"  處理 {layer_name} ({features.shape[1]} 維 -> 2 維)...")
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            max_iter=1000,
            init='pca'
        )
        tsne_results[layer_name] = tsne.fit_transform(features)
        print(f"    完成！")

    # 視覺化顏色設定
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    # 繪製單獨的 t-SNE 圖
    print("\n繪製 t-SNE 視覺化圖...")

    for layer_name, tsne_data in tsne_results.items():
        plt.figure(figsize=(10, 8))

        for class_idx in range(num_classes):
            mask = test_labels == class_idx
            plt.scatter(
                tsne_data[mask, 0],
                tsne_data[mask, 1],
                c=[colors[class_idx]],
                label=f"{class_names[class_idx]} ({chinese_labels[class_idx]})",
                alpha=0.7,
                s=30
            )

        plt.xlabel('t-SNE 維度 1', fontsize=12)
        plt.ylabel('t-SNE 維度 2', fontsize=12)
        plt.title(f'{title_prefix}t-SNE 視覺化 - {layer_name}\n(perplexity={perplexity}, 測試集 {test_size} 張)',
                  fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(output_dir, f'tsne_{layer_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已儲存: {save_path}")

    # 繪製三圖合併版
    print("\n繪製合併比較圖...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax_idx, (layer_name, tsne_data) in enumerate(tsne_results.items()):
        ax = axes[ax_idx]

        for class_idx in range(num_classes):
            mask = test_labels == class_idx
            ax.scatter(
                tsne_data[mask, 0],
                tsne_data[mask, 1],
                c=[colors[class_idx]],
                label=f"{class_names[class_idx]}",
                alpha=0.7,
                s=20
            )

        ax.set_xlabel('t-SNE 維度 1', fontsize=11)
        ax.set_ylabel('t-SNE 維度 2', fontsize=11)
        ax.set_title(f'{layer_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if ax_idx == 2:
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(f'{title_prefix}t-SNE 特徵視覺化比較 (測試集 {test_size} 張, perplexity={perplexity})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path_comparison = os.path.join(output_dir, 'tsne_comparison.png')
    plt.savefig(save_path_comparison, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已儲存: {save_path_comparison}")

    print("\nt-SNE 視覺化完成！")


# ============================================================================
# K-Fold 交叉驗證訓練迴圈
# ============================================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# 儲存結果
fold_results = []
all_val_preds = []
all_val_labels_list = []

target_lr = 0.002
min_lr = 1e-5

for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels)):
    print(f"\n{'='*80}")
    print(f"Fold {fold+1}/{N_FOLDS}")
    print(f"{'='*80}")

    fold_start_time = time.time()

    # 建立 fold 輸出目錄
    fold_dir = os.path.join(output_dir, f'fold_{fold+1}')
    os.makedirs(fold_dir, exist_ok=True)

    # 分割資料
    fold_train_paths = all_paths[train_idx]
    fold_train_labels = all_labels[train_idx]
    fold_val_paths = all_paths[val_idx]
    fold_val_labels = all_labels[val_idx]

    print(f"  訓練集: {len(fold_train_paths)} 張")
    print(f"  驗證集: {len(fold_val_paths)} 張")

    # 計算類別權重
    fold_class_weights = compute_class_weight('balanced', classes=np.unique(fold_train_labels), y=fold_train_labels)
    fold_class_weights = dict(enumerate(fold_class_weights))

    # 建立 Dataset
    train_ds = create_dataset(fold_train_paths, fold_train_labels, augment=True, shuffle=True)
    val_ds = create_dataset(fold_val_paths, fold_val_labels, augment=True, shuffle=False)

    # 計算 steps
    augmented_train_size = len(fold_train_paths) * AUGMENT_FACTOR
    steps_per_epoch = augmented_train_size // batch_size

    # 建立新模型（每折重新初始化）
    keras.backend.clear_session()
    model = build_epsanet_large(
        input_shape=INPUT_SHAPE,
        num_classes=num_classes,
        dropout_rate=DROPOUT_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # 學習率調度
    warmup_steps = int(0.1 * epochs * steps_per_epoch)
    lr_schedule = WarmupCosineDecay(
        warmup_steps=warmup_steps,
        target_lr=target_lr,
        total_steps=epochs * steps_per_epoch,
        min_lr=min_lr
    )

    model.compile(
        optimizer=SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')],
        jit_compile=True
    )

    # Checkpoint
    model_path = os.path.join(fold_dir, f'best_model_fold{fold+1}.keras')
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # 訓練
    print(f"\n開始訓練 Fold {fold+1}...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=fold_class_weights,
        callbacks=[checkpoint],
        verbose=2
    )

    # 載入最佳模型
    best_model = models.load_model(model_path)

    # 評估驗證集（不擴增）
    val_ds_eval = create_dataset(fold_val_paths, fold_val_labels, augment=False, shuffle=False)
    preds = best_model.predict(val_ds_eval)
    pred_labels = np.argmax(preds, axis=1)

    fold_acc = np.sum(pred_labels == fold_val_labels) / len(fold_val_labels)
    _, _, f1_scores_fold, _ = precision_recall_fscore_support(fold_val_labels, pred_labels, average=None)
    cm_fold = confusion_matrix(fold_val_labels, pred_labels)

    fold_duration = time.time() - fold_start_time

    print(f"\nFold {fold+1} 結果:")
    print(f"  驗證準確率: {fold_acc:.4f}")
    print(f"  訓練時間: {str(timedelta(seconds=int(fold_duration)))}")

    # 儲存結果
    fold_results.append({
        'fold': fold + 1,
        'accuracy': fold_acc,
        'f1_scores': f1_scores_fold,
        'confusion_matrix': cm_fold,
        'history': history.history,
        'duration': fold_duration
    })

    all_val_preds.extend(pred_labels)
    all_val_labels_list.extend(fold_val_labels)

    # 繪製訓練曲線
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Fold {fold+1} Loss Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Fold {fold+1} Accuracy Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(history.history['precision'], label='Train Precision', linewidth=2)
    plt.plot(history.history['val_precision'], label='Val Precision', linewidth=2)
    plt.plot(history.history['recall'], label='Train Recall', linewidth=2, linestyle='--')
    plt.plot(history.history['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Fold {fold+1} Precision & Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    loss_gap = np.array(history.history['val_loss']) - np.array(history.history['loss'])
    acc_gap = np.array(history.history['val_accuracy']) - np.array(history.history['accuracy'])
    plt.plot(loss_gap, label='Val-Train Loss Gap', linewidth=2)
    plt.plot(acc_gap, label='Val-Train Acc Gap', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Gap (Val - Train)', fontsize=12)
    plt.title(f'Fold {fold+1} Overfitting Analysis', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, f'training_curves_fold{fold+1}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 繪製混淆矩陣
    plt.figure(figsize=(10, 8))
    cm_norm = cm_fold.astype('float') / cm_fold.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_fold, annot=False, fmt='d', cmap='Blues', xticklabels=chinese_labels, yticklabels=chinese_labels)
    for j in range(cm_fold.shape[0]):
        for k in range(cm_fold.shape[1]):
            plt.text(k + 0.5, j + 0.5, f"{cm_fold[j, k]}\n({cm_norm[j, k]:.1f}%)",
                     ha='center', va='center', color='black', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.ylabel('True Label', fontsize=13)
    plt.title(f'Fold {fold+1} 混淆矩陣 (Acc: {fold_acc:.4f})', fontsize=15, fontweight='bold')
    plt.savefig(os.path.join(fold_dir, f'confusion_matrix_fold{fold+1}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # 生成 t-SNE 視覺化（針對此 fold 的模型）
    # ========================================================================
    generate_tsne_visualization(
        best_model, test_ds, test_labels, fold_dir, test_size,
        title_prefix=f"Fold {fold+1} - "
    )

    print(f"  輸出已保存至: {fold_dir}")

# ============================================================================
# K-Fold 結果彙整
# ============================================================================
print(f"\n{'='*80}")
print("K-Fold 交叉驗證結果彙整")
print("="*80)

accuracies = [r['accuracy'] for r in fold_results]
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print(f"\n各 Fold 準確率:")
for r in fold_results:
    print(f"  Fold {r['fold']}: {r['accuracy']:.4f}")

print(f"\n{N_FOLDS}-Fold 平均準確率: {mean_acc:.4f} ± {std_acc:.4f}")

# 整體混淆矩陣（合併所有 fold 驗證集預測）
overall_cm = confusion_matrix(all_val_labels_list, all_val_preds)

plt.figure(figsize=(10, 8))
cm_norm = overall_cm.astype('float') / overall_cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(overall_cm, annot=False, fmt='d', cmap='Blues', xticklabels=chinese_labels, yticklabels=chinese_labels)
for j in range(overall_cm.shape[0]):
    for k in range(overall_cm.shape[1]):
        plt.text(k + 0.5, j + 0.5, f"{overall_cm[j, k]}\n({cm_norm[j, k]:.1f}%)",
                 ha='center', va='center', color='black', fontsize=10)
plt.xlabel('Predicted Label', fontsize=13)
plt.ylabel('True Label', fontsize=13)
plt.title(f'{N_FOLDS}-Fold 整體混淆矩陣 (Avg Acc: {mean_acc:.4f} ± {std_acc:.4f})', fontsize=15, fontweight='bold')
plt.savefig(os.path.join(output_dir, 'overall_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n整體混淆矩陣已保存: {os.path.join(output_dir, 'overall_confusion_matrix.png')}")

# K-Fold 比較圖
plt.figure(figsize=(10, 6))
folds = [r['fold'] for r in fold_results]
plt.bar(folds, accuracies, color='steelblue', alpha=0.8)
plt.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.4f}')
plt.fill_between([0.5, N_FOLDS + 0.5], mean_acc - std_acc, mean_acc + std_acc,
                 color='red', alpha=0.1, label=f'±Std: {std_acc:.4f}')
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title(f'{N_FOLDS}-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.xticks(folds)
plt.ylim([min(accuracies) - 0.05, max(accuracies) + 0.05])
plt.grid(True, alpha=0.3, axis='y')
plt.savefig(os.path.join(output_dir, 'kfold_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"K-Fold 比較圖已保存: {os.path.join(output_dir, 'kfold_comparison.png')}")

# 繪製所有 Fold 的 Accuracy 曲線彙整圖（Train vs Val 分開左右）
colors = plt.cm.tab10(np.linspace(0, 1, N_FOLDS))

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
for i, r in enumerate(fold_results):
    epochs = range(1, len(r['history']['accuracy']) + 1)
    plt.plot(epochs, r['history']['accuracy'], color=colors[i], linewidth=2,
             label=f'Fold {r["fold"]}')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title(f'{N_FOLDS}-Fold Train Accuracy', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for i, r in enumerate(fold_results):
    epochs = range(1, len(r['history']['val_accuracy']) + 1)
    plt.plot(epochs, r['history']['val_accuracy'], color=colors[i], linewidth=2,
             label=f'Fold {r["fold"]} (Best: {max(r["history"]["val_accuracy"]):.4f})')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title(f'{N_FOLDS}-Fold Validation Accuracy', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'kfold_accuracy_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"K-Fold Accuracy 曲線圖已保存: {os.path.join(output_dir, 'kfold_accuracy_curves.png')}")

# 繪製所有 Fold 的 Loss 曲線彙整圖（Train vs Val 分開左右）
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
for i, r in enumerate(fold_results):
    epochs = range(1, len(r['history']['loss']) + 1)
    plt.plot(epochs, r['history']['loss'], color=colors[i], linewidth=2,
             label=f'Fold {r["fold"]}')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title(f'{N_FOLDS}-Fold Train Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for i, r in enumerate(fold_results):
    epochs = range(1, len(r['history']['val_loss']) + 1)
    plt.plot(epochs, r['history']['val_loss'], color=colors[i], linewidth=2,
             label=f'Fold {r["fold"]} (Best: {min(r["history"]["val_loss"]):.4f})')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title(f'{N_FOLDS}-Fold Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'kfold_loss_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"K-Fold Loss 曲線圖已保存: {os.path.join(output_dir, 'kfold_loss_curves.png')}")

# ============================================================================
# 測試集最終評估（使用最佳 fold 模型）
# ============================================================================
print(f"\n{'='*80}")
print("測試集最終評估")
print("="*80)

# 找出最佳 fold
best_fold_idx = np.argmax(accuracies)
best_fold = fold_results[best_fold_idx]
best_model_path = os.path.join(output_dir, f'fold_{best_fold["fold"]}', f'best_model_fold{best_fold["fold"]}.keras')

print(f"\n使用最佳 Fold 模型 (Fold {best_fold['fold']}, Val Acc: {best_fold['accuracy']:.4f})")

best_model = models.load_model(best_model_path)

# 原始測試集評估
preds = best_model.predict(test_ds)
pred_labels = np.argmax(preds, axis=1)

precision, recall, f1_scores, support = precision_recall_fscore_support(test_labels, pred_labels, average=None)
cm = confusion_matrix(test_labels, pred_labels)
test_acc_original = np.sum(pred_labels == test_labels) / len(test_labels)

print(f"\n原始測試集表現:")
print(f"  準確率: {test_acc_original:.4f}")
print(f"\n各類別 F1 分數:")
for i, cls_name in enumerate(class_names):
    print(f"  {cls_name}: {f1_scores[i]:.4f} (Support: {support[i]})")

# 擴增測試集評估
print(f"\n擴增測試集評估（循環位移 + 底部噪聲）:")
preds_aug = best_model.predict(test_ds_augmented)
pred_labels_aug = np.argmax(preds_aug, axis=1)

precision_aug, recall_aug, f1_scores_aug, support_aug = precision_recall_fscore_support(
    test_labels_augmented, pred_labels_aug, average=None
)
cm_aug = confusion_matrix(test_labels_augmented, pred_labels_aug)
test_acc_augmented = np.sum(pred_labels_aug == test_labels_augmented) / len(test_labels_augmented)

print(f"  準確率: {test_acc_augmented:.4f}")
print(f"  樣本數: {len(test_labels_augmented)} 張 (原始 {test_size} 張 x {AUGMENT_FACTOR})")

# 測試集比較
print(f"\n測試集比較:")
print(f"  原始測試集準確率: {test_acc_original:.4f}")
print(f"  擴增測試集準確率: {test_acc_augmented:.4f}")
print(f"  差異: {(test_acc_augmented - test_acc_original)*100:+.2f}%")

# 原始測試集混淆矩陣
plt.figure(figsize=(10, 8))
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=chinese_labels, yticklabels=chinese_labels)
for j in range(cm.shape[0]):
    for k in range(cm.shape[1]):
        plt.text(k + 0.5, j + 0.5, f"{cm[j, k]}\n({cm_norm[j, k]:.1f}%)",
                 ha='center', va='center', color='black', fontsize=11)
plt.xlabel('Predicted Label', fontsize=13)
plt.ylabel('True Label', fontsize=13)
plt.title(f'測試集混淆矩陣 - 原始 (Acc: {test_acc_original:.4f})', fontsize=15, fontweight='bold')
save_path_cm_original = os.path.join(output_dir, 'test_confusion_matrix_original.png')
plt.savefig(save_path_cm_original, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n原始測試集混淆矩陣已保存: {save_path_cm_original}")

# 擴增測試集混淆矩陣
plt.figure(figsize=(10, 8))
cm_aug_norm = cm_aug.astype('float') / cm_aug.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_aug, annot=False, fmt='d', cmap='Greens', xticklabels=chinese_labels, yticklabels=chinese_labels)
for j in range(cm_aug.shape[0]):
    for k in range(cm_aug.shape[1]):
        plt.text(k + 0.5, j + 0.5, f"{cm_aug[j, k]}\n({cm_aug_norm[j, k]:.1f}%)",
                 ha='center', va='center', color='black', fontsize=11)
plt.xlabel('Predicted Label', fontsize=13)
plt.ylabel('True Label', fontsize=13)
plt.title(f'測試集混淆矩陣 - 擴增 (Acc: {test_acc_augmented:.4f})', fontsize=15, fontweight='bold')
save_path_cm_augmented = os.path.join(output_dir, 'test_confusion_matrix_augmented.png')
plt.savefig(save_path_cm_augmented, dpi=300, bbox_inches='tight')
plt.close()

print(f"擴增測試集混淆矩陣已保存: {save_path_cm_augmented}")

# ============================================================================
# 保存報告
# ============================================================================
training_end_time = time.time()
training_duration = training_end_time - training_start_time

report_path = os.path.join(output_dir, 'kfold_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write(f"EPSANet-Large {N_FOLDS}-Fold 交叉驗證訓練報告\n")
    f.write("="*80 + "\n\n")

    f.write(f"訓練時間: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"總訓練時長: {str(timedelta(seconds=int(training_duration)))}\n\n")

    f.write(f"K-Fold 設定:\n")
    f.write(f"  折數: {N_FOLDS}\n")
    f.write(f"  隨機種子: {RANDOM_STATE}\n")
    f.write(f"  K-Fold 資料集: {kfold_size} 張 (train + val 合併)\n")
    f.write(f"  測試集: {test_size} 張 (保留做最終評估)\n\n")

    f.write(f"輸入設定:\n")
    f.write(f"  輸入尺寸: {INPUT_WIDTH}x{INPUT_HEIGHT} (寬x高)\n")
    f.write(f"  TensorFlow 格式: {INPUT_SHAPE}\n")
    f.write(f"  多通道特徵增強: {'啟用' if USE_MULTICHANNEL else '停用'}\n")
    if USE_MULTICHANNEL:
        f.write(f"    通道 1: 原始圖像\n")
        f.write(f"    通道 2: Sobel X (垂直邊緣)\n")
        f.write(f"    通道 3: Sobel Y (水平邊緣)\n\n")

    f.write(f"模型架構:\n")
    f.write(f"  Base Model: EPSANet-Large (from scratch)\n")
    f.write(f"  通道配置: [128, 256, 512, 1024]\n")
    f.write(f"  PSA Groups: [32, 32, 32, 32]\n")
    f.write(f"  可訓練參數量: {trainable_params:,}\n")
    f.write(f"  Dropout: {DROPOUT_RATE}\n")
    f.write(f"  Weight Decay: {WEIGHT_DECAY}\n")
    f.write(f"  Loss: CategoricalCrossentropy (label_smoothing=0.05)\n\n")

    f.write(f"訓練配置:\n")
    f.write(f"  Batch Size: {batch_size}\n")
    f.write(f"  Epochs: {epochs}\n")
    f.write(f"  Learning Rate: 0 -> {target_lr} -> {min_lr} (Warmup + Cosine Decay)\n")
    f.write(f"  Optimizer: SGD (momentum=0.9, nesterov=True)\n\n")

    f.write(f"資料擴增:\n")
    f.write(f"  循環水平位移: 啟用\n")
    f.write(f"  擴增倍數: {AUGMENT_FACTOR}x\n")
    f.write(f"  底部噪聲: 啟用 (每 {AUGMENT_FACTOR} 張隨機 {NOISE_COUNT} 張加噪聲)\n\n")

    f.write("=" * 40 + "\n")
    f.write("各 Fold 結果:\n")
    f.write("=" * 40 + "\n")
    for r in fold_results:
        f.write(f"  Fold {r['fold']}: {r['accuracy']:.4f} (訓練時間: {str(timedelta(seconds=int(r['duration'])))})\n")

    f.write(f"\n{N_FOLDS}-Fold 平均準確率: {mean_acc:.4f} ± {std_acc:.4f}\n\n")

    f.write("=" * 40 + "\n")
    f.write(f"測試集結果 (使用 Fold {best_fold['fold']} 模型):\n")
    f.write("=" * 40 + "\n")
    f.write(f"  原始測試集準確率: {test_acc_original:.4f}\n")
    f.write(f"  擴增測試集準確率: {test_acc_augmented:.4f}\n")
    f.write(f"  差異: {(test_acc_augmented - test_acc_original)*100:+.2f}%\n\n")

    f.write(f"各類別 F1 分數（原始測試集）:\n")
    for i, cls_name in enumerate(class_names):
        f.write(f"  {cls_name} ({chinese_labels[i]}): {f1_scores[i]:.4f} (Support: {support[i]})\n")

    f.write(f"\n各類別 F1 分數（擴增測試集）:\n")
    for i, cls_name in enumerate(class_names):
        f.write(f"  {cls_name} ({chinese_labels[i]}): {f1_scores_aug[i]:.4f} (Support: {support_aug[i]})\n")

print(f"\n報告已保存: {report_path}")

# 最終摘要
print(f"\n{'='*80}")
print("訓練完成！")
print("="*80)
print(f"  {N_FOLDS}-Fold 平均準確率: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"  最佳 Fold: {best_fold['fold']} (Val Acc: {best_fold['accuracy']:.4f})")
print(f"  測試集準確率（原始）: {test_acc_original:.4f}")
print(f"  測試集準確率（擴增）: {test_acc_augmented:.4f}")
print(f"  總訓練時間: {str(timedelta(seconds=int(training_duration)))}")
print(f"\n輸出目錄: {output_dir}")
print(f"\n每個 Fold 資料夾包含:")
print(f"  - best_model_foldX.keras (最佳模型)")
print(f"  - training_curves_foldX.png (訓練曲線)")
print(f"  - confusion_matrix_foldX.png (混淆矩陣)")
print(f"  - tsne_global_avg_pool.png (t-SNE GAP層)")
print(f"  - tsne_head_fc1.png (t-SNE FC1層)")
print(f"  - tsne_head_fc2.png (t-SNE FC2層)")
print(f"  - tsne_comparison.png (t-SNE 比較圖)")
print("="*80)