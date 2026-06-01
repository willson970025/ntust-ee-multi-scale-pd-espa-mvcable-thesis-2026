# -*- coding: utf-8 -*-
"""
PRPD 局部放電分類模型訓練程式
使用 ResNet18 進行七類別分類
TensorFlow 2.20.0 版本

修改版本：
- 使用 ResNet18 替代 ResNet50
- 輸入尺寸改為 180x80
- 加入 Dropout 防止過擬合
- 多通道特徵增強：翻轉後二值化影像 + x 方向 Sobel 梯度響應 |G_x| + y 方向 Sobel 梯度響應 |G_y|
- 訓練完成後自動儲存錯誤分類的圖片
- 層命名與 2_epsanet_large.py 一致，方便遷移學習
- [新增] 為所有 Conv2D 層加入 L2 正則化 (Weight Decay)
- [新增] 黑白翻轉：將訊號點從黑色翻轉為白色，讓模型正確學習訊號分佈
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
# Weight Decay 設定 (新增)
# ============================================================================
WEIGHT_DECAY = 1e-5  # 從 1e-4 降到 1e-5


# ============================================================================
# ResNet18 模型定義 (Baseline，含 Weight Decay)
# ============================================================================

def build_resnet18(input_shape=(80, 180, 3), num_classes=7, dropout_rate=0.2,
                   weight_decay=WEIGHT_DECAY):
    """
    建立 ResNet18 模型（PRPD 七類別分類 baseline）

    - 從頭訓練，不使用 ImageNet 預訓練權重
    - 使用 BasicBlock，block 數 [2, 2, 2, 2]，通道 [64, 128, 256, 512]
    - stem 與分類頭層命名與原 EPSANet 主模型一致，方便遷移學習與 t-SNE 特徵提取
    - 所有 Conv2D / Dense 皆加上 L2 正則化 (Weight Decay)
    """
    reg = keras.regularizers.l2(weight_decay)

    def basic_block(x, planes, stride, name):
        identity = x

        out = layers.Conv2D(planes, 3, strides=stride, padding='same', use_bias=False,
                            kernel_regularizer=reg, name=f'{name}_conv1')(x)
        out = layers.BatchNormalization(name=f'{name}_bn1')(out)
        out = layers.ReLU(name=f'{name}_relu1')(out)

        out = layers.Conv2D(planes, 3, strides=1, padding='same', use_bias=False,
                            kernel_regularizer=reg, name=f'{name}_conv2')(out)
        out = layers.BatchNormalization(name=f'{name}_bn2')(out)

        in_channels = identity.shape[-1]
        if stride != 1 or in_channels != planes:
            identity = layers.Conv2D(planes, 1, strides=stride, use_bias=False,
                                     kernel_regularizer=reg, name=f'{name}_downsample_conv')(identity)
            identity = layers.BatchNormalization(name=f'{name}_downsample_bn')(identity)

        out = layers.Add(name=f'{name}_add')([out, identity])
        out = layers.ReLU(name=f'{name}_relu2')(out)
        return out

    blocks = [2, 2, 2, 2]
    channels = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]

    inputs = layers.Input(shape=input_shape)

    # Stem（命名與原 EPSANet 主模型一致）
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False,
                      kernel_initializer='he_normal', name='stem_conv',
                      kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(name='stem_relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='stem_pool')(x)

    # 四個 stage
    for stage_idx in range(4):
        for block_idx in range(blocks[stage_idx]):
            stride = strides[stage_idx] if block_idx == 0 else 1
            x = basic_block(x, channels[stage_idx], stride,
                            name=f'layer{stage_idx + 1}_block{block_idx}')

    # Grad-CAM 專用層（linear activation，不影響輸出）
    x = layers.Activation('linear', name='gradcam_target')(x)

    # 分類頭（命名與原 EPSANet 主模型完全一致，供 t-SNE 直接使用）
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(dropout_rate, name='head_dropout')(x)
    x = layers.Dense(1024, kernel_regularizer=reg, name='head_fc1')(x)
    x = layers.BatchNormalization(name='head_bn1')(x)
    x = layers.ReLU(name='head_relu1')(x)
    x = layers.Dense(512, kernel_regularizer=reg, name='head_fc2')(x)
    x = layers.BatchNormalization(name='head_bn2')(x)
    x = layers.ReLU(name='head_relu2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=outputs, name='ResNet18_PRPD')

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
    print(f"  通道 1: 翻轉後二值化影像")
    print(f"  通道 2: x 方向 Sobel 梯度響應 |G_x|")
    print(f"  通道 3: y 方向 Sobel 梯度響應 |G_y|")


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
data_dir = '/home/wills/GG/data_original' 
output_dir = '/home/wills/GG/0601/model_compare_resnet18'

print("=" * 80)
print(f"ResNet18 7類別模型訓練 - 輸入尺寸: {INPUT_WIDTH}x{INPUT_HEIGHT} - 含 Dropout + Weight Decay")
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
class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
class_weights = dict(zip(unique_classes, class_weights))

print(f"\n類別權重:")
for cls_idx, cls_name in enumerate(class_names):
    if cls_idx in class_weights:
        print(f"  {cls_name}: {class_weights[cls_idx]:.4f}")


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
batch_size = 128 # 從 256 改為 128
epochs = 100

# 訓練集：擴增（1 張變 14 張，每個 epoch 隨機位移與加噪）
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.cache()  # [優化] 快取載入後的圖片，避免重複 I/O
train_ds = train_ds.map(augment_with_circular_shift, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.unbatch()  # 展開：(10, H, W, C) -> 10 個 (H, W, C)
train_ds = train_ds.shuffle(train_size * AUGMENT_FACTOR).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 驗證集：未擴增（與 K-Fold 版一致）
# 作為 validation_data、ModelCheckpoint(val_accuracy) 與最佳權重選取的唯一依據，
# 確保 val_accuracy / val_loss 反映模型在原始相位分佈上的表現，而非擴增資料。
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.cache()  # [優化] 快取載入後的圖片
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 測試集：不擴增（保持原始圖）
test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# [新增] 測試集（擴增版）：僅作為相位平移與雜訊擾動下的「穩健性觀察」，
# 不作為主要效能依據（主要效能一律以原始未擴增測試集 test_ds 為準）。
test_ds_augmented = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_ds_augmented = test_ds_augmented.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds_augmented = test_ds_augmented.map(augment_with_circular_shift, num_parallel_calls=tf.data.AUTOTUNE)
test_ds_augmented = test_ds_augmented.unbatch()
test_ds_augmented = test_ds_augmented.cache()  # 擴增測試資料於第一次生成後固定，確保評估可重現
test_ds_augmented = test_ds_augmented.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# [新增] 擴增後測試集的標籤（每張原始圖複製 AUGMENT_FACTOR 次）
test_labels_augmented = np.repeat(test_labels, AUGMENT_FACTOR)

# 擴增後的資料量（僅訓練集與「穩健性觀察用」的擴增測試集會擴增；驗證集不擴增）
augmented_train_size = train_size * AUGMENT_FACTOR
augmented_test_size = test_size * AUGMENT_FACTOR
steps_per_epoch = math.ceil(augmented_train_size / batch_size)  # batch() 未設 drop_remainder，以 ceil 計算（與 K-Fold 版一致）

print(f"\n訓練配置:")
print(f"  輸入尺寸: {INPUT_WIDTH}x{INPUT_HEIGHT}")
print(f"  Batch Size: {batch_size}")
print(f"  Epochs: {epochs}")
print(f"  Steps per Epoch: {steps_per_epoch}")
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
print(f"  擴增後訓練集: {augmented_train_size} 張 (原始 {train_size} 張)")
print(f"  驗證集（未擴增）: {val_size} 張 (主要 validation 指標來源)")
print(f"  測試集（原始，主要效能）: {test_size} 張 (不擴增)")
print(f"  測試集（擴增，僅作穩健性觀察）: {augmented_test_size} 張")

# ============================================================================
# 建立 ResNet18 模型
# ============================================================================
model = build_resnet18(
    input_shape=INPUT_SHAPE,
    num_classes=num_classes,
    dropout_rate=DROPOUT_RATE,
    weight_decay=WEIGHT_DECAY  # [新增] 傳入 Weight Decay 參數
)

target_lr = 0.002 # 從 0.004 改為 0.002（線性縮放）
min_lr = 1e-5
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
    jit_compile=True  # [優化] XLA 編譯加速
)

print(f"\n模型架構 (ResNet18 Baseline + Dropout + Weight Decay + 多通道特徵增強):")
print(f"  輸入尺寸: {INPUT_SHAPE} (HxWxC)")
print(f"  Base Model: ResNet18 (BasicBlock, blocks=[2, 2, 2, 2])")
print(f"  通道配置: [64, 128, 256, 512]")
print(f"  架構: ResNet18 -> GAP -> Dropout({DROPOUT_RATE}) -> Dense(1024) -> BN -> ReLU")
print(f"         -> Dense(512) -> BN -> ReLU -> Dense({num_classes})")
print(f"  L2 Regularization (Weight Decay): {WEIGHT_DECAY} (應用於所有 Conv2D 和 Dense 層)")  # [修改] 更新說明
print(f"  Loss: CategoricalCrossentropy (label_smoothing=0.05)")
print(f"  Learning Rate: 0 -> {target_lr} -> {min_lr} (Warmup + Cosine Decay)")
print(f"  Optimizer: SGD (momentum=0.9, nesterov=True)")

model.summary()

trainable_params = sum([tf.reduce_prod(w.shape).numpy() for w in model.trainable_weights])
print(f"\n可訓練參數量: {trainable_params:,}")

# ============================================================================
# 回調函數
# ============================================================================
model_path = os.path.join(output_dir, 'best_model_resnet18_dropout_wd_180x80.keras')

# monitor='val_accuracy'，其 val_accuracy 來自「未擴增驗證集」(val_ds)，
# 因此最佳權重是依模型在原始相位分佈上的表現挑選，而非擴增驗證集。
checkpoint = ModelCheckpoint(
    model_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

callbacks = [checkpoint]

# 訓練
print(f"\n{'='*80}")
print(f"開始訓練 ResNet18 (含 Dropout + Weight Decay + 多通道特徵增強)...")
print(f"{'='*80}\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)

training_end_time = time.time()
training_end_datetime = datetime.now()
training_duration = training_end_time - training_start_time

actual_epochs = len(history.history['loss'])
print(f"\n訓練完成！")
print(f"訓練輪數: {actual_epochs}")
print(f"訓練時間: {str(timedelta(seconds=int(training_duration)))}")

best_model = models.load_model(model_path)

# ============================================================================
# 原始測試集評估（不擴增）
# ============================================================================
print(f"\n{'='*80}")
print("原始測試集評估（不擴增）— 主要效能依據")
print("="*80)

# 獲取測試集指標
test_loss, test_acc, test_precision, test_recall = best_model.evaluate(test_ds, verbose=0)

preds = best_model.predict(test_ds)
pred_labels = np.argmax(preds, axis=1)

precision, recall, f1_scores, support = precision_recall_fscore_support(test_labels, pred_labels, average=None)
cm = confusion_matrix(test_labels, pred_labels)
test_acc_original = np.sum(pred_labels == test_labels) / len(test_labels)

print(f"\n原始測試集表現（主要 accuracy / precision / recall / F1 / 混淆矩陣依據）:")
print(f"  準確率: {test_acc_original:.4f}")
print(f"  損失: {test_loss:.4f}")
print(f"\n各類別 F1 分數:")
for i, cls_name in enumerate(class_names):
    print(f"  {cls_name}: {f1_scores[i]:.4f} (Support: {support[i]})")

# ============================================================================
# [新增] 擴增測試集評估（循環位移 + 噪聲）—— 僅作輔助穩健性觀察，非主要效能依據
# ============================================================================
print(f"\n{'='*80}")
print("擴增測試集評估（循環位移 + 底部噪聲）")
print("【註】擴增測試集僅作輔助穩健性觀察，非主要效能依據")
print("="*80)

preds_aug = best_model.predict(test_ds_augmented)
pred_labels_aug = np.argmax(preds_aug, axis=1)

precision_aug, recall_aug, f1_scores_aug, support_aug = precision_recall_fscore_support(
    test_labels_augmented, pred_labels_aug, average=None
)
cm_aug = confusion_matrix(test_labels_augmented, pred_labels_aug)
test_acc_augmented = np.sum(pred_labels_aug == test_labels_augmented) / len(test_labels_augmented)

print(f"\n擴增測試集表現（僅作穩健性觀察）:")
print(f"  準確率: {test_acc_augmented:.4f}")
print(f"  樣本數: {len(test_labels_augmented)} 張 (原始 {test_size} 張 x {AUGMENT_FACTOR})")
print(f"\n各類別 F1 分數:")
for i, cls_name in enumerate(class_names):
    print(f"  {cls_name}: {f1_scores_aug[i]:.4f} (Support: {support_aug[i]})")

# ============================================================================
# 比較結果
# ============================================================================
print(f"\n{'='*80}")
print("測試集比較（原始為主要效能；擴增僅作穩健性觀察）")
print("="*80)
print(f"  原始測試集準確率（主要）: {test_acc_original:.4f}")
print(f"  擴增測試集準確率（穩健性觀察）: {test_acc_augmented:.4f}")
print(f"  穩健性差異（擴增 - 原始）: {(test_acc_augmented - test_acc_original)*100:+.2f}%")

# ============================================================================
# 繪製訓練曲線
# ============================================================================
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title(f'ResNet18 Loss Curve (Input: {INPUT_WIDTH}x{INPUT_HEIGHT}, WD={WEIGHT_DECAY})', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title(f'ResNet18 Accuracy Curve (Input: {INPUT_WIDTH}x{INPUT_HEIGHT}, WD={WEIGHT_DECAY})', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(history.history['precision'], label='Train Precision', linewidth=2)
plt.plot(history.history['val_precision'], label='Val Precision', linewidth=2)
plt.plot(history.history['recall'], label='Train Recall', linewidth=2, linestyle='--')
plt.plot(history.history['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('ResNet18 Precision & Recall Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
loss_gap = np.array(history.history['val_loss']) - np.array(history.history['loss'])
acc_gap = np.array(history.history['val_accuracy']) - np.array(history.history['accuracy'])
plt.plot(loss_gap, label='Val-Train Loss Gap', linewidth=2)
plt.plot(acc_gap, label='Val-Train Acc Gap', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Gap (Val - Train)', fontsize=12)
plt.title('ResNet18 Overfitting Analysis', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
save_path = os.path.join(output_dir, 'training_curves_resnet18_dropout_wd_180x80.png')  # [修改] 檔名加上 _wd
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n訓練曲線已保存: {save_path}")

# ============================================================================
# 混淆矩陣 1：原始測試集（不擴增）
# ============================================================================
plt.figure(figsize=(10, 8))
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=chinese_labels, yticklabels=chinese_labels)
for j in range(cm.shape[0]):
    for k in range(cm.shape[1]):
        plt.text(k + 0.5, j + 0.5, f"{cm[j, k]}\n({cm_norm[j, k]:.1f}%)",
                 ha='center', va='center', color='black', fontsize=11)
plt.xlabel('Predicted Label', fontsize=13)
plt.ylabel('True Label', fontsize=13)
plt.title(f'原始測試集混淆矩陣（主要效能） (Acc: {test_acc_original:.4f})', fontsize=15, fontweight='bold')
save_path_cm_original = os.path.join(output_dir, 'confusion_matrix_original.png')
plt.savefig(save_path_cm_original, dpi=300, bbox_inches='tight')
plt.close()

print(f"原始測試集混淆矩陣已保存: {save_path_cm_original}")

# ============================================================================
# [新增] 混淆矩陣 2：擴增測試集（循環位移 + 底部噪聲）
# ============================================================================
plt.figure(figsize=(10, 8))
cm_aug_norm = cm_aug.astype('float') / cm_aug.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_aug, annot=False, fmt='d', cmap='Greens', xticklabels=chinese_labels, yticklabels=chinese_labels)
for j in range(cm_aug.shape[0]):
    for k in range(cm_aug.shape[1]):
        plt.text(k + 0.5, j + 0.5, f"{cm_aug[j, k]}\n({cm_aug_norm[j, k]:.1f}%)",
                 ha='center', va='center', color='black', fontsize=11)
plt.xlabel('Predicted Label', fontsize=13)
plt.ylabel('True Label', fontsize=13)
plt.title(f'擴增測試集混淆矩陣（僅穩健性觀察） (Acc: {test_acc_augmented:.4f})', fontsize=15, fontweight='bold')
save_path_cm_augmented = os.path.join(output_dir, 'confusion_matrix_augmented.png')
plt.savefig(save_path_cm_augmented, dpi=300, bbox_inches='tight')
plt.close()

print(f"擴增測試集混淆矩陣已保存: {save_path_cm_augmented}")

# ============================================================================
# 保存結構化訓練日誌
# ============================================================================
log_path = os.path.join(output_dir, 'training_log.txt')
with open(log_path, 'w', encoding='utf-8') as f:
    # ========== 標題 ==========
    f.write("=" * 80 + "\n")
    f.write("ResNet18 訓練日誌\n")
    f.write("=" * 80 + "\n\n")

    # ========== 基本資訊 ==========
    f.write("【基本資訊】\n")
    f.write(f"開始時間: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"結束時間: {training_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"訓練時長: {str(timedelta(seconds=int(training_duration)))}\n")
    f.write(f"資料集: {os.path.basename(data_dir)}\n\n")

    # ========== 模型配置 ==========
    f.write("【模型配置】\n")
    f.write(f"架構: ResNet18\n")
    f.write(f"輸入: {INPUT_WIDTH}x{INPUT_HEIGHT}x3（翻轉後二值化影像 + x 方向 Sobel 梯度響應 |G_x| + y 方向 Sobel 梯度響應 |G_y|）\n")
    f.write(f"參數量: {trainable_params:,}\n")
    f.write(f"通道: [64, 128, 256, 512] -> GAP(512) -> FC(1024) -> FC(512) -> {num_classes}類\n")
    f.write(f"正則化: Dropout={DROPOUT_RATE}, L2={WEIGHT_DECAY}, LabelSmoothing=0.05\n\n")

    # ========== 訓練配置 ==========
    f.write("【訓練配置】\n")
    f.write(f"Batch: {batch_size}, Epochs: {epochs} (實際: {actual_epochs})\n")
    f.write(f"優化器: SGD (momentum=0.9, nesterov=True)\n")
    f.write(f"學習率: 0->{target_lr}->{min_lr} (Warmup {warmup_steps} steps + CosineDecay)\n")
    f.write(f"擴增: {AUGMENT_FACTOR}x循環位移 + {NOISE_COUNT}張噪聲/組\n\n")

    # ========== 資料集統計 ==========
    f.write("【資料集】\n")
    f.write(f"原始: Train={train_size}, Val={val_size}, Test={test_size} (共{total_images}張)\n")
    f.write(f"訓練集擴增後: {augmented_train_size} 張\n")
    f.write(f"驗證集未擴增，共 {val_size} 張（主要 validation 指標來源）\n\n")

    # ========== 資料使用邏輯（與 K-Fold 版一致） ==========
    f.write("【資料使用邏輯】\n")
    f.write("train_augment = true\n")
    f.write("validation_augment = false\n")
    f.write("test_augment_for_main_eval = false\n")
    f.write("augmented_test_for_robustness_only = true\n")
    f.write('checkpoint_monitor = "val_accuracy on unaugmented validation set"\n')
    f.write("說明: 訓練集擴增；驗證集未擴增（主要 validation / checkpoint / 最佳權重依據）；\n")
    f.write("      原始測試集為主要效能；擴增測試集僅作相位平移與雜訊擾動下的穩健性觀察。\n\n")

    # ========== 模型比較說明 ==========
    f.write("【模型比較說明】\n")
    f.write("本模型作為第 4.5 節不同模型辨識效能比較之 baseline。為確保比較公平性，"
            "本程式沿用 EPSANet-Large 主模型相同之資料分割、三通道輸入特徵、訓練集擴增策略、"
            "未擴增驗證集、最佳權重選取方式與原始測試集評估流程，僅將模型骨幹替換為 ResNet18。\n\n")

    # ========== 類別權重 ==========
    f.write("【類別權重】\n")
    for cls_idx, cls_name in enumerate(class_names):
        if cls_idx in class_weights:
            w = class_weights[cls_idx]
            status = "[!]過高" if w > 1.5 else ("[!]過低" if w < 0.7 else "")
            f.write(f"  {cls_name}: {w:.4f} {status}\n")
    f.write("\n")

    # ========== 訓練過程（每10個epoch + 關鍵epoch） ==========
    f.write("=" * 80 + "\n")
    f.write("訓練過程\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'Epoch':>5} {'T_Loss':>8} {'T_Acc':>7} {'V_Loss':>8} {'V_Acc':>7} {'Gap':>7} {'Note':<10}\n")
    f.write("-" * 60 + "\n")

    hist = history.history
    best_epoch = np.argmax(hist['val_accuracy']) + 1
    best_val_acc = max(hist['val_accuracy'])

    for epoch in range(actual_epochs):
        # 顯示規則: 前20個每個都記，之後每5個記一次，加上最佳epoch和最後epoch
        ep = epoch + 1
        show = (ep <= 20) or (ep % 5 == 0) or (ep == best_epoch) or (ep == actual_epochs)
        if show:
            t_loss = hist['loss'][epoch]
            t_acc = hist['accuracy'][epoch]
            v_loss = hist['val_loss'][epoch]
            v_acc = hist['val_accuracy'][epoch]
            gap = t_acc - v_acc  # 正值=過擬合

            note = ""
            if epoch + 1 == best_epoch:
                note = "*Best"
            elif gap > 0.05:
                note = "[!]過擬合"

            f.write(f"{epoch+1:>5} {t_loss:>8.4f} {t_acc:>7.4f} {v_loss:>8.4f} {v_acc:>7.4f} {gap:>+7.3f} {note:<10}\n")

    f.write("\n")

    # ========== 訓練摘要 ==========
    f.write("【訓練摘要】\n")
    f.write(f"最佳Epoch: {best_epoch} (Val Acc: {best_val_acc:.4f})\n")

    final_gap = hist['accuracy'][-1] - hist['val_accuracy'][-1]
    if final_gap > 0.1:
        gap_status = "嚴重過擬合"
    elif final_gap > 0.05:
        gap_status = "輕微過擬合"
    elif final_gap < -0.02:
        gap_status = "欠擬合"
    else:
        gap_status = "正常"
    f.write(f"最終Gap: {final_gap:+.4f} ({gap_status})\n")

    # 檢查是否提前收斂
    if actual_epochs < epochs:
        f.write(f"提前停止: 是 (第{actual_epochs}輪)\n")
    f.write("\n")

    # ========== 測試結果 ==========
    f.write("=" * 80 + "\n")
    f.write("測試結果\n")
    f.write("=" * 80 + "\n\n")

    f.write("【原始測試集（主要效能依據）】\n")
    f.write(f"Accuracy: {test_acc_original:.4f}, Loss: {test_loss:.4f}\n")
    f.write(f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}\n\n")

    # ========== 各類別表現 ==========
    f.write("【各類別表現】\n")
    f.write(f"{'Class':<4} {'中文':<8} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Support':>8} {'狀態':<12}\n")
    f.write("-" * 60 + "\n")

    # 計算平均 F1
    avg_f1 = np.mean(f1_scores)

    for i, cls_name in enumerate(class_names):
        p = precision[i]
        r = recall[i]
        f1 = f1_scores[i]
        sup = support[i]

        # 狀態判斷
        status = ""
        if f1 < avg_f1 - 0.1:
            status = "[!]F1偏低"
        if p < 0.8:
            status += " [!]Prec低"
        if r < 0.8:
            status += " [!]Recall低"
        if not status:
            status = "OK"

        f.write(f"{cls_name:<4} {chinese_labels[i]:<8} {p:>6.4f} {r:>7.4f} {f1:>6.4f} {sup:>8} {status:<12}\n")

    f.write(f"\n平均F1: {avg_f1:.4f}\n\n")

    # ========== 混淆矩陣熱點分析 ==========
    f.write("【常見誤分類】\n")
    misclass_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                misclass_pairs.append((i, j, cm[i, j]))

    # 排序取前5
    misclass_pairs.sort(key=lambda x: x[2], reverse=True)
    for true_idx, pred_idx, count in misclass_pairs[:5]:
        if count > 0:
            pct = count / support[true_idx] * 100
            f.write(f"  {class_names[true_idx]}->{class_names[pred_idx]}: {count}次 ({pct:.1f}%)\n")

    if not misclass_pairs or misclass_pairs[0][2] == 0:
        f.write("  無明顯誤分類\n")
    f.write("\n")

    # ========== 擴增測試集（僅穩健性觀察） ==========
    f.write("【擴增測試集（僅作輔助穩健性觀察，非主要效能依據）】\n")
    f.write(f"Accuracy: {test_acc_augmented:.4f} (穩健性差異 擴增-原始: {(test_acc_augmented - test_acc_original)*100:+.2f}%)\n")
    f.write(f"樣本數: {augmented_test_size} ({test_size}x{AUGMENT_FACTOR})\n\n")

    # ========== 建議 ==========
    f.write("=" * 80 + "\n")
    f.write("分析建議\n")
    f.write("=" * 80 + "\n")

    suggestions = []
    if final_gap > 0.05:
        suggestions.append("- 過擬合: 考慮增加Dropout、Weight Decay或資料擴增")
    if avg_f1 < 0.9:
        suggestions.append("- F1偏低: 檢查類別不平衡或增加訓練資料")
    if best_epoch < epochs * 0.5:
        suggestions.append("- 過早收斂: 考慮降低學習率或增加模型容量")
    if best_epoch > epochs * 0.9:
        suggestions.append("- 可能需更多訓練: 考慮增加Epochs")

    # 找出問題類別
    for i, f1 in enumerate(f1_scores):
        if f1 < avg_f1 - 0.15:
            suggestions.append(f"- {class_names[i]}類表現差: 檢查訓練資料品質或增加該類樣本")

    if suggestions:
        for s in suggestions:
            f.write(f"{s}\n")
    else:
        f.write("模型表現良好，無明顯問題。\n")

    f.write("\n" + "=" * 80 + "\n")

print(f"\n訓練日誌已保存: {log_path}")
print(f"最佳模型已保存: {model_path}")

# ============================================================================
# [新增] 模型比較 summary CSV（論文 4.5 不同模型辨識效能比較）
# ============================================================================
# 主要比較指標：original_test_accuracy / macro_f1 / weighted_f1（皆以原始未擴增測試集為準）
# augmented_test_accuracy 僅作穩健性觀察，不作為主要效能依據
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    test_labels, pred_labels, average='macro'
)
weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    test_labels, pred_labels, average='weighted'
)
best_epoch_summary = int(np.argmax(history.history['val_accuracy']) + 1)
best_val_acc_summary = float(max(history.history['val_accuracy']))

summary_row = {
    'model_name': 'ResNet18',
    'trainable_params': int(trainable_params),
    'best_epoch': best_epoch_summary,
    'best_val_accuracy': round(best_val_acc_summary, 6),
    'original_test_accuracy': round(float(test_acc_original), 6),
    'original_test_loss': round(float(test_loss), 6),
    'macro_precision': round(float(macro_precision), 6),
    'macro_recall': round(float(macro_recall), 6),
    'macro_f1': round(float(macro_f1), 6),
    'weighted_precision': round(float(weighted_precision), 6),
    'weighted_recall': round(float(weighted_recall), 6),
    'weighted_f1': round(float(weighted_f1), 6),
    'augmented_test_accuracy': round(float(test_acc_augmented), 6),
    'robustness_gap_augmented_minus_original': round(float(test_acc_augmented - test_acc_original), 6),
    'training_time_seconds': round(float(training_duration), 2),
}

summary_columns = [
    'model_name', 'trainable_params', 'best_epoch', 'best_val_accuracy',
    'original_test_accuracy', 'original_test_loss',
    'macro_precision', 'macro_recall', 'macro_f1',
    'weighted_precision', 'weighted_recall', 'weighted_f1',
    'augmented_test_accuracy', 'robustness_gap_augmented_minus_original',
    'training_time_seconds',
]

summary_csv_path = os.path.join(output_dir, 'model_result_summary.csv')
pd.DataFrame([summary_row], columns=summary_columns).to_csv(
    summary_csv_path, index=False, encoding='utf-8-sig'
)
print(f"\n模型比較 summary CSV 已保存: {summary_csv_path}")


# ============================================================================
# t-SNE 特徵視覺化
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
    print("生成 t-SNE 特徵視覺化圖...")
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
    print(f"輸出檔案:")
    for layer_name in feature_layers:
        if layer_name in feature_models:
            print(f"  - tsne_{layer_name}.png")
    print(f"  - tsne_comparison.png")


# 生成 t-SNE 視覺化
generate_tsne_visualization(best_model, test_ds, test_labels, output_dir, test_size)

print(f"\n{'='*80}")
print("訓練完成！")
print("="*80)

print(f"\n" + "="*80)
print("訓練摘要")
print("="*80)
print(f"  模型: ResNet18 (含 Weight Decay)")
print(f"  原始測試集準確率（主要效能）: {test_acc_original:.4f} ({test_acc_original*100:.2f}%)")
print(f"  擴增測試集準確率（僅穩健性觀察）: {test_acc_augmented:.4f} ({test_acc_augmented*100:.2f}%)")
print(f"  訓練輪數: {actual_epochs}")
print(f"  訓練時間: {str(timedelta(seconds=int(training_duration)))}")
print(f"  可訓練參數量: {trainable_params:,}")
print(f"  Weight Decay: {WEIGHT_DECAY}")
print(f"\n輸出檔案:")
print(f"  模型: {model_path}")
print(f"  訓練日誌: {log_path}")
print(f"  訓練曲線: {save_path}")
print(f"  混淆矩陣（原始）: {save_path_cm_original}")
print(f"  混淆矩陣（擴增）: {save_path_cm_augmented}")
print(f"  t-SNE 視覺化:")
print(f"    - {os.path.join(output_dir, 'tsne_global_avg_pool.png')}")
print(f"    - {os.path.join(output_dir, 'tsne_head_fc1.png')}")
print(f"    - {os.path.join(output_dir, 'tsne_head_fc2.png')}")
print(f"    - {os.path.join(output_dir, 'tsne_comparison.png')}")
print("="*80)