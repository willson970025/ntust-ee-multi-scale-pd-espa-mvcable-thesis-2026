# -*- coding: utf-8 -*-
"""
EPSANet-Large 模型架構視覺化腳本

提供 4 種視覺化方法：
  1. tf.keras.utils.plot_model（Keras 內建）
  2. visualkeras 層堆疊視覺化
  3. 自訂 matplotlib 方塊圖（無額外依賴）
  4. Netron 使用說明

使用範例：
  # 載入 .keras 模型，執行全部方法
  python visualize_model_architecture.py --model-path /path/to/model.keras

  # 只跑 matplotlib 方塊圖（零額外依賴）
  python visualize_model_architecture.py --model-path /path/to/model.keras --methods matplotlib

  # 不需 .keras 檔案，直接建構模型
  python visualize_model_architecture.py --mode build

  # 指定輸出目錄
  python visualize_model_architecture.py --model-path /path/to/model.keras --output-dir ./my_output
"""

# ============================================================================
# 環境設定
# ============================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import argparse
import numpy as np

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# 與訓練時一致，使用 mixed_float16 才能正確載入模型
tf.keras.mixed_precision.set_global_policy('mixed_float16')

import keras
from keras import layers, Model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# 啟用動態記憶體增長
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# ============================================================================
# 可選套件偵測
# ============================================================================
HAS_PYDOT = False
try:
    import pydot
    HAS_PYDOT = True
except ImportError:
    pass

HAS_VISUALKERAS = False
try:
    import visualkeras
    HAS_VISUALKERAS = True
except ImportError:
    pass


# ============================================================================
# 自訂層定義（從 epsanet50_all_in_one.py 複製，用於載入 .keras 模型）
# ============================================================================
WEIGHT_DECAY = 1e-5


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


@keras.saving.register_keras_serializable()
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """自訂預熱學習率調度器"""
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
# build_epsanet_large()（從 epsanet50_all_in_one.py 複製）
# ============================================================================
def build_epsanet_large(input_shape=(80, 180, 3), num_classes=7, dropout_rate=0.2,
                        weight_decay=WEIGHT_DECAY):
    """建立 EPSANet-Large 模型"""
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

    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False,
                      kernel_initializer='he_normal', name='stem_conv',
                      kernel_regularizer=keras.regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(name='stem_relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='stem_pool')(x)

    in_ch = 64
    for i in range(layers_config[0]):
        use_ds = (i == 0) and (in_ch != channels[0] * 4)
        x = EPSABlock(in_ch if i == 0 else channels[0] * 4, channels[0], stride=1,
                      use_downsample=use_ds, conv_kernels=layer_kernels[0],
                      conv_groups=conv_groups, weight_decay=weight_decay,
                      name=f'layer1_block{i}')(x)

    prev_ch = channels[0] * 4
    for i in range(layers_config[1]):
        stride = 2 if i == 0 else 1
        use_ds = (i == 0)
        x = EPSABlock(prev_ch if i == 0 else channels[1] * 4, channels[1], stride=stride,
                      use_downsample=use_ds, conv_kernels=layer_kernels[1],
                      conv_groups=conv_groups, weight_decay=weight_decay,
                      name=f'layer2_block{i}')(x)

    prev_ch = channels[1] * 4
    for i in range(layers_config[2]):
        stride = 2 if i == 0 else 1
        use_ds = (i == 0)
        x = EPSABlock(prev_ch if i == 0 else channels[2] * 4, channels[2], stride=stride,
                      use_downsample=use_ds, conv_kernels=layer_kernels[2],
                      conv_groups=conv_groups, weight_decay=weight_decay,
                      name=f'layer3_block{i}')(x)

    prev_ch = channels[2] * 4
    for i in range(layers_config[3]):
        stride = 2 if i == 0 else 1
        use_ds = (i == 0)
        x = EPSABlock(prev_ch if i == 0 else channels[3] * 4, channels[3], stride=stride,
                      use_downsample=use_ds, conv_kernels=layer_kernels[3],
                      conv_groups=conv_groups, weight_decay=weight_decay,
                      name=f'layer4_block{i}')(x)

    x = layers.Activation('linear', name='gradcam_target')(x)
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
# 中文字型設定
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


# ============================================================================
# Method 1: tf.keras.utils.plot_model
# ============================================================================
def method_plot_model(model, output_dir):
    """使用 tf.keras.utils.plot_model 生成架構圖"""
    print("\n" + "=" * 60)
    print("Method 1: tf.keras.utils.plot_model")
    print("=" * 60)

    if not HAS_PYDOT:
        print("  [跳過] 需要安裝 pydot 與 graphviz：")
        print("    pip install pydot")
        print("    sudo apt-get install graphviz   # Ubuntu/Debian")
        print("    brew install graphviz            # macOS")
        return

    # 1. 基本視圖
    path_basic = os.path.join(output_dir, 'model_architecture_basic.png')
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=path_basic,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            dpi=150
        )
        print(f"  基本視圖已儲存: {path_basic}")
    except Exception as e:
        print(f"  基本視圖生成失敗: {e}")

    # 2. 詳細視圖
    path_detailed = os.path.join(output_dir, 'model_architecture_detailed.png')
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=path_detailed,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_dtype=True,
            rankdir='TB',
            dpi=150
        )
        print(f"  詳細視圖已儲存: {path_detailed}")
    except Exception as e:
        print(f"  詳細視圖生成失敗: {e}")

    # 3. 展開視圖
    path_expanded = os.path.join(output_dir, 'model_architecture_expanded.png')
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=path_expanded,
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            rankdir='TB',
            dpi=150
        )
        print(f"  展開視圖已儲存: {path_expanded}")
        print("  (注意：自訂 Layer 子類別不會被展開，僅限 Functional/Sequential 子模型)")
    except Exception as e:
        print(f"  展開視圖生成失敗: {e}")


# ============================================================================
# Method 2: visualkeras
# ============================================================================
def method_visualkeras(model, output_dir):
    """使用 visualkeras 生成層堆疊視覺化"""
    print("\n" + "=" * 60)
    print("Method 2: visualkeras")
    print("=" * 60)

    if not HAS_VISUALKERAS:
        print("  [跳過] 需要安裝 visualkeras：")
        print("    pip install visualkeras")
        return

    # 修補新版 Keras 相容性：visualkeras 需要 layer.output_shape 屬性
    for layer in model.layers:
        if not hasattr(layer, 'output_shape') or not callable(getattr(layer, 'output_shape', None)):
            try:
                shape = layer.output.shape
                # 轉為 tuple，例如 (None, 80, 180, 3)
                layer.output_shape = tuple(shape)
            except Exception:
                pass

    path_layered = os.path.join(output_dir, 'model_visualkeras_layered.png')
    try:
        img = visualkeras.layered_view(
            model,
            legend=True,
            scale_xy=2,
            scale_z=0.5,
            spacing=30,
        )
        img.save(path_layered)
        print(f"  層堆疊圖已儲存: {path_layered}")
    except Exception as e:
        print(f"  visualkeras 生成失敗: {e}")


# ============================================================================
# Method 3: 自訂 matplotlib 方塊圖
# ============================================================================
def method_matplotlib(model, output_dir):
    """使用 matplotlib 繪製自訂方塊圖"""
    print("\n" + "=" * 60)
    print("Method 3: matplotlib 方塊圖")
    print("=" * 60)

    fig = plt.figure(figsize=(96, 64))

    # ------------------------------------------------------------------
    # 上半部：主流程方塊圖
    # ------------------------------------------------------------------
    ax_main = fig.add_axes([0.03, 0.52, 0.94, 0.44])
    ax_main.set_xlim(0, 100)
    ax_main.set_ylim(0, 10)
    ax_main.axis('off')
    ax_main.set_title('EPSANet-Large 主流程架構', fontsize=96, fontweight='bold', pad=50)

    # 定義方塊資料: (x, width, label, sublabel, color)
    blocks = [
        (1,   7,  'Input',       '80x180x3',        '#E8F5E9'),
        (10,  8,  'Stem',        'Conv7x7 s2\nBN+ReLU\nMaxPool3x3 s2', '#C8E6C9'),
        (20,  10, 'Layer 1',     '3x EPSABlock\n128ch -> 512ch\n20x45',  '#BBDEFB'),
        (32,  10, 'Layer 2',     '4x EPSABlock\n256ch -> 1024ch\n10x23', '#90CAF9'),
        (44,  10, 'Layer 3',     '6x EPSABlock\n512ch -> 2048ch\n5x12',  '#64B5F6'),
        (56,  10, 'Layer 4',     '3x EPSABlock\n1024ch -> 4096ch\n3x6',  '#42A5F5'),
        (68,  8,  'GAP',         'GlobalAvgPool\n4096-d',                 '#FFF9C4'),
        (78,  8,  'Head',        'Drop(0.2)\nFC(1024)+BN+ReLU\nFC(512)+BN+ReLU', '#FFE0B2'),
        (88,  8,  'Output',      'Dense(7)\nSoftmax',                     '#FFCDD2'),
    ]

    for (x, w, label, sublabel, color) in blocks:
        rect = mpatches.FancyBboxPatch(
            (x, 2), w, 6,
            boxstyle="round,pad=0.3",
            facecolor=color, edgecolor='#333333', linewidth=3.0
        )
        ax_main.add_patch(rect)
        ax_main.text(x + w/2, 6.5, label, ha='center', va='center',
                     fontsize=64, fontweight='bold')
        ax_main.text(x + w/2, 4.0, sublabel, ha='center', va='center',
                     fontsize=44, color='#333333')

    # 繪製連接箭頭
    arrow_xs = [8, 18, 30, 42, 54, 66, 76, 86]
    for ax_pos in arrow_xs:
        ax_main.annotate('', xy=(ax_pos + 2, 5), xytext=(ax_pos, 5),
                         arrowprops=dict(arrowstyle='->', color='#555555', lw=4.0))

    # ------------------------------------------------------------------
    # 下半部左：EPSABlock 內部結構
    # ------------------------------------------------------------------
    ax_block = fig.add_axes([0.03, 0.03, 0.45, 0.44])
    ax_block.set_xlim(0, 50)
    ax_block.set_ylim(0, 28)
    ax_block.axis('off')
    ax_block.set_title('EPSABlock 內部結構', fontsize=80, fontweight='bold', pad=35)

    # EPSABlock 主路徑
    block_items = [
        (18, 24, 14, 2.5, 'Conv1x1\n(out_ch)',       '#C8E6C9'),
        (18, 20, 14, 2.5, 'BN + ReLU',               '#E8F5E9'),
        (18, 16, 14, 2.5, 'PSAModule\n(多尺度注意力)', '#BBDEFB'),
        (18, 12, 14, 2.5, 'BN + ReLU',               '#E8F5E9'),
        (18, 8,  14, 2.5, 'Conv1x1\n(out_ch x 4)',   '#C8E6C9'),
        (18, 4,  14, 2.5, 'BN',                       '#E8F5E9'),
    ]
    for (x, y, w, h, label, color) in block_items:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.2",
            facecolor=color, edgecolor='#333333', linewidth=2.5
        )
        ax_block.add_patch(rect)
        ax_block.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=52)

    # 主路徑箭頭
    for y_from, y_to in [(24, 22.5), (20, 18.5), (16, 14.5), (12, 10.5), (8, 6.5)]:
        ax_block.annotate('', xy=(25, y_to), xytext=(25, y_from),
                          arrowprops=dict(arrowstyle='->', color='#555555', lw=3.5))

    # Residual connection (identity/downsample)
    ax_block.annotate('', xy=(35, 4), xytext=(35, 26.5),
                      arrowprops=dict(arrowstyle='->', color='#E53935', lw=4.0,
                                      connectionstyle='arc3,rad=0'))
    ax_block.text(37, 15, 'Identity\n(or Downsample\nConv1x1+BN)',
                  fontsize=48, color='#E53935', ha='left', va='center')

    # Add + ReLU
    circle = plt.Circle((25, 2), 1.2, fill=True, facecolor='#FFF9C4',
                         edgecolor='#333333', linewidth=2.5)
    ax_block.add_patch(circle)
    ax_block.text(25, 2, '+', ha='center', va='center', fontsize=72, fontweight='bold')
    ax_block.annotate('', xy=(25, 3.2), xytext=(25, 4),
                      arrowprops=dict(arrowstyle='->', color='#555555', lw=3.5))
    ax_block.annotate('', xy=(34, 2), xytext=(35, 4),
                      arrowprops=dict(arrowstyle='->', color='#E53935', lw=3.5))
    ax_block.text(28, 1.5, 'ReLU', fontsize=52, fontweight='bold')

    # 輸入/輸出標記
    ax_block.text(25, 27.5, 'Input (H, W, in_ch)', ha='center', va='center',
                  fontsize=56, fontweight='bold', color='#1565C0')
    ax_block.text(25, 0, 'Output (H\', W\', out_ch x 4)', ha='center', va='center',
                  fontsize=56, fontweight='bold', color='#1565C0')

    # ------------------------------------------------------------------
    # 下半部右：PSAModule 內部結構
    # ------------------------------------------------------------------
    ax_psa = fig.add_axes([0.52, 0.03, 0.45, 0.44])
    ax_psa.set_xlim(0, 50)
    ax_psa.set_ylim(0, 28)
    ax_psa.axis('off')
    ax_psa.set_title('PSAModule 內部結構 (Pyramid Squeeze Attention)', fontsize=72,
                     fontweight='bold', pad=35)

    # Input
    ax_psa.text(25, 27, 'Input (H, W, C)', ha='center', va='center',
                fontsize=56, fontweight='bold', color='#1565C0')

    # 4 parallel conv branches
    branch_colors = ['#E3F2FD', '#E8EAF6', '#F3E5F5', '#FCE4EC']
    branch_labels = ['Conv 3x3\nGroup', 'Conv 5x5\nGroup', 'Conv 7x7\nGroup', 'Conv 9x9\nGroup']
    branch_x = [3, 14, 25, 36]

    for bx, label, color in zip(branch_x, branch_labels, branch_colors):
        rect = mpatches.FancyBboxPatch(
            (bx, 22), 9, 3,
            boxstyle="round,pad=0.2",
            facecolor=color, edgecolor='#333333', linewidth=2.0
        )
        ax_psa.add_patch(rect)
        ax_psa.text(bx + 4.5, 23.5, label, ha='center', va='center', fontsize=44)
        ax_psa.annotate('', xy=(bx + 4.5, 22), xytext=(bx + 4.5, 22 - 0.3),
                         arrowprops=dict(arrowstyle='->', color='#555555', lw=2.0))

    # SE Weight for each branch
    for bx in branch_x:
        rect = mpatches.FancyBboxPatch(
            (bx, 17.5), 9, 3,
            boxstyle="round,pad=0.2",
            facecolor='#FFF9C4', edgecolor='#333333', linewidth=2.0
        )
        ax_psa.add_patch(rect)
        ax_psa.text(bx + 4.5, 19, 'SE Weight\n(Attn)', ha='center', va='center', fontsize=44)
        ax_psa.annotate('', xy=(bx + 4.5, 20.5), xytext=(bx + 4.5, 22),
                         arrowprops=dict(arrowstyle='->', color='#555555', lw=2.0))

    # Concat SE weights
    rect = mpatches.FancyBboxPatch(
        (8, 13), 34, 2.5,
        boxstyle="round,pad=0.2",
        facecolor='#FFE0B2', edgecolor='#333333', linewidth=2.0
    )
    ax_psa.add_patch(rect)
    ax_psa.text(25, 14.25, 'Concat SE Weights -> Softmax (channel attention)', ha='center',
                va='center', fontsize=48)

    for bx in branch_x:
        ax_psa.annotate('', xy=(bx + 4.5, 15.5), xytext=(bx + 4.5, 17.5),
                         arrowprops=dict(arrowstyle='->', color='#555555', lw=2.0))

    # Weighted sum
    rect = mpatches.FancyBboxPatch(
        (8, 8.5), 34, 2.5,
        boxstyle="round,pad=0.2",
        facecolor='#C8E6C9', edgecolor='#333333', linewidth=2.0
    )
    ax_psa.add_patch(rect)
    ax_psa.text(25, 9.75, 'Attention Weighted Sum: Concat(x_i * att_i)', ha='center',
                va='center', fontsize=48)
    ax_psa.annotate('', xy=(25, 11), xytext=(25, 13),
                     arrowprops=dict(arrowstyle='->', color='#555555', lw=3.5))

    # SE Module detail
    ax_psa.text(25, 6.5, 'SE Weight Module:', ha='center', va='center',
                fontsize=56, fontweight='bold')
    ax_psa.text(25, 5, 'GAP -> Conv1x1(C/16) -> ReLU -> Conv1x1(C) -> Sigmoid',
                ha='center', va='center', fontsize=48, color='#555555',
                style='italic')

    # Output
    ax_psa.text(25, 3, 'Output (H\', W\', C)', ha='center', va='center',
                fontsize=56, fontweight='bold', color='#1565C0')
    ax_psa.annotate('', xy=(25, 3.5), xytext=(25, 8.5),
                     arrowprops=dict(arrowstyle='->', color='#555555', lw=3.5))

    # 儲存
    path_block = os.path.join(output_dir, 'model_block_diagram.png')
    fig.savefig(path_block, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  方塊圖已儲存: {path_block}")


# ============================================================================
# Method 4: Netron 使用說明
# ============================================================================
def method_netron(model_path, output_dir):
    """輸出 Netron 使用說明並存為 .txt"""
    print("\n" + "=" * 60)
    print("Method 4: Netron 使用說明")
    print("=" * 60)

    instructions = """========================================
Netron 模型架構互動式視覺化工具
========================================

Netron 是最受歡迎的深度學習模型視覺化工具，支援互動式瀏覽、
縮放、搜尋層、查看權重等功能。

方法一：線上版（無需安裝）
  1. 前往 https://netron.app
  2. 將 .keras 檔案拖曳到網頁上即可

方法二：桌面版
  pip install netron
  netron /path/to/model.keras

方法三：Python API
  import netron
  netron.start('/path/to/model.keras')
  # 會自動在瀏覽器中開啟

支援格式：
  .keras, .h5, .pb, .tflite, .onnx, .pt, .pth 等

注意事項：
  - .keras 格式包含自訂層定義，Netron 可以正確讀取
  - 若使用 build 模式建構的模型，需先儲存為 .keras 檔案
========================================
"""

    print(instructions)

    path_txt = os.path.join(output_dir, 'netron_instructions.txt')
    with open(path_txt, 'w', encoding='utf-8') as f:
        f.write(instructions)
    print(f"  說明文字已儲存: {path_txt}")


# ============================================================================
# model.summary() 文字版
# ============================================================================
def save_model_summary(model, output_dir):
    """將 model.summary() 儲存為文字檔"""
    path_summary = os.path.join(output_dir, 'model_summary.txt')

    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    summary_text = '\n'.join(lines)

    with open(path_summary, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f"\nmodel.summary() 已儲存: {path_summary}")


# ============================================================================
# 主程式
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='EPSANet-Large 模型架構視覺化工具'
    )
    parser.add_argument('--mode', type=str, default='load', choices=['load', 'build'],
                        help='模型來源模式：load=載入 .keras 檔案，build=直接建構模型 (預設: load)')
    parser.add_argument('--model-path', type=str,
                        default='/home/cckuo/m11307u09/GG/1126/epsanet50/all in one/all_7class_6(G)_add_noise_mix_train/best_model_epsanet_large_dropout_wd_180x80.keras',
                        help='.keras 模型檔案路徑（mode=load 時必須指定）')
    parser.add_argument('--output-dir', type=str,
                        default='/home/cckuo/m11307u09/GG/1126/epsanet50/all in one/all_7class_6(G)_add_noise_mix_train/model_architecture_output',
                        help='輸出目錄')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['all'],
                        choices=['all', 'plot_model', 'visualkeras', 'matplotlib', 'netron'],
                        help='選擇視覺化方法 (預設: all)')

    args = parser.parse_args()

    # 判斷要執行的方法
    if 'all' in args.methods:
        run_methods = {'plot_model', 'visualkeras', 'matplotlib', 'netron'}
    else:
        run_methods = set(args.methods)

    print("=" * 60)
    print("EPSANet-Large 模型架構視覺化")
    print("=" * 60)

    # 設定中文字型
    setup_chinese_font()
    plt.rcParams['axes.unicode_minus'] = False

    # 建立輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"輸出目錄: {os.path.abspath(args.output_dir)}")

    # 載入或建構模型
    if args.mode == 'load':
        if args.model_path is None:
            parser.error("mode=load 時必須指定 --model-path")
        if not os.path.exists(args.model_path):
            print(f"錯誤：找不到模型檔案: {args.model_path}")
            return
        print(f"\n載入模型: {args.model_path}")
        model = keras.models.load_model(args.model_path, compile=False)
        print("模型載入成功！")
    else:
        print("\n直接建構 EPSANet-Large 模型...")
        model = build_epsanet_large()
        print("模型建構成功！")

    # 儲存 model.summary()
    save_model_summary(model, args.output_dir)

    # 執行各方法
    if 'plot_model' in run_methods:
        method_plot_model(model, args.output_dir)

    if 'visualkeras' in run_methods:
        method_visualkeras(model, args.output_dir)

    if 'matplotlib' in run_methods:
        method_matplotlib(model, args.output_dir)

    if 'netron' in run_methods:
        method_netron(args.model_path, args.output_dir)

    # 列出所有輸出檔案
    print("\n" + "=" * 60)
    print("輸出檔案一覽")
    print("=" * 60)
    if os.path.isdir(args.output_dir):
        for fname in sorted(os.listdir(args.output_dir)):
            fpath = os.path.join(args.output_dir, fname)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fname} ({size_kb:.1f} KB)")

    print("\n完成！")


if __name__ == '__main__':
    main()
