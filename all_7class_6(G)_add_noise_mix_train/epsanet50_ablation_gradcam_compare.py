# -*- coding: utf-8 -*-
"""
================================================================================
第 4.6 節「消融實驗 Grad-CAM 視覺化比較」專用程式
================================================================================

本檔目的（重要）：
- 「不」重新訓練任何模型，只做 inference + Grad-CAM + 圖片輸出。
- 讀取消融實驗已訓練完成的 best_model_*.keras，對「同一張原始（未擴增）測試圖譜」
  產生不同消融設定下的 Grad-CAM，比較各模型在同一圖譜上的關注區域差異，
  作為論文第 4.6 節「輔助性可解釋性觀察」。
- 完全沿用 epsanet50_ablation_all_in_one.py 內的模型定義、自訂 layer、前處理
  （make_load_image / generic_preprocess）與資料讀取邏輯，確保前處理一致。
- 不改變 train/val/test 分割，不對測試圖譜做任何資料擴增。

重要聲明（規範第七、九條）：
- Grad-CAM 為「定性」視覺化工具，各模型 heatmap 各自正規化到 0~1，
  顏色僅代表「該模型內部相對關注區域」，不可跨模型當成絕對強度量化比較，
  亦不可單獨作為「某模型較優」之證據。

執行方式：
  # 模式 A：自動從原始 test set 挑一張（優先所有模型皆正確；其次 baseline 正確且信心高）
  python3 epsanet50_ablation_gradcam_compare.py --auto_select

  # 模式 B：指定圖檔（true_label 可省略，省略時由資料夾名稱前兩碼推定）
  python3 epsanet50_ablation_gradcam_compare.py --image_path "/path/to/test/CT_xxx/img.png" --true_label CT

  # Grad-CAM 目標類別：pred（預設，用模型預測類別）或 true（用真實類別）
  python3 epsanet50_ablation_gradcam_compare.py --auto_select --target_mode true
================================================================================
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 確保不論從哪個工作目錄執行，都能找到同目錄下的 epsanet50_ablation_all_in_one.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import argparse
import csv

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # 無顯示環境也可輸出圖片
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 直接沿用消融主程式的模型/自訂 layer/前處理/資料邏輯（不重寫、不修改該檔）
# 匯入此模組會執行其模組層級設定（mixed precision、字型、GPU memory growth），
# 但「不會」觸發訓練（訓練在 epsanet50_ablation_all_in_one.py 的 main() 內）。
# ----------------------------------------------------------------------------
import epsanet50_ablation_all_in_one as abl
from epsanet50_ablation_all_in_one import (
    SEWeightModule, PSAModule, PSAModuleNoAttention, EPSABlock, WarmupCosineDecay,
    class_names, chinese_labels, INPUT_HEIGHT, INPUT_WIDTH, INPUT_MODE_CHANNELS,
    make_load_image, load_data_from_split_dir, data_dir, class_to_idx,
    generic_preprocess, build_epsanet_large,
)

import keras


# ============================================================================
# 要比較的模型（規範第一條）。刻意不放 epsa_inv_gx_gy，因其等同 baseline。
# ============================================================================
DEFAULT_ABLATION_ROOT = ('/home/cckuo/m11307u09/GG/1126/epsanet50/all in one/'
                         'all_7class_7(G)_add_noise_mix_train/ablation_results')

# 模型清單僅定義 exp_name 與 input_mode；model_path 於執行時依 ablation_root 解析，
# 以支援 --ablation_root 覆寫（見 resolve_paths）。
MODELS = [
    {'exp_name': 'baseline_epsa_3ch',  'input_mode': 'inv_gx_gy'},
    {'exp_name': 'no_attention_3ch',   'input_mode': 'inv_gx_gy'},
    {'exp_name': 'epsa_inv_only',      'input_mode': 'inv_only'},
    {'exp_name': 'epsa_inv_gx',        'input_mode': 'inv_gx'},
    {'exp_name': 'epsa_inv_gy',        'input_mode': 'inv_gy'},
]


def resolve_paths(ablation_root):
    """依 ablation_root 設定各模型 model_path，並回傳對應的 OUTPUT_DIR。"""
    for m in MODELS:
        m['model_path'] = os.path.join(
            ablation_root, m['exp_name'], f"best_model_{m['exp_name']}.keras")
    return os.path.join(ablation_root, 'gradcam_compare')


GRADCAM_TARGET_LAYER = 'gradcam_target'
HEATMAP_ALPHA = 0.4          # 疊圖透明度（規範第七條，建議 0.35~0.4）


# ----------------------------------------------------------------------------
# Mixed precision 安全載入 patch（不改變權重、不重新訓練）
#
# 原 EPSABlock.call() 在 mixed precision (mixed_float16) 下，殘差相加
# `out + identity` 可能因 out 為 float16、identity 為 float32 而觸發
# AddV2 dtype mismatch（Input 'y' has type float32 ... does not match float16）。
#
# 實測發現：即使在 CUSTOM_OBJECTS 指定子類，Keras 反序列化仍可能呼叫
# epsanet50_ablation_all_in_one.py 中「已註冊」的原始 EPSABlock.call()。
# 因此這裡直接 monkey patch 原始 EPSABlock.call，邏輯與原版完全相同，
# 唯一差異是相加前將 identity cast 成 out.dtype，確保兩者 dtype 一致。
# 不影響任何權重、輸出語意與 Grad-CAM 計算，也不修改原始檔。
# ----------------------------------------------------------------------------
def epsablock_safe_call(self, x, training=None):
    identity = x
    out = self.relu(self.bn1(self.conv1(x), training=training))
    out = self.relu(self.bn2(self.psa(out), training=training))
    out = self.bn3(self.conv3(out), training=training)

    if self.use_downsample:
        identity = self.downsample_bn(self.downsample_conv(x), training=training)

    identity = tf.cast(identity, out.dtype)
    return self.relu(out + identity)


# 重要：直接 patch 原始 EPSABlock 類別，因為 Keras 反序列化時仍可能使用已註冊的原始類別
EPSABlock.call = epsablock_safe_call


CUSTOM_OBJECTS = {
    'SEWeightModule': SEWeightModule,
    'PSAModule': PSAModule,
    'PSAModuleNoAttention': PSAModuleNoAttention,
    'EPSABlock': EPSABlock,
    'Custom>EPSABlock': EPSABlock,
    'WarmupCosineDecay': WarmupCosineDecay,
}


def pick_colormap():
    """colormap 建議 turbo；若系統不支援則退回 jet（規範第七條）。"""
    try:
        plt.get_cmap('turbo')
        return 'turbo'
    except Exception:
        return 'jet'


COLORMAP = pick_colormap()


# ============================================================================
# 影像處理工具
# ============================================================================
def load_iinv_base(image_path):
    """
    產生 Grad-CAM 疊圖統一底圖 I_inv（翻轉後灰階，0~255）。
    與 make_load_image 的第一通道一致：gray = 255 - grayscale。
    對 1/2/3 通道模型，底圖一律使用此 I_inv，以確保視覺比較公平。
    回傳 float32 ndarray，shape (INPUT_HEIGHT, INPUT_WIDTH)。
    """
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_png(raw, channels=3)
    gray = tf.image.rgb_to_grayscale(img)
    gray = tf.squeeze(gray, axis=-1)
    gray = tf.cast(gray, tf.float32)
    gray = 255.0 - gray
    arr = gray.numpy()
    if arr.shape != (INPUT_HEIGHT, INPUT_WIDTH):
        raise ValueError(
            f"圖片尺寸 {arr.shape} 與模型輸入 ({INPUT_HEIGHT},{INPUT_WIDTH}) 不一致：{image_path}")
    return arr


def load_original_base(image_path):
    """
    讀取原始 PRPD 圖譜作為 Grad-CAM 視覺化底圖。
    不做黑白翻轉，不做 generic_preprocess（與模型輸入前處理無關，純粹為了視覺呈現）。
    回傳適合 matplotlib 顯示的 RGB uint8 ndarray，shape (INPUT_HEIGHT, INPUT_WIDTH, 3)。
    若尺寸不是 INPUT_HEIGHT x INPUT_WIDTH x 3 則報錯。
    """
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_png(raw, channels=3)
    arr = img.numpy()
    if arr.shape != (INPUT_HEIGHT, INPUT_WIDTH, 3):
        raise ValueError(
            f"原始圖譜尺寸 {arr.shape} 與預期 ({INPUT_HEIGHT},{INPUT_WIDTH},3) 不一致：{image_path}")
    return arr


def build_model_input(image_path, input_mode):
    """
    用消融主程式的 make_load_image(input_mode) 產生與訓練完全一致的模型輸入。
    回傳 shape (1, H, W, C) 的 float32 tensor。
    """
    load_image = make_load_image(input_mode)
    image, _ = load_image(image_path, tf.constant(0, dtype=tf.int64))
    image = tf.expand_dims(image, axis=0)
    return image


# ============================================================================
# 模型載入
# ============================================================================
def check_model_files():
    """模型檔不存在時清楚列出缺哪個（規範第八條）。"""
    missing = [m for m in MODELS if not os.path.isfile(m['model_path'])]
    if missing:
        print("\n[錯誤] 以下模型檔不存在：")
        for m in missing:
            print(f"  - {m['exp_name']}: {m['model_path']}")
        raise FileNotFoundError("缺少上述模型檔，無法繼續。請確認在原訓練機器上執行。")


def load_one_model(model_path):
    model = keras.models.load_model(
        model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
    return model


def verify_gradcam_layer(model, exp_name):
    """找不到 gradcam_target 時報錯並列出所有 layer 名稱（規範第八條）。"""
    names = [l.name for l in model.layers]
    if GRADCAM_TARGET_LAYER not in names:
        print(f"\n[錯誤] 模型 {exp_name} 找不到 '{GRADCAM_TARGET_LAYER}' 層。所有 layer 名稱：")
        for n in names:
            print(f"  {n}")
        raise ValueError(f"{exp_name} 缺少 {GRADCAM_TARGET_LAYER} 層。")


def verify_input_channels(model, input_mode, exp_name):
    """input_mode 通道數與模型 input_shape 最後一維不一致時報錯（規範第八條）。"""
    expected_ch = INPUT_MODE_CHANNELS[input_mode]
    in_shape = model.inputs[0].shape
    model_ch = int(in_shape[-1])
    if model_ch != expected_ch:
        raise ValueError(
            f"{exp_name}: input_mode={input_mode} 期望通道數 {expected_ch}，"
            f"但模型輸入通道數為 {model_ch}（input_shape={in_shape}）。")


# ============================================================================
# 推論
# ============================================================================
def predict_single(model, image_path, input_mode):
    """單張圖推論，回傳 (pred_idx, confidence, probs(float32, shape=[num_classes]))。"""
    x = build_model_input(image_path, input_mode)
    probs = model(x, training=False)
    probs = tf.cast(probs, tf.float32).numpy()[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return pred_idx, confidence, probs


# ============================================================================
# Grad-CAM
# ============================================================================
def compute_gradcam(model, image_path, input_mode, target_idx):
    """
    以 gradcam_target 層為 target layer 計算 Grad-CAM。
    mixed precision 下 feature map / output 可能為 float16，計算前一律 cast float32。
    回傳 normalize 至 0~1 並 resize 到 (INPUT_HEIGHT, INPUT_WIDTH) 的 heatmap (float32)。
    """
    x = build_model_input(image_path, input_mode)

    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(GRADCAM_TARGET_LAYER).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        conv_out = tf.cast(conv_out, tf.float32)
        preds = tf.cast(preds, tf.float32)
        class_score = preds[:, target_idx]

    grads = tape.gradient(class_score, conv_out)
    if grads is None:
        raise RuntimeError(
            f"Grad-CAM 梯度為 None：目標類別分數對 '{GRADCAM_TARGET_LAYER}' 的 feature map "
            f"無梯度連接。請確認 '{GRADCAM_TARGET_LAYER}' 層確實位於模型輸出的計算路徑上。")
    grads = tf.cast(grads, tf.float32)

    # 4. 對梯度做 global average pooling 得 channel weights
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))          # (C,)
    conv_out = conv_out[0]                                   # (h, w, C)

    # 5. weights 與 feature map 加權求和
    cam = tf.reduce_sum(conv_out * weights, axis=-1)         # (h, w)

    # 6. ReLU
    cam = tf.nn.relu(cam)

    # 7. normalize 0~1（各模型各自正規化）
    cam_max = tf.reduce_max(cam)
    cam = tf.cond(cam_max > 0, lambda: cam / cam_max, lambda: cam)

    # 8. resize 到 80x180
    cam = tf.image.resize(cam[..., tf.newaxis], [INPUT_HEIGHT, INPUT_WIDTH])
    cam = tf.squeeze(cam, axis=-1)
    return cam.numpy().astype(np.float32)


# ============================================================================
# 自動選圖（模式 A）
# ============================================================================
def auto_select_image(loaded, auto_select_top_n=300, select_class=None):
    """
    從原始（未擴增）test set 選圖：
      若 select_class 不為 None，僅在該類別的測試樣本中挑選。
      在 baseline 高信心且預測正確的前 auto_select_top_n 張樣本中，
      優先挑「所有模型皆正確」的樣本；
      若無，退而求其次挑「baseline 正確且 confidence 最高」的樣本（並於 log 說明）；
      若連 baseline 正確的樣本都沒有，則挑 baseline confidence 最高者（並於 log 說明）。
    回傳 (image_path, true_idx)。
    """
    test_dir = os.path.join(data_dir, 'test')
    test_paths, test_labels = load_data_from_split_dir(test_dir, class_to_idx)
    if len(test_paths) == 0:
        raise FileNotFoundError(f"test set 為空：{test_dir}")

    # [select_class] 限定類別：僅保留 true label == select_class 的測試樣本
    if select_class is not None:
        if select_class not in class_to_idx:
            raise ValueError(
                f"--select_class '{select_class}' 不在合法類別 {class_names} 中。")
        target_class_idx = class_to_idx[select_class]
        mask = (test_labels == target_class_idx)
        test_paths = test_paths[mask]
        test_labels = test_labels[mask]
        if len(test_paths) == 0:
            raise FileNotFoundError(
                f"test set 中沒有 true label 為 '{select_class}' 的樣本，無法選圖。")
        print(f"\n[自動選圖] 限定類別 select_class={select_class}"
              f"（共 {len(test_paths)} 張該類別測試樣本），先以 baseline 全量推論排序...")
    else:
        print(f"\n[自動選圖] 原始 test set 共 {len(test_paths)} 張，先以 baseline 全量推論排序...")

    baseline = next(m for m in MODELS if m['exp_name'] == 'baseline_epsa_3ch')
    bmodel = loaded['baseline_epsa_3ch']['model']
    b_input_mode = baseline['input_mode']

    # baseline 全量推論（batched，效率較佳）
    load_image = make_load_image(b_input_mode)
    ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(128).prefetch(tf.data.AUTOTUNE)
    probs = bmodel.predict(ds, verbose=0)
    probs = np.asarray(probs, dtype=np.float32)
    b_pred = np.argmax(probs, axis=1)
    b_conf = probs[np.arange(len(probs)), b_pred]

    # 候選：baseline 預測正確，依 confidence 由高到低
    correct_mask = (b_pred == test_labels)
    cand_idx = np.where(correct_mask)[0]
    if len(cand_idx) == 0:
        # baseline 全錯（極不可能），退回挑 baseline confidence 最高者
        print("[自動選圖] 警告：baseline 在 test set 無任何正確樣本，改挑 confidence 最高者。")
        best = int(np.argmax(b_conf))
        return test_paths[best], int(test_labels[best])

    cand_idx = cand_idx[np.argsort(-b_conf[cand_idx])]  # confidence 高到低

    other_models = [m for m in MODELS if m['exp_name'] != 'baseline_epsa_3ch']

    # 在前 N 個高信心候選中尋找「所有模型皆正確」者
    top_n = min(auto_select_top_n, len(cand_idx))
    print(f"[自動選圖] 在前 {top_n} 個 baseline 高信心正確樣本中，尋找所有模型皆正確者...")
    for rank, i in enumerate(cand_idx[:top_n]):
        path = test_paths[i]
        true_idx = int(test_labels[i])
        all_correct = True
        for m in other_models:
            pidx, _, _ = predict_single(loaded[m['exp_name']]['model'], path, m['input_mode'])
            if pidx != true_idx:
                all_correct = False
                break
        if all_correct:
            print(f"[自動選圖] 找到所有模型皆正確的樣本（baseline 信心排名第 {rank+1}）。")
            return path, true_idx

    # 退而求其次：baseline 正確且信心最高者
    best = int(cand_idx[0])
    print("[自動選圖] 前段候選中找不到『所有模型皆正確』的樣本，"
          "退而求其次採用 baseline 正確且 confidence 最高的樣本。")
    return test_paths[best], int(test_labels[best])


def infer_true_label_from_path(image_path):
    """由路徑資料夾名稱前兩碼推定 true label（AH/CT/HT/TD/CD/ID/SD）。"""
    folder = os.path.basename(os.path.dirname(image_path))
    prefix = folder[:2]
    if prefix not in class_to_idx:
        raise ValueError(
            f"無法由資料夾名稱 '{folder}' 前兩碼推定 true label（前兩碼='{prefix}'）。"
            f"請改用 --true_label 明確指定。合法類別：{class_names}")
    return prefix


# ============================================================================
# 繪圖
# ============================================================================
def overlay_heatmap(ax, base_image, heatmap, base_is_rgb=True):
    """在底圖上疊 Grad-CAM heatmap（統一 colormap / alpha）。
    base_is_rgb=True：原始 RGB PRPD 圖譜；False：灰階（如 I_inv）。"""
    if base_is_rgb:
        ax.imshow(base_image, aspect='auto')
    else:
        ax.imshow(base_image, cmap='gray', aspect='auto')
    ax.imshow(heatmap, cmap=COLORMAP, alpha=HEATMAP_ALPHA, aspect='auto',
              vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])


def save_individual_heatmap(out_path, base_image, heatmap, title, base_is_rgb=True):
    fig, ax = plt.subplots(figsize=(4.5, 2.2))
    overlay_heatmap(ax, base_image, heatmap, base_is_rgb=base_is_rgb)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_compare_figure(out_path, base_image, results, true_cls, image_filename,
                        target_mode, with_title=True, base_is_rgb=True):
    """
    2x3 總圖：(a) Original PRPD，(b)~(f) 五個模型 Grad-CAM（疊在原始圖譜上）。
    with_title=True：子圖標題含 exp_name / Pred / Conf，總標題含 true class 與檔名。
    with_title=False：簡潔版，子圖只放 (a)(b).. 與模型名稱，供論文排版。
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    axes = axes.ravel()
    sub_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    # (a) 原始 PRPD 底圖
    if base_is_rgb:
        axes[0].imshow(base_image, aspect='auto')
    else:
        axes[0].imshow(base_image, cmap='gray', aspect='auto')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    if with_title:
        axes[0].set_title("(a) Original PRPD", fontsize=10)
    else:
        axes[0].set_title("(a) Original PRPD", fontsize=10)

    for k, r in enumerate(results):
        ax = axes[k + 1]
        overlay_heatmap(ax, base_image, r['heatmap'], base_is_rgb=base_is_rgb)
        if with_title:
            ax.set_title(
                f"{sub_labels[k+1]} {r['exp_name']}\n"
                f"Pred: {r['pred_cls']}  Conf: {r['pred_confidence']*100:.2f}%",
                fontsize=9)
        else:
            ax.set_title(f"{sub_labels[k+1]} {r['exp_name']}", fontsize=10)

    if with_title:
        fig.suptitle(
            f"消融實驗 Grad-CAM 同圖比較  |  True class: {true_cls}  |  "
            f"檔案: {image_filename}  |  target_mode={target_mode}",
            fontsize=12)
        # 提醒文字（規範第七、九條）
        fig.text(0.5, 0.005,
                 "註：各模型 heatmap 各自正規化至 0~1，顏色僅表該模型內部相對關注區域，"
                 "不可跨模型作絕對強度比較，亦非模型優劣之量化依據。",
                 ha='center', fontsize=8)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        fig.tight_layout()

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="第 4.6 節 消融實驗 Grad-CAM 同圖比較（僅 inference + Grad-CAM）")
    parser.add_argument('--auto_select', action='store_true',
                        help='模式 A：自動從原始 test set 挑一張（優先所有模型皆正確）')
    parser.add_argument('--image_path', type=str, default=None,
                        help='模式 B：指定圖檔路徑')
    parser.add_argument('--true_label', type=str, default=None,
                        help='模式 B 可選：指定 true label（AH/CT/HT/TD/CD/ID/SD）；'
                             '省略則由路徑資料夾名稱前兩碼推定')
    parser.add_argument('--target_mode', type=str, default='pred',
                        choices=['pred', 'true'],
                        help='Grad-CAM 目標類別：pred（預設）或 true')
    parser.add_argument('--ablation_root', type=str, default=DEFAULT_ABLATION_ROOT,
                        help='消融實驗結果根目錄；模型路徑與輸出資料夾皆以此為基準。'
                             '預設為原訓練機器的 ablation_results 路徑。')
    parser.add_argument('--auto_select_top_n', type=int, default=300,
                        help='自動選圖時，在 baseline 高信心且預測正確的前 N 張樣本中'
                             '尋找所有模型皆正確者（預設 300）。')
    parser.add_argument('--select_class', type=str, default=None,
                        help='自動選圖時限定 true label 類別（AH/CT/HT/TD/CD/ID/SD）；'
                             '僅在該類別測試樣本中選圖。')
    args = parser.parse_args()

    if not args.auto_select and not args.image_path:
        parser.error("請指定 --auto_select 或 --image_path 其中之一。")

    # 依 ablation_root 解析各模型 model_path 與輸出資料夾
    output_dir = resolve_paths(args.ablation_root)

    print("=" * 78)
    print("第 4.6 節 消融實驗 Grad-CAM 視覺化比較")
    print(f"colormap = {COLORMAP}  |  alpha = {HEATMAP_ALPHA}  |  target_mode = {args.target_mode}")
    print(f"ablation_root = {args.ablation_root}")
    print(f"select_class = {args.select_class}")
    print("visualization_base = Original PRPD image")
    print("model_input_note = Model input still follows ablation preprocessing; "
          "only visualization base image is changed.")
    print("=" * 78)

    os.makedirs(output_dir, exist_ok=True)

    # --- 檢查並載入所有模型 ---
    check_model_files()
    print("\n載入模型中...")
    loaded = {}
    load_ok = True
    for m in MODELS:
        try:
            model = load_one_model(m['model_path'])
            verify_gradcam_layer(model, m['exp_name'])
            verify_input_channels(model, m['input_mode'], m['exp_name'])
            loaded[m['exp_name']] = {'model': model, 'input_mode': m['input_mode'],
                                     'model_path': m['model_path']}
            print(f"  [OK] {m['exp_name']}  (input_mode={m['input_mode']}, "
                  f"input_shape={tuple(model.inputs[0].shape)})")
        except Exception as e:
            load_ok = False
            print(f"  [FAIL] {m['exp_name']}: {e}")
            raise
    print(f"\n所有模型載入狀態：{'全部成功' if load_ok else '有失敗'}")

    # --- 取得測試圖譜（模式 A / B）---
    if args.image_path:
        image_path = args.image_path
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"--image_path 指定的圖檔不存在：{image_path}")
        true_label = args.true_label if args.true_label else infer_true_label_from_path(image_path)
        if true_label not in class_to_idx:
            raise ValueError(f"--true_label '{true_label}' 不在合法類別 {class_names} 中。")
        true_idx = class_to_idx[true_label]
        print(f"\n[模式 B] 指定圖檔：{image_path}")
    else:
        image_path, true_idx = auto_select_image(
            loaded, auto_select_top_n=args.auto_select_top_n, select_class=args.select_class)
        true_label = class_names[true_idx]
        print(f"\n[模式 A] 自動選圖完成：{image_path}")

    true_cls = class_names[true_idx]
    image_filename = os.path.basename(image_path)
    print(f"  true_class = {true_cls} ({chinese_labels[true_idx]})")

    # --- 視覺化底圖 ---
    # 預設疊圖底圖：原始未翻轉 PRPD 圖譜（僅影響視覺呈現，不影響模型輸入與 Grad-CAM 計算）。
    base_original = load_original_base(image_path)
    # I_inv 底圖：保留作為檢查用。
    base_iinv = load_iinv_base(image_path)

    # 輸出 original_base_image.png（原始圖底圖）
    original_path = os.path.join(output_dir, 'original_base_image.png')
    fig_o, ax_o = plt.subplots(figsize=(4.5, 2.2))
    ax_o.imshow(base_original, aspect='auto')
    ax_o.set_xticks([])
    ax_o.set_yticks([])
    ax_o.set_title(f"Original PRPD  |  True: {true_cls}  |  {image_filename}", fontsize=9)
    fig_o.tight_layout()
    fig_o.savefig(original_path, dpi=300, bbox_inches='tight')
    plt.close(fig_o)

    # 輸出 iinv_base_image.png（檢查用）
    iinv_path = os.path.join(output_dir, 'iinv_base_image.png')
    fig_b, ax_b = plt.subplots(figsize=(4.5, 2.2))
    ax_b.imshow(base_iinv, cmap='gray', aspect='auto')
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    ax_b.set_title(f"I_inv base  |  True: {true_cls}  |  {image_filename}", fontsize=9)
    fig_b.tight_layout()
    fig_b.savefig(iinv_path, dpi=300, bbox_inches='tight')
    plt.close(fig_b)

    # --- 對每個模型推論 + Grad-CAM ---
    results = []
    print("\n各模型推論與 Grad-CAM：")
    for m in MODELS:
        info = loaded[m['exp_name']]
        model = info['model']
        input_mode = info['input_mode']

        pred_idx, conf, probs = predict_single(model, image_path, input_mode)
        pred_cls = class_names[pred_idx]
        correct = (pred_idx == true_idx)

        # Grad-CAM 目標類別
        target_idx = pred_idx if args.target_mode == 'pred' else true_idx
        target_cls = class_names[target_idx]

        heatmap = compute_gradcam(model, image_path, input_mode, target_idx)

        heatmap_path = os.path.join(output_dir, f"gradcam_{m['exp_name']}.png")
        save_individual_heatmap(
            heatmap_path, base_original, heatmap,
            f"{m['exp_name']}  Pred:{pred_cls} ({conf*100:.2f}%)  [GradCAM:{target_cls}]",
            base_is_rgb=True)

        results.append({
            'exp_name': m['exp_name'],
            'model_path': info['model_path'],
            'input_mode': input_mode,
            'pred_idx': pred_idx,
            'pred_cls': pred_cls,
            'pred_confidence': conf,
            'target_mode': args.target_mode,
            'gradcam_target_class': target_cls,
            'correct': correct,
            'heatmap': heatmap,
            'heatmap_path': heatmap_path,
        })
        mark = '正確' if correct else '錯誤'
        print(f"  {m['exp_name']:<18} Pred={pred_cls:<3} Conf={conf*100:6.2f}%  "
              f"({mark})  GradCAM target={target_cls}")

    # --- 總圖（含標題 / 簡潔版）---
    fig_path = os.path.join(output_dir, 'gradcam_compare_same_image.png')
    fig_path_no_title = os.path.join(output_dir, 'gradcam_compare_same_image_no_title.png')
    save_compare_figure(fig_path, base_original, results, true_cls, image_filename,
                        args.target_mode, with_title=True, base_is_rgb=True)
    save_compare_figure(fig_path_no_title, base_original, results, true_cls, image_filename,
                        args.target_mode, with_title=False, base_is_rgb=True)

    # --- 重組資料包（供 epsanet50_ablation_gradcam_compose_figures.py 單獨重畫兩張總圖用）---
    # 儲存原始 heatmap 陣列與底圖等，使兩張總圖可在「不需 TensorFlow / 不重跑模型」的情況下重建。
    compose_npz_path = os.path.join(output_dir, 'gradcam_compose_data.npz')
    np.savez(
        compose_npz_path,
        exp_names=np.array([r['exp_name'] for r in results], dtype=object),
        heatmaps=np.stack([r['heatmap'] for r in results]).astype(np.float32),
        base_original=base_original,
        base_iinv=base_iinv.astype(np.float32),
        pred_cls=np.array([r['pred_cls'] for r in results], dtype=object),
        pred_confidence=np.array([r['pred_confidence'] for r in results], dtype=np.float32),
        gradcam_target_class=np.array([r['gradcam_target_class'] for r in results], dtype=object),
        correct=np.array([r['correct'] for r in results], dtype=bool),
        true_cls=np.array(true_cls),
        image_filename=np.array(image_filename),
        image_path=np.array(image_path),
        target_mode=np.array(args.target_mode),
        select_class=np.array('' if args.select_class is None else args.select_class),
        colormap=np.array(COLORMAP),
        heatmap_alpha=np.array(HEATMAP_ALPHA, dtype=np.float32),
    )

    # --- metadata CSV ---
    csv_path = os.path.join(output_dir, 'gradcam_compare_metadata.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            'exp_name', 'model_path', 'input_mode', 'image_path', 'true_class',
            'pred_class', 'pred_confidence', 'target_mode', 'gradcam_target_class',
            'correct', 'output_heatmap_path'])
        for r in results:
            writer.writerow([
                r['exp_name'], r['model_path'], r['input_mode'], image_path, true_cls,
                r['pred_cls'], f"{r['pred_confidence']:.6f}", r['target_mode'],
                r['gradcam_target_class'], r['correct'], r['heatmap_path']])

    # --- selected_sample_info.txt ---
    info_path = os.path.join(output_dir, 'selected_sample_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("第 4.6 節 消融實驗 Grad-CAM 同圖比較 — 選用樣本資訊\n")
        f.write("=" * 70 + "\n")
        f.write(f"選用圖檔路徑 : {image_path}\n")
        f.write(f"true label   : {true_cls} ({chinese_labels[true_idx]})\n")
        f.write(f"select_class : {args.select_class}\n")
        f.write(f"target_mode  : {args.target_mode}\n")
        f.write(f"colormap     : {COLORMAP}   alpha: {HEATMAP_ALPHA}\n")
        f.write("visualization_base = Original PRPD image\n")
        f.write("model_input_note = Model input still follows ablation preprocessing; "
                "only visualization base image is changed.\n")
        f.write("-" * 70 + "\n")
        f.write("各模型預測結果：\n")
        for r in results:
            mark = '正確' if r['correct'] else '錯誤'
            f.write(f"  {r['exp_name']:<18} input_mode={r['input_mode']:<9} "
                    f"Pred={r['pred_cls']:<3} Conf={r['pred_confidence']*100:6.2f}%  "
                    f"({mark})  GradCAM target={r['gradcam_target_class']}\n")
        f.write("-" * 70 + "\n")
        f.write("提醒：Grad-CAM 僅為定性視覺化工具，各模型 heatmap 各自正規化至 0~1，\n")
        f.write("      顏色僅代表該模型內部相對關注區域，不能跨模型作絕對強度比較，\n")
        f.write("      亦不能單獨作為模型優劣之證據。\n")

    # --- 終端彙整回報 ---
    wrong = [r for r in results if not r['correct']]
    print("\n" + "=" * 78)
    print("完成。輸出檔案：")
    print(f"  原始 PRPD 底圖     : {original_path}")
    print(f"  I_inv 底圖（檢查） : {iinv_path}")
    print(f"  總圖（含標題）     : {fig_path}")
    print(f"  總圖（簡潔版）     : {fig_path_no_title}")
    print(f"  重組資料包 npz     : {compose_npz_path}")
    print(f"  metadata CSV       : {csv_path}")
    print(f"  樣本資訊 txt       : {info_path}")
    for r in results:
        print(f"  個別 heatmap       : {r['heatmap_path']}")
    print("-" * 78)
    print(f"選用圖檔: {image_path}")
    print(f"true label: {true_cls}")
    print(f"select_class: {args.select_class}")
    print("visualization_base = Original PRPD image")
    print("model_input_note = Model input still follows ablation preprocessing; "
          "only visualization base image is changed.")
    if wrong:
        print("以下模型對此圖預測與 true label 不一致（僅客觀列出，不作優劣解釋）：")
        for r in wrong:
            print(f"  - {r['exp_name']}: Pred={r['pred_cls']} (Conf {r['pred_confidence']*100:.2f}%)")
    else:
        print("所有模型對此圖皆預測正確。")
    print("=" * 78)


if __name__ == '__main__':
    main()
