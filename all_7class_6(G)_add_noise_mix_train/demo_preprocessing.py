#!/usr/bin/env python3
"""
demo_preprocessing.py - PRPD 資料前處理與擴增展示腳本

隨機從訓練集中取一張圖片，逐步展示：
1. 資料前處理流程（灰階、黑白翻轉、Sobel 邊緣、三通道合成）
2. 循環水平位移擴增（14x）
3. 8 種噪聲模擬

每個步驟存為個別圖片，並生成 matplotlib 總覽大圖（含中文標注）。
"""

import os
import glob
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============================================================================
# 設定區
# ============================================================================

# 資料路徑（請依實際環境修改）
DATA_DIR = '/home/cckuo/m11307u09/GG/1126/data_original'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')

# 輸出路徑
OUTPUT_DIR = '/home/cckuo/m11307u09/GG/1126/epsanet50/all in one/all_7class_6(G)_add_noise_mix_train/demo_preprocessing'

# 圖片尺寸
INPUT_WIDTH = 180
INPUT_HEIGHT = 80

# 擴增倍數
AUGMENT_FACTOR = 14  # 1 原始 + 13 隨機位移

# 類別定義
CLASS_NAMES = ['AH', 'CT', 'HT', 'TD', 'CD', 'ID', 'SD']
CHINESE_LABELS = ['空洞', '碳痕', '接頭異常', '不規則邊緣', '典型電暈放電', '典型內部放電', '典型表面放電']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

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

setup_chinese_font()
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 前處理函數
# ============================================================================

def load_and_show_preprocessing(image_path, output_dir):
    """
    逐步展示前處理流程，保存個別圖片和總覽大圖。

    回傳前處理完成的三通道影像 (H, W, 3)，數值範圍 0-255 uint8。
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 讀取原始 RGB 圖片
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"無法讀取圖片: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_dir, 'step1_original_rgb.png'), img_bgr)

    # Step 2: RGB -> 灰階
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, 'step2_grayscale.png'), gray)

    # Step 3: 黑白翻轉 (255 - gray)
    inverted = (255 - gray).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, 'step3_inverted.png'), inverted)

    # Step 4: Sobel X 邊緣
    gray_float = inverted.astype(np.float64)
    sobel_x = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)
    sx_max = sobel_x.max()
    sobel_x_norm = (sobel_x / sx_max * 255.0).astype(np.uint8) if sx_max > 0 else sobel_x.astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, 'step4_sobel_x.png'), sobel_x_norm)

    # Step 5: Sobel Y 邊緣
    sobel_y = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.abs(sobel_y)
    sy_max = sobel_y.max()
    sobel_y_norm = (sobel_y / sy_max * 255.0).astype(np.uint8) if sy_max > 0 else sobel_y.astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, 'step5_sobel_y.png'), sobel_y_norm)

    # Step 6: 三通道合成 [翻轉灰階, SobelX, SobelY]
    multichannel = np.stack([inverted, sobel_x_norm, sobel_y_norm], axis=-1)
    # 存為 PNG（BGR 順序給 OpenCV）
    cv2.imwrite(os.path.join(output_dir, 'step6_multichannel.png'),
                cv2.cvtColor(multichannel, cv2.COLOR_RGB2BGR))

    # 生成總覽圖 overview_preprocessing.png
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle('PRPD 資料前處理流程', fontsize=16, fontweight='bold')

    titles = [
        'Step 1: 原始 RGB 圖片',
        'Step 2: 灰階轉換',
        'Step 3: 黑白翻轉 (255-gray)',
        'Step 4: Sobel X 邊緣偵測',
        'Step 5: Sobel Y 邊緣偵測',
        'Step 6: 三通道合成\n[翻轉灰階, Sobel X, Sobel Y]',
    ]
    images = [img_rgb, gray, inverted, sobel_x_norm, sobel_y_norm, multichannel]
    cmaps = [None, 'gray', 'gray', 'gray', 'gray', None]

    for ax, title, img, cmap in zip(axes.flat, titles, images, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'overview_preprocessing.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  前處理總覽圖已儲存: overview_preprocessing.png")

    return multichannel


# ============================================================================
# 循環水平位移擴增函數
# ============================================================================

def show_augmentation(preprocessed_image, output_dir):
    """
    生成 1 張原始 + 13 張隨機水平循環位移，保存個別圖片和總覽大圖。
    """
    os.makedirs(output_dir, exist_ok=True)
    aug_images = []
    shift_values = [0]  # 第一張為原始

    # 生成 13 個隨機位移量
    for _ in range(AUGMENT_FACTOR - 1):
        shift_values.append(random.randint(0, INPUT_WIDTH - 1))

    for i, shift in enumerate(shift_values):
        shifted = np.roll(preprocessed_image, shift, axis=1)
        aug_images.append(shifted)

        if i == 0:
            fname = 'aug_00_original.png'
        else:
            fname = f'aug_{i:02d}_shift.png'
        cv2.imwrite(os.path.join(output_dir, fname), shifted[:, :, 0])

    # 生成總覽圖 overview_augmentation.png (3x5 佈局，共 15 格，用 14 個)
    rows, cols = 2, 7
    fig, axes = plt.subplots(rows, cols, figsize=(21, 7))
    fig.suptitle('循環水平位移擴增 (14x)', fontsize=16, fontweight='bold')

    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx < len(aug_images):
            ax.imshow(aug_images[idx][:, :, 0], cmap='gray')
            if idx == 0:
                ax.set_title('原始 (shift=0)', fontsize=9)
            else:
                ax.set_title(f'位移={shift_values[idx]}px', fontsize=9)
        else:
            ax.axis('off')
            continue
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'overview_augmentation.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  擴增總覽圖已儲存: overview_augmentation.png")


# ============================================================================
# 噪聲模擬函數
# ============================================================================

def _find_signal_rows_from_bottom(gray_channel):
    """
    由下往上尋找有訊號的列（像素值 > 0 的列），回傳行號列表（由下往上排序）。
    """
    height = gray_channel.shape[0]
    signal_rows = []
    for r in range(height - 1, -1, -1):
        if np.any(gray_channel[r] > 0):
            signal_rows.append(r)
    return signal_rows


def show_noise_types(preprocessed_image, output_dir):
    """
    展示 8 種噪聲類型，保存個別圖片和總覽大圖。
    """
    os.makedirs(output_dir, exist_ok=True)
    gray_ch = preprocessed_image[:, :, 0]
    fill_value = preprocessed_image.max()
    signal_rows = _find_signal_rows_from_bottom(gray_ch)
    height, width = preprocessed_image.shape[:2]

    noise_images = []
    noise_descriptions = [
        'Type 1: 第1訊號列塗白',
        'Type 2: 第2訊號列塗白',
        'Type 3: 第3訊號列塗白',
        'Type 4: 第1列+上方2列塗白',
        'Type 5: 第1列上下加隨機點',
        'Type 6: 第2列上下加隨機點',
        'Type 7: 第3列上下加隨機點',
        'Type 8: 3列區塊邊界加隨機點',
    ]

    shift_values = []
    for noise_type in range(1, 9):
        # 先做隨機水平循環位移，再加噪聲（與訓練流程一致）
        shift = random.randint(0, INPUT_WIDTH - 1)
        shift_values.append(shift)
        img = np.roll(preprocessed_image.copy(), shift, axis=1)

        if noise_type <= 4:
            # Types 1-4: 塗白指定列
            rows_to_fill = _get_rows_to_fill(noise_type, signal_rows, height)
            for r in rows_to_fill:
                if 0 <= r < height:
                    img[r, :, :] = fill_value
        else:
            # Types 5-8: 隨機像素點
            img = _add_random_pixels(img, noise_type, signal_rows, fill_value, height, width)

        noise_images.append(img)
        cv2.imwrite(os.path.join(output_dir, f'noise_type{noise_type}.png'),
                    img[:, :, 0])

    # 生成總覽圖 overview_noise.png (2x4 佈局)
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    fig.suptitle('8 種底部噪聲模擬（含水平循環位移）', fontsize=16, fontweight='bold')

    for ax, img, desc, sv in zip(axes.flat, noise_images, noise_descriptions, shift_values):
        ax.imshow(img[:, :, 0], cmap='gray')
        ax.set_title(f'{desc}\n(shift={sv}px)', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'overview_noise.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  噪聲總覽圖已儲存: overview_noise.png")


def _get_rows_to_fill(noise_type, signal_rows, height):
    """根據噪聲類型回傳要塗白的行號列表。"""
    if noise_type == 1:
        return [signal_rows[0]] if len(signal_rows) >= 1 else []
    elif noise_type == 2:
        return [signal_rows[1]] if len(signal_rows) >= 2 else []
    elif noise_type == 3:
        return [signal_rows[2]] if len(signal_rows) >= 3 else []
    elif noise_type == 4:
        if len(signal_rows) >= 1:
            start = signal_rows[0]
            return [r for r in [start, start - 1, start - 2] if 0 <= r < height]
        return []
    return []


def _add_random_pixels(img, noise_type, signal_rows, fill_value, height, width):
    """對指定列的上下相鄰列加入隨機像素點。"""

    def add_pixels_to_row(image, row_idx, num_pixels):
        if row_idx < 0 or row_idx >= height:
            return image
        actual_pixels = min(num_pixels, width)
        x_coords = np.random.choice(width, size=actual_pixels, replace=False)
        image[row_idx, x_coords, :] = fill_value
        return image

    def process_single_row(target_row):
        result = img.copy()
        if target_row < 0:
            return result
        above_row = target_row - 1
        below_row = target_row + 1
        num_above = random.randint(20, 30)
        num_below = random.randint(4, 10)
        result = add_pixels_to_row(result, above_row, num_above)
        result = add_pixels_to_row(result, below_row, num_below)
        return result

    if noise_type == 5:
        target = signal_rows[0] if len(signal_rows) >= 1 else -1
        return process_single_row(target)
    elif noise_type == 6:
        target = signal_rows[1] if len(signal_rows) >= 2 else -1
        return process_single_row(target)
    elif noise_type == 7:
        target = signal_rows[2] if len(signal_rows) >= 3 else -1
        return process_single_row(target)
    elif noise_type == 8:
        if len(signal_rows) < 1:
            return img.copy()
        bottom_row = signal_rows[0]
        top_row = bottom_row - 2
        above_row = top_row - 1
        below_row = bottom_row + 1
        num_above = random.randint(20, 30)
        num_below = random.randint(4, 10)
        result = img.copy()
        result = add_pixels_to_row(result, above_row, num_above)
        result = add_pixels_to_row(result, below_row, num_below)
        return result

    return img.copy()


# ============================================================================
# 主程式
# ============================================================================

def main():
    print("=" * 60)
    print("PRPD 資料前處理與擴增展示")
    print("=" * 60)

    # 1. 掃描 train/ 目錄，收集所有 .png 圖片路徑
    if not os.path.isdir(TRAIN_DIR):
        print(f"錯誤：找不到訓練目錄: {TRAIN_DIR}")
        print("請修改腳本開頭的 DATA_DIR 變數。")
        return

    all_images = []
    subdirs = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    for subdir in subdirs:
        class_prefix = subdir[:2]
        if class_prefix in CLASS_TO_IDX:
            paths = glob.glob(os.path.join(TRAIN_DIR, subdir, '*.png'))
            for p in paths:
                all_images.append((p, class_prefix))

    if not all_images:
        print(f"錯誤：在 {TRAIN_DIR} 中找不到任何 .png 圖片")
        return

    print(f"找到 {len(all_images)} 張訓練圖片")

    # 2. 隨機選取一張
    chosen_path, chosen_class = random.choice(all_images)
    class_idx = CLASS_TO_IDX[chosen_class]
    chinese_name = CHINESE_LABELS[class_idx]

    print(f"\n選取圖片: {chosen_path}")
    print(f"類別: {chosen_class} ({chinese_name})")

    # 3. 建立輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"輸出目錄: {os.path.abspath(OUTPUT_DIR)}")

    # 4. 執行前處理展示
    print(f"\n--- 步驟一：資料前處理 ---")
    preprocessed = load_and_show_preprocessing(chosen_path, OUTPUT_DIR)
    print(f"  個別圖片已儲存: step1~step6")

    # 5. 執行擴增展示
    print(f"\n--- 步驟二：循環水平位移擴增 ---")
    show_augmentation(preprocessed, OUTPUT_DIR)
    print(f"  個別圖片已儲存: aug_00~aug_13")

    # 6. 執行噪聲展示
    print(f"\n--- 步驟三：噪聲模擬 ---")
    show_noise_types(preprocessed, OUTPUT_DIR)
    print(f"  個別圖片已儲存: noise_type1~noise_type8")

    # 7. 印出所有輸出檔案清單
    print(f"\n{'=' * 60}")
    print("所有輸出檔案：")
    print(f"{'=' * 60}")
    output_files = sorted(os.listdir(OUTPUT_DIR))
    for f in output_files:
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")
    print(f"\n共 {len(output_files)} 個檔案")
    print("完成！")


if __name__ == '__main__':
    main()
