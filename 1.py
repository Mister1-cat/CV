import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os

plt.rcParams['font.family'] = 'DejaVu Sans'

def rgb_to_grayscale(image_rgb):
    if len(image_rgb.shape) == 2:
        return image_rgb.copy()
    height, width, _ = image_rgb.shape
    gray = np.zeros((height, width), dtype=np.uint8)
    for r in range(height):
        for c in range(width):
            R, G, B = image_rgb[r, c]
            val = 0.299 * R + 0.587 * G + 0.114 * B
            gray[r, c] = int(round(val))
    return gray

def pad_image(image, pad_size, mode='edge'):
    if len(image.shape) == 2:
        return np.pad(image, pad_size, mode=mode)
    else:
        padded = np.zeros((image.shape[0]+2*pad_size, image.shape[1]+2*pad_size, image.shape[2]),
                          dtype=image.dtype)
        for ch in range(image.shape[2]):
            padded[:,:,ch] = np.pad(image[:,:,ch], pad_size, mode=mode)
        return padded

def median_filter(image, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным")
    if len(image.shape) == 2:
        pad = kernel_size // 2
        padded = pad_image(image, pad)
        filtered = np.zeros_like(image)
        h, w = image.shape
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kernel_size, j:j+kernel_size].flatten()
                filtered[i, j] = np.median(window)
        return filtered.astype(np.uint8)
    else:
        result = np.zeros_like(image)
        for ch in range(3):
            result[:,:,ch] = median_filter(image[:,:,ch], kernel_size)
        return result

def gaussian_kernel(size, sigma):
    kernel = np.zeros(size, dtype=np.float32)
    center = size // 2
    sum_val = 0.0
    for i in range(size):
        x = i - center
        kernel[i] = math.exp(-(x**2) / (2 * sigma**2))
        sum_val += kernel[i]
    return kernel / sum_val

def gaussian_filter(image, kernel_size=5, sigma=1.0):
    if kernel_size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным")
    kernel = gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2
    if len(image.shape) == 2:
        padded = pad_image(image, pad)
        h, w = image.shape
        temp = np.zeros_like(image, dtype=np.float32)
        for i in range(h):
            for j in range(w):
                window = padded[i+pad, j:j+kernel_size]
                temp[i, j] = np.sum(window * kernel)
        padded2 = pad_image(temp, pad)
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(h):
            for j in range(w):
                window = padded2[i:i+kernel_size, j+pad]
                result[i, j] = np.sum(window * kernel)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = np.zeros_like(image, dtype=np.float32)
        for ch in range(3):
            result[:,:,ch] = gaussian_filter(image[:,:,ch], kernel_size, sigma)
        return result.astype(np.uint8)

def erosion_binary(binary_image, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным")
    pad = kernel_size // 2
    padded = pad_image(binary_image, pad, mode='constant')
    h, w = binary_image.shape
    eroded = np.zeros_like(binary_image)
    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            if np.all(window == 255):
                eroded[i, j] = 255
    return eroded

def dilation_binary(binary_image, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным")
    pad = kernel_size // 2
    padded = pad_image(binary_image, pad, mode='constant')
    h, w = binary_image.shape
    dilated = np.zeros_like(binary_image)
    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            if np.any(window == 255):
                dilated[i, j] = 255
    return dilated

def threshold_binarize(image, threshold):
    if len(image.shape) == 2:
        binary = np.zeros_like(image)
        binary[image > threshold] = 255
        return binary
    else:
        gray = rgb_to_grayscale(image)
        return threshold_binarize(gray, threshold)

def histogram_equalization(image):
    if len(image.shape) == 2:
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        cdf_normalized = cdf_normalized.astype(np.uint8)
        equalized = cdf_normalized[image]
        return equalized.astype(np.uint8)
    else:
        result = np.zeros_like(image)
        for ch in range(3):
            result[:,:,ch] = histogram_equalization(image[:,:,ch])
        return result

def rotate_90_manual_single(image_2d, times=1):
    times = times % 4
    if times == 0:
        return image_2d.copy()
    rotated = image_2d.copy()
    for _ in range(times):
        h, w = rotated.shape
        transposed = np.zeros((w, h), dtype=rotated.dtype)
        for i in range(h):
            for j in range(w):
                transposed[j, i] = rotated[i, j]
        flipped = np.zeros_like(transposed)
        for i in range(transposed.shape[0]):
            flipped[i, :] = transposed[i, ::-1]
        rotated = flipped
    return rotated

def rotate_90(image, times=1):
    times = times % 4
    if times == 0:
        return image.copy()
    
    if len(image.shape) == 2:
        return rotate_90_manual_single(image, times)
    else:
        channels = []
        for ch in range(image.shape[2]):
            rotated_ch = rotate_90_manual_single(image[:, :, ch], times)
            channels.append(rotated_ch)
        return np.stack(channels, axis=2)

def create_figure_pair(original, result, title_left, title_right, suptitle, enlarge_pixels=False):
    figsize = (12, 6) if enlarge_pixels else (10, 5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    cmap1 = 'gray' if len(original.shape) == 2 else None
    cmap2 = 'gray' if len(result.shape) == 2 else None
    interp = 'nearest' if enlarge_pixels else 'auto'
    ax1.imshow(original, cmap=cmap1, interpolation=interp)
    ax1.set_title(title_left)
    ax1.axis('off')
    ax2.imshow(result, cmap=cmap2, interpolation=interp)
    ax2.set_title(title_right)
    ax2.axis('off')
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig

def main():
    image_path = '123.jpg'
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден.")
        return

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = rgb_to_grayscale(img_rgb)

    operations = [
        ("Медианный фильтр (ядро 5)",      lambda: median_filter(img_rgb, 5)),
        ("Фильтр Гаусса (σ=1.5)",          lambda: gaussian_filter(img_rgb, 5, 1.5)),
        ("Эрозия",     lambda: erosion_binary(threshold_binarize(gray, 127), 3)),
        ("Дилатация",  lambda: dilation_binary(threshold_binarize(gray, 127), 3)),
        ("Бинаризация (RGB, порог 127)",   lambda: threshold_binarize(img_rgb, 127)),
        ("Выравнивание гистограммы",       lambda: histogram_equalization(img_rgb)),
        ("Поворот на 90°",                 lambda: rotate_90(img_rgb, 1)),
        ("Поворот на 180°",                lambda: rotate_90(img_rgb, 2)),
    ]

    main_figures = []
    for idx, (name, func) in enumerate(operations, start=1):
        result = func()
        fig = create_figure_pair(img_rgb, result,
                                  "Исходное изображение", f"{name}",
                                  f"{name}")
        main_figures.append(fig)

    if main_figures:
        print("Закройте все, чтобы продолжить.")
        plt.show()

if __name__ == "__main__":
    main()

    print("\n")
    demo_figures = []
    counter = 1

    # 1. Перевод в серый
    h_demo, w_demo = 48, 48
    rgb_demo = np.random.randint(0, 256, (h_demo, w_demo, 3), dtype=np.uint8)
    gray_demo = rgb_to_grayscale(rgb_demo)
    fig = create_figure_pair(rgb_demo, gray_demo,
                             "Исходный RGB 48×48", "Серое изображение",
                             f" Перевод в серый", enlarge_pixels=True)
    demo_figures.append(fig)
    counter += 1

    # 2. Паддинг edge
    img_big = np.arange(49*49, dtype=np.uint8).reshape(49,49) + 50
    padded_edge = pad_image(img_big, 2, mode='edge')
    fig = create_figure_pair(img_big, padded_edge, "Исходная матрица 49×49", "Паддинг (edge, pad=2)",
                             f" Дополнение краёв (edge)", enlarge_pixels=True)
    demo_figures.append(fig)
    counter += 1

    # 3. Паддинг constant
    padded_const = pad_image(img_big, 2, mode='constant')
    fig = create_figure_pair(img_big, padded_const, "Исходная матрица 49×49", "Паддинг (constant, pad=2)",
                             f" Дополнение краёв (constant)", enlarge_pixels=True)
    demo_figures.append(fig)
    counter += 1

    # 4. Медианный фильтр
    np.random.seed(42)
    clean_big = np.tile(np.linspace(30, 200, 49, dtype=np.uint8), (49,1))
    noisy_big = clean_big.copy()
    for _ in range(100):  
        r, c = np.random.randint(0, 49, 2)
        noisy_big[r, c] = np.random.choice([0, 255])
    median_big = median_filter(noisy_big, kernel_size=3)
    fig = create_figure_pair(noisy_big, median_big, "Зашумлённое", "Медианный фильтр",
                             f" Медианный фильтр", enlarge_pixels=True)
    demo_figures.append(fig)
    counter += 1

    # 5. Фильтр Гаусса
    edge_big = np.zeros((49,49), dtype=np.uint8)
    edge_big[:, :24] = 30
    edge_big[:, 24:] = 200
    blurred_big = gaussian_filter(edge_big, kernel_size=5, sigma=1.5)
    fig = create_figure_pair(edge_big, blurred_big, "Резкий перепад", "Размытие Гаусса",
                             f" Фильтр Гаусса", enlarge_pixels=True)
    demo_figures.append(fig)
    counter += 1

    # 6. Эрозия
    square_big = np.zeros((49,49), dtype=np.uint8)
    square_big[2:47, 2:47] = 255  
    eroded_big = erosion_binary(square_big, kernel_size=3)
    fig = create_figure_pair(square_big, eroded_big, "Белый квадрат 45×45", "Эрозия",
                             f" Эрозия", enlarge_pixels=True)
    demo_figures.append(fig)
    counter += 1

    # 7. Дилатация
    dot_big = np.zeros((49,49), dtype=np.uint8)
    dot_big[24, 24] = 255  
    dilated_big = dilation_binary(dot_big, kernel_size=3)
    fig = create_figure_pair(dot_big, dilated_big, "Одиночная точка", "Дилатация",
                             f" Дилатация", enlarge_pixels=True)
    demo_figures.append(fig)
    counter += 1

    # 8. Бинаризация
    grad_big = np.tile(np.linspace(30, 220, 49, dtype=np.uint8), (49,1))
    binary_big = threshold_binarize(grad_big, threshold=100)
    fig = create_figure_pair(grad_big, binary_big, "Градиент яркости", "Бинаризация (порог 100)",
                             f" Пороговая бинаризация", enlarge_pixels=True)
    demo_figures.append(fig)
    counter += 1

    # 9. Выравнивание гистограммы
    h, w = 49, 49
    low_contrast_big = np.full((h, w), 130, dtype=np.uint8)  
    low_contrast_big[20:30, 20:30] = 100                        
    equalized_big = histogram_equalization(low_contrast_big)

    # Ручное создание фигуры, чтобы избежать автонормировки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(low_contrast_big, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    ax1.set_title("Низкоконтрастное (фон 130, квадрат 100)")
    ax1.axis('off')
    ax2.imshow(equalized_big, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    ax2.set_title("После выравнивания")
    ax2.axis('off')
    fig.suptitle(f"{counter}. Выравнивание гистограммы")
    fig.tight_layout()

    demo_figures.append(fig)
    counter += 1

    # 10. Поворот 90°
    rot_big = np.arange(49*49, dtype=np.uint8).reshape(49,49) + 50
    rot90_big = rotate_90(rot_big, times=1)
    fig = create_figure_pair(rot_big, rot90_big, "Исходная 49×49", "Поворот 90°",
                             f" Поворот на 90°", enlarge_pixels=True)
    demo_figures.append(fig)
    counter += 1

    # 11. Поворот 180°
    rot180_big = rotate_90(rot_big, times=2)
    fig = create_figure_pair(rot_big, rot180_big, "Исходная 49×49", "Поворот 180°",
                             f" Поворот на 180°", enlarge_pixels=True)
    demo_figures.append(fig)

    if demo_figures:
        print(f"Закройте все для завершения.")
        plt.show()

    print("")