Бондарев Игорь, 8Е21.

Для 1 лабораторной работы по CV необходимо реализовать базовый минимум операций над изображениями
Входное изображение в формате RGB. Использовать методы OpenCV для реализации операций нельзя. Допустимы только методы cv2.imread() и cv2.imshow(). Все методы должны быть реализованы вручную.

1. Фильтры
<br>1.1 Медианный фильтр
<br>1.2 Фильтр гаусса

2. Морфологические операции
<br>2.1 Эрозия
<br>2.2 Дилатация

3. Прочие операции
<br>3.1 пороговая бинаризация (для rgb и grayscale изображения)
<br>3.2 выравнивание гистограммы
<br>3.3 поворот изображений на угол кратный 90 градусов

    
![jpg](README_files/123.jpg)


## Подготовка изображения и вспомогательные операции

Цветное изображение загружается через OpenCV. Для работы с яркостью реализован перевод в оттенки серого по фотометрической формуле.

### Перевод в оттенки серого

```python
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
```

![png](README_files/Figure_11.png)

###  Паддинг (дополнение краёв)

При свёрточных и морфологических операциях окно фильтра должно полностью находиться внутри изображения. Чтобы итоговый размер не уменьшался, края дополняются (padding). Реализованы два режима:

- edge – крайние пиксели копируются наружу (зеркальное продолжение границы);

- constant – добавляются нули (чёрные пиксели).

```python
def pad_image(image, pad_size, mode='edge'):
    if len(image.shape) == 2:
        return np.pad(image, pad_size, mode=mode)
    else:
        padded = np.zeros((image.shape[0]+2*pad_size, image.shape[1]+2*pad_size, image.shape[2]),
                          dtype=image.dtype)
        for ch in range(image.shape[2]):
            padded[:,:,ch] = np.pad(image[:,:,ch], pad_size, mode=mode)
        return padded
```

![png](README_files/Figure_22.png)
![png](README_files/Figure_33.png)

# 1. ФИЛЬТРЫ

### 1.1 Медианный фильтр

Медианный фильтр предназначен для подавления импульсных шумов. В отличие от линейного усреднения он не размывает границы, потому что медиана выбирает одно из реальных значений окрестности, а не вычисляет среднее.
<p>
Возьмём массив x = [2 80 6 3] и окно размером 3.
Чтобы не терять краевые элементы, сначала зеркально расширим массив: [2 2 80 6 3 3]

- окно (2 2 80), сортируем: (2 2 80), медиана = 2
- окно (2 80 6), сортируем: (2 6 80), медиана = 6
- окно (80 6 3), сортируем: (2 6 80), медиана = 6
- окно (6 3 3), сортируем: (3 3 6), медиана = 3

Выход: [2, 6, 6, 3]. Выброс 80 полностью убран, а перепад стал плавнее.

Для изображения мы действуем квадратным окном kernel_size × kernel_size (нечётным). Для каждого пикселя:

1. Извлекаем значения всех пикселей под окном.
2. Разворачиваем их в одномерный массив.
3. Находим медиану (np.median).
4. Присваиваем центральному пикселю.

```python
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
```

Цветное изображение обрабатывается поканально: та же процедура применяется к R, G и B по отдельности.

Результат на реальном фото (ядро 5×5):

![png](README_files/Figure_1.png)

Синтетический тест (ядро 3×3, матрица 49×49)

![png](README_files/Figure_44.png)


### 1.2 Фильтр гаусса

Фильтр Гаусса (размытие по Гауссу) – это низкочастотный фильтр, подавляющий мелкие детали и высокочастотный шум. В основе лежит двумерная функция Гаусса:

```text
G(x,y) = (1/(2πσ²)) * exp(-(x² + y²) / (2σ²))
```

Но для вычислений мы пользуемся свойством сепарабельности: двумерную свёртку можно заменить последовательной одномерной свёрткой по строкам и столбцам. Это значительно ускоряет вычисления.

#### Построение одномерного ядра

Для размера size (нечётного) и параметра σ генерируем ядро, веса которого убывают по кривой Гаусса, и нормализуем так, чтобы сумма весов была равна 1.


```python
def gaussian_kernel(size, sigma):
    kernel = np.zeros(size, dtype=np.float32)
    center = size // 2
    sum_val = 0.0
    for i in range(size):
        x = i - center
        kernel[i] = math.exp(-(x**2) / (2 * sigma**2))
        sum_val += kernel[i]
    return kernel / sum_val
```

Пример весов ядра размера 5 с σ=1.0:
[0.054, 0.244, 0.403, 0.244, 0.054]

Сначала свёртка по горизонтали (для каждой строки), затем по вертикали (для каждого столбца). На каждом шаге для пикселя вычисляется взвешенная сумма соседей с весами из ядра. Цветное изображение обрабатывается поканально.


```python
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
```


Результат на реальном фото (σ=1.5, ядро 5×5):

![png](README_files/Figure_2.png)
    
Синтетический тест:

![png](README_files/Figure_55.png)


# 2. Морфологические операции

### 2.1 Эрозия

Значение выходного пикселя является минимальным значением всех пикселей в окружении. В бинарном изображении пиксель установлен в 0 если какой-либо из соседних пикселей имеет значение 0. Эрозия «съедает» белые области. Центральный пиксель становится белым (255) только в том случае, если все пиксели под структурным элементом (окном) – белые.


```python
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
```

Мелкие белые точки и тонкие линии исчезают, объекты уменьшаются.

Пример на реальном фото (ядро 3×3):

![png](README_files/Figure_3.png)

Синтетический тест: белый квадрат 45×45 на чёрном фоне, ядро 3×3:

![png](README_files/Figure_66.png)

### 2.2 Дилатация

Значение выходного пикселя является максимальным значением всех пикселей в окружении. В бинарном изображении пиксель установлен в 1 если какой-либо из соседних пикселей имеет значение 1. Дилатация – операция, обратная эрозии. Она расширяет белые области. Центральный пиксель становится белым, если хотя бы один пиксель в окне белый.


```python
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
```

Пример на реальном фото:

![png](README_files/Figure_4.png)

Синтетический тест: одиночная белая точка в центре чёрного поля, ядро 3×3:

![png](README_files/Figure_77.png)


# 3. Прочие операции

### 3.1 Пороговая бинаризация

Бинаризация преобразует полутоновое (или цветное) изображение в строго двухцветное. Для каждого пикселя сравнивается его яркость с заданным порогом threshold. Если яркость больше порога – пиксель становится белым (255), иначе чёрным (0). Цветное изображение сначала переводится в серый, чтобы решение принималось только по яркости.

```python
def threshold_binarize(image, threshold):
    if len(image.shape) == 2:
        binary = np.zeros_like(image)
        binary[image > threshold] = 255
        return binary
    else:
        gray = rgb_to_grayscale(image)
        return threshold_binarize(gray, threshold)
```

Бинаризация цветного RGB (порог 127):

![png](README_files/Figure_4.png)

Синтетический пример – горизонтальный градиент яркости от 30 до 220, порог=100:

![png](README_files/Figure_88.png)


### 3.2 Выравнивание гистограммы

Выравнивание гистограммы предназначено для улучшения контраста. Если яркости сконцентрированы в узком диапазоне (например, снимок слишком тёмный), метод «растягивает» гистограмму на весь диапазон 0–255.

1. Строим гистограмму h(k) – количество пикселей с яркостью k.
2. Вычисляем кумулятивную функцию распределения (CDF) – накопленную сумму гистограммы.
3. Нормализуем CDF так, чтобы минимальное значение стало 0, а максимальное – 255.
4. Преобразуем яркость каждого пикселя: s = cdf_normalized[r], где r – исходная яркость.

```python
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
```

Пример на реальном фото:

![png](README_files/Figure_6.png)


### 3.3 Поворот на угол, кратный 90°

Поворот на 90°, 180°, 270° осуществляется без интерполяции, поскольку координаты пикселей просто переставляются. Согласно заданию, запрещено использовать готовые функции OpenCV, поэтому реализация, состоит из двух элементарных операций:

- Транспонирование – строки становятся столбцами;
- Отражение строк – каждая строка переворачивается справа налево.

Эти два действия и дают поворот на 90° по часовой стрелке. Поворот на 180° – это два таких поворота подряд, на 270° – три.

Ниже приведена полная ручная реализация функции rotate_90. Она работает как с одноканальными, так и с RGB изображениями.

```python
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
```

Примеры поворотов реального фото:
Поворот на 90°:
![png](README_files/Figure_7.png)

Поворот на 180°:
![png](README_files/Figure_8.png)



Лабораторная работа №2. Визуальная одометрия (навигация)
Цель: Разработать систему визуальной одометрии (навигации) по группе фотографий.
Ход работы: сделайте не менее 8 фото с переносом камеры или ноутбука по квадрату. Используя данные фотографии реализуйте следующее:

<p> 1.	Определите на каждой фотографии ключевые точки </p>
<p>2.	Отфильтруйте самые наилучшие применяю адаптивный радиус и локальные максимумы, не забудьте так же выровнять по яркости изображения.</p>
<p>3.	Постройте по каждой точке дескриптор (можете использовать любой, рекомендуется SIFT)</p>
<p>4.	Сопоставьте два соседних изображения на предмет соответствия ключевых точек. То есть определите пары одинаковых точек.</p>
<p>5.	Постройте модель преобразования изображений, учитывайте только поворот и сдвиг.</p>
<p>6.	С учетом полученных моделей постройте траекторию движения камеры.</p>

## Подготовка данных

Изображения загружаются, преобразуются в RGB, приводятся к одинаковой ширине 1080px. Затем переводятся в оттенки серого и сглаживаются фильтром Гаусса для подавления шума. Функции  взяты из первой лабораторной работы.

![png](README_files/Figure_21.png)

![png](README_files/Figure_222.png)

## 1. Детектирование ключевых точек

Идея алгоритма Харриса: угол – это область, где интенсивность сильно меняется в двух направлениях одновременно. В отличие от однородных участков (градиент ~0) и краёв (градиент только в одном направлении).

### Вычисление градиентов

Для каждого пикселя вычисляются частные производные по горизонтали и вертикали центральными разностями:

```python
def harris_keypoints(image, minor_size=5, k=0.04, threshold_ratio=0.01):
    if len(image.shape) == 3:
        image = intensity_grayscale(image)
    image = image.astype(np.float32)
    h, w = image.shape
    Ix = np.zeros_like(image)
    Iy = np.zeros_like(image)
    for r in range(1, h-1):
        for c in range(1, w-1):
            Ix[r,c] = (image[r,c+1] - image[r,c-1]) / 2.0
            Iy[r,c] = (image[r+1,c] - image[r-1,c]) / 2.0

    pad = minor_size // 2
    R = np.zeros_like(image)
    for r in range(pad, h-pad):
        for c in range(pad, w-pad):
            sum_Ix2 = 0.0
            sum_Iy2 = 0.0
            sum_Ixy = 0.0
            for i in range(-pad, pad+1):
                for j in range(-pad, pad+1):
                    gx = Ix[r+i, c+j]
                    gy = Iy[r+i, c+j]
                    sum_Ix2 += gx*gx
                    sum_Iy2 += gy*gy
                    sum_Ixy += gx*gy
            det = sum_Ix2*sum_Iy2 - sum_Ixy**2
            trace = sum_Ix2 + sum_Iy2
            R[r,c] = det - k*(trace**2)

    R_max = np.max(R)
    threshold = threshold_ratio * R_max
    keypoints = []
    for r in range(pad, h-pad):
        for c in range(pad, w-pad):
            if R[r,c] > threshold:
                local_max = True
                for i in range(-1,2):
                    for j in range(-1,2):
                        if R[r+i,c+j] > R[r,c]:
                            local_max = False
                if local_max:
                    keypoints.append((r,c))
    return keypoints, Ix, Iy
```

### Матрица вторых моментов и отклик угла

ля каждого пикселя в окне `minor_size × minor_size` суммируются $I_x^2$, $I_y^2$ и $I_x I_y$.

Затем вычисляется мера «угловатости»:

$$R = \det(M) - k \cdot (\text{trace}(M))^2$$

где матрица $M$ определяется как:

$$M = \begin{bmatrix} 
\sum I_x^2 & \sum I_x I_y \\ 
\sum I_x I_y & \sum I_y^2 
\end{bmatrix}$$

#### Параметры

- **minor_size** — размер окна для анализа
- **k** — эмпирический коэффициент 
- $I_x$ — градиент изображения по оси X
- $I_y$ — градиент изображения по оси Y
- $\det(M)$ — определитель матрицы $M$
- $\text{trace}(M)$ — след матрицы $M$ (сумма диагональных элементов)

```python
    pad = minor_size // 2
    R = np.zeros_like(image)
    for r in range(pad, h-pad):
        for c in range(pad, w-pad):
            sum_Ix2 = 0.0
            sum_Iy2 = 0.0
            sum_Ixy = 0.0
            for i in range(-pad, pad+1):
                for j in range(-pad, pad+1):
                    gx = Ix[r+i, c+j]
                    gy = Iy[r+i, c+j]
                    sum_Ix2 += gx*gx
                    sum_Iy2 += gy*gy
                    sum_Ixy += gx*gy
            det = sum_Ix2*sum_Iy2 - sum_Ixy**2
            trace = sum_Ix2 + sum_Iy2
            R[r,c] = det - k*(trace**2)
```

### Матрица вторых моментов и отклик угла

Оставляем точки, у которых R превышает порог (threshold_ratio * R_max), и которые являются локальными максимумами в окрестности 3×3.

```python
    R_max = np.max(R)
    threshold = threshold_ratio * R_max
    keypoints = []
    for r in range(pad, h-pad):
        for c in range(pad, w-pad):
            if R[r,c] > threshold:
                local_max = True
                for i in range(-1,2):
                    for j in range(-1,2):
                        if R[r+i,c+j] > R[r,c]:
                            local_max = False
                if local_max:
                    keypoints.append((r,c))
    return keypoints, Ix, Iy
```

## 2. Фильтрация ключевых точек

Некоторые точки могут быть вызваны шумом (одинокие выбросы). Фильтр удаляет точки, вокруг которых в радиусе radius находится менее min_neighbors других ключевых точек.

```python
def filter_isolated_points(keypoints, radius=10, min_neighbors=5):
    filtered = []
    for i, p in enumerate(keypoints):
        neighbors = 0
        for j, q in enumerate(keypoints):
            if i == j:
                continue
            if np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2) < radius:
                neighbors += 1
        if neighbors >= min_neighbors:
            filtered.append(p)
    return filtered
```

После фильтрации остаются только точки, принадлежащие заметным структурам.

![png](README_files/Figure1.png)
![png](README_files/Figure3.png) 
![png](README_files/Figure5.png)
![png](README_files/Figure7.png)


## 3. SIFT-подобный дескриптор

# 3.1 Вычисление доминирующей ориентации

Для точки берется окно 16×16 пикселей. В каждом пикселе рассчитывается магнитуда и угол градиента, затем строится гистограмма из 36 бинов (шаг 10°). Вклад пикселя взвешивается гауссовым окном (σ=8). Пик гистограммы задаёт доминирующий угол точки.

```python
def compute_keypoint_orientations(keypoints, Ix, Iy,
                                  orientation_window_size=16, num_bins=36):
    h, w = Ix.shape
    half = orientation_window_size // 2
    sigma = half
    oriented = []
    for (r,c) in keypoints:
        if r-half < 0 or r+half >= h or c-half < 0 or c+half >= w:
            continue
        hist = np.zeros(num_bins)
        bin_width = 360.0 / num_bins
        for i in range(-half, half+1):
            for j in range(-half, half+1):
                gx = Ix[r+i, c+j]
                gy = Iy[r+i, c+j]
                mag = math.sqrt(gx*gx + gy*gy)
                angle_deg = math.degrees(math.atan2(gy, gx)) % 360
                gauss_weight = math.exp(-(i*i + j*j) / (2*sigma*sigma))
                bin_idx = int(angle_deg / bin_width) % num_bins
                hist[bin_idx] += mag * gauss_weight
        peak_bin = np.argmax(hist)
        dominant_angle = math.radians((peak_bin + 0.5) * bin_width)
        oriented.append((r, c, dominant_angle))
    return oriented
```

# 3.2 Вычисление доминирующей ориентации

Область 16×16 вокруг точки разбивается на 4×4 блока. В каждом блоке строится гистограмма градиентов по 8 направлениям. Угол градиента пересчитывается относительно доминирующей ориентации точки, что даёт инвариантность к повороту. Полученный вектор из 4×4×8 = 128 элементов нормализуется, затем значения обрезаются сверху (порог 0.2) и нормализуются снова – это повышает устойчивость к изменению освещения.

```python
def compute_sift_descriptors(oriented_keypoints, Ix, Iy,
                             patch_size=16, num_spatial_bins=4,
                             num_orientation_bins=8):
    h, w = Ix.shape
    half = patch_size // 2
    cell_size = patch_size // num_spatial_bins
    bin_width = 360.0 / num_orientation_bins
    valid_kp = []
    descriptors = []
    for (r, c, dominant_angle) in oriented_keypoints:
        if r-half < 0 or r+half >= h or c-half < 0 or c+half >= w:
            continue
        histograms = []
        for bi in range(num_spatial_bins):
            row = []
            for bj in range(num_spatial_bins):
                row.append(np.zeros(num_orientation_bins))
            histograms.append(row)
        for i in range(-half, half+1):
            for j in range(-half, half+1):
                gx = Ix[r+i, c+j]
                gy = Iy[r+i, c+j]
                magnitude = math.sqrt(gx*gx + gy*gy)
                raw_angle = math.degrees(math.atan2(gy, gx))
                relative_angle = (raw_angle - math.degrees(dominant_angle)) % 360
                bi = min((i+half)//cell_size, num_spatial_bins-1)
                bj = min((j+half)//cell_size, num_spatial_bins-1)
                bin_idx = int(relative_angle / bin_width) % num_orientation_bins
                histograms[bi][bj][bin_idx] += magnitude
        desc = []
        for bi in range(num_spatial_bins):
            for bj in range(num_spatial_bins):
                for val in histograms[bi][bj]:
                    desc.append(val)
        desc = np.array(desc, dtype=float)
        norm = np.sqrt(np.sum(desc * desc))
        if norm > 1e-6:
            desc /= norm
        desc = np.clip(desc, 0, 0.2)
        norm2 = np.sqrt(np.sum(desc * desc))
        if norm2 > 1e-6:
            desc /= norm2
        valid_kp.append((r, c, dominant_angle))
        descriptors.append(desc)
    return valid_kp, np.array(descriptors)
```

В результате каждая точка характеризуется 128-мерным вектором.

## 4. Сопоставление точек между кадрами

Для каждой точки первого кадра ищется наилучшее и второе по близости совпадение во втором кадре по евклидову расстоянию между дескрипторами. Тест Лоу отсеивает неоднозначные соответствия: если best_dist / second_best_dist > 0.75, матч отбрасывается.

```python
def match_descriptors(kp_a, desc_a, kp_b, desc_b, ratio=0.75):
    matches = []
    for i in range(len(desc_a)):
        best_dist = float('inf')
        second_dist = float('inf')
        best_j = -1
        for j in range(len(desc_b)):
            dist = euclidean_distance(desc_a[i], desc_b[j])
            if dist < best_dist:
                second_dist = best_dist
                best_dist = dist
                best_j = j
            elif dist < second_dist:
                second_dist = dist
        if second_dist > 1e-6 and best_dist / second_dist < ratio:
            r_a, c_a = kp_a[i][0], kp_a[i][1]
            r_b, c_b = kp_b[best_j][0], kp_b[best_j][1]
            matches.append(((r_a, c_a), (r_b, c_b)))
    return matches
```

![png](README_files/Figure11.png)
![png](README_files/Figure33.png)
![png](README_files/Figure55.png)
![png](README_files/Figure77.png)
    
## 5. Вычисление модели преобразования

Мы предполагаем, что движение камеры — это жёсткое тело (rigid transformation):  поворот на угол `θ` и плоский сдвиг `(tx, ty)`. Модель считается по точечным соответствиям с помощью **RANSAC** для отсеивания выбросов.

```python
def estimate_rotation_translation(matches, ransac_iterations=500, inlier_threshold=5.0):
    if len(matches) < 2:
        print("Недостаточно матчей")
        return 0.0, 0.0, 0.0, []
    def fit_model(sample):
        n = len(sample)
        cax = sum(m[0][1] for m in sample) / n
        cay = sum(m[0][0] for m in sample) / n
        cbx = sum(m[1][1] for m in sample) / n
        cby = sum(m[1][0] for m in sample) / n
        dot = 0.0
        cross = 0.0
        for ((r_a,c_a),(r_b,c_b)) in sample:
            ax = c_a - cax; ay = r_a - cay
            bx = c_b - cbx; by = r_b - cby
            dot   += ax*bx + ay*by
            cross += ax*by - ay*bx
        angle = math.atan2(cross, dot)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        tx = cbx - (cos_a*cax - sin_a*cay)
        ty = cby - (sin_a*cax + cos_a*cay)
        return angle, tx, ty
    def count_inliers(all_matches, angle, tx, ty):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        inliers = []
        for ((r_a,c_a),(r_b,c_b)) in all_matches:
            xp = cos_a*c_a - sin_a*r_a + tx
            yp = sin_a*c_a + cos_a*r_a + ty
            if math.sqrt((xp-c_b)**2 + (yp-r_b)**2) < inlier_threshold:
                inliers.append(((r_a,c_a),(r_b,c_b)))
        return inliers
    best_angle, best_tx, best_ty = 0.0, 0.0, 0.0
    best_inliers = []
    for _ in range(ransac_iterations):
        sample = random.sample(matches, min(4, len(matches)))
        angle, tx, ty = fit_model(sample)
        inliers = count_inliers(matches, angle, tx, ty)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_angle, best_tx, best_ty = angle, tx, ty
    if len(best_inliers) >= 2:
        best_angle, best_tx, best_ty = fit_model(best_inliers)
    print(f"  Угол: {math.degrees(best_angle):.2f}°, сдвиг: tx={best_tx:.1f}, ty={best_ty:.1f}, inliers={len(best_inliers)}")
    return best_angle, best_tx, best_ty, best_inliers

```

Для каждой пары кадров выводится угол в градусах и сдвиг в пикселях.

Пример вывода для одной из пар:

```text
Трансформация 0 → 1:
  Угол: -3.25°, сдвиг: tx=150.5, ty=-6.0, inliers=24
```
    
## 6. Построение траектории
    
# 6.1 Накопление трансформаций

Последовательно применяем преобразования, начиная с первой позиции (0,0). Ошибки суммируются, и образуется дрейф.

```python
def build_trajectory(transforms):
    positions = [(0.0, 0.0)]
    angles = [0.0]
    cam_x, cam_y = 0.0, 0.0
    global_angle = 0.0
    for local_angle, tx, ty in transforms:
        global_angle += local_angle
        cam_x += -tx
        cam_y += ty
        positions.append((cam_x, cam_y))
        angles.append(global_angle)
    return positions, angles
```

![png](README_files/Figure_f1.png)

# 6.2 Накопление трансформаций

Второй метод не накапливает ошибки: на каждом кадре независимо вычисляется центр масс ключевых точек. Затем берётся разность с первым кадром.

```python
def build_trajectories_from_keypoints(all_keypoints):
    centroids = []
    for kps in all_keypoints:
        if len(kps) == 0:
            centroids.append(None)
            continue
        mean_c = sum(kp[1] for kp in kps) / len(kps)
        mean_r = sum(kp[0] for kp in kps) / len(kps)
        centroids.append((mean_c, mean_r))
    origin = next((c for c in centroids if c is not None), (0.0, 0.0))
    x0, y0 = origin
    obj_positions = []
    cam_positions = []
    for c in centroids:
        if c is None:
            obj_positions.append(None)
            cam_positions.append(None)
        else:
            dx = c[0] - x0
            dy = c[1] - y0
            obj_positions.append(( dx,  dy))
            cam_positions.append((-dx, -dy))
    return obj_positions, cam_positions, centroids
```

Так как центроид вычисляется заново для каждого кадра, накопление ошибок отсутствует.

![png](README_files/Figure_f4.png)






    


# Вывод
Задачи лабораторной работы выполнены в полном объеме

Костин Арсений, 8Е21, вариант 3.

# Лабораторная работа №3. Работа с видеопотоком

<p>Цель: Научиться анализировать видеопоток.
<p>Ход работы: получить видеопоток с Web-камеры и определить перемещающийся в кадре объект. Используя данные видеопотока реализуйте следующее:
<p> 1. Реализуйте получение данных с Web-камеры
<p> 2. Реализуйте алгоритм вычитания фона
<p> 3. Реализуйте определение движущегося предмета
<p> 4. Постройте траекторию движения объекта.
<p> 5. Проведите тестирование на тестовом видео.
<p> Проверка работоспособности: будет осуществляться на специальном видео, предоставленным преподавателем. Траектория движения, для которых недоступна.


```python
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image
%matplotlib inline
import math
import time
import lab1_functions as lb1
import lab2_functions as lb2
from collections import deque
import os
print(os.getcwd())
print(os.listdir())
```

    /Users/arseniikostin/cv-labs-sem8/labs
    ['sample_image2.png', 'sample_image3.png', 'gradient.png', 'lab3.py', 'histfunc.png', 'lab2.py', 'output.gif', 'sample_image4.jpg', 'sample_image5.jpg', 'lab1_functions.py', 'sequence1.jpeg', 'sequence6.jpeg', 'sequence7.jpeg', '__pycache__', 'doodles.ipynb', 'sequence8.jpeg', 'lab2.ipynb', 'sequence4.jpeg', 'harris1.png', 'sequence5.jpeg', 'gpt-stripfunctions.py', 'gaussfunc.png', 'lab1.py', 'lab3.ipynb', 'sequence2.jpeg', 'clean.ipynb', 'stitch.py', 'lab1.ipynb', 'lab3_v3.ipynb', 'clearoutput.py', 'sample_image.jpg', 'sequence3.jpeg', 'lab2_functions.py', 'combined.ipynb']


# 3.1 Получить видеопоток с веб-камеры

Запись разбита на два отдельных шага — фон и движение — чтобы можно было переснять каждый независимо.

### Шаг 1 — запись фона

Запускаем ячейку, убираем всё из кадра и ждём 3 секунды. Консоль тикает каждую секунду. После записи показываем первый и последний кадр — проверяем что в кадре пусто.


```python
BG_SECONDS  = 3
FPS_APPROX  = 10
N_BG_FRAMES = BG_SECONDS * FPS_APPROX

cap = cv2.VideoCapture(2)

print('ФОН')
print(f'Держите кадр ПУСТЫМ в течение {BG_SECONDS} секунд...')

bg_frames = []
for i in range(N_BG_FRAMES):
    ret, frame = cap.read()
    if ret:
        bg_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if i % FPS_APPROX == 0:
        print('...')
    cv2.waitKey(100)

cap.release()
print(f'Записано кадров фона: {len(bg_frames)}')

f, axarr = plt.subplots(1, 2, figsize=(12, 5))
axarr[0].imshow(bg_frames[0])
axarr[0].set_title('Фон начало')
axarr[1].imshow(bg_frames[-1])
axarr[1].set_title('Фон конец')
for ax in axarr: ax.axis('off')
plt.suptitle('в кадре не должно быть лишних объектов')
plt.show()
```

    === ЗАПИСЬ ФОНА ===
    Держите кадр ПУСТЫМ в течение 3 секунд...
    ...
    ...
    ...
    Записано кадров фона: 30



    
![png](combined345_files/combined345_4_1.png)
    


### Шаг 2 — запись движения

После старта идёт обратный отсчёт 3 ... 1, вносим объект и плавно двигаем его по кадру 5 секунд.


```python
COUNTDOWN_SEC  = 3
MOTION_SECONDS = 5
N_MOT_FRAMES   = MOTION_SECONDS * FPS_APPROX

cap = cv2.VideoCapture(2)

print('запись')
for s in range(COUNTDOWN_SEC, 0, -1):
    print(f'  Старт через {s}...')
    time.sleep(1)
print('двигаем')

motion_frames = []
for i in range(N_MOT_FRAMES):
    ret, frame = cap.read()
    if ret:
        motion_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if i % FPS_APPROX == 0:
        print(f'...')
    cv2.waitKey(100)

cap.release()
frames_sequence = bg_frames + motion_frames
print(f'Записано кадров движения: {len(motion_frames)}')

step = max(1, len(motion_frames) // 4)
f, axarr = plt.subplots(1, 4, figsize=(18, 4))
for i, ax in enumerate(axarr):
    idx = min(i * step, len(motion_frames) - 1)
    ax.imshow(motion_frames[idx])
    ax.set_title(f'Кадр движения #{idx}')
    ax.axis('off')
plt.suptitle('объект должен быть виден и двигаться')
plt.show()
```

    запись
      Старт через 3...
      Старт через 2...
      Старт через 1...
    двигаем
    ...
    ...
    ...
    ...
    ...
    Записано кадров движения: 50



    
![png](combined345_files/combined345_6_1.png)
    


# 3.2 Инициализировать вычитатель фона

Берём все кадры фона и считаем попиксельное среднее — это и есть наша модель фона. Логика та же, что и при любом усреднении: случайные отклонения из-за шума камеры компенсируют друг друга, остаётся стабильная картина пустой сцены.


```python
def build_background(frames):
    bg = np.zeros_like(frames[0], dtype=float)
    for f in frames:
        bg += f.astype(float)
    bg /= len(frames)
    return bg

background = build_background(bg_frames)

plt.figure(figsize=(6, 4))
plt.imshow(background.astype(np.uint8))
plt.title('Модель фона')
plt.axis('off')
plt.show()
```


    
![png](combined345_files/combined345_8_0.png)
    


# 3.3 Применить вычитание фона к кадру

Для каждого кадра считаем абсолютную разность с фоном по каждому каналу RGB. Потом усредняем разность по трём каналам — получаем одноканальное изображение, где яркость пикселя = насколько сильно он отличается от фона.


```python
THRESHOLD = 10

def subtract_background(frame, bg, threshold):
    diff      = np.abs(frame.astype(float) - bg)
    diff_gray = (diff[:, :, 0] + diff[:, :, 1] + diff[:, :, 2]) / 3.0
    mask      = np.zeros(diff_gray.shape, dtype=np.uint8)
    mask[diff_gray > threshold] = 255
    return diff_gray, mask
```

# 3.4 Получить маску переднего плана (foreground mask)

Применяем вычитание к тестовому кадру. Всё что ярче порога — бинаризуется в белый (255), остальное — чёрный (0). Белые пиксели — кандидаты в «движущийся объект».


```python
frame_test = motion_frames[len(motion_frames) // 2]

diff_gray_test, mask_test = subtract_background(frame_test, background, THRESHOLD)

f, axarr = plt.subplots(1, 3, figsize=(15, 5))
axarr[0].imshow(frame_test)
axarr[0].set_title('Кадр с движением')
axarr[1].imshow(diff_gray_test, cmap='gray')
axarr[1].set_title('Разность |кадр - фон|')
axarr[2].imshow(mask_test, cmap='gray')
axarr[2].set_title(f'Бинарная маска (порог = {THRESHOLD})')
for ax in axarr: ax.axis('off')
plt.show()
```


    
![png](combined345_files/combined345_12_0.png)
    


# 3.5 Очистить маску (морфология, шумоподавление)

Сырая маска шумная — на ней много мелких белых пятен от перепадов освещения и шума камеры. Применяем морфологическое открытие: сначала эрозия уничтожает мелкие пятна, потом дилатация возвращает размер оставшимся (настоящим) объектам.

В лабе 1 эрозия и дилатация были реализованы через тройные питоновские циклы — на одном изображении это нормально, но на 50 кадрах видео работало бы более 5 минут. Поэтому здесь делаем то же самое, но через numpy-операции: скользящее минимальное/максимальное по окну через `np.lib.stride_tricks`. Результат идентичный — только быстро.


```python
def fast_erode(mask_bin, size=5):
    pad = size // 2
    padded = np.pad(mask_bin, pad, mode='constant', constant_values=1)
    h, w = mask_bin.shape

    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return windows.min(axis=(-2, -1)).astype(np.uint8)

def fast_dilate(mask_bin, size=5):
    pad = size // 2
    padded = np.pad(mask_bin, pad, mode='constant', constant_values=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return windows.max(axis=(-2, -1)).astype(np.uint8)

def clean_mask(mask_uint8, morph_size=5):
    mask_bin = (mask_uint8 // 255).astype(np.uint8)
    eroded   = fast_erode(mask_bin,  morph_size)
    dilated  = fast_dilate(eroded,   morph_size)
    return (dilated * 255).astype(np.uint8)

mask_cleaned = clean_mask(mask_test, morph_size=5)

f, axarr = plt.subplots(1, 2, figsize=(10, 5))
axarr[0].imshow(mask_test,    cmap='gray')
axarr[0].set_title('Маска до очистки')
axarr[1].imshow(mask_cleaned, cmap='gray')
axarr[1].set_title('Маска после эрозии + дилатации')
for ax in axarr: ax.axis('off')
plt.show()
```


    
![png](combined345_files/combined345_14_0.png)
    


# 3.6 Найти контуры движущихся объектов

Контурный пиксель — белый пиксель маски, у которого хотя бы один из четырёх соседей чёрный. То есть он стоит на границе объекта. Проходим по всей маске и собираем такие пиксели в список.


```python
def find_contour_pixels(mask_uint8):
    m    = mask_uint8
    inner = m[1:-1, 1:-1]
    is_white    = inner == 255
    has_black_neighbor = (
        (m[0:-2, 1:-1] == 0) |
        (m[2:,   1:-1] == 0) |
        (m[1:-1, 0:-2] == 0) |
        (m[1:-1, 2:]   == 0)
    )
    contour_map = is_white & has_black_neighbor
    rows, cols  = np.where(contour_map)
    return list(zip(rows + 1, cols + 1))

contour_pixels = find_contour_pixels(mask_cleaned)

frame_with_contour = frame_test.copy()
for (r, c) in contour_pixels:
    frame_with_contour[r, c] = [255, 0, 0]

plt.figure(figsize=(7, 5))
plt.imshow(frame_with_contour)
plt.title(f'Контур объекта (найдено {len(contour_pixels)} пикселей)')
plt.axis('off')
plt.show()
```


    
![png](combined345_files/combined345_16_0.png)
    


# 3.7 Определить и отфильтровать объекты по размеру

На маске может быть несколько белых областей — часть из них шум, который не убрала морфология. Чтобы найти отдельные объекты, используем обход в ширину (BFS): стартуем из непосещённого белого пикселя, обходим все связные с ним — это один объект. Повторяем для всех оставшихся. Компоненты с площадью меньше `min_area` отбрасываем как шум.


```python
def connected_components(mask_uint8, min_area=300):
    visited    = np.zeros_like(mask_uint8, dtype=bool)
    h, w       = mask_uint8.shape
    components = []

    for r in range(h):
        for c in range(w):
            if mask_uint8[r, c] == 255 and not visited[r, c]:
                queue     = deque([(r, c)])
                visited[r, c] = True
                component = []
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if mask_uint8[nr, nc] == 255 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                if len(component) >= min_area:
                    components.append(component)

    return components

components = connected_components(mask_cleaned, min_area=300)
print(f'Найдено объектов после фильтрации по площади: {len(components)}')
for i, comp in enumerate(components):
    print(f'  Объект {i}: {len(comp)} пикселей')
```

    Найдено объектов после фильтрации по площади: 4
      Объект 0: 39430 пикселей
      Объект 1: 2696 пикселей
      Объект 2: 384 пикселей
      Объект 3: 1549 пикселей


# 3.8 Вычислить центроид и bounding box объекта

Центроид — «центр масс» компоненты, то есть среднее по строкам и столбцам всех её пикселей. Bounding box — минимальный охватывающий прямоугольник, находим через min/max строк и столбцов.


```python
def get_centroid_and_bbox(component):
    px = np.array(component)
    centroid_r = px[:, 0].mean()
    centroid_c = px[:, 1].mean()
    r0, c0 = px[:, 0].min(), px[:, 1].min()
    r1, c1 = px[:, 0].max(), px[:, 1].max()
    return (centroid_c, centroid_r), (r0, c0, r1, c1)

vis_frame = frame_test.copy()
for comp in components:
    (cx, cy), (r0, c0, r1, c1) = get_centroid_and_bbox(comp)
    vis_frame[r0, c0:c1] = [255, 0, 0]
    vis_frame[r1, c0:c1] = [255, 0, 0]
    vis_frame[r0:r1, c0] = [255, 0, 0]
    vis_frame[r0:r1, c1] = [255, 0, 0]
    cr, cc = int(cy), int(cx)
    for d in range(-6, 7):
        if 0 <= cr+d < vis_frame.shape[0]: vis_frame[cr+d, cc] = [0, 255, 0]
        if 0 <= cc+d < vis_frame.shape[1]: vis_frame[cr, cc+d] = [0, 255, 0]

plt.figure(figsize=(8, 6))
plt.imshow(vis_frame)
plt.title('Bounding box (красный) и центроид (зелёный)')
plt.axis('off')
plt.show()
```


    
![png](combined345_files/combined345_20_0.png)
    


# 3.9 Накопить координаты центроида и построить траекторию

Запускаем полный конвейер по всем кадрам движения. На каждом кадре: вычитание фона → очистка маски → поиск компонент → берём самую большую → запоминаем центроид. Если объект не найден — пишем `None`.

Траекторию строим через `lb2.draw_trajectory_generic` из второй лабы.


```python
MIN_AREA   = 300
trajectory = []

for idx, frame in enumerate(motion_frames):
    _, mask = subtract_background(frame, background, THRESHOLD)
    mask_cl = clean_mask(mask, morph_size=5)
    comps   = connected_components(mask_cl, min_area=MIN_AREA)

    if comps:
        largest     = max(comps, key=lambda c: len(c))
        (cx, cy), _ = get_centroid_and_bbox(largest)
        trajectory.append((cx, cy))
    else:
        trajectory.append(None)

detected = sum(1 for t in trajectory if t is not None)
print(f'Кадров движения обработано: {len(motion_frames)}')
print(f'Кадров с обнаруженным объектом: {detected}')

if detected == 0:
    print('Объект не найден')

valid_traj  = [(i, p) for i, p in enumerate(trajectory) if p is not None]
pos_list    = [p for p in trajectory if p is not None]
labels_list = [str(i) for i, _ in valid_traj]

if pos_list:
    lb2.draw_trajectory_generic(
        pos_list,
        image_labels=labels_list,
        title='Траектория движущегося объекта',
        color='darkorange'
    )
```

    Кадров движения обработано: 50
    Кадров с обнаруженным объектом: 50



    
![png](combined345_files/combined345_22_1.png)
    


# 3.10 Отобразить результаты (оригинал + маска + траектория + bounding box)

Финальная сводная визуализация. Берём первый кадр, где объект обнаружен, и показываем четыре этапа рядом. На последнем — рисуем накопленную траекторию линиями через алгоритм Брезенхема (рисует прямую попиксельно, без `cv2.line`).


```python
def bresenham_line(img, x0, y0, x1, y1, color):
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if 0 <= y0 < img.shape[0] and 0 <= x0 < img.shape[1]:
            img[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 <  dx: err += dx; y0 += sy

best_idx   = next((i for i, t in enumerate(trajectory) if t is not None), 0)
best_frame = motion_frames[best_idx]

_, mask_best = subtract_background(best_frame, background, THRESHOLD)
mask_best_cl = clean_mask(mask_best, morph_size=5)
comps_best   = connected_components(mask_best_cl, min_area=MIN_AREA)

frame_result = best_frame.copy()
for comp in comps_best:
    (cx, cy), (r0, c0, r1, c1) = get_centroid_and_bbox(comp)
    frame_result[r0, c0:c1] = [255, 0, 0]
    frame_result[r1, c0:c1] = [255, 0, 0]
    frame_result[r0:r1, c0] = [255, 0, 0]
    frame_result[r0:r1, c1] = [255, 0, 0]
    cr, cc = int(cy), int(cx)
    for d in range(-6, 7):
        if 0 <= cr+d < frame_result.shape[0]: frame_result[cr+d, cc] = [0, 255, 0]
        if 0 <= cc+d < frame_result.shape[1]: frame_result[cr, cc+d] = [0, 255, 0]

prev_pt = None
for pt in trajectory:
    if pt is None:
        prev_pt = None
        continue
    px, py = int(pt[0]), int(pt[1])
    if prev_pt is not None:
        bresenham_line(frame_result, int(prev_pt[0]), int(prev_pt[1]), px, py, [255, 220, 0])
    if 0 <= py < frame_result.shape[0] and 0 <= px < frame_result.shape[1]:
        frame_result[py, px] = [255, 255, 0]
    prev_pt = pt

f, axarr = plt.subplots(1, 4, figsize=(22, 5))
axarr[0].imshow(best_frame)
axarr[0].set_title(f'Оригинал (кадр #{best_idx})')
axarr[1].imshow(mask_best, cmap='gray')
axarr[1].set_title('Маска (до очистки)')
axarr[2].imshow(mask_best_cl, cmap='gray')
axarr[2].set_title('Маска (после морфологии)')
axarr[3].imshow(frame_result)
axarr[3].set_title('Bbox + траектория')
for ax in axarr: ax.axis('off')
plt.tight_layout()
plt.show()
```


    
![png](combined345_files/combined345_24_0.png)
    


# Вывод

В ходе лабораторной работы реализован полный конвейер обнаружения и трекинга движущегося объекта без использования готовых алгоритмов OpenCV.

Модель фона строится как попиксельное среднее по кадрам пустой сцены. Вычитание фона выполняется через абсолютную разность с последующей бинаризацией по порогу. Морфологическое открытие (эрозия + дилатация) реализовано через скользящие numpy-окна (`stride_tricks`) — это сохраняет логику операций из лабы 1, но работает на всём видео за секунды вместо минут. Поиск объектов выполнен BFS по связным компонентам с фильтрацией по площади. Центроид и bounding box вычисляются аналитически через min/max/mean по координатам пикселей. Траектория строится через `lb2.draw_trajectory_generic`, отрисовка линий — алгоритмом Брезенхема.

Костин Арсений, 8Е21, вариант 3.

# Лабораторная работа №4. Разработка алгоритма определения лиц.

<p>Цель: на практике закрепить полученные в ходе курса знания, в том числе по машинному обучению и нейронным сетям для решения задачи детектирования лиц и классификации лиц на мужчин и женщин.
<p>Ход работы: в ходе первой и второй лабораторной каждый из студентов собрал свои фотографии. Данную выборку можно использовать в качестве обучающей выборки для синтеза алгоритмов. Разметку данных каждый студент проводит сам. Алгоритм детектирования и классификации может быть любым.

<p><b>Выбранный метод: HOG + Linear SVM</b>

<p>HOG (Histogram of Oriented Gradients) описывает форму объекта через распределение направлений градиентов — ровно тот же принцип, что использовался в детекторе Харриса в лабе 2, только там мы считали $I_x$, $I_y$ для поиска углов, а здесь строим из них гистограммы для описания внешнего вида патча.

<p>SVM (Support Vector Machine) ищет гиперплоскость, максимально разделяющую два класса в пространстве HOG-признаков.


```python
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image
%matplotlib inline
import math
import os
import pickle

import lab1_functions as lb1
import lab2_functions as lb2
import lab3_functions as lb3
from collections import deque

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import fetch_lfw_people

# Настройка графиков по ГОСТ: шрифт с засечками, 14pt
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size':        14,
    'axes.titlesize':   14,
    'axes.labelsize':   14,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  12,
    'figure.dpi':       100,
})

print(os.getcwd())
print(os.listdir())
```

    /Users/arseniikostin/cv-labs-sem8/labs
    ['sample_image2.png', 'sample_image3.png', 'gradient.png', 'histfunc.png', 'lab2.py', 'output.gif', 'sample_image4.jpg', 'sample_image5.jpg', 'lab1_functions.py', 'sequence1.jpeg', 'sequence6.jpeg', 'sequence7.jpeg', '__pycache__', 'detector_model.pkl', 'doodles.ipynb', 'sequence8.jpeg', 'lb5cv.png', 'lab2.ipynb', 'sequence4.jpeg', 'gender_model.pkl', 'harris1.png', 'sequence5.jpeg', 'lab4test.py', 'lab4.ipynb', 'gpt-stripfunctions.py', 'live_camera.py', 'gaussfunc.png', 'lab1.py', 'lab4_styled.ipynb', 'lab3.ipynb', 'sequence2.jpeg', 'signals.csv', 'clean.ipynb', 'lab3_functions.py', 'stitch.py', 'lab4_functions.py', 'lab1.ipynb', 'lab5.ipynb', 'lab5_styled.ipynb', 'clearoutput.py', 'sample_image.jpg', 'sequence3.jpeg', 'lab2_functions.py', 'combined.ipynb']


# 4.1 Загрузка датасета

Используем LFW (Labeled Faces in the Wild) — открытый датасет из 13 000+ фотографий лиц публичных людей. Параметр `min_faces_per_person=20` оставляет только тех, кого достаточно много — так модель лучше обобщается и не переобучается на единичные примеры.

Разметку по полу делаем вручную через словарь `GENDER_LABELS` — это и есть ручная разметка данных согласно условию лабы.


```python
print('Загружаем датасет LFW...')
lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.5, color=True)

print(f'Изображений: {lfw.images.shape[0]}')
print(f'Размер патча: {lfw.images.shape[1]}x{lfw.images.shape[2]}')
print(f'Персон: {len(lfw.target_names)}')
print('Имена:', lfw.target_names)
```

    Загружаем датасет LFW...
    Изображений: 3023
    Размер патча: 62x47
    Персон: 62
    Имена: ['Alejandro Toledo' 'Alvaro Uribe' 'Amelie Mauresmo' 'Andre Agassi'
     'Angelina Jolie' 'Ariel Sharon' 'Arnold Schwarzenegger'
     'Atal Bihari Vajpayee' 'Bill Clinton' 'Carlos Menem' 'Colin Powell'
     'David Beckham' 'Donald Rumsfeld' 'George Robertson' 'George W Bush'
     'Gerhard Schroeder' 'Gloria Macapagal Arroyo' 'Gray Davis'
     'Guillermo Coria' 'Hamid Karzai' 'Hans Blix' 'Hugo Chavez' 'Igor Ivanov'
     'Jack Straw' 'Jacques Chirac' 'Jean Chretien' 'Jennifer Aniston'
     'Jennifer Capriati' 'Jennifer Lopez' 'Jeremy Greenstock' 'Jiang Zemin'
     'John Ashcroft' 'John Negroponte' 'Jose Maria Aznar'
     'Juan Carlos Ferrero' 'Junichiro Koizumi' 'Kofi Annan' 'Laura Bush'
     'Lindsay Davenport' 'Lleyton Hewitt' 'Luiz Inacio Lula da Silva'
     'Mahmoud Abbas' 'Megawati Sukarnoputri' 'Michael Bloomberg' 'Naomi Watts'
     'Nestor Kirchner' 'Paul Bremer' 'Pete Sampras' 'Recep Tayyip Erdogan'
     'Ricardo Lagos' 'Roh Moo-hyun' 'Rudolph Giuliani' 'Saddam Hussein'
     'Serena Williams' 'Silvio Berlusconi' 'Tiger Woods' 'Tom Daschle'
     'Tom Ridge' 'Tony Blair' 'Vicente Fox' 'Vladimir Putin' 'Winona Ryder']


Присваиваем метки пола вручную. 0 — мужчина, 1 — женщина. Это разметка данных.


```python
GENDER_LABELS = {
    'Ariel Sharon': 0, 'Colin Powell': 0, 'Donald Rumsfeld': 0,
    'George W Bush': 0, 'Gerhard Schroeder': 0, 'Hugo Chavez': 0,
    'Tony Blair': 0, 'Junichiro Koizumi': 0, 'Jean Chretien': 0,
    'John Ashcroft': 0, 'Vladmir Putin': 0, 'Hamid Karzai': 0,
    'Luiz Inacio Lula da Silva': 0, 'Jacques Chirac': 0, 'Jiang Zemin': 0,
    'Vicente Fox': 0, 'Silvio Berlusconi': 0, 'Alejandro Toledo': 0,
    'John Snow': 0, 'Arnold Schwarzenegger': 0,
    'Lleyton Hewitt': 0, 'Andre Agassi': 0, 'Tiger Woods': 0,
    'Jennifer Aniston': 1, 'Halle Berry': 1, 'Laura Bush': 1,
    'Serena Williams': 1, 'Winona Ryder': 1,
    'Gloria Macapagal Arroyo': 1, 'Condoleezza Rice': 1,
}

gender_labels = []
valid_indices = []

for i, target_id in enumerate(lfw.target):
    name = lfw.target_names[target_id]
    if name in GENDER_LABELS:
        gender_labels.append(GENDER_LABELS[name])
        valid_indices.append(i)

images_valid  = lfw.images[valid_indices]
gender_labels = np.array(gender_labels)

print(f'Изображений с разметкой пола: {len(images_valid)}')
print(f'Мужчин: {(gender_labels == 0).sum()}, Женщин: {(gender_labels == 1).sum()}')

f, axes = plt.subplots(2, 8, figsize=(16, 5))
for row, gender in enumerate([0, 1]):
    idxs = np.where(gender_labels == gender)[0][:8]
    for col, idx in enumerate(idxs):
        axes[row, col].imshow(images_valid[idx])
        name = lfw.target_names[lfw.target[valid_indices[idx]]]
        axes[row, col].set_title(name.split()[-1], fontsize=7)
        axes[row, col].axis('off')
axes[0, 0].set_ylabel('Мужчины', fontsize=10)
axes[1, 0].set_ylabel('Женщины', fontsize=10)
plt.suptitle('Примеры из датасета LFW с ручной разметкой пола')
plt.tight_layout()
plt.show()
```

    Изображений с разметкой пола: 2026
    Мужчин: 1844, Женщин: 182



    
![png](combined345_files/combined345_33_1.png)
    


# 4.2 HOG-дескриптор

HOG — Histogram of Oriented Gradients. Алгоритм пошагово:

**Шаг 1 — градиенты.** Для каждого пикселя считаем $I_x$ и $I_y$ через центральные разности — ровно так же, как в лабе 2 при вычислении детектора Харриса. Из градиентов получаем магнитуду и угол:

$$M = \sqrt{I_x^2 + I_y^2}, \quad \theta = \arctan\!\left(\frac{I_y}{I_x}\right) \bmod 180°$$

Угол берём по модулю 180° (неориентированный) — нам не важно смотрит ли граница вверх или вниз.

**Шаг 2 — ячейки.** Делим изображение на клетки 8×8 пикселей. В каждой ячейке строим гистограмму из 9 бинов по углам (0–180°), взвешенную по магнитуде.

**Шаг 3 — блоки.** Объединяем соседние 2×2 ячейки в блок и нормируем его вектор на L2-норму. Нормировка делает дескриптор устойчивым к изменению освещения.

**Результат** — конкатенация всех нормированных блоков.

### Шаг 1: градиенты

Перевод в grayscale через взвешенную сумму каналов, потом центральные разности.


```python
def to_gray(image_float):
    if len(image_float.shape) == 3:
        return (0.299*image_float[:,:,0] + 0.587*image_float[:,:,1] + 0.114*image_float[:,:,2]) * 255.0
    return image_float * 255.0

def compute_gradients(gray):
    Ix = np.zeros_like(gray)
    Iy = np.zeros_like(gray)
    Ix[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2.0
    Iy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2.0
    magnitude = np.sqrt(Ix**2 + Iy**2)
    angle     = np.degrees(np.arctan2(Iy, Ix)) % 180.0
    return Ix, Iy, magnitude, angle

sample = images_valid[0]
gray   = to_gray(sample)
Ix, Iy, magnitude, angle = compute_gradients(gray)

f, axarr = plt.subplots(1, 4, figsize=(16, 4))
axarr[0].imshow(sample)
axarr[0].set_title('Исходное фото')
axarr[1].imshow(gray, cmap='gray')
axarr[1].set_title('Grayscale')
axarr[2].imshow(magnitude, cmap='hot')
axarr[2].set_title('Магнитуда градиента')
axarr[3].imshow(angle, cmap='hsv')
axarr[3].set_title('Угол градиента, °')
for ax in axarr: ax.axis('off')
plt.tight_layout()
plt.show()
```


    
![png](combined345_files/combined345_36_0.png)
    


### Шаг 2: гистограммы по ячейкам

Сетка ячеек 8×8 пикселей. В каждой — 9-бинная гистограмма, где вес пикселя равен его магнитуде. Посмотрим на гистограммы нескольких ячеек — у ячеек с выраженными краями (брови, контур лица) будут пики в конкретных бинах.


```python
CELL_SIZE = 8
NUM_BINS  = 9
BIN_WIDTH = 180.0 / NUM_BINS

def build_cell_histograms(magnitude, angle, cell_size=8, num_bins=9):
    h, w  = magnitude.shape
    n_cy  = h // cell_size
    n_cx  = w // cell_size
    hists = np.zeros((n_cy, n_cx, num_bins))
    for cy in range(n_cy):
        for cx in range(n_cx):
            r0, r1 = cy*cell_size, (cy+1)*cell_size
            c0, c1 = cx*cell_size, (cx+1)*cell_size
            hist, _ = np.histogram(angle[r0:r1, c0:c1], bins=num_bins,
                                   range=(0, 180), weights=magnitude[r0:r1, c0:c1])
            hists[cy, cx] = hist
    return hists

cell_hists = build_cell_histograms(magnitude, angle, CELL_SIZE, NUM_BINS)
print(f'Сетка ячеек: {cell_hists.shape[0]}×{cell_hists.shape[1]}, бинов на ячейку: {NUM_BINS}')

bins_x = np.arange(NUM_BINS) * BIN_WIDTH
f, axes = plt.subplots(3, 4, figsize=(14, 8))
for i, ax in enumerate(axes.flat):
    cy = (i // 4) + 2
    cx = (i %  4) + 1
    ax.bar(bins_x, cell_hists[cy, cx], width=BIN_WIDTH*0.9, color='steelblue')
    ax.set_title(f'Ячейка [{cy},{cx}]', fontsize=9)
    ax.set_xlabel('Угол, °', fontsize=8)
    ax.set_xticks(bins_x)
    ax.set_xticklabels([f'{int(b)}' for b in bins_x], fontsize=7)
plt.suptitle('Гистограммы ориентированных градиентов по ячейкам')
plt.tight_layout()
plt.show()
```

    Сетка ячеек: 7×5, бинов на ячейку: 9



    
![png](combined345_files/combined345_38_1.png)
    


### Шаг 3: блочная нормировка и финальный дескриптор

Объединяем соседние 2×2 ячейки в блок, нормируем вектор блока:

$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\sqrt{\|\mathbf{v}\|^2 + \varepsilon}}$$

Маленький $\varepsilon = 10^{-6}$ защищает от деления на ноль в пустых ячейках. После нормировки все блоки конкатенируем в один вектор — это и есть HOG-дескриптор изображения.


```python
BLOCK_SIZE = 2

def normalize_blocks(cell_hists, block_size=2):
    n_cy, n_cx, num_bins = cell_hists.shape
    descriptor = []
    for by in range(n_cy - block_size + 1):
        for bx in range(n_cx - block_size + 1):
            block = cell_hists[by:by+block_size, bx:bx+block_size].flatten()
            norm  = np.sqrt(np.sum(block**2) + 1e-6)
            descriptor.append(block / norm)
    return np.concatenate(descriptor)

def hog_descriptor(image_float, cell_size=8, num_bins=9, block_size=2):
    gray  = to_gray(image_float)
    _, _, magnitude, angle = compute_gradients(gray)
    hists = build_cell_histograms(magnitude, angle, cell_size, num_bins)
    return normalize_blocks(hists, block_size)

desc_test = hog_descriptor(sample)
print(f'Размер HOG-дескриптора для патча {sample.shape[0]}×{sample.shape[1]}: {len(desc_test)}')
```

    Размер HOG-дескриптора для патча 62×47: 864


# 4.3 Извлечение признаков

Вычисляем HOG для каждого изображения датасета. Каждое изображение → вектор признаков. Все векторы складываем в матрицу `X_faces` (изображений × признаков).


```python
print('Вычисляем HOG для лиц...')
X_faces = []
for i, img in enumerate(images_valid):
    X_faces.append(hog_descriptor(img))
    if (i+1) % 200 == 0:
        print(f'  {i+1}/{len(images_valid)}')

X_faces  = np.array(X_faces)
y_gender = gender_labels.copy()

print(f'Матрица признаков: {X_faces.shape}')
```

    Вычисляем HOG для лиц...
      200/2026
      400/2026
      600/2026
      800/2026
      1000/2026
      1200/2026
      1400/2026
      1600/2026
      1800/2026
      2000/2026
    Матрица признаков: (2026, 864)


## Негативные примеры для детектора лицо/не-лицо

Детектору нужно видеть оба класса — и лица, и не-лица. Берём те же фотографии LFW и создаём из них патчи, которые заведомо не являются лицом:

- **вертикальный переворот** — нос смотрит вверх, лоб внизу, структура нарушена
- **сдвиг вниз + шум сверху** — лоб обрезан, вместо него случайный шум
- **поворот на 90°** — ориентация полностью нарушена
- **поворот на 180°** — перевёрнутое лицо, иная градиентная структура
- **угловой кроп** — угол кадра LFW, там фон или плечи
- **случайный шум** — никакой структуры вообще

Каждый тип берёт разные исходные изображения, поэтому в train и test попадают разные патчи и SVM не может их просто запомнить.


```python
img_h, img_w = lfw.images.shape[1], lfw.images.shape[2]
N_NEG    = len(images_valid)
all_imgs = lfw.images
np.random.seed(42)

def make_negatives(all_imgs, img_h, img_w, n_total):
    n_imgs   = len(all_imgs)
    per_type = n_total // 6 + 1
    negatives = []
    idx = np.random.permutation(n_imgs)

    for i in range(per_type):
        patch = all_imgs[idx[i % n_imgs]].copy()
        negatives.append(patch[::-1, :, :])

    for i in range(per_type):
        patch   = all_imgs[idx[(i + per_type) % n_imgs]].copy()
        shift   = max(1, img_h // 3)
        shifted = np.zeros_like(patch)
        shifted[shift:, :, :]  = patch[:img_h - shift, :, :]
        shifted[:shift, :, :]  = np.random.rand(shift, img_w, 3).astype(np.float32)
        negatives.append(shifted)

    for i in range(per_type):
        patch   = all_imgs[idx[(i + 2*per_type) % n_imgs]].copy()
        rotated = np.transpose(patch, (1, 0, 2))
        if rotated.shape[0] < img_h or rotated.shape[1] < img_w:
            rotated = np.pad(rotated, ((0, max(0, img_h-rotated.shape[0])),
                                       (0, max(0, img_w-rotated.shape[1])),
                                       (0, 0)), mode='edge')
        negatives.append(rotated[:img_h, :img_w, :])

    for i in range(per_type):
        patch = all_imgs[idx[(i + 3*per_type) % n_imgs]].copy()
        negatives.append(patch[::-1, ::-1, :])

    for i in range(per_type):
        src = all_imgs[idx[(i + 4*per_type) % n_imgs]]
        big = np.kron(src, np.ones((2, 2, 1)))
        h_b, w_b = big.shape[:2]
        negatives.append(np.clip(big[h_b-img_h:, w_b-img_w:, :], 0, 1).astype(np.float32))

    for i in range(per_type):
        negatives.append(np.random.rand(img_h, img_w, 3).astype(np.float32))

    return negatives[:n_total]

print('Генерируем негативные примеры (6 типов трансформаций)...')
neg_patches = make_negatives(all_imgs, img_h, img_w, N_NEG)

print('Вычисляем HOG для негативов...')
X_neg = []
for i, patch in enumerate(neg_patches):
    X_neg.append(hog_descriptor(patch))
    if (i+1) % 500 == 0:
        print(f'  {i+1}/{N_NEG}')

X_neg = np.array(X_neg)
y_neg = np.full(N_NEG, -1)

X_detect = np.vstack([X_faces, X_neg])
y_detect = np.concatenate([np.ones(len(X_faces), dtype=int), y_neg])

print(f'Всего для детектора: {X_detect.shape}')
print(f'  лиц: {(y_detect==1).sum()},  не-лиц: {(y_detect==-1).sum()}')

per_type = N_NEG // 6
labels_types = ['вертик. флип', 'сдвиг вниз', 'поворот 90°', 'поворот 180°', 'угловой кроп', 'шум']
f, axes = plt.subplots(6, 4, figsize=(10, 14))
for t in range(6):
    for j in range(4):
        axes[t, j].imshow(np.clip(neg_patches[t * per_type + j], 0, 1))
        axes[t, j].axis('off')
    axes[t, 0].set_ylabel(labels_types[t], fontsize=9)
plt.suptitle('Негативные примеры — 6 типов трансформаций')
plt.tight_layout()
plt.show()
```

    Генерируем негативные примеры (6 типов трансформаций)...
    Вычисляем HOG для негативов...
      500/2026
      1000/2026
      1500/2026
      2000/2026
    Всего для детектора: (4052, 864)
      лиц: 2026,  не-лиц: 2026



    
![png](combined345_files/combined345_44_1.png)
    


# 4.4 Обучение SVM

Обучаем два независимых классификатора.

**Детектор** — бинарный: лицо (+1) или не-лицо (−1). Принимает решение в каждой позиции скользящего окна.

**Классификатор пола** — бинарный: мужчина (0) или женщина (1). Применяется только к патчам, которые детектор уже признал лицом.

Перед обучением масштабируем признаки через `StandardScaler` — вычитаем среднее и делим на стандартное отклонение. Без этого бины с большими значениями будут доминировать и SVM будет работать хуже.

### Детектор лицо/не-лицо


```python
X_tr, X_te, y_tr, y_te = train_test_split(
    X_detect, y_detect, test_size=0.2, random_state=42, stratify=y_detect
)

print('Обучаем детектор...')
detector_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    LinearSVC(C=0.1, max_iter=2000))
])
detector_pipe.fit(X_tr, y_tr)

y_pred = detector_pipe.predict(X_te)
print(f'Точность на тесте: {accuracy_score(y_te, y_pred):.3f}')
print(classification_report(y_te, y_pred, target_names=['не-лицо', 'лицо']))
```

    Обучаем детектор...
    Точность на тесте: 0.994
                  precision    recall  f1-score   support
    
         не-лицо       1.00      0.99      0.99       406
            лицо       0.99      1.00      0.99       405
    
        accuracy                           0.99       811
       macro avg       0.99      0.99      0.99       811
    weighted avg       0.99      0.99      0.99       811
    


### Классификатор пола: мужчина / женщина


```python
X_tr_g, X_te_g, y_tr_g, y_te_g = train_test_split(
    X_faces, y_gender, test_size=0.2, random_state=42, stratify=y_gender
)

print('Обучаем классификатор пола...')
gender_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    LinearSVC(C=1.0, max_iter=2000))
])
gender_pipe.fit(X_tr_g, y_tr_g)

y_pred_g = gender_pipe.predict(X_te_g)
print(f'Точность на тесте: {accuracy_score(y_te_g, y_pred_g):.3f}')
print(classification_report(y_te_g, y_pred_g, target_names=['мужчина', 'женщина']))

with open('detector_model.pkl', 'wb') as f:
    pickle.dump(detector_pipe, f)
with open('gender_model.pkl', 'wb') as f:
    pickle.dump(gender_pipe, f)
print('Модели сохранены.')
```

    Обучаем классификатор пола...
    Точность на тесте: 0.943
                  precision    recall  f1-score   support
    
         мужчина       0.97      0.96      0.97       370
         женщина       0.67      0.72      0.69        36
    
        accuracy                           0.94       406
       macro avg       0.82      0.84      0.83       406
    weighted avg       0.95      0.94      0.94       406
    
    Модели сохранены.


    /Users/arseniikostin/cv-labs-sem8/venv/lib/python3.14/site-packages/sklearn/svm/_base.py:1258: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      warnings.warn(


# 4.5 Скользящее окно и пирамида масштабов

Детектор обучен на патчах фиксированного размера. Чтобы находить лица разного размера на любом изображении, используем две идеи.

**Пирамида масштабов** — уменьшаем изображение с коэффициентом 0.85 на каждом шаге. На каждом масштабе запускаем скользящее окно. Маленькое окно таким образом «видит» и большие лица — просто уменьшенные.

**Скользящее окно** — перемещаем окно с шагом `step` по строкам и столбцам, для каждой позиции считаем HOG и спрашиваем детектор.

**Non-Maximum Suppression (NMS)** — одно лицо даёт десятки срабатываний в соседних позициях. NMS оставляет только прямоугольник с наибольшей уверенностью из всех перекрывающихся. Перекрытие измеряем через IoU (Intersection over Union):

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$


```python
def sliding_window(image_uint8, win_h, win_w, step=16):
    h, w = image_uint8.shape[:2]
    for r in range(0, h - win_h + 1, step):
        for c in range(0, w - win_w + 1, step):
            yield r, c, image_uint8[r:r+win_h, c:c+win_w]

def image_pyramid(image_uint8, scale=0.85, min_size=64):
    img    = image_uint8.copy()
    factor = 1.0
    while True:
        yield img, factor
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h < min_size or new_w < min_size:
            break
        img    = cv2.resize(img, (new_w, new_h))
        factor *= scale

def iou(boxA, boxB):
    r0 = max(boxA[0], boxB[0]); c0 = max(boxA[1], boxB[1])
    r1 = min(boxA[2], boxB[2]); c1 = min(boxA[3], boxB[3])
    inter = max(0, r1-r0) * max(0, c1-c0)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def nms(detections, iou_thresh=0.3):
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if iou(best[1:], d[1:]) < iou_thresh]
    return kept

def detect_and_classify(image_uint8, detector, gender_clf,
                        win_h, win_w, step=16, scale=0.85,
                        det_threshold=0.5, iou_thresh=0.3):
    detections = []
    for img_scaled, factor in image_pyramid(image_uint8, scale=scale):
        for r, c, patch in sliding_window(img_scaled, win_h, win_w, step):
            desc  = hog_descriptor(patch.astype(np.float32) / 255.0).reshape(1, -1)
            score = detector.decision_function(desc)[0]
            if score > det_threshold:
                r0, c0 = int(r/factor), int(c/factor)
                r1, c1 = int((r+win_h)/factor), int((c+win_w)/factor)
                detections.append((score, r0, c0, r1, c1))

    detections = nms(detections, iou_thresh)

    results = []
    for score, r0, c0, r1, c1 in detections:
        crop = image_uint8[r0:r1, c0:c1]
        if crop.size == 0:
            continue
        face_f = cv2.resize(crop, (win_w, win_h)).astype(np.float32) / 255.0
        gender = gender_clf.predict(hog_descriptor(face_f).reshape(1, -1))[0]
        results.append((r0, c0, r1, c1, gender))

    return results

WIN_H = images_valid.shape[1]
WIN_W = images_valid.shape[2]
print(f'Размер окна детектора: {WIN_H}×{WIN_W} пикселей')
```

    Размер окна детектора: 62×47 пикселей


# 4.6 Тестирование на фотографиях

Берём случайные изображения из тестовой выборки (те, что модель не видела при обучении) и классифицируем пол. Рамка красная — мужчина, синяя — женщина.


```python
test_imgs_idx = np.random.choice(len(X_te_g), 4, replace=False)

f, axes = plt.subplots(1, 4, figsize=(18, 5))
for i, idx in enumerate(test_imgs_idx):
    orig_idx = valid_indices[idx]
    image_u8 = (lfw.images[orig_idx] * 255).astype(np.uint8)

    pred_g = gender_pipe.predict(X_faces[idx].reshape(1, -1))[0]
    true_g = y_gender[idx]

    color_border = [255, 0, 0] if pred_g == 0 else [0, 0, 255]
    img_show = image_u8.copy()
    img_show[0:4, :]  = color_border
    img_show[-4:, :]  = color_border
    img_show[:, 0:4]  = color_border
    img_show[:, -4:]  = color_border

    label_pred = 'Муж' if pred_g == 0 else 'Жен'
    label_true = 'Муж' if true_g == 0 else 'Жен'

    axes[i].imshow(img_show)
    axes[i].set_title(f'Предсказание: {label_pred}\nИстина: {label_true}', fontsize=10)
    axes[i].axis('off')

plt.suptitle('Классификация пола (красный — мужчина, синий — женщина)')
plt.tight_layout()
plt.show()
```


    
![png](combined345_files/combined345_53_0.png)
    


# 4.7 Детектирование в реальном времени с веб-камеры

Живая камера вынесена в отдельный скрипт `live_camera.py` — он открывает окно cv2 и работает пока не нажать **Q**. Перед запуском убедитесь, что ячейка 4.4 выполнена и файлы `detector_model.pkl`, `gender_model.pkl` сохранены в папке с лабами.

**Запуск из терминала:**
```bash
python live_camera.py
```

Параметры в начале скрипта: `CAM_INDEX`, `DET_THRESHOLD`, `STEP`, `SCALE_DOWN`.

# 4.8 Оценка качества

Строим матрицы ошибок для обоих классификаторов на тестовой выборке.


```python
from sklearn.metrics import confusion_matrix

y_pred_det = detector_pipe.predict(X_te)
cm_det     = confusion_matrix(y_te, y_pred_det, labels=[-1, 1])

y_pred_gen = gender_pipe.predict(X_te_g)
cm_gen     = confusion_matrix(y_te_g, y_pred_gen, labels=[0, 1])

f, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm, title, labels in [
    (axes[0], cm_det, 'Детектор лицо / не-лицо', ['не-лицо', 'лицо']),
    (axes[1], cm_gen, 'Классификатор пола',       ['мужчина', 'женщина'])
]:
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_ylabel('Истина')
    ax.set_xlabel('Предсказание')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)

plt.tight_layout()
plt.show()

print(f'Точность детектора:           {accuracy_score(y_te, y_pred_det):.3f}')
print(f'Точность классификатора пола: {accuracy_score(y_te_g, y_pred_gen):.3f}')
```


    
![png](combined345_files/combined345_56_0.png)
    


    Точность детектора:           0.994
    Точность классификатора пола: 0.943


# Вывод

В ходе лабораторной работы реализован полный конвейер детектирования лиц и классификации пола на основе HOG + Linear SVM.

HOG-дескриптор вычисляется вручную: градиенты через центральные разности (как в лабе 2), гистограммы ориентаций по ячейкам, блочная L2-нормировка. Детектор на LinearSVC разделяет патчи на «лицо» и «не-лицо», второй SVM классифицирует пол. Для локализации лиц применяется пирамида масштабов и скользящее окно, дублирующиеся срабатывания убираются NMS по порогу IoU. Негативные примеры синтезируются из самого датасета через 6 типов трансформаций, что обеспечивает реальную разнообразность и корректные метрики.