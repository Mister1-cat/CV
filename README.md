Бондарев Игорь, 8Е21.
## 1 лабораторная работа

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


vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv Подготовка изображения и вспомогательные операции

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


## 2 лабораторная работа
Визуальная одометрия (навигация)
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

### 3.1 Вычисление доминирующей ориентации

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

### 3.2 Вычисление доминирующей ориентации

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
    
### 6.1 Накопление трансформаций

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

### 6.2 Накопление трансформаций

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

# 3 Лабораторная работа. 
Работа с видеопотоком
<p>Цель: Научиться анализировать видеопоток.
<p>Ход работы: получить видеопоток с Web-камеры и определить перемещающийся в кадре объект. Используя данные видеопотока реализуйте следующее:

<p> 1. Реализуйте получение данных с Web-камеры
<p> 2. Реализуйте алгоритм вычитания фона
<p> 3. Реализуйте определение движущегося предмета
<p> 4. Постройте траекторию движения объекта.
<p> 5. Проведите тестирование на тестовом видео.

### 1 Запись фона

Запись разбита на два шага: сначала фиксируется пустой фон (3 секунды), затем фиксируется движение объекта (5 секунд). Используются `cv2.VideoCapture` и обратный отсчёт для синхронизации.


```python
BG_SECONDS = 3
MOTION_SECONDS = 5
FPS_APPROX = 10
N_BG_FRAMES = BG_SECONDS * FPS_APPROX
N_MOT_FRAMES = MOTION_SECONDS * FPS_APPROX

THRESHOLD = 15
MORPH_SIZE = 5
CAM_ID = 0   

print("ЗАПИСЬ ФОНА")
cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError("Камера не найдена")

bg_frames = []
for i in range(N_BG_FRAMES):
    ret, frame = cap.read()
    if ret:
        img = frame[:, :, ::-1].copy()   # BGR -> RGB
        bg_frames.append(img)
    if i % FPS_APPROX == 0:
        print(f"  кадр {i}")
    cv2.waitKey(100)
cap.release()
print(f"Захвачено кадров фона: {len(bg_frames)}")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(bg_frames[0]);  plt.title('Фон: начало')
plt.subplot(1,2,2); plt.imshow(bg_frames[-1]); plt.title('Фон: конец')
plt.show()
```

    ЗАПИСЬ ФОНА
        кадр 0
        кадр 10
        кадр 20
    Захвачено кадров фона: 30

    
![png](README_files/Безымянный1.png)
    
```python
print("\nЗАПИСЬ ДВИЖЕНИЯ")
cap = cv2.VideoCapture(CAM_ID)
motion_frames = []
for i in range(N_MOT_FRAMES):
    ret, frame = cap.read()
    if ret:
        img = frame[:, :, ::-1].copy()
        motion_frames.append(img)
    if i % FPS_APPROX == 0:
        print(f"  кадр {i}")
    cv2.waitKey(100)
cap.release()
print(f"Захвачено кадров движения: {len(motion_frames)}")

step = max(1, len(motion_frames)//4)
indices = list(range(0, len(motion_frames), step))[:4]
plt.figure(figsize=(16,4))
for i, idx in enumerate(indices):
    plt.subplot(1,4,i+1); plt.imshow(motion_frames[idx]); plt.title(f'Движение {idx}')
plt.show()
```
    ЗАПИСЬ ДВИЖЕНИЯ
        кадр 0
        кадр 10
        кадр 20
        кадр 30
        кадр 40
    Захвачено кадров движения: 50

    
![png](README_files/Безымянный2.png)



### 2 Модель фона

Модель фона строится попиксельным усреднением всех кадров пустой сцены. Случайный шум матрицы камеры при усреднении подавляется, остаётся стабильная картина фона.


```python
def construct_background(sequence):
    accum = np.zeros_like(sequence[0], dtype=np.float64)
    for f in sequence:
        accum += f.astype(np.float64)
    accum /= len(sequence)
    return accum

background = construct_background(bg_frames)
```

    
![png](README_files/Безымянный3.png)
    


### 3 Вычитание фона и бинарная маска

Для каждого кадра вычисляется абсолютная разность по каждому каналу RGB между текущим кадром и моделью фона. Затем разность усредняется по трём каналам, формируя одноканальное изображение 
$$D = \frac{|R - R_{bg}| + |G - G_{bg}| + |B - B_{bg}|}{3}$$

Пиксели, для которых D превышает порог THRESHOLD, считаются передним планом и получают значение 255 (белый), остальные — 0 (чёрный).

```python
def difference_mask(frame, bg, thr):
    d = np.abs(frame.astype(np.float64) - bg.astype(np.float64))
    gray = (d[:,:,0] + d[:,:,1] + d[:,:,2]) / 3.0
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[gray > thr] = 255
    return gray, mask
```


    
![png](README_files/Безымянный4.png)
    


### 4 Морфологическая очистка маски

Сырая маска содержит множество ложных срабатываний из-за шумов и мелких изменений освещения. Для их удаления применяется морфологическое открытие: сначала эрозия, затем дилатация с квадратным структурным элементом размера MORPH_SIZE × MORPH_SIZE.

#### Эрозия

Центральный пиксель остаётся белым (255) только если все пиксели под окном белые. Это удаляет мелкие белые пятна.

```python
def erode_manual(mask, ksize):
    h, w = mask.shape
    pad = ksize // 2
    bin_mask = mask > 0
    padded = np.pad(bin_mask, pad, mode='constant', constant_values=True)
    res = np.zeros_like(mask, dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            window = padded[y:y+ksize, x:x+ksize]
            res[y, x] = 255 if np.all(window) else 0
    return res
```


#### Дилатация

Центральный пиксель становится белым, если хотя бы один пиксель под окном белый. Это восстанавливает размер объектов, «съеденных» эрозией.

```python
def dilate_manual(mask, ksize):
    h, w = mask.shape
    pad = ksize // 2
    bin_mask = mask > 0
    padded = np.pad(bin_mask, pad, mode='constant', constant_values=False)
    res = np.zeros_like(mask, dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            window = padded[y:y+ksize, x:x+ksize]
            res[y, x] = 255 if np.any(window) else 0
    return res
```

```python
def clean_mask(mask, ksize):
    return dilate_manual(erode_manual(mask, ksize), ksize)
```

![png](README_files/Безымянный5.png)


### 5 Поиск объекта на маске

Для нахождения движущегося объекта на очищенной маске используется обход в ширину (BFS) для выделения всех связных областей белых пикселей. Из них выбирается область с максимальной площадью. Для нее вычисляются:

- центроид – среднее арифметическое координат всех пикселей области;

- bounding box – минимальный и максимальный номера строк и столбцов;

- граничные пиксели – пиксели области, у которых хотя бы один из четырёх соседей не принадлежит области (контур).

```python
def extract_blob_info(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    best_set = None
    best_cx = best_cy = None
    best_bbox = None
    best_border = None

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 255 and not seen[y][x]:
                q = deque()
                q.append((x, y))
                seen[y][x] = True
                pixels = set()
                pixels.add((x, y))
                while q:
                    cx, cy = q.popleft()
                    for nx, ny in [(cx-1,cy),(cx+1,cy),(cx,cy-1),(cx,cy+1)]:
                        if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] == 255 and not seen[ny][nx]:
                            seen[ny][nx] = True
                            q.append((nx, ny))
                            pixels.add((nx, ny))
                if len(pixels) > len(best_set or []):
                    best_set = pixels
                    xs = [p[0] for p in pixels]
                    ys = [p[1] for p in pixels]
                    best_cx = int(np.mean(xs))
                    best_cy = int(np.mean(ys))
                    best_bbox = (min(xs), min(ys), max(xs), max(ys))
                    # граничные пиксели
                    best_border = set()
                    for (px, py) in pixels:
                        is_border = False
                        for nx, ny in [(px-1,py),(px+1,py),(px,py-1),(px,py+1)]:
                            if (nx, ny) not in pixels:
                                is_border = True
                                break
                        if is_border:
                            best_border.add((px, py))
    return best_set, (best_cx, best_cy) if best_set else None, best_bbox, best_border
```


    
![png](README_files/Безымянный6.png)
    


### 6 Траектория движения

Для каждого кадра движения находится центроид самого большого объекта, если объект присутствует. Последовательность центроидов формирует траекторию.

Вспомогательные функции рисования:

```python
def draw_line(pic, p1, p2, color):
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    while True:
        if 0 <= x1 < pic.shape[1] and 0 <= y1 < pic.shape[0]:
            pic[y1, x1] = color
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x1 += sx
        if e2 <= dx:
            err += dx
            y1 += sy

def draw_rect(pic, x0, y0, x1, y1, color):
    for x in range(x0, x1+1):
        pic[y0, x] = color
        pic[y1, x] = color
    for y in range(y0, y1+1):
        pic[y, x0] = color
        pic[y, x1] = color

def draw_spot(pic, centre, r, color):
    cx, cy = centre
    for dx in range(-r, r+1):
        for dy in range(-r, r+1):
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < pic.shape[1] and 0 <= ny < pic.shape[0]:
                pic[ny, nx] = color
```

Накопление координат центроида и отрисовка траектории на первом кадре:

```python
positions = []
for frame in motion_frames:
    _, msk = difference_mask(frame, background, THRESHOLD)
    msk_cl = clean_mask(msk, MORPH_SIZE)
    _, ct, _, _ = extract_blob_info(msk_cl)
    if ct:
        positions.append(ct)

if positions:
    trail_img = motion_frames[0].copy()
    for i in range(1, len(positions)):
        draw_line(trail_img, positions[i-1], positions[i], [255, 0, 0])
    for pt in positions:
        draw_spot(trail_img, pt, 2, [255, 255, 0])

    plt.figure(figsize=(8,6))
    plt.imshow(trail_img)
    plt.title('Траектория движения')
    plt.axis('off')
    plt.show()
```
    
![png](README_files/Безымянный7.png)
    


### 7 Графики изменения координат

Для анализа движения строятся графики координаты X и Y центроида от номера кадра.


```python
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    idxs = list(range(len(positions)))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(idxs, xs, 'r.-')
    plt.xlabel('Кадр')
    plt.ylabel('X')
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(idxs, ys, 'b.-')
    plt.xlabel('Кадр')
    plt.ylabel('Y')
    plt.grid(True)
    plt.suptitle('Координаты центроида')
    plt.tight_layout()
    plt.show()
```


![png](README_files/Безымянный8.png)
    

# Вывод

В ходе лабораторной работы реализован полный конвейер обнаружения и отслеживания движущегося объекта в видеопотоке без использования готовых функций компьютерного зрения (кроме захвата кадров). Построена модель фона усреднением, реализовано вычитание фона с пороговой бинаризацией, написаны морфологические операции эрозии и дилатации для очистки маски, применён BFS для поиска связных областей и вычисления центроида. По последовательности центроидов построена траектория и графики координат.

Все операции выполнены вручную с использованием базовых циклов и numpy, что соответствует требованиям задания.


## 4 лабораторная работа 
Разработка алгоритма определения лиц.

<p>Цель: на практике закрепить полученные в ходе курса знания, в том числе по машинному обучению и нейронным сетям для решения задачи детектирования лиц и классификации лиц на мужчин и женщин.

<p><b>Выбранный метод: LBP + Linear SVM</b>

<p>LBP (Local Binary Patterns) описывает локальную текстуру изображения путём сравнения центрального пикселя с его 8 соседями. LBP вычислительно дешевый, устойчив к монотонным изменениям освещённости и хорошо выделяет структурные особенности кожи, контуров лица и глаз, что делает его удобным для задач детекции при ограниченных вычислительных ресурсах.

<p>SVM (Support Vector Machine) ищет гиперплоскость, максимально разделяющую два класса в пространстве HOG-признаков.

### 1 Загрузка датасета и ручная разметка

Используем датасет LFW (Labeled Faces in the Wild). Параметр min_faces_per_person=15 отфильтровывает персоны с малым количеством снимков, оставляя только репрезентативную выборку. Разметка по полу выполняется вручную через словарь GENDER_LABELS, что соответствует требованию о ручной аннотации данных.

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.datasets import fetch_lfw_people
import os
import pickle

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'font.size': 12,
    'figure.dpi': 100,
})

print('Загружаем датасет LFW')
lfw = fetch_lfw_people(min_faces_per_person=15, resize=0.5, color=True)
print(f'Изображений: {lfw.images.shape[0]}')
print(f'Размер патча: {lfw.images.shape[1]}x{lfw.images.shape[2]}')

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

valid_indices = []
gender_labels = []
for i, tid in enumerate(lfw.target):
    name = lfw.target_names[tid]
    if name in GENDER_LABELS:
        valid_indices.append(i)
        gender_labels.append(GENDER_LABELS[name])

images_valid = lfw.images[valid_indices]
y_gender = np.array(gender_labels)
print(f'Отфильтровано {len(images_valid)} изображений')
print(f'Мужчин {(y_gender==0).sum()}, Женщин {(y_gender==1).sum()}')

f, axes = plt.subplots(2, 8, figsize=(16, 5))
for row, gender in enumerate([0, 1]):
    idxs = np.where(y_gender == gender)[0][:8]
    for col, idx in enumerate(idxs):
        axes[row, col].imshow(images_valid[idx])
        axes[row, col].axis('off')
        name = lfw.target_names[lfw.target[valid_indices[idx]]]
        axes[row, col].set_title(name.split()[-1], fontsize=7)
axes[0,0].set_ylabel('Мужчины'); axes[1,0].set_ylabel('Женщины')
plt.suptitle('Примеры из датасета LFW с ручной разметкой')
plt.tight_layout(); plt.show()
```

![png](README_files/Безымянный11.png)


### 2 LBP-дескриптор и предобработка

Для работы с яркостью реализуем ручной перевод в оттенки серого по фотометрической формуле. Изображения приводятся к единому размеру 48×48 с помощью билинейной интерполяции. Билинейное изменение размера вычисляет взвешенную сумму четырёх ближайших пикселей исходного изображения, что сохраняет плавность переходов без появления «лесенок».

```python
def to_gray_manual(image_float):
    if image_float.ndim == 3:
        return (0.299*image_float[:,:,0] + 0.587*image_float[:,:,1] + 0.114*image_float[:,:,2]) * 255.0
    return image_float * 255.0

def bilinear_resize(image, target_h, target_w):
    orig_h, orig_w = image.shape[:2]
    if target_h == orig_h and target_w == orig_w:
        return image
    
    sy, sx = orig_h / target_h, orig_w / target_w
    gy, gx = np.mgrid[0:target_h, 0:target_w]
    
    src_y = gy * sy
    src_x = gx * sx
    y0, x0 = np.floor(src_y).astype(int), np.floor(src_x).astype(int)
    y1 = np.minimum(y0 + 1, orig_h - 1)
    x1 = np.minimum(x0 + 1, orig_w - 1)
    
    dy, dx = src_y - y0, src_x - x0
    w1 = (1-dy) * (1-dx)
    w2 = dy * (1-dx)
    w3 = (1-dy) * dx
    w4 = dy * dx
    
    if image.ndim == 2:
        out = image[y0, x0]*w1 + image[y1, x0]*w2 + image[y0, x1]*w3 + image[y1, x1]*w4
    else:
        out = np.stack([
            image[y0, x0]*w1 + image[y1, x0]*w2 + image[y0, x1]*w3 + image[y1, x1]*w4
            for i in range(3)], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)
```

LBP (Local Binary Patterns) сравнивает центральный пиксель с 8 соседями. Если сосед ярче центра, записывается 1, иначе 0. 8 бит сдвигаются в соответствии с позицией соседа и объединяются в байт. Для всего патча строится нормированная гистограмма на 256 бинов, которая становится финальным вектором признаков.


```python
def compute_lbp_manual(gray_uint8):
    p = np.pad(gray_uint8, 1, mode='edge')
    
    tl = p[:-2, :-2]; t = p[:-2, 1:-1]; tr = p[:-2, 2:]
    l  = p[1:-1, :-2];                 r = p[1:-1, 2:]
    bl = p[2:, :-2];   b = p[2:, 1:-1];  br = p[2:, 2:]
    c  = p[1:-1, 1:-1]
    
    lbp = ((t >= c) << 7) | ((tr >= c) << 6) | ((r >= c) << 5) | \
          ((br >= c) << 4) | ((b >= c) << 3) | ((bl >= c) << 2) | \
          ((l >= c) << 1) | ((tl >= c) << 0)
          
    hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    return (hist / (hist.sum() + 1e-6)).flatten()
```
    


### 3 Извлечение признаков и генерация негативов

Детектору необходимо видеть оба класса. В качестве негативов («не-лицо») используются те же изображения LFW, но подвергнутые 6 типам искажений: вертикальный флип, сдвиг с шумом, транспонирование, поворот на 180°, угловой кроп и чистый шум. Это гарантирует, что модель учится различать структуру лица, а не запоминает фоновые артефакты.


```python
TARGET_SIZE = (48, 48)
print('Вычисляем LBP-дескрипторы')
X_faces = []
for i, img in enumerate(images_valid):
    gray = to_gray_manual(img).astype(np.uint8)
    gray_res = bilinear_resize(gray, TARGET_SIZE[1], TARGET_SIZE[0])
    X_faces.append(compute_lbp_manual(gray_res))
    if (i+1) % 200 == 0:
        print(f'  Обработано: {i+1}/{len(images_valid)}')

X_faces = np.array(X_faces)
print(f'Матрица признаков лиц {X_faces.shape}')
```
    Вычисляем LBP-дескрипторы
        Обработано: 200/2059
        Обработано: 400/2059
        Обработано: 600/2059
        Обработано: 800/2059
        Обработано: 1000/2059
        Обработано: 1200/2059
        Обработано: 1400/2059
        Обработано: 1600/2059
        Обработано: 1800/2059
        Обработано: 2000/2059
    Матрица признаков лиц (2059, 256)

```python
print('Генерируем синтетические негативы из LFW')
np.random.seed(42)
N_NEG = len(images_valid)
negatives = []

def make_negatives(all_imgs, n_total):
    per_type = n_total // 6 + 1
    neg_list = []
    idx = np.random.permutation(len(all_imgs))
    
    for i in range(per_type):
        patch = all_imgs[idx[i % n_imgs]].copy()
        # Вертикальный флип
        neg_list.append(np.flipud(patch))
    for i in range(per_type):
        patch = all_imgs[idx[(i+per_type)%n_imgs]].copy()
        # Сдвиг + шум сверху
        patch_shifted = np.roll(patch, 20, axis=0)
        patch_shifted[:20] = np.random.rand(20, patch.shape[1], 3).astype(np.float32)
        neg_list.append(patch_shifted)
    for i in range(per_type):
        patch = all_imgs[idx[(i+2*per_type)%n_imgs]].copy()
        # Транспонирование 90 град
        neg_list.append(np.transpose(patch, (1, 0, 2)))
    for i in range(per_type):
        patch = all_imgs[idx[(i+3*per_type)%n_imgs]].copy()
        # Переворот 180
        neg_list.append(np.flipud(np.fliplr(patch)))
    for i in range(per_type):
        patch = all_imgs[idx[(i+4*per_type)%n_imgs]].copy()
        # Угловой кроп (растягиваем и берём угол)
        big = np.kron(patch, np.ones((2, 2, 1)))
        neg_list.append(big[-62:, -47:, :])
    for i in range(per_type):
        # Чистый шум
        neg_list.append(np.random.rand(62, 47, 3).astype(np.float32))
        
    return neg_list[:n_total]

n_imgs = len(images_valid)
neg_patches = make_negatives(images_valid, N_NEG)

print('Вычисляем LBP для негативов')
X_neg = []
for i, patch in enumerate(neg_patches):
    gray = to_gray_manual(patch).astype(np.uint8)
    gray_res = bilinear_resize(gray, TARGET_SIZE[1], TARGET_SIZE[0])
    X_neg.append(compute_lbp_manual(gray_res))
    if (i+1) % 500 == 0:
        print(f'  Обработано негативов: {i+1}/{N_NEG}')

X_neg = np.array(X_neg)
y_neg = np.full(N_NEG, -1)
X_detect = np.vstack([X_faces, X_neg])
y_detect = np.concatenate([np.ones(len(X_faces), dtype=int), y_neg])

print(f'Всего для детектора: {X_detect.shape} (лица: {(y_detect==1).sum()}, не-лица: {(y_detect==-1).sum()})')
```
    Генерируем синтетические негативы из LFW
    Вычисляем LBP для негативов
        Обработано негативов: 500/2059
        Обработано негативов: 1000/2059
        Обработано негативов: 1500/2059
        Обработано негативов: 2000/2059
    Всего для детектора: (4118, 256) (лица: 2059, не-лица: 2059)

![png](README_files/Безымянный17.png)
    


### 4 Обучение SVM-классификаторов

Перед обучением признаки масштабируются StandardScaler (вычитание среднего, деление на СКО), что критично для линейных SVM. Обучаются две модели:
1. Детектор лицо/не-лицо (y ∈ {-1, 1})
2. Классификатор пола (y ∈ {0, 1})


```python
print('Обучаем детектор лицо / не-лицо')
X_tr, X_te, y_tr, y_te = train_test_split(
    X_detect, y_detect, test_size=0.2, random_state=42, stratify=y_detect
)

detector_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(C=1.0, max_iter=3000))
])
detector_pipe.fit(X_tr, y_tr)
y_pred_det = detector_pipe.predict(X_te)
print(f'Точность детектора: {accuracy_score(y_te, y_pred_det):.3f}')
print(classification_report(y_te, y_pred_det, target_names=['не-лицо', 'лицо']))

print('\nОбучаем классификатор пола')
X_tr_g, X_te_g, y_tr_g, y_te_g = train_test_split(
    X_faces, y_gender, test_size=0.2, random_state=42, stratify=y_gender
)

gender_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(C=1.0, max_iter=3000))
])
gender_pipe.fit(X_tr_g, y_tr_g)
y_pred_g = gender_pipe.predict(X_te_g)
print(f'Точность классификатора пола: {accuracy_score(y_te_g, y_pred_g):.3f}')
print(classification_report(y_te_g, y_pred_g, target_names=['мужчина', 'женщина']))

with open('lbp_detector.pkl', 'wb') as f: pickle.dump(detector_pipe, f)
with open('lbp_gender.pkl', 'wb') as f: pickle.dump(gender_pipe, f)
```

![png](README_files/Безымянный14.png)


### 5 Пирамида масштабов, скользящее окно и NMS

Детектор обучен на фиксированном окне 48×48. Для поиска лиц произвольного размера реализуется:
- Пирамида масштабов: изображение последовательно уменьшается в 0.85 раз.
- Скользящее окно: окно сдвигается с шагом step по строкам и столбцам.
- NMS (Non-Maximum Suppression): убираются дублирующиеся прямоугольники. Перекрытие считается через IoU (Intersection over Union), оставляется только окно с наибольшим score.


```python
def image_pyramid_manual(image_uint8, scale=0.85, min_size=64):
    img = image_uint8.copy()
    factor = 1.0
    while True:
        yield img, factor
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h < min_size or new_w < min_size:
            break
        img = bilinear_resize(img, new_h, new_w)
        factor *= scale

def sliding_window_manual(image_uint8, win_h, win_w, step=16):
    h, w = image_uint8.shape[:2]
    for r in range(0, h - win_h + 1, step):
        for c in range(0, w - win_w + 1, step):
            yield r, c, image_uint8[r:r+win_h, c:c+win_w]

def iou_manual(boxA, boxB):
    r0 = max(boxA[0], boxB[0]); c0 = max(boxA[1], boxB[1])
    r1 = min(boxA[2], boxB[2]); c1 = min(boxA[3], boxB[3])
    inter = max(0, r1-r0) * max(0, c1-c0)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def nms_manual(detections, iou_thresh=0.3):
    if not detections: return []
    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if iou_manual(best[1:], d[1:]) < iou_thresh]
    return kept

def detect_and_classify_lbp(image_uint8, detector, gender_clf,
                        win_h, win_w, step=16, scale=0.85,
                        det_threshold=0.5, iou_thresh=0.3):
    detections = []
    for img_scaled, factor in image_pyramid_manual(image_uint8, scale=scale):
        for r, c, patch in sliding_window_manual(img_scaled, win_h, win_w, step):
            gray = to_gray_manual(patch).astype(np.uint8)
            gray_res = bilinear_resize(gray, win_w, win_h)
            desc = compute_lbp_manual(gray_res).reshape(1, -1)
            
            score = detector.decision_function(desc)[0]
            if score > det_threshold:
                r0 = int(r / factor); c0 = int(c / factor)
                r1 = int((r+win_h) / factor); c1 = int((c+win_w) / factor)
                detections.append((score, r0, c0, r1, c1))
                
    detections = nms_manual(detections, iou_thresh)
    
    results = []
    for score, r0, c0, r1, c1 in detections:
        crop = image_uint8[r0:r1, c0:c1]
        if crop.size == 0: continue
        gray = to_gray_manual(crop).astype(np.uint8)
        gray_res = bilinear_resize(gray, win_w, win_h)
        desc = compute_lbp_manual(gray_res).reshape(1, -1)
        gender = gender_clf.predict(desc)[0]
        results.append((r0, c0, r1, c1, gender))
    return results
```


### 6 Визуализация и тестирование

Для проверки качества случайно выбираются изображения из тестовой выборки. Рамка окрашивается в красный (мужчина) или синий (женщина).


```python
test_imgs_idx = np.random.choice(len(X_te_g), 4, replace=False)
f, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, idx in enumerate(test_imgs_idx):
    orig_idx = valid_indices[idx]
    img = (lfw.images[orig_idx] * 255).astype(np.uint8)
    pred = gender_pipe.predict(X_faces[idx].reshape(1, -1))[0]
    true = y_gender[idx]
    color = [255, 0, 0] if pred == 0 else [0, 0, 255]
    img_vis = img.copy()
    img_vis[:3, :] = img_vis[-3:, :] = img_vis[:, :3] = img_vis[:, -3:] = color
    axes[i].imshow(img_vis)
    axes[i].set_title(f'Предсказание: {"Муж" if pred==0 else "Жен"}\nИстина: {"Муж" if true==0 else "Жен"}')
    axes[i].axis('off')
plt.suptitle('LBP + SVM: Классификация пола на тесте')
plt.tight_layout(); plt.show()
```

![png](README_files/Безымянный16.png)


### 7 Оценка качества
Строим матрицы ошибок для обоих классификаторов на тестовой выборке.


```python
y_pred_det = detector_pipe.predict(X_te)
cm_det = confusion_matrix(y_te, y_pred_det, labels=[-1, 1])
y_pred_gen = gender_pipe.predict(X_te_g)
cm_gen = confusion_matrix(y_te_g, y_pred_gen, labels=[0, 1])

f, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm, title, labels in [
    (axes[0], cm_det, 'Детектор лицо / не-лицо', ['не-лицо', 'лицо']),
    (axes[1], cm_gen, 'Классификатор пола', ['мужчина', 'женщина'])
]:
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title); ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_ylabel('Истина'); ax.set_xlabel('Предсказание')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
plt.tight_layout(); plt.show()

print(f'Точность детектора:           {accuracy_score(y_te, y_pred_det):.3f}')
print(f'Точность классификатора пола: {accuracy_score(y_te_g, y_pred_gen):.3f}')
```

![png](README_files/Безымянный15.png)

# Вывод

В ходе лабораторной работы реализован полный конвейер детектирования лиц и классификации пола на основе LBP + Linear SVM. Все операции предобработки, извлечения признаков и постобработки написаны вручную без использования готовых CV-библиотек. Синтетические негативы обеспечивают сбалансированность классов и повышают обобщающую способность детектора.