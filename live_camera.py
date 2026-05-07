#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cv2
import pickle
import sys
import os

# ⚙️ Параметры для баланса Скорость / Точность
CAM_INDEX = 0          # Индекс камеры (0 = встроенная, 1/2 = внешняя)
DET_THRESHOLD = 1.2    # Порог уверенности. Чем выше, тем меньше ложных лиц
STEP = 16              # Шаг скользящего окна (пикс.). Больше = быстрее, но грубее
SCALE_DOWN = 0.3       # Уменьшение входного кадра для скорости (0.3 = 30% от оригинала)
PYRAMID_SCALE = 0.9    # Коэффициент уменьшениsя пирамиды
IOU_THRESH = 0.0       # Порог перекрытия для NMS
WIN_H = 48             # Высота окна (должна совпадать с TARGET_SIZE из ноутбука)
WIN_W = 48             # Ширина окна

# ─────────────────────────────────────────────────────────────
# 🛠 РУЧНЫЕ ФУНКЦИИ КОМПЬЮТЕРНОГО ЗРЕНИЯ (из ноутбука)
# ─────────────────────────────────────────────────────────────
def to_gray_manual(image):
    # image может быть uint8 [0..255] или float [0..1]
    if image.dtype == np.uint8:
        if image.ndim == 3:
            # яркость для uint8 (умножение на 1.0, чтобы получить float, потом приведём)
            gray = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
        else:
            gray = image.astype(np.float64)
    else:
        # предполагаем float [0..1]
        if image.ndim == 3:
            gray = (0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]) * 255.0
        else:
            gray = image * 255.0
    return gray  # возвращает float, будет потом .astype(np.uint8)

def bilinear_resize(image, target_h, target_w):
    orig_h, orig_w = image.shape[:2]
    if target_h == orig_h and target_w == orig_w:
        return image
    
    sy, sx = orig_h / target_h, orig_w / target_w
    gy, gx = np.mgrid[0:target_h, 0:target_w]
    src_y, src_x = gy * sy, gx * sx
    y0, x0 = np.floor(src_y).astype(int), np.floor(src_x).astype(int)
    y1 = np.minimum(y0 + 1, orig_h - 1)
    x1 = np.minimum(x0 + 1, orig_w - 1)
    dy, dx = src_y - y0, src_x - x0
    
    w1 = (1-dy)*(1-dx)
    w2 = dy*(1-dx)
    w3 = (1-dy)*dx
    w4 = dy*dx
    
    # 🔧 ФИКС: добавляем третье измерение весам, если изображение цветное (3 канала)
    if image.ndim == 3:
        w1 = w1[:, :, np.newaxis]
        w2 = w2[:, :, np.newaxis]
        w3 = w3[:, :, np.newaxis]
        w4 = w4[:, :, np.newaxis]
        
    out = image[y0, x0]*w1 + image[y1, x0]*w2 + image[y0, x1]*w3 + image[y1, x1]*w4
    return np.clip(out, 0, 255).astype(np.uint8)

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

def image_pyramid_manual(image_uint8, scale=0.8, min_size=64):
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

def detect_and_classify_lbp(image_uint8, detector, gender_clf, win_h, win_w, step, scale, det_threshold, iou_thresh):
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

def draw_results(frame_bgr, results, threshold):
    img = frame_bgr.copy()
    for r0, c0, r1, c1, gender in results:
        # BGR цвета: Синий = Мужчина, Розовый = Женщина
        color = (255, 0, 0) if gender == 0 else (135, 0, 255)
        label = 'Male' if gender == 0 else 'Female'
        cv2.rectangle(img, (c0, r0), (c1, r1), color, 2)
        cv2.putText(img, label, (c0, r0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    cv2.putText(img, f'Threshold: {threshold:.2f} [+/-] to adjust', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f'Faces: {len(results)} [Q] to quit', (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img

def main():
    global DET_THRESHOLD
    # Загрузка моделей
    for fname in ['lbp_detector.pkl', 'lbp_gender.pkl']:
        if not os.path.exists(fname):
            print(f'[ERROR] Файл {fname} не найден.')
            print('Сначала выполните ячейки обучения в ноутбуке, чтобы сохранить модели.')
            sys.exit(1)
            
    with open('lbp_detector.pkl', 'rb') as f: detector = pickle.load(f)
    with open('lbp_gender.pkl', 'rb') as f: gender_clf = pickle.load(f)
    print('✅ Модели LBP+SVM загружены.')

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f'[ERROR] Не удалось открыть камеру с индексом {CAM_INDEX}')
        sys.exit(1)

    cv2.namedWindow('LBP Face Detector', cv2.WINDOW_NORMAL)
    print('Управление: Q — выход, + — поднять порог, - — снизить порог')
    
    threshold = DET_THRESHOLD
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Уменьшаем кадр для скорости
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * SCALE_DOWN), int(h * SCALE_DOWN)))
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        # Детектируем
        results = detect_and_classify_lbp(small_rgb, detector, gender_clf, WIN_H, WIN_W, step=STEP, scale=PYRAMID_SCALE, det_threshold=threshold, iou_thresh=IOU_THRESH)
        
        # Возвращаем координаты к исходному масштабу
        results_full = [(int(r0/SCALE_DOWN), int(c0/SCALE_DOWN), int(r1/SCALE_DOWN), int(c1/SCALE_DOWN), g) for r0, c0, r1, c1, g in results]
        
        frame_drawn = draw_results(frame, results_full, threshold)
        cv2.imshow('LBP Face Detector', frame_drawn)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'): break
        elif key == ord('+') or key == ord('='):
            threshold = min(threshold + 0.05, 100.0)
            print(f'Порог: {threshold:.2f}')
        elif key == ord('-'):
            threshold = max(threshold - 0.05, -1.0)
            print(f'Порог: {threshold:.2f}')
            
    cap.release()
    cv2.destroyAllWindows()
    print('Завершено.')

if __name__ == '__main__':
    main()