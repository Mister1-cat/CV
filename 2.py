import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import random

def intensity_grayscale(image):
    if len(image.shape) == 2:
        return image.copy()
    return (0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]).astype(np.uint8)

def gauss_kernel(size, sigma):
    k = np.zeros(size, dtype=np.float32)
    c = size // 2
    s = 0.0
    for i in range(size):
        x = i - c
        k[i] = math.exp(-0.5 * (x/sigma)**2)
        s += k[i]
    return k / s

def gaussian_2d(image, sigma, minor_size=21):
    k = gauss_kernel(minor_size, sigma)
    pad = minor_size // 2
    h, w = image.shape
    if len(image.shape) == 2:
        padded = np.pad(image, pad, mode='edge')
        temp = np.zeros_like(image, dtype=np.float32)
        for i in range(h):
            for j in range(w):
                temp[i,j] = np.sum(padded[i+pad, j:j+minor_size] * k)
        padded2 = np.pad(temp, pad, mode='edge')
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(h):
            for j in range(w):
                result[i,j] = np.sum(padded2[i:i+minor_size, j+pad] * k)
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        res = np.zeros_like(image, dtype=np.float32)
        for ch in range(3):
            res[:,:,ch] = gaussian_2d(image[:,:,ch], sigma, minor_size)
        return res.astype(np.uint8)

def hist_equalize(image):
    if len(image.shape) != 2:
        image = intensity_grayscale(image)
    hist, _ = np.histogram(image.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_norm = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_norm = cdf_norm.astype(np.uint8)
    return cdf_norm[image]

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

def draw_keypoints(image, keypoints, Ix=None, Iy=None, show_vectors=False, vector_scale=5):
    if len(image.shape) == 2:
        display = np.stack([image, image, image], axis=-1).astype(np.uint8)
    else:
        display = image.copy()
    plt.figure(figsize=(8,6))
    plt.imshow(display)
    for (r,c) in keypoints:
        plt.scatter(c, r, s=20, color='red')
        if show_vectors and Ix is not None and Iy is not None:
            gx = Ix[r,c]
            gy = Iy[r,c]
            plt.arrow(c, r, gx*vector_scale, gy*vector_scale,
                      head_width=3, length_includes_head=True, color='lime')
    plt.axis('off')
    plt.show()

def show_images(images, rows=2, cols=4, figsize=(12,6), titles=None, suptitle=None):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=9)
            ax.axis('off')
        else:
            ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

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

def euclidean_distance(a, b):
    return math.sqrt(np.sum((a-b)**2))

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

def draw_matches(img_a, img_b, matches, max_display=50):
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    h_combined = max(h_a, h_b)
    combined = np.zeros((h_combined, w_a + w_b, 3), dtype=np.uint8)
    if len(img_a.shape) == 2:
        combined[:h_a, :w_a] = np.stack([img_a]*3, axis=-1)
    else:
        combined[:h_a, :w_a] = img_a
    if len(img_b.shape) == 2:
        combined[:h_b, w_a:w_a + w_b] = np.stack([img_b]*3, axis=-1)
    else:
        combined[:h_b, w_a:w_a + w_b] = img_b
    plt.figure(figsize=(14,6))
    plt.imshow(combined)
    np.random.seed(42)
    displayed = matches[:max_display]
    for (r_a,c_a), (r_b,c_b) in displayed:
        color = (np.random.random(), np.random.random(), np.random.random())
        plt.scatter(c_a, r_a, s=15, color=color)
        plt.scatter(c_b + w_a, r_b, s=15, color=color)
        plt.plot([c_a, c_b + w_a], [r_a, r_b], color=color, linewidth=0.8, alpha=0.7)
    plt.axis('off')
    plt.title(f'Показано {len(displayed)} из {len(matches)} соответствий')
    plt.tight_layout()
    plt.show()

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

def draw_trajectory(positions, angles=None, image_labels=None):
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    plt.figure(figsize=(8,8))
    plt.plot(xs, ys, 'steelblue', linewidth=1.5)
    plt.scatter(xs, ys, color='steelblue', s=40)
    if angles:
        span = max(max(xs)-min(xs), max(ys)-min(ys))
        arrow_len = span*0.06 + 5
        for (x,y), a in zip(positions, angles):
            dx = math.cos(a) * arrow_len
            dy = math.sin(a) * arrow_len
            plt.annotate('', xy=(x+dx, y+dy), xytext=(x,y),
                         arrowprops=dict(arrowstyle='->', color='tomato', lw=1.5))
    labels = image_labels if image_labels else [str(i) for i in range(len(positions))]
    for i, (x,y) in enumerate(positions):
        plt.annotate(labels[i], (x,y), textcoords='offset points', xytext=(6,6),
                     fontsize=9, color='dimgray')
    plt.scatter([xs[0]], [ys[0]], color='green', s=100, label='Старт')
    plt.scatter([xs[-1]], [ys[-1]], color='red', s=100, label='Финиш')
    closure = math.sqrt((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2)
    plt.plot([xs[-1], xs[0]], [ys[-1], ys[0]], '--', color='gray', alpha=0.6,
             label=f'Ошибка замыкания: {closure:.1f}px')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.title('Траектория камеры')
    plt.xlabel('X (пиксели)')
    plt.ylabel('Y (пиксели)')
    plt.tight_layout()
    plt.show()

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

def draw_trajectory_generic(positions, image_labels=None,
                             title='Траектория', color='steelblue'):
    valid = [(i, p) for i, p in enumerate(positions) if p is not None]
    idxs  = [v[0] for v in valid]
    xs    = [v[1][0] for v in valid]
    ys    = [v[1][1] for v in valid]
    labels = image_labels if image_labels else [str(i) for i in range(len(positions))]
    plt.figure(figsize=(8,8))
    plt.plot(xs, ys, color=color, linewidth=1.5)
    plt.scatter(xs, ys, color=color, s=40)
    for idx, x, y in zip(idxs, xs, ys):
        plt.annotate(labels[idx], (x,y), textcoords='offset points', xytext=(6,6),
                     fontsize=9, color='dimgray')
    plt.scatter([xs[0]], [ys[0]], color='green', s=100, label='Старт')
    plt.scatter([xs[-1]], [ys[-1]], color='red',  s=100, label='Финиш')
    closure = math.sqrt((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2)
    plt.plot([xs[-1], xs[0]], [ys[-1], ys[0]], '--', color='gray', alpha=0.6,
             label=f'Ошибка замыкания: {closure:.1f}px')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.title(title)
    plt.xlabel('X (пиксели)')
    plt.ylabel('Y (пиксели)')
    plt.tight_layout()
    plt.show()

def draw_both_trajectories(obj_positions, cam_positions, image_labels=None):
    labels = image_labels if image_labels else \
             [str(i) for i in range(len(obj_positions))]
    fig, axes = plt.subplots(1, 2, figsize=(14,7))
    configs = [
        (obj_positions, 'darkorange', 'Траектория объекта'),
        (cam_positions, 'steelblue',  'Траектория камеры'),
    ]
    for ax, (positions, color, title) in zip(axes, configs):
        valid = [(i, p) for i, p in enumerate(positions) if p is not None]
        idxs  = [v[0] for v in valid]
        xs    = [v[1][0] for v in valid]
        ys    = [v[1][1] for v in valid]
        ax.plot(xs, ys, color=color, linewidth=1.5)
        ax.scatter(xs, ys, color=color, s=40)
        for idx, x, y in zip(idxs, xs, ys):
            ax.annotate(labels[idx], (x,y), textcoords='offset points', xytext=(6,6),
                        fontsize=9, color='dimgray')
        ax.scatter([xs[0]], [ys[0]], color='green', s=100, label='Старт')
        ax.scatter([xs[-1]], [ys[-1]], color='red',  s=100, label='Финиш')
        closure = math.sqrt((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2)
        ax.plot([xs[-1], xs[0]], [ys[-1], ys[0]], '--', color='gray', alpha=0.6,
                label=f'Ошибка замыкания: {closure:.1f}px')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_title(title)
        ax.set_xlabel('X (пиксели)')
        ax.set_ylabel('Y (пиксели)')
    plt.tight_layout()
    plt.show()

images = []
for i in range(1, 9):
    img = cv2.imread(f'sequence{i}.jpg')
    if img is None:
        raise FileNotFoundError(f'sequence{i}.jpg не найден')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    new_w = 1080
    new_h = int(h * new_w / w)
    img = cv2.resize(img, (new_w, new_h))
    images.append(img)

grays = [intensity_grayscale(img) for img in images]

processed = [gaussian_2d(gray, 1.2, 21) for gray in grays]

show_images(grays, 2, 4, suptitle='Исходные кадры')
plt.show()

show_images(processed, 2, 4, suptitle='Обработанные кадры')
plt.show()

all_keypoints = []
all_descriptors = []
for idx, img in enumerate(processed):
    kps, Ix, Iy = harris_keypoints(img, minor_size=3)
    kps = filter_isolated_points(kps, radius=30, min_neighbors=3)
    draw_keypoints(img, kps[:100])
    oriented = compute_keypoint_orientations(kps, Ix, Iy)
    valid_kp, descs = compute_sift_descriptors(oriented, Ix, Iy)
    all_keypoints.append(valid_kp)
    all_descriptors.append(descs)
    print(f'Кадр {idx+1}: точек {len(valid_kp)}')

all_matches = []
for i in range(len(processed)-1):
    print(f'\nПара {i+1} → {i+2}')
    matches = match_descriptors(all_keypoints[i], all_descriptors[i],
                                all_keypoints[i+1], all_descriptors[i+1])
    print(f'  Найдено матчей: {len(matches)}')
    all_matches.append(matches)
    draw_matches(processed[i], processed[i+1], matches)

transforms = []
for i, matches in enumerate(all_matches):
    print(f'\nТрансформация {i} → {i+1}:')
    angle, tx, ty, inliers = estimate_rotation_translation(matches)
    transforms.append((angle, tx, ty))

positions, angles = build_trajectory(transforms)
labels = [f'{i+1}' for i in range(len(positions))]
draw_trajectory(positions, angles, labels)

obj_positions, cam_positions, centroids = build_trajectories_from_keypoints(all_keypoints)
draw_trajectory_generic(obj_positions, labels, 'Траектория объекта', 'darkorange')
draw_trajectory_generic(cam_positions, labels, 'Траектория камеры', 'steelblue')
draw_both_trajectories(obj_positions, cam_positions, labels)

print('\nСводка:')
for i, (ang, tx, ty) in enumerate(transforms):
    print(f'{i}→{i+1}: угол={math.degrees(ang):6.2f}°, tx={tx:6.1f}, ty={ty:6.1f}')