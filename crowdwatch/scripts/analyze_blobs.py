import cv2
import numpy as np

video_path = "data/uploads/4k_video_Crowd_waiting_to_get_into_queue_evening_shot_tear_temple_Free_Video_Stock_Footage.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Current logic in clip_ebc.py
k_size = max(11, int(min(h, w) * 0.045) | 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
saliency = cv2.GaussianBlur(cv2.add(bh, th), (11, 11), 2.0)

p_med = float(np.median(saliency))
p_std = float(np.std(saliency))
thresh = p_med + 0.35 * p_std
ped_signal = np.maximum(saliency - thresh, 0.0)

_, bin_mask = cv2.threshold(ped_signal, 0.40 * p_std, 255, cv2.THRESH_BINARY)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8))

print(f"k_size: {k_size}, p_med: {p_med:.2f}, p_std: {p_std:.2f}, num connected components: {num_labels - 1}")

areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
print(f"Areas: min={min(areas)}, median={np.median(areas):.1f}, max={max(areas)}, sum={sum(areas)}")
print(f"Total frame pixels: {w*h}")
print(f"Foreground blob coverage: {sum(areas)/(w*h)*100:.1f}%")

# Now let's see how many people the formula calculated:
total_count = 0.0
for i in range(1, num_labels):
    cx, cy = centroids[i]
    area = stats[i, cv2.CC_STAT_AREA]
    norm_y = cy / float(h)
    expected_person_area = 480.0 + 1050.0 * (norm_y ** 1.3)
    if area >= max(60.0, expected_person_area * 0.15):
        ppl = max(0.5, area / expected_person_area)
        total_count += ppl

print(f"Calculated headcount: {total_count:.1f}")
