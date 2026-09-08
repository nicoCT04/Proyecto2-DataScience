"""Extract visual features from the Bundesliga clips for the exploratory analysis.

For every clip in data/clips/ we sample a fixed number of frames and compute a
set of per-frame features that describe the video without any labels:

- brightness / contrast  (mean and std of the grayscale frame)
- saturation             (mean HSV saturation)
- green_ratio            (share of pixels classified as pitch)
- edge_density           (share of Canny-edge pixels: scene detail proxy)
- player_count           (non-green blobs of player size inside the pitch)
- motion                 (mean abs difference vs. the previous sampled frame)

The result is written to data/frame_features.csv (one row per sampled frame).
Frames are read on the fly and not stored.
"""

import csv
import os
import sys

import cv2
import numpy as np

CLIPS_DIR = "data/clips"
OUT_CSV = "data/frame_features.csv"
FRAMES_PER_CLIP = 10

# HSV thresholds for the grass; H is 0-180 in OpenCV.
GREEN_LO = np.array([35, 40, 40])
GREEN_HI = np.array([90, 255, 255])


def pitch_mask(hsv):
    """Binary mask of the grass pixels."""
    return cv2.inRange(hsv, GREEN_LO, GREEN_HI)


def count_players(frame, green):
    """Count player-sized non-green blobs that sit inside the pitch area.

    The pitch area is the green mask morphologically closed so that players
    (holes in the grass) get filled in; players are then the non-green pixels
    within that area. Returns the count and the blob bounding boxes.
    """
    area = cv2.morphologyEx(green, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    non_green = cv2.bitwise_and(area, cv2.bitwise_not(green))
    non_green = cv2.morphologyEx(non_green, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(non_green, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if 40 <= a <= 3000 and h >= w * 0.6 and h <= 120:
            boxes.append((x, y, w, h))
    return len(boxes), boxes


def frame_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = pitch_mask(hsv)
    edges = cv2.Canny(gray, 100, 200)
    players, _ = count_players(frame, green)
    return {
        "brightness": round(float(gray.mean()), 2),
        "contrast": round(float(gray.std()), 2),
        "saturation": round(float(hsv[:, :, 1].mean()), 2),
        "green_ratio": round(float(green.mean() / 255), 4),
        "edge_density": round(float(edges.mean() / 255), 4),
        "player_count": players,
    }, gray


def process_clip(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(total * 0.05, total * 0.95, FRAMES_PER_CLIP).astype(int)
    rows, prev_gray = [], None
    for k, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        feats, gray = frame_features(frame)
        motion = 0.0 if prev_gray is None else round(
            float(cv2.absdiff(gray, prev_gray).mean()), 3)
        prev_gray = gray
        feats.update({"frame_pos": k, "time_s": round(float(fi) / 25, 2),
                      "motion": motion})
        rows.append(feats)
    cap.release()
    return rows


def main(limit=None):
    files = sorted(f for f in os.listdir(CLIPS_DIR) if f.endswith(".mp4"))
    if limit:
        files = files[:limit]
    fields = ["file", "match_id", "clip_idx", "frame_pos", "time_s",
              "brightness", "contrast", "saturation", "green_ratio",
              "edge_density", "player_count", "motion"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for j, name in enumerate(files):
            match_id, clip_idx = name.replace(".mp4", "").split("_")
            for r in process_clip(os.path.join(CLIPS_DIR, name)):
                r.update({"file": name, "match_id": match_id, "clip_idx": int(clip_idx)})
                writer.writerow(r)
            if (j + 1) % 20 == 0:
                print(f"  {j + 1}/{len(files)} clips", flush=True)
    print(f"Done -> {OUT_CSV}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
