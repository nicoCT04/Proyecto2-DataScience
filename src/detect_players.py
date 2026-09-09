"""Detect players with a pretrained YOLOv8 model and study team colors.

For a sample of frames per clip we run person detection (COCO class 0), count
the players, and take the dominant jersey colour of each detection (mean hue of
the torso crop). Jersey hues are then clustered into two teams with k-means so
we can check whether the two sides are visually separable.

Outputs:
- data/player_detections.csv   one row per sampled frame (player_count, ...)
- data/jersey_colors.csv        one row per detected player (hue, sat, val)
- outputs/06_player_detection_example.png   an annotated frame
- outputs/07_jersey_hue_clusters.png        team colour clustering
"""

import csv
import os

import cv2
import numpy as np
from sklearn.cluster import KMeans

CLIPS_DIR = "data/clips"
FRAMES_PER_CLIP = 3
CONF = 0.35


def torso_hsv(frame, box):
    """Mean HSV of the upper-middle of a person box (the jersey region)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h = y2 - y1
    crop = frame[y1 + int(0.15 * h):y1 + int(0.55 * h), x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hsv = hsv[hsv[:, 1] > 40]  # drop washed-out / white pixels for a stable hue
    if len(hsv) < 20:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return hsv.mean(axis=0)


def main():
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    files = sorted(f for f in os.listdir(CLIPS_DIR) if f.endswith(".mp4"))
    det_rows, jersey_rows = [], []
    example_saved = False

    for j, name in enumerate(files):
        match_id, clip_idx = name.replace(".mp4", "").split("_")
        cap = cv2.VideoCapture(os.path.join(CLIPS_DIR, name))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for k, fi in enumerate(np.linspace(total * 0.2, total * 0.8, FRAMES_PER_CLIP).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            res = model(frame, classes=[0], conf=CONF, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
            det_rows.append({"file": name, "match_id": match_id,
                             "clip_idx": int(clip_idx), "frame_pos": k,
                             "player_count": len(boxes)})
            for b in boxes:
                hsv = torso_hsv(frame, b)
                if hsv is not None:
                    jersey_rows.append({"file": name, "match_id": match_id,
                                        "hue": round(float(hsv[0]), 1),
                                        "sat": round(float(hsv[1]), 1),
                                        "val": round(float(hsv[2]), 1)})
            if not example_saved and len(boxes) >= 15:
                annot = frame.copy()
                for b in boxes:
                    x1, y1, x2, y2 = [int(v) for v in b]
                    cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.imwrite("outputs/06_player_detection_example.png", annot)
                example_saved = True
        cap.release()
        if (j + 1) % 25 == 0:
            print(f"  {j + 1}/{len(files)} clips", flush=True)

    with open("data/player_detections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "match_id", "clip_idx",
                                          "frame_pos", "player_count"])
        w.writeheader(); w.writerows(det_rows)
    with open("data/jersey_colors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "match_id", "hue", "sat", "val"])
        w.writeheader(); w.writerows(jersey_rows)

    # Team colour clustering on jersey hue (mapped to a circle so red wraps).
    hues = np.array([r["hue"] for r in jersey_rows])
    feats = np.column_stack([np.cos(np.radians(hues * 2)), np.sin(np.radians(hues * 2))])
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(feats)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for c, col in zip([0, 1], ["#c53030", "#2b6cb0"]):
        ax.hist(hues[km.labels_ == c], bins=36, range=(0, 180), alpha=0.7,
                color=col, label=f"Equipo {c + 1}")
    ax.set(xlabel="Matiz de camiseta (H, 0-180)", ylabel="Jugadores detectados",
           title="Separación de equipos por color de camiseta (k-means)")
    ax.legend()
    fig.tight_layout(); fig.savefig("outputs/07_jersey_hue_clusters.png"); plt.close(fig)

    print(f"Detections: {len(det_rows)} frames | players: {len(jersey_rows)}")


if __name__ == "__main__":
    main()
