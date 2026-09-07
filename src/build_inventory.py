"""Build a metadata inventory of the Bundesliga clips using ffprobe.

Reads every .mp4 in data/clips/ and writes data/clips_inventory.csv with one
row per clip (resolution, fps, duration, size, match id). The clips themselves
are not versioned; they are re-downloaded from the Hugging Face dataset.
"""

import csv
import json
import os
import subprocess

CLIPS_DIR = "data/clips"
OUT_CSV = "data/clips_inventory.csv"


def probe(path):
    """Return the ffprobe stream/format info for a single video."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames:format=duration,size",
        "-of", "json", path,
    ]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    stream, fmt = data["streams"][0], data["format"]
    num, den = (stream["avg_frame_rate"].split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else 0.0
    return stream, fmt, fps


def main():
    files = sorted(f for f in os.listdir(CLIPS_DIR) if f.endswith(".mp4"))
    rows = []
    for name in files:
        stream, fmt, fps = probe(os.path.join(CLIPS_DIR, name))
        match_id, clip_idx = name.replace(".mp4", "").split("_")
        rows.append({
            "file": name,
            "match_id": match_id,
            "clip_idx": int(clip_idx),
            "width": stream["width"],
            "height": stream["height"],
            "fps": round(fps, 3),
            "duration_s": round(float(fmt["duration"]), 3),
            "size_mb": round(int(fmt["size"]) / 1e6, 2),
        })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} clips to {OUT_CSV}")


if __name__ == "__main__":
    main()
