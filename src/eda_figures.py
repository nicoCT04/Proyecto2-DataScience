"""Generate the exploratory figures for the data-description section.

Reads data/clips_inventory.csv and data/train.csv and writes the PNG figures
used in the report into outputs/. train.csv is used only as context: its
annotations describe 12 full matches (not the 200 clips) and give the pass
taxonomy and temporal density of the events.
"""

import ast
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def parse_attrs(value):
    """Parse an event_attributes cell like "['pass', 'openplay']" into a list."""
    try:
        return ast.literal_eval(value) if isinstance(value, str) and value.strip() else []
    except (ValueError, SyntaxError):
        return []


def main():
    inv = pd.read_csv("data/clips_inventory.csv")
    train = pd.read_csv("data/train.csv")

    # Fig 1 - clip size distribution
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(inv["size_mb"], bins=20, color="#2b6cb0", edgecolor="white")
    ax.set(xlabel="Clip size (MB)", ylabel="Number of clips",
           title="Size distribution of the 200 clips")
    fig.tight_layout(); fig.savefig("outputs/01_clip_size_hist.png"); plt.close(fig)

    # Fig 2 - event types in train.csv (context)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    train["event"].value_counts().plot(kind="bar", color="#2f855a",
                                        edgecolor="white", ax=ax)
    ax.set(xlabel="Event type", ylabel="Frequency",
           title="Annotated events in train.csv (context)")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig("outputs/02_event_types.png"); plt.close(fig)

    # Fig 3 - attributes of 'play' events (pass taxonomy)
    plays = train[train["event"] == "play"]
    counts = Counter()
    for attrs in plays["event_attributes"]:
        counts.update(parse_attrs(attrs))
    series = pd.Series(dict(counts)).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    series.plot(kind="barh", color="#c05621", edgecolor="white", ax=ax)
    ax.set(xlabel="Frequency", title="Attributes of 'play' events (pass types)")
    fig.tight_layout(); fig.savefig("outputs/03_play_attributes.png"); plt.close(fig)

    # Fig 4 - temporal density of passes in the busiest match
    passes = plays[plays["event_attributes"].apply(lambda x: "pass" in parse_attrs(x))]
    video = passes["video_id"].value_counts().index[0]
    sub = passes[passes["video_id"] == video]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(sub["time"] / 60, bins=30, color="#6b46c1", edgecolor="white")
    ax.set(xlabel="Match minute", ylabel="Passes",
           title=f"Temporal density of passes - video {video}")
    fig.tight_layout(); fig.savefig("outputs/04_pass_temporal_density.png"); plt.close(fig)

    print("Figures written to outputs/")


if __name__ == "__main__":
    main()
