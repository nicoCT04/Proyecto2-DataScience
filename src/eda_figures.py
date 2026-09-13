"""Generate the exploratory figures for the data-description section.

Reads data/clips_inventory.csv and data/train.csv and writes the PNG figures
used in the report into outputs/. train.csv is used only as context: its
annotations describe 12 full matches (not the 200 clips) and give the pass
taxonomy and temporal density of the events.
"""

import ast

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
    ax.set(xlabel="Tamaño del clip (MB)", ylabel="Número de clips",
           title="Distribución del tamaño de los 200 clips")
    fig.tight_layout(); fig.savefig("outputs/01_clip_size_hist.png"); plt.close(fig)

    # Fig 2 - event types in train.csv (context)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    train["event"].value_counts().plot(kind="bar", color="#2f855a",
                                        edgecolor="white", ax=ax)
    ax.set(xlabel="Tipo de evento", ylabel="Frecuencia",
           title="Eventos anotados en train.csv (contexto)")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig("outputs/02_event_types.png"); plt.close(fig)

    # Fig 3 - 'play' events: action type (pass / cross) x context
    # Each play carries two attributes: the action and its context. Counting
    # the attributes independently mixes passes and crosses, so cross them.
    plays = train[train["event"] == "play"]
    attrs = plays["event_attributes"].apply(parse_attrs)
    table = pd.crosstab(attrs.str[1], attrs.str[0])
    table = table.loc[table.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    table.plot(kind="barh", stacked=True, color=["#c05621", "#2b6cb0"],
               edgecolor="white", ax=ax)
    for i, total in enumerate(table.sum(axis=1)):
        ax.text(total, i, f" {total:,}", va="center", fontsize=9)
    ax.set(xlabel="Frecuencia", ylabel="Contexto",
           title="Jugadas (play) por acción y contexto")
    ax.set_xlim(0, table.sum(axis=1).max() * 1.15)
    ax.legend(title="Acción")
    fig.tight_layout(); fig.savefig("outputs/03_play_attributes.png"); plt.close(fig)

    # Fig 4 - temporal density of passes in the busiest match
    passes = plays[plays["event_attributes"].apply(lambda x: "pass" in parse_attrs(x))]
    video = passes["video_id"].value_counts().index[0]
    sub = passes[passes["video_id"] == video]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(sub["time"] / 60, bins=30, color="#6b46c1", edgecolor="white")
    ax.set(xlabel="Minuto del video", ylabel="Pases",
           title=f"Densidad temporal de pases - video {video}")
    fig.tight_layout(); fig.savefig("outputs/04_pass_temporal_density.png"); plt.close(fig)

    print("Figures written to outputs/")


if __name__ == "__main__":
    main()
