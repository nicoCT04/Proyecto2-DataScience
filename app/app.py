"""Aplicación del Proyecto 2: clasificación de acciones en video de la Bundesliga.

Ejecutar desde la raíz del repositorio:  streamlit run app/app.py

Páginas:
  1. Resumen            la historia del proyecto: problema, datos, hallazgos y resultado
  2. Explorar datos     variables de los 762 subclips tras la limpieza (las de los modelos)
  3. Clasificar video   sube un video o usa un ejemplo; elige uno, varios o todos los modelos
  4. Rendimiento        métricas en el partido de prueba (se pueden ocultar)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # PyTorch + XGBoost en macOS (ver src/inference.py)

import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import theme as th  # noqa: E402
from theme import CLASS_COLORS, CSS, chip, kpi  # noqa: E402

RES = ROOT / "outputs" / "results"
SAMPLES = Path(__file__).resolve().parent / "samples"
LABELS = ["background", "challenge", "play", "throwin"]
LABELS_ES = {"background": "Fondo", "challenge": "Disputa", "play": "Pase", "throwin": "Saque de banda"}
MODELS = ["logreg", "rf", "xgb", "gru"]
NAMES = {"logreg": "Regresión logística", "rf": "Random Forest", "xgb": "XGBoost",
         "gru": "GRU (ResNet18)", "todos": "Todos (promedio)"}
CLASS_ORDER = {"label": LABELS}
ES_COLORS = {LABELS_ES[k]: v for k, v in CLASS_COLORS.items()}
LEGEND_TOP = dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text="")

# Variables de los modelos agrupadas por familia, con su descripción
FAMILIES = {
    "Color e iluminación": ("brightness", "contrast", "saturation", "green_ratio", "green_window"),
    "Bordes y cortes": ("edge_density", "shot_cut"),
    "Movimiento global": ("motion",),
    "Cámara y flujo óptico": ("cam_motion", "cam_accel", "player_flow", "flow_coherence"),
    "Jugadores (YOLO)": ("players",),
    "Balón (YOLO)": ("ball",),
}
DESCRIPTIONS = {
    "green_window": "Cobertura media de césped en la ventana (0-1); bajo 0.30 la toma no muestra la cancha",
    "brightness": "Brillo: media de gris (0-255)", "contrast": "Contraste: desviación de gris",
    "saturation": "Saturación media (canal S de HSV)", "green_ratio": "Proporción de píxeles de césped por cuadro",
    "edge_density": "Proporción de píxeles de borde (Canny)", "shot_cut": "1 si hay un corte de cámara en la ventana",
    "motion": "Diferencia absoluta media entre cuadros consecutivos",
    "motion_peak_pos": "Posición relativa (0-1) del pico de movimiento dentro de la ventana",
    "cam_motion": "Paneo de cámara: mediana del flujo óptico de Farnebäck (px/cuadro)",
    "cam_accel": "Cambio máximo del paneo entre cuadros (arranques y frenadas de cámara)",
    "player_flow": "Movimiento propio de jugadores y balón: flujo óptico menos el paneo",
    "flow_coherence": "Coherencia de dirección del flujo (1 = todo se mueve igual)",
    "players_mean": "Personas detectadas por YOLOv8n por cuadro", "players_std": "Variación del conteo de personas",
    "players_spread": "Dispersión espacial de los jugadores (x horizontal, y vertical; fracción del cuadro)",
    "ball_visible": "Fracción de cuadros en que YOLO ve el balón", "ball_speed": "Velocidad del balón (fracción del cuadro por paso)",
    "ball_y_mean": "Altura media del balón en la imagen (0 arriba, 1 abajo)",
    "ball_edge_dist": "Distancia mínima del balón al borde de la imagen",
    "ball_player_dist": "Distancia media del balón al jugador más cercano",
}

st.set_page_config(page_title="Detección de acciones · Bundesliga", page_icon="⚽", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Datos en caché
# ----------------------------------------------------------------------------
@st.cache_data
def load_features():
    df = pd.read_csv(ROOT / "data" / "subclip_features.csv")
    df["conjunto"] = df["split"].map({"train": "Entrenamiento", "valid": "Prueba"})
    df["clase"] = df["label"].map(LABELS_ES)
    return df


@st.cache_data
def load_results():
    out = {"metrics": pd.read_csv(RES / "metrics.csv"), "per_class": pd.read_csv(RES / "per_class.csv"),
           "importance": pd.read_csv(RES / "feature_importance.csv"),
           "predictions": pd.read_csv(RES / "test_predictions.csv"),
           "curves": pd.read_csv(RES / "tuning_curves.csv"),
           "tuning": json.loads((RES / "tuning.json").read_text())}
    out["confusion"] = {k: pd.read_csv(RES / f"confusion_{k}.csv", index_col=0) for k in MODELS}
    return out


@st.cache_resource(show_spinner="Cargando modelos y redes preentrenadas…")
def load_predictor():
    import inference
    return inference.Predictor()


def feature_cols(df):
    import modeling
    return [c for c in modeling.feature_columns(df) if c not in ("conjunto", "clase")]


def family_of(col):
    for fam, prefixes in FAMILIES.items():
        if col.startswith(prefixes):
            return fam
    return "Otras"


def describe(col):
    for key in sorted(DESCRIPTIONS, key=len, reverse=True):
        if col.startswith(key):
            suffix = col[len(key):].strip("_")
            stat = {"mean": "media", "std": "desviación", "max": "máximo", "min": "mínimo",
                    "p90_max": "máximo del percentil 90", "x": "eje x", "y": "eje y"}.get(suffix, suffix)
            return DESCRIPTIONS[key] + (f" — {stat} en la ventana" if stat else "")
    return col


def best_model(metrics):
    return metrics.sort_values("f1_macro").iloc[-1]["modelo"]


# ----------------------------------------------------------------------------
# 1. Resumen
# ----------------------------------------------------------------------------
def page_summary():
    df, R = load_features(), load_results()
    m = R["metrics"].set_index("modelo")
    best = best_model(R["metrics"])
    st.title("¿Se puede reconocer un pase en video de fútbol?")
    st.markdown(
        "Reto 21 · CC3084 Data Science · UVG. Clasificamos ventanas cortas de video de la Bundesliga en "
        f"cuatro acciones: {chip('play', 'Pase')} {chip('challenge', 'Disputa')} "
        f"{chip('throwin', 'Saque de banda')} {chip('background', 'Fondo')}", unsafe_allow_html=True)

    c = st.columns(4)
    c[0].markdown(kpi(f"{len(df)}", "subclips etiquetados (762 ventanas de 1.6 s)"), unsafe_allow_html=True)
    c[1].markdown(kpi(f"{df.match_id.nunique()}", "partidos: 7 para entrenar, 1 de prueba"), unsafe_allow_html=True)
    c[2].markdown(kpi(f"{len(feature_cols(df))}", "variables por ventana + 16×512 embeddings"),
                  unsafe_allow_html=True)
    c[3].markdown(kpi(f"{m.loc[best, 'f1_macro']:.2f}", f"F1 macro de {NAMES[best]} en el partido nuevo "
                                                        "(azar = 0.25)"), unsafe_allow_html=True)

    st.subheader("La historia en cinco pasos")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        steps = [
            ("1 · Los datos", "Subclips recortados alrededor de eventos anotados por la DFL. Cada uno se reduce a "
                              "una ventana de 1.62 s (16 cuadros) que termina justo después de la acción."),
            ("2 · Preprocesamiento", "Se recorta el marcador, se mide césped, color, bordes, flujo óptico "
                                     "(separando el paneo de cámara del movimiento propio), jugadores y balón con "
                                     "YOLOv8 y apariencia con ResNet18. La duración del recorte se excluyó: delataba "
                                     "la clase sin describir la acción."),
            ("3 · Lo que mostraron los datos", "El saque de banda es la acción más quieta: la cámara casi no se "
                                               "mueve y hay menos flujo de jugadores. Pase y disputa se parecen "
                                               "mucho entre sí. Y cada partido se ve distinto: el de prueba muestra "
                                               "menos césped (0.57) que 6 de los 7 de entrenamiento (0.74-0.88)."),
            ("4 · Los modelos", "Tres modelos tabulares sobre las variables (regresión logística, Random Forest, "
                                "XGBoost) y una GRU sobre la secuencia de embeddings, ajustados con validación "
                                "cruzada por partido."),
            ("5 · El resultado", f"{NAMES[best]} es el mejor en el partido nuevo (F1 {m.loc[best, 'f1_macro']:.2f}): "
                                 "reconoce saques y pases mejor que el azar, pero confunde la disputa con el fondo. "
                                 "La GRU no generaliza y casi siempre predice saque de banda."),
        ]
        for title, text in steps:
            st.markdown(f'<div class="step"><b>{title}</b><br>{text}</div>', unsafe_allow_html=True)
    with right:
        tab1, tab2 = st.tabs(["Movimiento por clase", "Cada partido se ve distinto"])
        with tab1:
            fig = px.box(df, x="clase", y="cam_motion_mean", color="clase", color_discrete_map=ES_COLORS,
                         category_orders={"clase": [LABELS_ES[k] for k in LABELS]}, points=False,
                         labels={"cam_motion_mean": "Paneo de cámara medio (px/cuadro)", "clase": ""},
                         title="El saque de banda casi no mueve la cámara")
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            g = df.groupby(["match_id", "conjunto"], as_index=False)["green_window"].mean()
            g = g.sort_values("green_window")
            fig = px.bar(g, x="green_window", y="match_id", color="conjunto", orientation="h",
                         color_discrete_map={"Entrenamiento": th.NEUTRAL, "Prueba": th.HIGHLIGHT},
                         labels={"green_window": "Cobertura media de césped", "match_id": "Partido", "conjunto": ""},
                         title="Césped visible por partido: el encuadre cambia entre encuentros")
            fig.update_layout(height=380, legend=LEGEND_TOP)
            st.plotly_chart(fig, use_container_width=True)
    st.caption("Use el menú de la izquierda para explorar los datos, clasificar un video o revisar el rendimiento.")


# ----------------------------------------------------------------------------
# 2. Explorar datos
# ----------------------------------------------------------------------------
def page_explore():
    df = load_features()
    cols = feature_cols(df)
    st.title("Explorar los datos")
    st.markdown("Las **762 ventanas** después de la limpieza, con las **48 variables** con que se entrenaron los "
                "modelos tabulares. No hay valores faltantes; la duración del subclip y el contexto previo "
                "(`duration_s`, `lead_s`) se muestran aparte porque se excluyeron de los modelos.")

    f1, f2, f3 = st.columns([1, 1.4, 1.6])
    sets = f1.multiselect("Conjunto", ["Entrenamiento", "Prueba"], default=["Entrenamiento", "Prueba"])
    classes = f2.multiselect("Clases", LABELS, default=LABELS, format_func=LABELS_ES.get)
    matches = f3.multiselect("Partidos", sorted(df.match_id.unique()), default=sorted(df.match_id.unique()))
    d = df[df.conjunto.isin(sets) & df.label.isin(classes) & df.match_id.isin(matches)]
    if d.empty:
        st.warning("Ningún subclip cumple los filtros.")
        return
    counts = d.groupby("clase").size()
    st.markdown(" ".join(chip(k, f"{LABELS_ES[k]}: {counts.get(LABELS_ES[k], 0)}") for k in LABELS)
                + f'<span class="muted"> · {len(d)} subclips</span>', unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["Una variable", "Relación entre dos", "Por partido", "Correlaciones", "Tabla"])
    order = {"clase": [LABELS_ES[k] for k in LABELS if k in classes]}
    with t1:
        a, b = st.columns([1, 2.4])
        fam = a.selectbox("Familia", list(FAMILIES), index=3)
        options = [c for c in cols if family_of(c) == fam] + (["duration_s", "lead_s"] if fam == "Color e iluminación" else [])
        var = a.selectbox("Variable", options, index=0)
        kind = a.radio("Gráfica", ["Caja", "Histograma", "Violín"], horizontal=True)
        a.info(describe(var) if var not in ("duration_s", "lead_s") else
               "Variable del recorte (excluida de los modelos por filtrar la etiqueta).")
        stats = d.groupby("clase")[var].describe()[["mean", "50%", "std"]].round(3)
        a.dataframe(stats.rename(columns={"mean": "media", "50%": "mediana", "std": "desv."}), use_container_width=True)
        common = dict(color="clase", color_discrete_map=ES_COLORS, category_orders=order, labels={var: var, "clase": ""})
        if kind == "Caja":
            fig = px.box(d, x="clase", y=var, points="outliers", hover_data=["file", "match_id"], **common)
        elif kind == "Violín":
            fig = px.violin(d, x="clase", y=var, box=True, **common)
        else:
            fig = px.histogram(d, x=var, barmode="overlay", opacity=0.6, nbins=40, **common)
        fig.update_layout(height=470, title=f"{var} por clase", legend=LEGEND_TOP)
        b.plotly_chart(fig, use_container_width=True)
    with t2:
        a, b, c = st.columns(3)
        x = a.selectbox("Eje x", cols, index=cols.index("cam_motion_mean"))
        y = b.selectbox("Eje y", cols, index=cols.index("player_flow_mean"))
        logx = c.toggle("Escala logarítmica", value=False)
        fig = px.scatter(d, x=x, y=y, color="clase", color_discrete_map=ES_COLORS, category_orders=order,
                         symbol="conjunto", hover_data=["file", "match_id"], log_x=logx, log_y=logx, opacity=0.75,
                         labels={"clase": "Clase", "conjunto": "Conjunto"})
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color=th.SURFACE)))
        fig.update_layout(height=520, title=f"{y} vs. {x}")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Las clases se superponen mucho: ninguna pareja de variables las separa, lo que anticipa el "
                   "desempeño modesto de los modelos.")
    with t3:
        var = st.selectbox("Variable", cols, index=cols.index("green_window"), key="var_match")
        fig = px.box(d, x="match_id", y=var, color="conjunto",
                     color_discrete_map={"Entrenamiento": th.NEUTRAL, "Prueba": th.HIGHLIGHT},
                     labels={"match_id": "Partido", "conjunto": ""}, points=False)
        fig.update_layout(height=450, title=f"{var} por partido", legend=LEGEND_TOP)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Cada partido tiene su propia iluminación, encuadre y uniformes. Por eso la validación se hace "
                   "por partido: mezclar ventanas de un mismo encuentro en entrenamiento y prueba inflaría las cifras.")
    with t4:
        fam_sel = st.multiselect("Familias", list(FAMILIES), default=["Movimiento global", "Cámara y flujo óptico",
                                                                      "Jugadores (YOLO)", "Balón (YOLO)"])
        sub = [c for c in cols if family_of(c) in fam_sel]
        if len(sub) > 1:
            corr = d[sub].corr().round(2)
            fig = px.imshow(corr, color_continuous_scale=[[0, "#c0392b"], [0.5, "#f0efec"], [1, "#1c5cab"]],
                            zmin=-1, zmax=1, aspect="auto", title="Correlación de Pearson")
            fig.update_layout(height=620)
            st.plotly_chart(fig, use_container_width=True)
    with t5:
        show = ["file", "conjunto", "clase", "match_id"] + cols
        st.dataframe(d[show], use_container_width=True, height=420)
        st.download_button("Descargar CSV filtrado", d[show].to_csv(index=False).encode(), "subclips_filtrados.csv")
        with st.expander("Diccionario de variables"):
            st.dataframe(pd.DataFrame({"variable": cols, "familia": [family_of(c) for c in cols],
                                       "descripción": [describe(c) for c in cols]}),
                         use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# 3. Clasificar video
# ----------------------------------------------------------------------------
def prob_chart(tab, title):
    long = tab.rename(index=NAMES).reset_index(names="modelo").melt(id_vars="modelo", var_name="label",
                                                                    value_name="prob")
    long["clase"] = long["label"].map(LABELS_ES)
    fig = px.bar(long, y="modelo", x="prob", color="clase", orientation="h", color_discrete_map=ES_COLORS,
                 category_orders={"clase": [LABELS_ES[k] for k in LABELS], "modelo": list(tab.rename(index=NAMES).index)},
                 labels={"prob": "Probabilidad", "modelo": "", "clase": ""}, text_auto=".0%")
    fig.update_traces(marker_line_color=th.SURFACE, marker_line_width=2, textfont_color="white",
                      textposition="inside")
    fig.update_layout(uniformtext=dict(minsize=10, mode="hide"))
    fig.update_layout(barmode="stack", height=110 + 55 * len(tab), title=title, xaxis_tickformat=".0%",
                      xaxis_range=[0, 1], legend=LEGEND_TOP)
    return fig


def timeline_chart(out, model):
    fig = go.Figure()
    for lab in LABELS:
        fig.add_trace(go.Scatter(x=out["t_s"], y=out[f"{model}_{lab}"], name=LABELS_ES[lab], mode="lines+markers",
                                 line=dict(color=CLASS_COLORS[lab], width=2), marker=dict(size=7),
                                 hovertemplate="%{x:.1f} s · %{y:.0%}"))
    for t in out.loc[~out["field_view"], "t_s"]:
        fig.add_vrect(x0=t - 0.25, x1=t + 0.25, fillcolor=th.NEUTRAL, opacity=0.25, line_width=0)
    fig.update_layout(height=380, hovermode="x unified", yaxis_tickformat=".0%", yaxis_range=[0, 1],
                      xaxis_title="Segundo del video (fin de la ventana)", yaxis_title="Probabilidad",
                      title=f"Probabilidad por clase a lo largo del video — {NAMES[model]}",
                      legend=LEGEND_TOP)
    return fig


def page_classify():
    import inference
    import preprocessing as pp

    st.title("Clasificar un video")
    st.markdown("Suba un video `.mp4` de una transmisión o use un ejemplo del partido de prueba. El recorte, el "
                "muestreo de cuadros y la extracción de variables ocurren automáticamente, igual que en el "
                "entrenamiento.")
    samples = pd.read_csv(SAMPLES / "samples.csv") if (SAMPLES / "samples.csv").exists() else pd.DataFrame()

    left, right = st.columns([1, 1.3], gap="large")
    with left:
        source = st.radio("Video", ["Ejemplo del partido de prueba", "Subir un video"], horizontal=True)
        true_label, path = None, None
        if source.startswith("Ejemplo") and not samples.empty:
            pick = st.selectbox("Ejemplo", samples.index,
                                format_func=lambda i: f"{LABELS_ES[samples.label[i]]} — {samples.file[i]}")
            path, true_label = SAMPLES / samples.file[pick], samples.label[pick]
            st.caption("Ejemplos elegidos entre los que XGBoost acierta (uno por clase) para ilustrar el flujo; el "
                       "desempeño real está en la página de rendimiento.")
        else:
            up = st.file_uploader("Video (.mp4, .mov)", type=["mp4", "mov", "avi", "mkv"])
            if up is not None:
                tmp = Path(tempfile.gettempdir()) / "proyecto2_uploads"
                tmp.mkdir(exist_ok=True)
                path = tmp / Path(up.name).name
                path.write_bytes(up.getbuffer())
        if path is None:
            st.info("Elija un ejemplo o suba un video para empezar.")
            return
        st.video(str(path))
        info = pp.video_info(path)
        event_t = inference.subclip_event_time(path, info)
        st.caption(f"{info.width}×{info.height} · {info.fps:.0f} fps · {info.duration_s:.1f} s"
                   + (f" · evento anotado en {event_t:.2f} s" if event_t is not None else ""))

    with right:
        choice = st.radio("Modelos", ["Todos", "Elegir"], horizontal=True)
        models = MODELS if choice == "Todos" else st.multiselect(
            "Modelos a usar", MODELS, default=["xgb"], format_func=NAMES.get)
        if not models:
            st.warning("Elija al menos un modelo.")
            return
        modes = ["Momento del evento" if event_t is not None else "Centro del video", "Momento concreto",
                 "Recorrer todo el video"]
        mode = st.radio("¿Qué clasificar?", modes, horizontal=True)
        t_sel = None
        if mode == "Momento concreto":
            hi = max(info.duration_s - pp.AFTER_S, 0.1)
            t_sel = st.slider("Segundo (fin de la ventana de 1.6 s)", 0.0, float(hi),
                              float(event_t if event_t is not None else hi / 2), 0.1)
        stride = st.select_slider("Paso entre ventanas (s)", [0.25, 0.5, 1.0, 2.0], value=0.5) \
            if mode.startswith("Recorrer") else None
        if not st.button("Clasificar", type="primary", use_container_width=True):
            return
        predictor = load_predictor()
        if mode.startswith("Recorrer"):
            bar = st.progress(0.0, text="Analizando ventanas…")
            out, _ = inference.scan_video(path, predictor, models, stride,
                                          progress=lambda i, n: bar.progress(i / n, text=f"Ventana {i} de {n}"))
            bar.empty()
            keys = models + (["todos"] if len(models) > 1 else [])
            shown = st.selectbox("Modelo en la línea de tiempo", keys, index=len(keys) - 1, format_func=NAMES.get)
            st.plotly_chart(timeline_chart(out, shown), use_container_width=True)
            summary = pd.DataFrame({NAMES[k]: out[f"{k}_pred"].map(LABELS_ES).value_counts() for k in keys}).fillna(0)
            st.markdown("**Ventanas asignadas a cada clase**")
            st.dataframe(summary.astype(int), use_container_width=True)
            st.caption(f"{len(out)} ventanas · {out.extract_s.mean():.1f} s de extracción por ventana · las franjas "
                       "grises marcan ventanas sin vista de cancha (césped < 30 %), donde la predicción no es fiable.")
            return
        with st.spinner("Extrayendo variables (flujo óptico, YOLO, ResNet18)…"):
            tab, feats = inference.classify_moment(path, predictor, t_sel, models)
        final = "todos" if "todos" in tab.index else tab.index[0]
        pred = tab.loc[final].idxmax()
        verdict = f"Predicción ({NAMES[final]}): {chip(pred, LABELS_ES[pred])} con {tab.loc[final, pred]:.0%}"
        if true_label:
            verdict += f" &nbsp;·&nbsp; etiqueta real: {chip(true_label, LABELS_ES[true_label])}"
        st.markdown(verdict, unsafe_allow_html=True)
        if not feats["field_view"]:
            st.warning("La ventana casi no muestra la cancha (primer plano o repetición); la predicción es poco fiable.")
        st.plotly_chart(prob_chart(tab, "Probabilidad que asigna cada modelo"), use_container_width=True)
        agree = tab.drop(index="todos", errors="ignore").idxmax(axis=1)
        st.markdown("**Voto de cada modelo:** " + " ".join(
            f"{NAMES[k]} → {chip(v, LABELS_ES[v])}" for k, v in agree.items()), unsafe_allow_html=True)
        st.caption(f"Ventana {feats['t_s'] - pp.BEFORE_S:.2f}–{feats['t_s'] + pp.AFTER_S:.2f} s · extracción "
                   f"{feats['extract_s']:.1f} s")
        with st.expander("Variables extraídas de esta ventana"):
            df = load_features()
            row = pd.Series({c: feats[c] for c in feature_cols(df)}, name="valor").to_frame()
            row["mediana en entrenamiento"] = df[df.split == "train"][row.index].median()
            st.dataframe(row.round(4), use_container_width=True)


# ----------------------------------------------------------------------------
# 4. Rendimiento
# ----------------------------------------------------------------------------
def page_performance():
    R = load_results()
    metrics = R["metrics"].set_index("modelo")
    best = best_model(R["metrics"])
    st.title("Rendimiento de los modelos")
    st.markdown("Todas las cifras se miden en el **partido de prueba 3c993bd2** (362 subclips), que ningún modelo "
                "vio durante el ajuste. La métrica principal es el **F1 macro**: pondera igual las cuatro clases.")
    show = st.toggle("Mostrar métricas de rendimiento", value=True)
    if not show:
        st.info("Métricas ocultas. Active el interruptor para verlas.")
        return
    selected = st.multiselect("Modelos a comparar", MODELS, default=MODELS, format_func=NAMES.get)
    if not selected:
        return
    m = metrics.loc[selected]
    cols = st.columns(len(selected))
    for c, k in zip(cols, selected):
        c.markdown(kpi(f"{m.loc[k, 'f1_macro']:.3f}", f"{NAMES[k]} · F1 macro (exactitud "
                                                       f"{m.loc[k, 'accuracy']:.0%})"), unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6 = st.tabs(["Comparación", "Por clase", "Matriz de confusión", "Precisión-recall",
                                      "Importancia de variables", "Ajuste y tiempos"])
    with t1:
        fig = go.Figure()
        fig.add_bar(x=[NAMES[k] for k in selected], y=m["cv_f1_macro"], name="Validación cruzada (train)",
                    marker_color=th.NEUTRAL, error_y=dict(array=m["cv_f1_macro_std"], color=th.MUTED))
        fig.add_bar(x=[NAMES[k] for k in selected], y=m["f1_macro"], name="Partido de prueba",
                    marker_color=[th.HIGHLIGHT if k == best else th.DARK_NEUTRAL for k in selected],
                    text=m["f1_macro"].round(3), textposition="outside")
        fig.add_hline(y=0.25, line_dash="dash", line_color=th.MUTED, annotation_text="azar (4 clases)")
        fig.update_layout(barmode="group", height=430, yaxis_title="F1 macro", yaxis_range=[0, 0.6],
                          title="F1 macro: validación cruzada vs. partido nuevo",
                          legend=LEGEND_TOP)
        st.plotly_chart(fig, use_container_width=True)
        table = m[["f1_macro", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "ap_macro",
                   "cv_f1_macro"]].rename(index=NAMES, columns={
                       "f1_macro": "F1 macro", "accuracy": "Exactitud", "balanced_accuracy": "Exactitud balanceada",
                       "precision_macro": "Precisión macro", "recall_macro": "Recall macro",
                       "ap_macro": "AP macro", "cv_f1_macro": "F1 macro CV"})
        st.dataframe(table.style.format("{:.3f}").highlight_max(color="#cde2fb"), use_container_width=True)
        st.caption("XGBoost mantiene en el partido nuevo casi el mismo F1 que en validación cruzada; la GRU cae de "
                   "0.32 a 0.11 porque predice saque de banda para casi todo: sus embeddings de apariencia no "
                   "transfieren a un partido con otro encuadre.")
    with t2:
        pc = R["per_class"][R["per_class"].modelo.isin(selected)].copy()
        metric = st.radio("Métrica", ["f1", "precision", "recall"], horizontal=True,
                          format_func={"f1": "F1", "precision": "Precisión", "recall": "Recall"}.get)
        pv = pc.pivot(index="modelo", columns="clase", values=metric).loc[selected, LABELS]
        fig = px.imshow(pv.rename(index=NAMES, columns=LABELS_ES), text_auto=".2f", zmin=0, zmax=1,
                        color_continuous_scale=th.SEQ_SCALE, aspect="auto",
                        title=f"{metric.upper() if metric == 'f1' else metric.capitalize()} por clase")
        fig.update_layout(height=120 + 70 * len(selected))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("En XGBoost, el pase y el saque de banda son las clases mejor reconocidas; la disputa es casi invisible "
                   "para todos los modelos: se confunde con el fondo porque ambas son jugadas sin un patrón de "
                   "movimiento distintivo en 1.6 s.")
    with t3:
        a, b = st.columns([1, 2.5])
        k = a.radio("Modelo", selected, index=selected.index(best) if best in selected else 0, format_func=NAMES.get)
        norm = a.toggle("Normalizar por fila (recall)", value=True)
        cm = R["confusion"][k].loc[LABELS, LABELS]
        z = cm.div(cm.sum(1), axis=0) if norm else cm
        fig = px.imshow(z.rename(index=LABELS_ES, columns=LABELS_ES), text_auto=".0%" if norm else "d",
                        color_continuous_scale=th.SEQ_SCALE, zmin=0, zmax=1 if norm else None,
                        labels=dict(x="Predicha", y="Real", color="Proporción" if norm else "Conteo"),
                        title=f"Matriz de confusión — {NAMES[k]}")
        fig.update_layout(height=470)
        b.plotly_chart(fig, use_container_width=True)
        a.metric("Aciertos", f"{np.trace(cm.to_numpy())} de {cm.to_numpy().sum()}")
    with t4:
        from sklearn.metrics import average_precision_score, precision_recall_curve
        k = st.radio("Modelo", selected, horizontal=True, key="pr_model",
                     index=selected.index(best) if best in selected else 0, format_func=NAMES.get)
        pr = R["predictions"]
        fig = go.Figure()
        for lab in LABELS:
            yt, sc = (pr["label"] == lab).to_numpy(), pr[f"p_{k}_{lab}"].to_numpy()
            p, r, _ = precision_recall_curve(yt, sc)
            fig.add_scatter(x=r, y=p, mode="lines", line=dict(color=CLASS_COLORS[lab], width=2),
                            name=f"{LABELS_ES[lab]} (AP {average_precision_score(yt, sc):.2f}, azar {yt.mean():.2f})")
        fig.update_layout(height=470, xaxis_title="Recall", yaxis_title="Precisión", yaxis_range=[0, 1.02],
                          title=f"Curvas precisión-recall — {NAMES[k]}", legend=dict(x=0.55, y=0.98))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Cuando el modelo está muy seguro de que hay un pase, suele acertar (precisión alta a recall "
                   "bajo): es útil para encontrar los pases más claros, no para detectarlos todos.")
    with t5:
        tab_models = [k for k in selected if k != "gru"]
        if not tab_models:
            st.info("La importancia por permutación se calculó solo para los modelos tabulares.")
        else:
            a, b = st.columns([1, 2.5])
            k = a.radio("Modelo", tab_models, format_func=NAMES.get,
                        index=tab_models.index(best) if best in tab_models else 0)
            n = a.slider("Variables", 5, 30, 15)
            imp = R["importance"][R["importance"].modelo == k].nlargest(n, "importancia").iloc[::-1]
            imp["familia"] = imp["variable"].map(family_of)
            fig = px.bar(imp, x="importancia", y="variable", orientation="h", error_x="std",
                         hover_data=["familia"], title=f"Importancia por permutación — {NAMES[k]}",
                         labels={"importancia": "Caída del F1 macro al permutar", "variable": ""})
            fig.update_traces(marker_color=th.HIGHLIGHT)
            fig.update_layout(height=140 + 24 * n)
            b.plotly_chart(fig, use_container_width=True)
            a.caption("Las variables de flujo óptico (movimiento de jugadores y de cámara) y la dispersión de "
                      "los jugadores son las que más aportan: confirman lo visto en el análisis exploratorio.")
    with t6:
        cur = R["curves"]
        k = st.radio("Modelo", [x for x in selected if x != "gru"] or ["xgb"], horizontal=True, key="tune",
                     format_func=NAMES.get)
        c = cur[cur.modelo == k].astype({"valor": float})
        fig = go.Figure()
        fig.add_scatter(x=c["valor"], y=c["f1_train"], name="Entrenamiento", mode="lines+markers",
                        line=dict(color=th.NEUTRAL, width=2))
        fig.add_scatter(x=c["valor"], y=c["f1_cv"], name="CV por partido", mode="lines+markers",
                        line=dict(color=th.HIGHLIGHT, width=2),
                        error_y=dict(array=c["f1_cv_std"], color=th.HIGHLIGHT, thickness=1))
        fig.update_layout(height=400, yaxis_range=[0, 1.05], xaxis_type="log" if k == "logreg" else "linear",
                          xaxis_title=c["parametro"].iloc[0], yaxis_title="F1 macro",
                          title=f"Curva de validación — {NAMES[k]} (mejores: {R['tuning'][k]['best_params']})",
                          legend=LEGEND_TOP)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Los árboles memorizan el entrenamiento (F1 ≈ 1) mientras la validación por partido se queda "
                   "en ~0.3: el límite no es la capacidad del modelo, sino que las variables cambian de un partido "
                   "a otro.")
        t = m[["fit_s", "infer_us_per_clip", "f1_macro"]].rename(index=NAMES).reset_index(names="modelo")
        fig = px.scatter(t, x="infer_us_per_clip", y="f1_macro", text="modelo", log_x=True, size="fit_s",
                         size_max=30, labels={"infer_us_per_clip": "Inferencia por subclip (µs, log)",
                                              "f1_macro": "F1 macro", "fit_s": "Ajuste (s)"},
                         title="Costo vs. desempeño (sin contar ~0.5-2 s de extracción de variables por ventana)")
        fig.update_traces(marker_color=th.DARK_NEUTRAL, textposition="top center")
        fig.update_layout(height=380, yaxis_range=[0, 0.5])
        st.plotly_chart(fig, use_container_width=True)


pages = [st.Page(page_summary, title="Resumen", icon="📖", default=True),
         st.Page(page_explore, title="Explorar datos", icon="🔎", url_path="explorar"),
         st.Page(page_classify, title="Clasificar video", icon="🎬", url_path="clasificar"),
         st.Page(page_performance, title="Rendimiento", icon="📊", url_path="rendimiento")]
with st.sidebar:
    st.markdown("### ⚽ Acciones en video\nBundesliga · Proyecto 2\n\nCC3084 Data Science · UVG")
st.navigation(pages).run()
