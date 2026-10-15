"""Paleta y plantilla de Plotly de la aplicación (la misma del informe).

Colores de clase validados para daltonismo: las tres acciones (pase, disputa, saque)
superan la separación mínima entre todos los pares en protanopia, deuteranopia y
tritanopia; el fondo, que significa "sin acción", va en gris neutro. Los modelos no
llevan un color propio: se identifican por su nombre en el eje y solo el mejor se
resalta, para no competir con los colores de clase.
"""
import plotly.graph_objects as go
import plotly.io as pio

CLASS_COLORS = {"background": "#8b8a85", "challenge": "#eb6834", "play": "#2a78d6", "throwin": "#1baf7a"}
INK, MUTED, GRID, SURFACE, PANEL = "#1f1f1d", "#6b6a66", "#e4e3df", "#fcfcfb", "#f0efec"
HIGHLIGHT, NEUTRAL, DARK_NEUTRAL = "#2a78d6", "#b9b8b2", "#5b5a55"
SEQ_SCALE = [[0, "#f4f8fd"], [0.25, "#9ec5f4"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]]

pio.templates["proyecto"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Helvetica, Arial, sans-serif", color=INK, size=13),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    colorway=[CLASS_COLORS[k] for k in ["play", "challenge", "throwin", "background"]],
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, tickfont=dict(color=MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK)),
    hoverlabel=dict(bgcolor="white", font=dict(color=INK), bordercolor=GRID),
    margin=dict(l=10, r=10, t=90, b=10),
    title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
))
pio.templates.default = "proyecto"

CSS = f"""
<style>
.kpi {{background:{PANEL}; border-radius:12px; padding:14px 16px; height:100%;}}
.kpi .v {{font-size:1.9rem; font-weight:700; color:{INK}; line-height:1.1}}
.kpi .l {{font-size:.85rem; color:{MUTED}}}
.chip {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:.85rem;
        color:{INK}; font-weight:600; margin-right:4px; background:{PANEL}}}
.chip i {{display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px}}
.step {{border-left:3px solid {HIGHLIGHT}; padding:2px 0 2px 12px; margin-bottom:10px}}
.step b {{color:{INK}}}
.muted {{color:{MUTED}; font-size:.9rem}}
</style>
"""


def chip(label_key: str, text: str) -> str:
    return f'<span class="chip"><i style="background:{CLASS_COLORS[label_key]}"></i>{text}</span>'


def kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>'
