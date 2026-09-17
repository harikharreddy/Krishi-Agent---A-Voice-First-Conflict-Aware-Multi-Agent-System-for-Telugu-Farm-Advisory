"""
System architecture diagram for the PDF report. Generated fresh -- no
diagram image existed anywhere in the repo before this. Stages and
order are taken directly from docs/evaluation_and_validation.md Sec.
1's pipeline description plus backend/voice.py's real call order
(normalize_numerals_te() runs before TTS as a real, separate function
call -- see shared/text_normalization.py -- not an invented stage).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

PRIMARY = "#2f5d4f"
PRIMARY_DARK = "#1c3f38"
SURFACE = "#eef2ec"
ARROW = "#5c655c"
INK = "#1e2a22"

fig, ax = plt.subplots(figsize=(13, 9.5))
ax.set_xlim(0, 125)
ax.set_ylim(0, 100)
ax.axis("off")


def node(x, y, w, h, label, sub=None, hub=False, fontsize=10.5):
    rounding = max(2.2, min(w, h) * 0.16)
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                          linewidth=1.6 if hub else 1.1,
                          edgecolor=PRIMARY, facecolor=PRIMARY if hub else SURFACE)
    ax.add_patch(box)
    color = "white" if hub else INK
    yoff = h * 0.16 if sub else 0
    ax.text(x + w / 2, y + h / 2 + yoff, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color)
    if sub:
        ax.text(x + w / 2, y + h / 2 - h * 0.28, sub, ha="center", va="center", fontsize=7.6, color=color)
    return dict(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2)


def arrow(n1, n2, p1=None, p2=None, curve=0.0, style="-", color=ARROW):
    if p1 is None:
        p1 = (n1["x"] + n1["w"], n1["cy"])
    if p2 is None:
        p2 = (n2["x"], n2["cy"])
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                                  linewidth=1.3, color=color, linestyle=style,
                                  connectionstyle=f"arc3,rad={curve}"))


# ---- Title ----
ax.text(62.5, 95, "Krishi-Agent Pipeline", fontsize=17, fontweight="bold",
        color=PRIMARY_DARK, ha="center")
ax.text(62.5, 90.5, "voice-first, conflict-aware, confidence-hedged multi-agent farm advisory (Telugu)",
        fontsize=9.5, color="#5c655c", ha="center")

# ---- Row 1: input -> ASR -> Intent Router ----
n_input = node(2, 68, 20, 16, "Farmer's\nQuestion", "voice or text\n+ optional leaf photo", fontsize=10)
n_asr = node(28, 68, 20, 16, "ASR", "indic-conformer-600m\n(voice path only)", fontsize=10)
n_router = node(54, 64, 28, 20, "Intent Router", "qwen2.5:7b-instruct (Ollama)", hub=True, fontsize=12)

arrow(n_input, n_asr)
arrow(n_asr, n_router)
arrow(None, None, p1=(12, 68), p2=(58, 84), curve=-0.25, style=(0, (3, 2)), color="#9aa398")
ax.text(30, 86.5, "text path skips ASR", fontsize=8, color="#78806f", ha="center")

# ---- Middle: three conditional agents, feeding a tall Conflict Resolver ----
n_disease = node(58, 49, 24, 10, "Disease Agent", "EfficientNetB0 / ResNet18", fontsize=9.5)
n_weather = node(58, 36, 24, 10, "Weather Agent", "OpenWeatherMap + hedging", fontsize=9.5)
n_price = node(58, 23, 24, 10, "Price Agent", "commodity API + hedging", fontsize=9.5)
n_resolver = node(92, 23, 26, 36, "Conflict\nResolver", "deterministic rule table\n13/13 synthetic tests", hub=True, fontsize=11)

for n in (n_disease, n_weather, n_price):
    arrow(None, n, p1=(n_router["cx"], n_router["y"]), p2=(n["x"], n["cy"]), curve=0.12)
    arrow(n, n_resolver, p2=(n_resolver["x"], n["cy"]))
ax.text(45, 62, "conditional / parallel", fontsize=8, color="#78806f", ha="center")

# ---- Row 2 (S-curve, right to left): Phrasing -> Numeral Norm -> TTS -> Output ----
n_phrase = node(90, 6, 22, 15, "Phrasing\nTemplates", "10 templates +\nunresolved-conflict fallback", fontsize=9.3)
n_numeral = node(64, 6, 22, 15, "Numeral\nNormalization", "normalize_numerals_te()", fontsize=9.3)
n_tts = node(38, 6, 22, 15, "TTS", "indic-parler-tts", fontsize=10)
n_output = node(2, 6, 22, 15, "Spoken\nTelugu Answer", fontsize=9.7)

arrow(None, n_phrase, p1=(n_resolver["cx"], n_resolver["y"]), p2=(n_phrase["cx"], n_phrase["y"] + n_phrase["h"]))
arrow(None, None, p1=(n_phrase["x"], n_phrase["cy"]), p2=(n_numeral["x"] + n_numeral["w"], n_numeral["cy"]))
arrow(None, None, p1=(n_numeral["x"], n_numeral["cy"]), p2=(n_tts["x"] + n_tts["w"], n_tts["cy"]))
arrow(None, None, p1=(n_tts["x"], n_tts["cy"]), p2=(n_output["x"] + n_output["w"], n_output["cy"]))

# ---- Legend ----
legend_elems = [
    Line2D([0], [0], marker="s", color="none", markerfacecolor=PRIMARY, markersize=13, label="Decision hub (LLM / rule engine)"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=SURFACE, markeredgecolor=PRIMARY, markersize=13, label="Deterministic / model stage"),
    Line2D([0], [0], color="#9aa398", linestyle=(0, (3, 2)), label="Text-only shortcut"),
]
ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(0.005, 0.32),
          frameon=False, fontsize=8.5)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "architecture_diagram.png"), dpi=160, facecolor="white")
plt.close(fig)
print("Saved architecture_diagram.png")
