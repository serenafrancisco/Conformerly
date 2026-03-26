"""plots_ids.py

Interactive binary occurrence heatmaps for IMHB and π–π stacking ID matrices.

Embeds Plotly.js directly inside a components.html() iframe so we can set
an explicit pixel width that guarantees all conformers are always rendered,
while still providing the full Plotly toolbar (PNG download, pan, zoom, etc.).

Two public functions:
    imhb_occurrence_heatmap(csv_path)  →  (html, height_px, legend_df) | (None, None, None)
    pi_occurrence_heatmap(csv_path)    →  (html, height_px, legend_df) | (None, None, None)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ── Brand colours ─────────────────────────────────────────────────────────────
_BLUE  = "#0079b0"
_TEAL  = "#00a896"
_EMPTY = "#dde3e8"

# ── Metadata columns ──────────────────────────────────────────────────────────
_IMHB_META = ["hb_label", "n_conformers_present", "frequency",
               "distance_range_class", "pseudoring_type"]
_PI_META   = ["pi_label", "n_conformers_present", "frequency",
               "pi_class", "patch_a", "patch_b"]

# px per conformer cell — determines figure width
_CELL_PX = 12
# px for the frequency bar panel (right side, as fraction of total width)
_FREQ_DOMAIN_FRAC = 0.18


def _build_html(
    df: pd.DataFrame,
    label_col: str,
    meta_cols: list,
    color: str,
    title: str,
) -> tuple[str, int]:

    df = df.sort_values("frequency", ascending=False).reset_index(drop=True)
    conformer_cols = [c for c in df.columns if c not in meta_cols]
    n_conf  = len(conformer_cols)
    n_rows  = len(df)

    labels  = df[label_col].tolist()
    matrix  = df[conformer_cols].values.tolist()   # list of lists

    # ── figure dimensions ────────────────────────────────────────────────────
    row_h    = max(20, min(40, 400 // max(n_rows, 1)))
    fig_h    = n_rows * row_h + 310
    max_label_len = max((len(str(l)) for l in df[label_col]), default=10)
    left_margin   = max(160, min(max_label_len * 7, 320))
    iframe_h = fig_h + 40

    hmap_x2  = 1.0 - _FREQ_DOMAIN_FRAC      # right edge of heatmap domain
    freq_x1  = hmap_x2 + 0.02               # left edge of freq bar domain

    # ── hover text matrix ────────────────────────────────────────────────────
    hover = []
    for i, row in df.iterrows():
        row_hover = []
        meta_parts = []
        for m in meta_cols:
            if m == label_col:
                continue
            meta_parts.append(f"{m.replace('_', ' ')}: <b>{row[m]}</b>")
        meta_str = "<br>".join(meta_parts)
        for j, col in enumerate(conformer_cols):
            val     = int(row[col])
            conf_no = j + 1
            status  = "✓ present" if val else "✗ absent"
            row_hover.append(
                f"<b>{row[label_col]}</b><br>"
                f"Conformer: <b>{conf_no}</b><br>"
                f"Status: <b>{status}</b><br>"
                f"{meta_str}<extra></extra>"
            )
        hover.append(row_hover)

    # ── colour scale: 0 → empty, 1 → brand colour ────────────────────────────
    colorscale = [
        [0.0, _EMPTY],
        [0.499, _EMPTY],
        [0.5, color],
        [1.0, color],
    ]

    # ── Plotly trace dicts (serialised to JSON) ───────────────────────────────
    heatmap_trace = {
        "type": "heatmap",
        "z": matrix,
        "x": list(range(1, n_conf + 1)),
        "y": labels,
        "colorscale": colorscale,
        "zmin": 0, "zmax": 1,
        "showscale": False,
        "hovertemplate": hover,
        "hoverinfo": "text",
        "xgap": 1, "ygap": 2,
        "xaxis": "x", "yaxis": "y",
    }

    freq_trace = {
        "type": "bar",
        "x": df["frequency"].tolist(),
        "y": labels,
        "orientation": "h",
        "marker": {"color": color, "opacity": 0.8},
        "hovertemplate": "%{y}<br>frequency: %{x:.1%}<extra></extra>",
        "xaxis": "x2", "yaxis": "y",
        "showlegend": False,
    }

    layout = {
        "title": {"text": title, "font": {"size": 13, "color": "#333"}, "x": 0},
        "height": fig_h,
        "margin": {"l": left_margin, "r": 20, "t": 110, "b": 110},
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "xaxis": {
            "domain": [0, hmap_x2],
            "title": "Conformer index",
            "tickmode": "array",
            "tickvals": list(range(1, n_conf + 1)),
            "ticktext": [str(i) for i in range(1, n_conf + 1)],
            "tickangle": 90,
            "tickfont": {"size": 8},
            "showgrid": False,
            "zeroline": False,
            "range": [0.5, n_conf + 0.5],
            "constrain": "domain",
        },
        "xaxis2": {
            "domain": [freq_x1, 1.0],
            "title": "Frequency",
            "tickformat": ".0%",
            "range": [0, 1],
            "showgrid": True,
            "gridcolor": "rgba(200,200,200,0.3)",
            "zeroline": False,
        },
        "yaxis": {
            "autorange": "reversed",
            "showgrid": False,
            "tickfont": {"size": 10, "family": "monospace"},
        },
        "dragmode": "pan",
    }

    config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": title.replace(" ", "_"),
            "height": fig_h,
            "scale": 2,
        },
    }

    traces_json = json.dumps([heatmap_trace, freq_trace])
    layout_json = json.dumps(layout)
    config_json = json.dumps(config)

    # iframe height = figure height + horizontal scrollbar room
    iframe_h = fig_h + 40

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background: transparent; overflow: hidden; }}
  #plt {{ width: 100%; }}
</style>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
</head>
<body>
<div id="plt"></div>
<script>
  var layout = {layout_json};
  layout.width = window.innerWidth;
  Plotly.newPlot("plt", {traces_json}, layout, {config_json});
  window.addEventListener("resize", function() {{
    Plotly.relayout("plt", {{width: window.innerWidth}});
  }});
</script>
</body>
</html>"""

    return html, iframe_h


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def imhb_occurrence_heatmap(csv_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return None, None, None
    missing = [c for c in _IMHB_META if c not in df.columns]
    if missing:
        raise ValueError(f"Missing IMHB metadata columns: {missing}")

    html, height = _build_html(df, "hb_label", _IMHB_META, _BLUE,
                                "IMHB occurrence across conformers")

    df_s = df.sort_values("frequency", ascending=False).reset_index(drop=True)
    leg  = df_s[["hb_label", "distance_range_class", "pseudoring_type",
                  "n_conformers_present", "frequency"]].copy()
    leg.columns = ["Label", "Distance class", "Pseudoring type", "# conformers", "Frequency"]
    leg["Frequency"] = leg["Frequency"].map(lambda x: f"{x:.1%}")
    return html, height, leg


def pi_occurrence_heatmap(csv_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return None, None, None
    if "pi_label" not in df.columns:
        return None, None, None
    for col in _PI_META:
        if col not in df.columns:
            df[col] = ""

    html, height = _build_html(df, "pi_label", _PI_META, _TEAL,
                                "π–π stacking occurrence across conformers")

    df_s = df.sort_values("frequency", ascending=False).reset_index(drop=True)
    leg  = df_s[["pi_label", "pi_class", "patch_a", "patch_b",
                  "n_conformers_present", "frequency"]].copy()
    leg.columns = ["Label", "Stacking type", "Ring system A", "Ring system B",
                   "# conformers", "Frequency"]
    leg["Frequency"] = leg["Frequency"].map(lambda x: f"{x:.1%}")
    return html, height, leg