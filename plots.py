"""plots.py

Plotting module for Chameleons v1.2.1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Column and label constants ─────────────────────────────────────────────────
PSA_MOL  = "3D-PSA"
PSA_SASA = "3D-PSA(SA)"
RGYR     = "Rgyr_AA"

NUMERIC_COLS: List[str] = [
    RGYR, PSA_MOL, PSA_SASA,
    "IMHB_Tot", "IMHB_SR", "IMHB_MR", "IMHB_LR",
    "Pi_Tot", "Pi_FF", "Pi_EF",
]

AXIS_LABELS: dict = {
    RGYR:       "Rgyr \u2014 all atoms (\u00c5)",
    PSA_MOL:    "3D-PSA \u2014 mol. surface, probe 0 (\u00c5\u00b2)",
    PSA_SASA:   "3D-PSA \u2014 SASA, probe 1.4 \u00c5 (\u00c5\u00b2)",
    "IMHB_Tot": "Total IMHBs",
    "Pi_Tot":   "Total \u03c0\u2013\u03c0 contacts",
    "Solvent":  "Solvent",
}

_PALETTE = px.colors.qualitative.Pastel   
_PLOT_PX  = 640                           

# Smart heuristic mapping for common solvents
_HEURISTIC_COLORS = {
    "water": "#3b82f6", "h2o": "#3b82f6",          
    "chcl3": "#eab308", "chloroform": "#eab308",   
    "dmso": "#8b5cf6",                             
    "methanol": "#10b981", "meoh": "#10b981",      
    "ethanol": "#14b8a6", "etoh": "#14b8a6",       
    "toluene": "#f97316",                          
    "benzene": "#ef4444",                          
    "gas": "#9ca3af", "vacuum": "#9ca3af"          
}


def run_plots(
    tsv_path: Path,
    out_dir: Path,
    quiet: bool = False,
) -> List[Path]:
    if not tsv_path.exists():
        if not quiet:
            print(f"  WARNING [plots] TSV not found: {tsv_path}", file=sys.stderr)
        return []

    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except Exception as ex:
        if not quiet:
            print(f"  WARNING [plots] could not read {tsv_path.name}: {ex}", file=sys.stderr)
        return []

    df   = prepare_df(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = tsv_path.stem
    written: List[Path] = []

    if RGYR in df.columns and not df[RGYR].isna().all():
        try:
            fig = conformational_landscape_interactive(df)
            p   = out_dir / f"{stem}_landscape.html"
            fig.write_html(str(p), include_plotlyjs="cdn")
            written.append(p)
            if not quiet:
                print(f"  [plots] wrote: {p.name}")
        except Exception as ex:
            if not quiet:
                print(f"  WARNING [plots] landscape failed: {ex}", file=sys.stderr)

    return written

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _empty_figure(msg: str = "No data available.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="gray"))
    fig.update_layout(height=300, xaxis_visible=False, yaxis_visible=False,
                      margin=dict(l=20, r=20, t=30, b=20))
    return fig


def conformational_landscape_interactive(df: pd.DataFrame) -> go.Figure:
    x_options = [c for c in [PSA_MOL, PSA_SASA] if c in df.columns and not df[c].isna().all()]
    if not x_options:
        return _empty_figure("No 3D-PSA columns found.")

    psa_check_cols = [c for c in x_options]
    plot_df = df.dropna(subset=[RGYR]).copy()
    plot_df = plot_df[plot_df[psa_check_cols].notna().any(axis=1)].copy()
    if plot_df.empty:
        return _empty_figure("No valid data rows.")

    if PSA_MOL in plot_df.columns and PSA_SASA in plot_df.columns:
        plot_df[PSA_MOL]  = plot_df[PSA_MOL].fillna(plot_df[PSA_SASA])
        plot_df[PSA_SASA] = plot_df[PSA_SASA].fillna(plot_df[PSA_MOL])

    # ── Handle discrete solvent colors ────────────────────────────────────────
    if "Solvent" in plot_df.columns:
        solvents_ordered = sorted(plot_df["Solvent"].dropna().unique().tolist())
    else:
        solvents_ordered = ["unknown"]
        plot_df["Solvent"] = "unknown"

    n_sol = len(solvents_ordered)

    assigned_colors = []
    used_indices = 0
    for s in solvents_ordered:
        sl = s.lower()
        if sl in _HEURISTIC_COLORS:
            assigned_colors.append(_HEURISTIC_COLORS[sl])
        else:
            assigned_colors.append(_PALETTE[used_indices % len(_PALETTE)])
            used_indices += 1

    # ── Handle continuous metrics ─────────────────────────────────────────────
    if "IMHB_Tot" in plot_df.columns and not plot_df["IMHB_Tot"].isna().all():
        plot_df["IMHB_Tot"] = plot_df["IMHB_Tot"].fillna(0).astype(int)
        imhb_vals = plot_df["IMHB_Tot"].tolist()
    else:
        plot_df["IMHB_Tot"] = 0
        imhb_vals = [0] * len(plot_df)

    if "Pi_Tot" in plot_df.columns and not plot_df["Pi_Tot"].isna().all():
        plot_df["Pi_Tot"] = plot_df["Pi_Tot"].fillna(0).astype(int)
        pi_vals = plot_df["Pi_Tot"].tolist()
    else:
        plot_df["Pi_Tot"] = 0
        pi_vals = [0] * len(plot_df)

    hover_cols = ["Conformer", "Solvent"]
    for col in ["IMHB_Tot", "IMHB_SR", "IMHB_MR", "IMHB_LR",
                "Pi_Tot", "Pi_FF", "Pi_EF"]:
        if col in plot_df.columns and not plot_df[col].isna().all():
            plot_df[col] = plot_df[col].fillna(0).astype(int)
            if col not in hover_cols:
                hover_cols.append(col)

    hover_parts = [
        "<b>Conformer %{customdata[0]}</b>",
        "Solvent: %{customdata[1]}",
    ]
    custom_data_cols = hover_cols[:]
    for i, col in enumerate(hover_cols[2:], start=2):
        hover_parts.append(f"{col}: %{{customdata[{i}]}}")

    hover_template = "<br>".join(hover_parts) + "<extra></extra>"
    x_default = x_options[0]

    fig = go.Figure()

    sol_color_arrays = []
    imhb_color_arrays = []
    pi_color_arrays = []

    # Map each solvent to a distinct trace to trigger a real discrete legend
    for i, sol in enumerate(solvents_ordered):
        sol_df = plot_df[plot_df["Solvent"] == sol].copy()
        
        sol_color_arrays.append(assigned_colors[i])
        imhb_color_arrays.append(sol_df["IMHB_Tot"].tolist())
        pi_color_arrays.append(sol_df["Pi_Tot"].tolist())
        
        customdata = sol_df[custom_data_cols].values.tolist()
        
        fig.add_trace(go.Scatter(
            x=sol_df[x_default].tolist(),
            y=sol_df[RGYR].tolist(),
            mode="markers",
            name=sol,
            marker=dict(
                size=9,
                color=assigned_colors[i],
                line=dict(width=0.8, color="DarkSlateGrey"),
            ),
            customdata=customdata,
            hovertemplate=hover_template,
            showlegend=True,
        ))

    x_buttons = []
    for xcol in x_options:
        x_arrays_per_trace = [plot_df[plot_df["Solvent"] == sol][xcol].tolist() for sol in solvents_ordered]
        x_buttons.append(dict(
            label=AXIS_LABELS.get(xcol, xcol),
            method="update",
            args=[
                {"x": x_arrays_per_trace},
                {"xaxis.title.text": AXIS_LABELS.get(xcol, xcol)},
            ],
        ))

    cb_imhb = dict(title=dict(text="IMHB total"), lenmode="fraction", len=0.75)
    cb_pi = dict(title=dict(text="π–π total"), lenmode="fraction", len=0.75)

    color_buttons = [
        dict(
            label="Colour: Solvent",
            method="restyle",
            args=[{
                "marker.color": sol_color_arrays,
                "marker.colorscale": [None] * n_sol,
                "marker.showscale": [False] * n_sol,
                "showlegend": [True] * n_sol,
            }],
        ),
        dict(
            label="Colour: IMHB count",
            method="restyle",
            args=[{
                "marker.color": imhb_color_arrays,
                "marker.colorscale": ["Teal"] * n_sol,
                "marker.cmin": [min(imhb_vals)] * n_sol,
                "marker.cmax": [max(imhb_vals)] * n_sol,
                "marker.colorbar": [cb_imhb] * n_sol,
                "marker.showscale": [True] + [False] * (n_sol - 1),
                "showlegend": [False] * n_sol,
            }],
        ),
        dict(
            label="Colour: π–π count",
            method="restyle",
            args=[{
                "marker.color": pi_color_arrays,
                "marker.colorscale": ["Plasma"] * n_sol,
                "marker.cmin": [min(pi_vals)] * n_sol,
                "marker.cmax": [max(pi_vals)] * n_sol,
                "marker.colorbar": [cb_pi] * n_sol,
                "marker.showscale": [True] + [False] * (n_sol - 1),
                "showlegend": [False] * n_sol,
            }],
        ),
    ]

    menu_style = dict(
        bgcolor="white", bordercolor="#CCCCCC", borderwidth=1, font=dict(size=12),
        direction="down", showactive=True, yanchor="top", y=1.12,
    )

    fig.update_layout(
        updatemenus=[
            dict(**menu_style, buttons=x_buttons, x=0.0, xanchor="left", pad=dict(r=10)),
            dict(**menu_style, buttons=color_buttons, x=0.52, xanchor="left", pad=dict(r=10)),
        ],
        width=_PLOT_PX, height=_PLOT_PX, margin=dict(l=65, r=80, t=90, b=65),
        xaxis=dict(
            title=AXIS_LABELS.get(x_default, x_default), showgrid=True, gridcolor="#EEEEEE",
            zeroline=False, mirror=True, showline=True, linecolor="#BBBBBB",
        ),
        yaxis=dict(
            title=AXIS_LABELS.get(RGYR, RGYR), showgrid=True, gridcolor="#EEEEEE",
            zeroline=False, mirror=True, showline=True, linecolor="#BBBBBB",
        ),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        legend=dict(
            title="Solvent",
            orientation="v",
            yanchor="top", y=1,
            xanchor="left", x=1.02
        )
    )
    return fig

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate Chameleons plots from a master TSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("molecule_name", help="Molecule prefix (e.g. ARV-110).")
    p.add_argument("--tsv-path", type=Path, default=None)
    p.add_argument("--out-dir",  type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    return p

def _main(argv: list) -> int:
    args      = _build_parser().parse_args(argv)
    res_dir   = Path("./results").resolve()
    tsv_path  = args.tsv_path or (res_dir / f"{args.molecule_name}.tsv")
    out_dir   = args.out_dir  or (res_dir / args.molecule_name)
    if not tsv_path.exists():
        print(f"ERROR: TSV not found at {tsv_path}", file=sys.stderr)
        return 2
    written = run_plots(tsv_path=tsv_path, out_dir=out_dir, quiet=args.quiet)
    if not args.quiet:
        print(f"\n[plots] {len(written)} plot(s) written to {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))