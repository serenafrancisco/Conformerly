#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pages/analysis.py — Conformerly
Analysis page: file upload, parameter configuration, and results display.

Place this file inside a `pages/` directory next to streamlit_app.py.
All sibling analysis modules (imhb.py, pi.py, rgyr.py, psa3d.py,
mol2_io.py, results_manager.py, plots.py, viewer2D.py, viewer3D.py)
must live in the same root directory as streamlit_app.py.
"""

from __future__ import annotations

import io
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from components.footer import show_footer

import streamlit as st

# ── Make sibling modules importable ──────────────────────────────────────────
# pages/ is one level below the project root, so we walk up one directory.
_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pandas as pd
import streamlit.components.v1 as components

from mol2_io import split_mol2_blocks, Mol2Block
from imhb import (
    run_imhb,
    parse_atoms_and_bonds,
    IMHB_MAX_DIST_HA,
    IMHB_MIN_ANGLE_DHA,
    IMHB_MIN_ANGLE_ACCEPTOR,
    IMHB_MIN_ANGLE_STRICT,
    IMHB_MIN_BOND_SEPARATION,
)
from pi import run_pi
from rgyr import run_rgyr
from psa3d import run_psa3d, PSA3D_PROBE_SASA, PSA3D_N_SPHERE_POINTS
from results_manager import generate_tsv_summary
from plots import conformational_landscape_interactive, prepare_df


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULTS: Dict[str, object] = {
    "do_imhb": True,
    "do_pi":   True,
    # 3D PSA
    "psa3d_probe_sasa": float(PSA3D_PROBE_SASA),
    "psa3d_n_points":   int(PSA3D_N_SPHERE_POINTS),
    # IMHB
    "hb_max_dist":         float(IMHB_MAX_DIST_HA),
    "hb_min_angle":        float(IMHB_MIN_ANGLE_DHA),
    "hb_min_angle_acc":    float(IMHB_MIN_ANGLE_ACCEPTOR),
    "hb_min_angle_strict": float(IMHB_MIN_ANGLE_STRICT),
    "hb_min_bond_sep":     int(IMHB_MIN_BOND_SEPARATION),
    # π–π
    "pi_ff_max_dist":    4.4,
    "pi_ef_max_dist":    5.5,
    "pi_parallel_angle": 30.0,
    "pi_t_angle_min":    60.0,
    "pi_min_bond_sep":   2,
}


def _init_session() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "results"  not in st.session_state:
        st.session_state["results"]  = None
    if "_tmp_dir" not in st.session_state:
        st.session_state["_tmp_dir"] = None


def _reset_to_defaults() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


_init_session()


# ══════════════════════════════════════════════════════════════════════════════
# MINIMAL CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
.cover-banner {
    width: 100%;
    height: 190px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1rem;
    overflow: hidden;
    background: var(--background-color, #0a1628);
    background-image:
        radial-gradient(ellipse 55% 60% at 15% 55%,
            rgba(96,165,250,0.08) 0%, transparent 70%),
        radial-gradient(ellipse 40% 50% at 82% 38%,
            rgba(167,139,250,0.07) 0%, transparent 70%);
    box-shadow: 0 6px 30px rgba(0,0,0,0.18);
}
.cover-wordmark {
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    opacity: 0.07;
    user-select: none;
}
.fmt-error {
    border-left: 4px solid #e53e3e;
    border-radius: 6px;
    padding: 0.55rem 0.9rem;
    margin: 0.3rem 0;
    font-size: 0.86rem;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Analysis Settings")
    st.caption("Rgyr and 3D-PSA are always computed.")

    st.checkbox("Intramolecular H-bonds (IMHB)", key="do_imhb")
    st.checkbox("Aromatic π–π Stacking",         key="do_pi")

    st.divider()
    st.subheader("Parameter Settings")

    # ── 3D PSA ────────────────────────────────────────────────────────────────
    with st.expander("3D Polar Surface Area", expanded=False):
        st.number_input(
            "Solvent probe radius (Å)",
            min_value=0.0, max_value=5.0,
            step=0.1, format="%.1f",
            key="psa3d_probe_sasa",
            help=(
                "Probe radius for the SASA-based 3D PSA (Shrake–Rupley algorithm). "
                "Default 1.4 Å approximates a water molecule. "
                "The molecular-surface variant always uses probe = 0 Å (not configurable)."
            ),
        )
        st.number_input(
            "Fibonacci sphere points per atom",
            min_value=120, max_value=5000,
            step=120,
            key="psa3d_n_points",
            help=(
                "Number of test points per atom's expanded sphere. "
                "Higher = more accurate but slower. Default: 960. "
                "Values below 480 may introduce significant numerical noise."
            ),
        )

    # ── IMHB ──────────────────────────────────────────────────────────────────
    imhb_label = "Intramolecular H-Bonds (IMHB)" + (
        "" if st.session_state.get("do_imhb", True) else "  *(disabled)*"
    )
    with st.expander(imhb_label, expanded=False):
        if not st.session_state.get("do_imhb", True):
            st.caption("Enable IMHB above to use these parameters.")
        else:
            st.caption("Defaults match Schrödinger Maestro H-bond criteria.")
            st.number_input(
                "Max H···A distance (Å)",
                min_value=1.5, max_value=5.0,
                step=0.1, format="%.1f",
                key="hb_max_dist",
                help=(
                    "Maximum distance between the hydrogen and the acceptor heavy atom. "
                    "Pairs beyond this threshold are always rejected. "
                    "Maestro default: 2.8 Å."
                ),
            )
            st.number_input(
                "Min D–H···A donor angle (°)",
                min_value=90.0, max_value=180.0,
                step=1.0, format="%.1f",
                key="hb_min_angle",
                help=(
                    "Minimum angle at the bridging hydrogen (vertex = H). "
                    "180° = perfectly linear; 120° captures most intramolecular contacts. "
                    "Maestro default: 120°."
                ),
            )
            st.number_input(
                "Min X–A···H acceptor angle (°)",
                min_value=60.0, max_value=180.0,
                step=1.0, format="%.1f",
                key="hb_min_angle_acc",
                help=(
                    "Minimum angle at the acceptor atom (vertex = A). "
                    "Enforces lone-pair geometry. Maestro default: 90°."
                ),
            )
            st.number_input(
                "Strict D–H···A angle — close contacts (°)",
                min_value=90.0, max_value=180.0,
                step=1.0, format="%.1f",
                key="hb_min_angle_strict",
                help=(
                    "Tighter donor-angle cutoff applied when D and A are separated "
                    "by fewer bonds than the threshold below. "
                    "Guards against false positives in strained rings. "
                    "Maestro default: 130°."
                ),
            )
            st.number_input(
                "Bond-sep. threshold for strict angle",
                min_value=1, max_value=15,
                step=1,
                key="hb_min_bond_sep",
                help=(
                    "If D and A are fewer than this many bonds apart, "
                    "the stricter angle above is applied. "
                    "Maestro default: 5 bonds."
                ),
            )

    # ── π–π stacking ──────────────────────────────────────────────────────────
    pi_label = "Aromatic π–π Stacking" + (
        "" if st.session_state.get("do_pi", True) else "  *(disabled)*"
    )
    with st.expander(pi_label, expanded=False):
        if not st.session_state.get("do_pi", True):
            st.caption("Enable π–π stacking above to use these parameters.")
        else:
            st.caption("Defaults match Schrödinger Maestro π–π criteria.")
            st.number_input(
                "Face-to-face max centroid dist (Å)",
                min_value=1.0, max_value=10.0,
                step=0.1, format="%.1f",
                key="pi_ff_max_dist",
                help="Maximum centroid-to-centroid distance for face-to-face π–π. Maestro default: 4.4 Å.",
            )
            st.number_input(
                "Edge-to-face max centroid dist (Å)",
                min_value=1.0, max_value=10.0,
                step=0.1, format="%.1f",
                key="pi_ef_max_dist",
                help="Maximum centroid-to-centroid distance for edge-to-face (T-shaped) π–π. Maestro default: 5.5 Å.",
            )
            st.number_input(
                "Max inter-plane angle — face-to-face (°)",
                min_value=0.0, max_value=45.0,
                step=1.0, format="%.1f",
                key="pi_parallel_angle",
                help="Maximum inter-plane normal angle for face-to-face stacking. Maestro default: 30°.",
            )
            st.number_input(
                "Min inter-plane angle — edge-to-face (°)",
                min_value=45.0, max_value=90.0,
                step=1.0, format="%.1f",
                key="pi_t_angle_min",
                help="Minimum inter-plane normal angle for T-shaped stacking. Maestro default: 60°.",
            )
            st.number_input(
                "Min bond separation (ring exclusion)",
                min_value=0, max_value=5,
                step=1,
                key="pi_min_bond_sep",
                help=(
                    "Pairs at this bond separation or closer are excluded. "
                    "Default 2: excludes fused rings (dist 0), directly bonded rings (dist 1), "
                    "and one-atom-bridged rings Ph\u2013X\u2013Ph (dist 2). "
                    "Only rings with 3+ bonds of separation are evaluated. "
                    "sp2-bridged pairs one step beyond the threshold are also excluded automatically."
                ),
            )

    # ── Reset ─────────────────────────────────────────────────────────────────
    st.divider()
    st.button(
        "Reset to Defaults", 
        icon=":material/refresh:", 
        use_container_width=True, 
        type="secondary",
        on_click=_reset_to_defaults
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE — HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.image("figures/logo-banner.png", use_container_width=True)

# ↓ Add this to kill the gap
st.markdown("""
    <style>
        [data-testid="stImage"] { margin-bottom: -12rem; }
        h1:first-of-type { margin-top: 0; }
    </style>
""", unsafe_allow_html=True)

st.title("Run Analysis")
st.caption(
    "Conformational ensemble analysis · Rgyr · 3D-PSA · IMHB · π–π stacking"
)
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
    <h3 style='margin-bottom:0; display:flex; align-items:center; gap:8px'>
        <span class='material-symbols-rounded' style='font-size:1.4rem'>upload</span>
        Upload Input Files
    </h3>
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet"/>
""", unsafe_allow_html=True)
st.markdown(
    "Upload one or more multi-conformer **MOL2** files. "
    "Files **must** follow the naming convention: `<molecule_name>_<solvent>.mol2`  \n"
    "Examples of recognized formats: `ARV110_H2O.mol2`, `ARV-110_CHCl3.mol2`, `ARV_110_DMSO.mol2`."
)

st.caption(
    "Multiple solvents for the same molecule are grouped automatically."
)

uploaded_files = st.file_uploader(
    "Drop MOL2 files here",
    type=["mol2"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

_VALID_STEM = re.compile(r"^.+_.+$")

format_errors: List[str] = []
valid_files:   List      = []

if uploaded_files:
    for uf in uploaded_files:
        stem = Path(uf.name).stem
        if not _VALID_STEM.match(stem):
            format_errors.append(uf.name)
        else:
            valid_files.append(uf)

for bad in format_errors:
    st.markdown(
        f'<div class="fmt-error">'
        f'⚠  <b>{bad}</b> — does not match '
        f'<code>&lt;molecule_name&gt;_&lt;solvent&gt;.mol2</code>. '
        f'This file will be skipped.'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("""
    <style>
    :root {
        --primary-color: #0079b0 !important;
    }
    .stCheckbox label span:first-child {
        border-color: #0079b0 !important;
    }
    .stCheckbox input:checked ~ label span:first-child,
    .stCheckbox [aria-checked="true"] span:first-child,
    .stCheckbox label span:first-child[data-checked="true"],
    .stCheckbox input[type="checkbox"]:checked + label span:first-child,
    .stCheckbox label:has(input:checked) span:first-child {
        background-color: #0079b0 !important;
        border-color: #0079b0 !important;
    }
    .stButton button[kind="primary"],
    .stButton button[data-testid="baseButton-primary"],
    button[kind="primary"] {
        background: #0079b0 !important;
        background-color: #0079b0 !important;
        color: white !important;
        border: none !important;
        font-size: 1.6rem !important;
        padding: 1.2rem 2rem !important;
        font-weight: 700 !important;
    }
    .stButton button[kind="primary"] p,
    .stButton button[data-testid="baseButton-primary"] p,
    button[kind="primary"] p {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    .stButton button[kind="primary"] span[data-testid="stIconMaterial"],
    .stButton button[data-testid="baseButton-primary"] span[data-testid="stIconMaterial"],
    button[kind="primary"] span[data-testid="stIconMaterial"] {
        font-size: 1.6rem !important;
    }
    .stButton button[kind="primary"]:hover,
    .stButton button[data-testid="baseButton-primary"]:hover {
        background: #005f8a !important;
        background-color: #005f8a !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

run_clicked = False
if valid_files:
    st.markdown("")
    run_clicked = st.button("Run Analysis", icon=":material/rocket_launch:", type="primary")
elif uploaded_files and not valid_files:
    st.error("No valid files to process — please fix the naming errors shown above.")
else:
    st.info("Upload at least one correctly named MOL2 file to start.")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_mol_solvent(filename: str) -> Tuple[str, str]:
    """Split '<mol>_<solvent>.mol2' on the last underscore."""
    stem  = Path(filename).stem
    parts = stem.rsplit("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (stem, "unknown")


def _make_zip(mol_data: dict) -> bytes:
    """Bundle all CSVs / TSV for one molecule into an in-memory ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for _sol, sd in mol_data["solvents"].items():
            for csv_p in sd.get("csvs", []):
                p = Path(csv_p)
                if p.exists():
                    try:
                        zf.write(p, arcname=p.name)
                    except Exception:
                        pass
        tsv = mol_data.get("tsv_path")
        if tsv and Path(tsv).exists():
            try:
                zf.write(tsv, arcname=Path(tsv).name)
            except Exception:
                pass
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _run_analysis(valid_files: list, tmp_dir: Path) -> dict:
    """
    Run the full Chameleons pipeline for every uploaded file.

    Returns a dict keyed by molecule name; each value contains:
        solvents   : { solvent → { csvs, html_3d, warnings } }
        tsv_path, tsv_df, plotly_fig, html_2d, [optional warning strings]
    """
    ss = st.session_state
    do_imhb_val = ss.get("do_imhb", True)
    do_pi_val   = ss.get("do_pi", True)
    show_viewers = do_imhb_val or do_pi_val

    mol_files: Dict[str, List] = defaultdict(list)
    for uf in valid_files:
        mol_name, solvent = _parse_mol_solvent(uf.name)
        mol_files[mol_name].append((uf, solvent))

    total      = len(valid_files)
    progress   = st.progress(0, text="Starting…")
    status     = st.empty()
    file_count = 0
    results    = {}

    for mol_name, file_list in mol_files.items():
        mol_dir = tmp_dir / mol_name
        mol_dir.mkdir(parents=True, exist_ok=True)

        mol_data: dict = {
            "mol_name":   mol_name,
            "solvents":   {},
            "tsv_path":   None,
            "tsv_df":     None,
            "plotly_fig": None,
            "html_2d":    None,
        }

        # ── Per-solvent file ───────────────────────────────────────────────
        for uf, solvent in file_list:
            file_count += 1
            pct = int(file_count / total * 100)
            status.markdown(
                f"**Computing…** &nbsp;`{uf.name}`&nbsp; "
                f"({file_count} / {total})"
            )
            progress.progress(pct, text=f"{pct}%")

            mol2_path = mol_dir / uf.name
            mol2_path.write_bytes(uf.getvalue())

            out_prefix = f"{mol_name}_{solvent}"
            blocks     = split_mol2_blocks(mol2_path)

            sd: dict = {"csvs": [], "html_3d": None, "warnings": []}

            if not blocks:
                sd["warnings"].append(f"No molecule blocks found in {uf.name}.")
                mol_data["solvents"][solvent] = sd
                continue

            # Rgyr — always
            try:
                p = run_rgyr(blocks=blocks, out_dir=mol_dir, out_prefix=out_prefix)
                sd["csvs"].append(p)
            except Exception as exc:
                sd["warnings"].append(f"Rgyr failed: {exc}")

            # 3D PSA — always
            try:
                p = run_psa3d(
                    blocks=blocks,
                    out_dir=mol_dir,
                    out_prefix=out_prefix,
                    probe_sasa=float(ss["psa3d_probe_sasa"]),
                    n_sphere_points=int(ss["psa3d_n_points"]),
                )
                sd["csvs"].append(p)
            except Exception as exc:
                sd["warnings"].append(f"3D PSA failed: {exc}")

            # IMHB — optional
            if do_imhb_val:
                try:
                    detail, summary, hbond_ids = run_imhb(
                        blocks=blocks,
                        out_dir=mol_dir,
                        out_prefix=out_prefix,
                        hb_max_dist=float(ss["hb_max_dist"]),
                        hb_min_angle=float(ss["hb_min_angle"]),
                        hb_min_angle_strict=float(ss["hb_min_angle_strict"]),
                        hb_min_bond_sep=int(ss["hb_min_bond_sep"]),
                        hb_min_angle_acceptor=float(ss["hb_min_angle_acc"]),
                    )
                    sd["csvs"].extend([detail, summary, hbond_ids])
                except Exception as exc:
                    sd["warnings"].append(f"IMHB failed: {exc}")

            # π–π — optional
            if do_pi_val:
                try:
                    outs = run_pi(
                        blocks=blocks,
                        out_dir=mol_dir,
                        out_prefix=out_prefix,
                        min_bond_sep=int(ss["pi_min_bond_sep"]),
                        use_pi_criteria=True,
                        pi_ff_max_dist=float(ss["pi_ff_max_dist"]),
                        pi_ef_max_dist=float(ss["pi_ef_max_dist"]),
                        pi_parallel_angle=float(ss["pi_parallel_angle"]),
                        pi_t_angle_min=float(ss["pi_t_angle_min"]),
                    )
                    sd["csvs"].extend(outs)
                except Exception as exc:
                    sd["warnings"].append(f"π–π stacking failed: {exc}")

            # 3D viewer HTML - skip completely if both interactions disabled
            if show_viewers:
                try:
                    from viewer3D import (
                        build_ensemble_scene,
                        build_stats              as _bs3,
                        generate_html            as _gh3,
                        compute_rdkit_properties as _rp3,
                        enumerate_donors_acceptors as _da3,
                    )
                    scene = build_ensemble_scene(blocks, do_imhb=do_imhb_val, do_pi=do_pi_val)
                    if scene:
                        atoms_d = parse_atoms_and_bonds(blocks[0].text)
                        stats3  = _bs3(atoms_d, len(blocks))
                        rdkp3   = _rp3(blocks[0].text)
                        # Feed empty dicts if disabled to save cycles
                        da3     = _da3(atoms_d, blocks[0].text) if do_imhb_val else {"donors": [], "acceptors": []}
                        out_files_data = []
                        for csv_p in sd["csvs"]:
                            try:
                                out_files_data.append({
                                    "name":    csv_p.name,
                                    "content": csv_p.read_text(encoding="utf-8"),
                                })
                            except Exception:
                                pass
                        sd["html_3d"] = _gh3(
                            scene, stats3, rdkp3, da3, out_prefix,
                            output_files=out_files_data,
                        )
                except Exception as exc:
                    sd["warnings"].append(f"3D viewer failed: {exc}")

            mol_data["solvents"][solvent] = sd

        # ── TSV summary ────────────────────────────────────────────────────
        try:
            tsv_path = generate_tsv_summary(mol_name, mol_dir, tmp_dir)
            mol_data["tsv_path"] = tsv_path
            mol_data["tsv_df"]   = pd.read_csv(tsv_path, sep="\t")
        except Exception as exc:
            mol_data["tsv_warning"] = f"TSV generation failed: {exc}"

        # ── Plotly conformational landscape ───────────────────────────────
        if mol_data["tsv_df"] is not None:
            try:
                fig = conformational_landscape_interactive(
                    prepare_df(mol_data["tsv_df"])
                )
                mol_data["plotly_fig"] = fig
            except Exception:
                pass

        # ── 2D topology viewer ─────────────────────────────────────────────
        # Skip completely if both interactions disabled
        if show_viewers:
            try:
                from viewer2D import (
                    generate_2d_svg,
                    build_stats                as _bs2,
                    generate_html              as _gh2,
                    compute_rdkit_properties   as _rp2,
                    enumerate_donors_acceptors as _da2,
                    detect_ring_systems_for_viz,
                    _HAS_RDKIT                 as _2d_rdkit,
                )
                if _2d_rdkit and file_list:
                    first_mol2   = mol_dir / file_list[0][0].name
                    first_blocks = split_mol2_blocks(first_mol2)
                    if first_blocks:
                        blk     = first_blocks[0]
                        atoms_d = parse_atoms_and_bonds(blk.text)
                        if atoms_d:
                            from rdkit import Chem as _Chem
                            rdkit_mol = _Chem.MolFromMol2Block(
                                blk.text, sanitize=True, removeHs=False
                            )
                            rdkp2    = _rp2(rdkit_mol) if rdkit_mol else {}
                            
                            # Feed empty data to geometry/renderer if disabled
                            da2      = _da2(atoms_d, blk.text) if do_imhb_val else {"donors": [], "acceptors": []}
                            ring_sys = detect_ring_systems_for_viz(blk.text) if do_pi_val else []
                            
                            rs_info  = [
                                {
                                    "label":      rs["label"],
                                    "n_atoms":    rs["n_atoms"],
                                    "n_patches":  rs["n_patches"],
                                    "color":      rs["color"],
                                    "atom_names": rs.get("atom_names", []),
                                    "patches":    rs.get("patches", []),
                                }
                                for rs in ring_sys
                            ]
                            stats2 = _bs2(atoms_d, blk, len(first_blocks))
                            svg    = generate_2d_svg(
                                blk.text,
                                {d["id"] for d in da2["donors"]},
                                {a["id"] for a in da2["acceptors"]},
                                ring_sys,
                            )
                            mol_data["html_2d"] = _gh2(
                                svg, stats2, rdkp2, da2, rs_info, mol_name
                            )
            except Exception as exc:
                mol_data["html_2d_warning"] = f"2D viewer failed: {exc}"

        results[mol_name] = mol_data

    progress.progress(100, text=":material/check_circle: Done")
    status.empty()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def _subsection(icon: str, label: str) -> None:
    st.markdown(f"##### {icon} {label}")


def _display_results(results: dict) -> None:
    from plots_ids import imhb_occurrence_heatmap, pi_occurrence_heatmap

    st.header("Results")

    show_viewers = st.session_state.get("do_imhb", True) or st.session_state.get("do_pi", True)
    do_imhb      = st.session_state.get("do_imhb", True)
    do_pi        = st.session_state.get("do_pi",   True)

    for mol_name, mol_data in results.items():
        with st.expander(mol_name, icon=":material/hub:", expanded=True):

            # ── 2D topology ───────────────────────────────────────────────────
            if show_viewers:
                _subsection(":material/map:", "2D Molecular Topology")
                if mol_data.get("html_2d"):
                    components.html(mol_data["html_2d"], height=700, scrolling=True)
                elif mol_data.get("html_2d_warning"):
                    st.warning(mol_data["html_2d_warning"])
                else:
                    st.info("2D viewer unavailable (RDKit required).")
                st.markdown("<br><br>", unsafe_allow_html=True)

            # ── Conformational landscape ──────────────────────────────────────
            _subsection(":material/scatter_plot:", "Conformational Landscape")
            if mol_data.get("plotly_fig") is not None:
                st.plotly_chart(
                    mol_data["plotly_fig"],
                    use_container_width=False,
                    config={"displayModeBar": True, "scrollZoom": True},
                )
            else:
                st.info(
                    "Scatter plot unavailable — TSV data missing or "
                    "no 3D-PSA / Rgyr columns found."
                )
            st.markdown("<br><br>", unsafe_allow_html=True)

            # ── IMHB occurrence heatmap ───────────────────────────────────────
            if do_imhb:
                _subsection(":material/key_visualizer:", "IMHB Occurrence")
                st.caption(
                    "Binary heatmap of intramolecular hydrogen bonds across all conformers. "
                    "Rows sorted by descending frequency. Scroll horizontally to see all conformers."
                )
                imhb_solvents = []
                for sol, sd in mol_data["solvents"].items():
                    for csv_p in sd.get("csvs", []):
                        p = Path(csv_p)
                        if p.name.endswith("_hbond_ids.csv"):
                            imhb_solvents.append((sol, p))
                            break

                if imhb_solvents:
                    tabs = st.tabs([f":material/water_drop: {sol}" for sol, _ in imhb_solvents])
                    for tab, (sol, p) in zip(tabs, imhb_solvents):
                        with tab:
                            try:
                                html, height, legend_df = imhb_occurrence_heatmap(p)
                                if html:
                                    components.html(html, height=height + 20, scrolling=False)
                                    st.caption("Legend")
                                    st.dataframe(legend_df, use_container_width=True, hide_index=True)
                                else:
                                    st.info(f"No IMHBs detected in {sol}.")
                            except Exception as exc:
                                st.warning(f"IMHB heatmap failed for {sol}: {exc}")
                else:
                    st.info("No IMHB occurrence data available.")
                st.markdown("<br><br>", unsafe_allow_html=True)

            # ── π–π occurrence heatmap ────────────────────────────────────────
            if do_pi:
                _subsection(":material/key_visualizer:", "π–π Stacking Occurrence")
                st.caption(
                    "Binary heatmap of aromatic π–π stacking interactions across all conformers. "
                    "Rows sorted by descending frequency. Scroll horizontally to see all conformers."
                )
                pi_solvents = []
                for sol, sd in mol_data["solvents"].items():
                    for csv_p in sd.get("csvs", []):
                        p = Path(csv_p)
                        if p.name.endswith("_pi_label_ids.csv"):
                            pi_solvents.append((sol, p))
                            break

                if pi_solvents:
                    tabs = st.tabs([f":material/water_drop: {sol}" for sol, _ in pi_solvents])
                    for tab, (sol, p) in zip(tabs, pi_solvents):
                        with tab:
                            try:
                                html, height, legend_df = pi_occurrence_heatmap(p)
                                if html:
                                    components.html(html, height=height + 20, scrolling=False)
                                    st.caption("Legend")
                                    st.dataframe(legend_df, use_container_width=True, hide_index=True)
                                else:
                                    st.info(f"No π–π stacking detected in {sol}.")
                            except Exception as exc:
                                st.warning(f"π–π heatmap failed for {sol}: {exc}")
                else:
                    st.info("No π–π stacking occurrence data available.")
                st.markdown("<br><br>", unsafe_allow_html=True)

            # ── 3D viewer ─────────────────────────────────────────────────────
            if show_viewers:
                _subsection(":material/3d_rotation:", "3D Conformer Viewer")
                solvents_3d = [
                    (sol, sd)
                    for sol, sd in mol_data["solvents"].items()
                    if sd.get("html_3d")
                ]
                if solvents_3d:
                    tabs = st.tabs([f":material/water_drop: {sol}" for sol, _ in solvents_3d])
                    for tab, (sol, sd) in zip(tabs, solvents_3d):
                        with tab:
                            components.html(sd["html_3d"], height=740, scrolling=False)
                            for w in sd.get("warnings", []):
                                st.warning(w)
                else:
                    st.info("3D viewer unavailable.")
                    for sol, sd in mol_data["solvents"].items():
                        for w in sd.get("warnings", []):
                            st.warning(f"[{sol}]  {w}")
                st.markdown("<br><br>", unsafe_allow_html=True)

            # ── Summary TSV ───────────────────────────────────────────────────
            _subsection(":material/table_view:", "Summary Table")
            st.markdown("Conformers for all solvents are listed together with their properties.")
            if mol_data.get("tsv_df") is not None:
                df = mol_data["tsv_df"]
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    label="Download TSV",
                    icon=":material/download:",
                    data=df.to_csv(sep="\t", index=False).encode("utf-8"),
                    file_name=f"{mol_name}.tsv",
                    mime="text/tab-separated-values",
                    key=f"dl_tsv_{mol_name}",
                )
            elif mol_data.get("tsv_warning"):
                st.warning(mol_data["tsv_warning"])
            else:
                st.info("Summary table unavailable.")
            st.markdown("<br><br>", unsafe_allow_html=True)

            # ── ZIP of all CSVs ───────────────────────────────────────────────
            _subsection(":material/folder_zip:", "Download All CSVs")
            st.markdown('''
            * **`<molecule_name>.tsv`** — Summary table in TSV format.
            * **`<molecule_name>_<solvent>_rgyr.csv`** — Detailed Rgyr-related data.
            * **`<molecule_name>_<solvent>_3dpsa.csv`** — Detailed 3D-PSA-related data.
            * **`<molecule_name>_<solvent>_hbonds.csv`** — Thorough report on IMHBs.
            * **`<molecule_name>_<solvent>_hbonds_ids.csv`** — Frequency matrix of IMHBs.
            * **`<molecule_name>_<solvent>_hbonds_summary.csv`** — Summarized information on IMHBs.
            * **`<molecule_name>_<solvent>_ar_rings.csv`** — Aromatic rings information and geometry.
            * **`<molecule_name>_<solvent>_pi_stacking.csv`** — Information on π-π stackings.
            * **`<molecule_name>_<solvent>_pi_label_ids.csv`** — Frequency matrix of π-π stackings.
            * **`<molecule_name>_<solvent>_pi_summary.csv`** — Summarized information on π-π stackings.
            ''')
            try:
                zip_bytes = _make_zip(mol_data)
                st.download_button(
                    label="Download all CSVs (ZIP)",
                    icon=":material/download:",
                    data=zip_bytes,
                    file_name=f"{mol_name}_results.zip",
                    mime="application/zip",
                    key=f"dl_zip_{mol_name}",
                )
            except Exception as exc:
                st.warning(f"Could not create ZIP archive: {exc}")
            st.markdown("<br><br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if run_clicked and valid_files:
    prev = st.session_state.get("_tmp_dir")
    if prev:
        try:
            shutil.rmtree(prev, ignore_errors=True)
        except Exception:
            pass

    tmp = tempfile.mkdtemp(prefix="conformerly_")
    st.session_state["_tmp_dir"] = tmp

    try:
        st.session_state["results"] = _run_analysis(valid_files, Path(tmp))
    except Exception as exc:
        st.error(f"Analysis pipeline failed unexpectedly: {exc}")
        st.session_state["results"] = None

if st.session_state.get("results"):
    _display_results(st.session_state["results"])


show_footer(image="figures/footer.png", caption="")