#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""viewer3D.py

Molecular 3D structure viewer.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mol2_io import split_mol2_blocks, Mol2Block
from imhb import (
    parse_atoms_and_bonds,
    find_imhbs,
    identify_donors_acceptors_rdkit,
    _HAS_RDKIT as _IMHB_HAS_RDKIT,
    IMHB_MAX_DIST_HA,
    IMHB_MIN_ANGLE_DHA,
    IMHB_MIN_ANGLE_ACCEPTOR,
    IMHB_MIN_ANGLE_STRICT,
    IMHB_MIN_BOND_SEPARATION,
)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

try:
    from pi import (
        mol_from_mol2_block,
        detect_aromatic_ring_systems,
        compute_pair_outputs,
        coords_for_atoms,
        centroid as _centroid,
        fit_plane_normal,
    )
    _HAS_PI = True
except Exception:
    _HAS_PI = False

try:
    from rgyr import compute_rgyr as _compute_rgyr
    from rgyr import _parse_atoms as _parse_atoms_rgyr
    _HAS_RGYR = True
except Exception:
    _HAS_RGYR = False

try:
    from psa3d import (
        compute_3dpsa as _compute_3dpsa,
        compute_per_atom_exposure as _compute_exposure,
        _parse_atoms_and_bonds as _parse_psa3d,
        PSA3D_PROBE_SASA,
    )
    _HAS_PSA3D = True
except Exception:
    _HAS_PSA3D = False


ELEMENT_COLORS: Dict[str, str] = {
    "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "S": "#FFFF30",
    "F": "#90E050", "Cl": "#1FF01F", "Br": "#A62929", "I": "#940094",
    "P": "#FF8000", "H": "#FFFFFF",
}


def compute_rdkit_properties(block_text: str) -> dict:
    if not _HAS_RDKIT:
        return {}
    mol = Chem.MolFromMol2Block(block_text, sanitize=True, removeHs=False)
    if mol is None:
        return {}
    mol_noH = Chem.RemoveHs(mol)
    mw = Descriptors.ExactMolWt(mol_noH)
    tpsa = Descriptors.TPSA(mol_noH)
    n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol_noH)
    hba = rdMolDescriptors.CalcNumHBA(mol_noH)
    hbd = rdMolDescriptors.CalcNumHBD(mol_noH)
    logp = Descriptors.MolLogP(mol_noH)
    n_rings = rdMolDescriptors.CalcNumRings(mol_noH)
    n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol_noH)
    formula = rdMolDescriptors.CalcMolFormula(mol_noH)
    return {
        "molecular_formula": formula,
        "molecular_weight": round(mw, 3),
        "logP": round(logp, 2),
        "TPSA": round(tpsa, 2),
        "n_rotatable_bonds": n_rotatable,
        "n_HBD": hbd,
        "n_HBA": hba,
        "n_rings": n_rings,
        "n_aromatic_rings": n_aromatic_rings,
    }


def enumerate_donors_acceptors(atoms: Dict[int, dict], block_text: str) -> dict:
    donors: List[dict] = []
    acceptors: List[dict] = []
    donor_ids: set = set()
    acceptor_ids: set = set()

    if _HAS_RDKIT and _IMHB_HAS_RDKIT:
        try:
            donor_ids, acceptor_ids = identify_donors_acceptors_rdkit(block_text)
        except Exception:
            pass

    for atom_id, a in sorted(atoms.items()):
        sybyl = a["type"]
        if atom_id in acceptor_ids:
            acceptors.append({"id": atom_id, "name": a["name"], "type": sybyl})
        if atom_id in donor_ids:
            h_count = sum(1 for nb in a["bonds"] if atoms.get(nb, {}).get("element") == "H")
            donors.append({"id": atom_id, "name": a["name"], "type": sybyl, "n_H": h_count})
    return {"donors": donors, "acceptors": acceptors}


def build_ensemble_scene(blocks: List[Mol2Block], do_imhb: bool = True, do_pi: bool = True) -> dict:
    if not blocks:
        return {}

    first_atoms = parse_atoms_and_bonds(blocks[0].text)
    if not first_atoms:
        return {}

    atom_list = []
    bond_list = []
    seen_bonds = set()

    for aid, a in sorted(first_atoms.items()):
        elem = a["element"]
        atom_list.append({
            "id": aid,
            "name": a["name"],
            "element": elem,
            "type": a["type"],
            "color": ELEMENT_COLORS.get(elem, "#FF69B4"),
        })
        for nb_id in a["bonds"]:
            key = (min(aid, nb_id), max(aid, nb_id))
            if key not in seen_bonds:
                seen_bonds.add(key)
                bond_list.append({"from": key[0], "to": key[1]})

    conformers = []
    names = []
    hbonds_per_conf: List[List[dict]] = []
    pi_per_conf: List[List[dict]] = []
    rgyr_per_conf: List[dict] = []
    psa3d_per_conf: List[dict] = []
    exposure_per_conf: List[List[float]] = []

    pi_params = {
        "pi_ff_max_dist": 4.4,
        "pi_ef_max_dist": 5.5,
        "pi_parallel_angle": 30.0,
        "pi_t_angle_min": 60.0,
    }

    rdkit_donor_ids = None
    rdkit_acceptor_ids = None
    if do_imhb and _HAS_RDKIT and _IMHB_HAS_RDKIT:
        try:
            rdkit_donor_ids, rdkit_acceptor_ids = identify_donors_acceptors_rdkit(blocks[0].text)
        except Exception:
            pass

    for blk in blocks:
        atoms = parse_atoms_and_bonds(blk.text)
        if not atoms or len(atoms) != len(first_atoms):
            continue

        coords = []
        for aid in sorted(atoms.keys()):
            c = atoms[aid]["coords"]
            coords.append([round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)])
        conformers.append(coords)
        names.append(blk.name)

        if do_imhb:
            try:
                hb_rows = find_imhbs(
                    atoms=atoms, max_dist_ha=IMHB_MAX_DIST_HA, min_angle_dha=IMHB_MIN_ANGLE_DHA,
                    min_angle_strict=IMHB_MIN_ANGLE_STRICT, min_bond_separation=IMHB_MIN_BOND_SEPARATION,
                    min_angle_acceptor=IMHB_MIN_ANGLE_ACCEPTOR, rdkit_donor_ids=rdkit_donor_ids, rdkit_acceptor_ids=rdkit_acceptor_ids,
                )
                hbonds_json = []
                for r in hb_rows:
                    h_id = r["hydrogen_id"]
                    a_id = r["acceptor_id"]
                    h_coords = atoms[h_id]["coords"]
                    a_coords = atoms[a_id]["coords"]
                    hbonds_json.append({
                        "label": r["hb_label"], "donor_name": r["donor_name"], "donor_type": r["donor_type"],
                        "hydrogen_name": r["hydrogen_name"], "hydrogen_id": h_id, "acceptor_name": r["acceptor_name"],
                        "acceptor_type": r["acceptor_type"], "acceptor_id": a_id, "donor_id": r["donor_id"],
                        "dist_HA": round(r["dist_HA"], 3), "dist_DA": round(r.get("dist_DA", float("nan")), 3),
                        "angle_DHA": round(r["angle_DHA"], 1), "angle_XAH": round(r["angle_XAH"], 1),
                        "bond_sep": r["bond_separation_D_A"], "range_class": r["distance_range_class"],
                        "pseudoring_type": r.get("pseudoring_type", ""),
                        "h_xyz": [round(float(h_coords[0]), 4), round(float(h_coords[1]), 4), round(float(h_coords[2]), 4)],
                        "a_xyz": [round(float(a_coords[0]), 4), round(float(a_coords[1]), 4), round(float(a_coords[2]), 4)],
                    })
                hbonds_per_conf.append(hbonds_json)
            except Exception:
                hbonds_per_conf.append([])
        else:
            hbonds_per_conf.append([])

        if do_pi:
            try:
                if _HAS_PI:
                    mol = mol_from_mol2_block(blk.text)
                    moieties, patches_by_system = detect_aromatic_ring_systems(mol)
                    pi_rows, _, _ = compute_pair_outputs(mol=mol, moieties=moieties, patches_by_system=patches_by_system, min_bond_sep=1, use_pi_criteria=True, pi_params=pi_params)
                    pi_json = []
                    for r in pi_rows:
                        pa_label = r.get("patch_a", "")
                        pb_label = r.get("patch_b", "")
                        ca = cb = None
                        for sys_label, plist in patches_by_system.items():
                            for p in plist:
                                if p.label == pa_label: ca = _centroid(coords_for_atoms(mol, p.atom_indices)).tolist()
                                if p.label == pb_label: cb = _centroid(coords_for_atoms(mol, p.atom_indices)).tolist()
                        if ca and cb:
                            pi_json.append({
                                "moiety_a": r.get("moiety_a", ""), "moiety_b": r.get("moiety_b", ""),
                                "patch_a": pa_label, "patch_b": pb_label, "pi_class": r.get("pi_class", ""),
                                "centroid_dist": round(float(r.get("patch_centroid_dist_A", 0)), 3),
                                "plane_angle": round(float(r.get("plane_angle_deg", 0)), 1),
                                "ca_xyz": [round(float(ca[0]), 4), round(float(ca[1]), 4), round(float(ca[2]), 4)],
                                "cb_xyz": [round(float(cb[0]), 4), round(float(cb[1]), 4), round(float(cb[2]), 4)],
                            })
                    pi_per_conf.append(pi_json)
                else: 
                    pi_per_conf.append([])
            except Exception: 
                pi_per_conf.append([])
        else:
            pi_per_conf.append([])

        try:
            if _HAS_RGYR:
                rgyr_atoms = _parse_atoms_rgyr(blk.text)
                rg_all, _, _ = _compute_rgyr(rgyr_atoms, heavy_only=False)
                rg_heavy, _, _ = _compute_rgyr(rgyr_atoms, heavy_only=True)
                rgyr_per_conf.append({"rgyr_all": round(rg_all, 3), "rgyr_heavy": round(rg_heavy, 3)})
            else: rgyr_per_conf.append({})
        except Exception: rgyr_per_conf.append({})

        try:
            if _HAS_PSA3D:
                psa_atoms = _parse_psa3d(blk.text)
                sasa_tot, psa_sasa, _, n_polar = _compute_3dpsa(psa_atoms, probe_radius=1.4, n_sphere_points=480)
                mol_tot, psa_mol, _, _ = _compute_3dpsa(psa_atoms, probe_radius=0.0, n_sphere_points=480)
                psa3d_per_conf.append({
                    "psa3d_sasa": round(psa_sasa, 1), "psa3d_molsurf": round(psa_mol, 1),
                    "sasa_total": round(sasa_tot, 1), "molsurf_total": round(mol_tot, 1), "n_polar_atoms": n_polar,
                })
                exp, _ = _compute_exposure(psa_atoms, probe_radius=1.4, n_sphere_points=480)
                exposure_per_conf.append(exp)
            else:
                psa3d_per_conf.append({})
                exposure_per_conf.append([])
        except Exception:
            psa3d_per_conf.append({})
            exposure_per_conf.append([])

    return {
        "topology": {"atoms": atom_list, "bonds": bond_list},
        "conformers": conformers,
        "names": names,
        "hbonds": hbonds_per_conf,
        "pi_stacking": pi_per_conf,
        "rgyr": rgyr_per_conf,
        "psa3d": psa3d_per_conf,
        "surface_exposure": exposure_per_conf,
    }

def build_stats(atoms: Dict[int, dict], total_conformers: int) -> dict:
    elements: Dict[str, int] = {}
    sybyl_types: Dict[str, int] = {}
    seen = set()
    for aid, a in atoms.items():
        elements[a["element"]] = elements.get(a["element"], 0) + 1
        sybyl_types[a["type"]] = sybyl_types.get(a["type"], 0) + 1
        for nb in a["bonds"]:
            key = (min(aid, nb), max(aid, nb))
            seen.add(key)
    return {
        "total_conformers": total_conformers,
        "n_atoms": len(atoms),
        "n_bonds": len(seen),
        "n_heavy_atoms": sum(1 for a in atoms.values() if a["element"] != "H"),
        "elements": dict(sorted(elements.items())),
        "sybyl_types": dict(sorted(sybyl_types.items())),
    }


def generate_html(scene: dict, stats: dict, rdkit_props: dict,
                  donors_acceptors: dict, title: str,
                  output_files: Optional[List[Dict[str, str]]] = None) -> str:
    scene_json = json.dumps(scene)
    stats_json = json.dumps(stats)
    props_json = json.dumps(rdkit_props)
    da_json = json.dumps(donors_acceptors)
    files_json = json.dumps(output_files or [])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — 3D Viewer</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Source+Sans+3:wght@300;400;500;600;700&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  :root {{
    --bg: #0c0f13; --surface: #14181f; --surface2: #1a1f28;
    --border: #272d3a; --border-light: #333b4d;
    --text: #e2e8f0; --text-dim: #8892a4; --text-faint: #5a6478;
    --accent: #60a5fa; --accent-dim: #3b82f6;
    --green: #34d399; --amber: #fbbf24; --rose: #fb7185;
    --hb-color: #fbbf24; --pi-color: #60a5fa; --surf-color: #34d399;
    --mono: 'IBM Plex Mono', monospace; --sans: 'Source Sans 3', system-ui, sans-serif;
  }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}

  .topbar {{
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 10px 20px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
  }}
  .topbar h1 {{ font-size: 14px; font-weight: 600; letter-spacing: -0.3px; }}
  .badge {{
    background: linear-gradient(135deg, var(--accent-dim), var(--accent));
    color: #fff; font-size: 10px; font-weight: 600;
    padding: 2px 8px; border-radius: 10px; font-family: var(--mono);
    text-transform: uppercase; letter-spacing: 0.5px;
  }}
  .slider-container {{
    display: flex; align-items: center; gap: 10px;
    background: var(--surface2); padding: 4px 12px; border-radius: 6px; border: 1px solid var(--border);
  }}
  .slider-container input[type=range] {{ width: 180px; accent-color: var(--accent); }}

  .toggle-group {{ display: flex; align-items: center; gap: 8px; }}
  .toggle-label {{ font-size: 11px; font-family: var(--mono); color: var(--text-dim); }}
  .toggle {{ position: relative; width: 34px; height: 18px; cursor: pointer; }}
  .toggle input {{ opacity: 0; width: 0; height: 0; }}
  .toggle .slider {{
    position: absolute; inset: 0; background: var(--border); border-radius: 9px;
    transition: background 0.2s;
  }}
  .toggle .slider::before {{
    content: ''; position: absolute; height: 14px; width: 14px;
    left: 2px; bottom: 2px; background: var(--text-dim); border-radius: 50%;
    transition: transform 0.2s, background 0.2s;
  }}
  .toggle input:checked + .slider {{ background: var(--accent-dim); }}
  .toggle input:checked + .slider::before {{ transform: translateX(16px); background: #fff; }}

  .main {{ flex: 1; display: flex; overflow: hidden; }}
  .viewport {{ flex: 1; position: relative; background: #0A0E13; }}
  #viewer {{ width: 100%; height: 100%; }}
  .viewport-hint {{
    position: absolute; bottom: 10px; left: 10px;
    font-size: 11px; color: rgba(255,255,255,0.25); font-family: var(--mono); pointer-events: none;
  }}

  .panel {{
    width: 380px; min-width: 380px; background: var(--surface);
    border-left: 1px solid var(--border); overflow-y: auto;
    scrollbar-width: thin; scrollbar-color: var(--border) transparent;
  }}
  .section {{ padding: 14px 18px; border-bottom: 1px solid var(--border); }}
  .section-title {{
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.2px; color: var(--text-faint); margin-bottom: 10px;
  }}
  .prop-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }}
  .prop-item {{
    display: flex; flex-direction: column; gap: 1px;
    padding: 6px 8px; border-radius: 6px; background: var(--surface2);
    border: 1px solid transparent; transition: border-color 0.15s;
  }}
  .prop-item:hover {{ border-color: var(--border-light); }}
  .prop-label {{ font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.4px; }}
  .prop-value {{ font-family: var(--mono); font-size: 13px; font-weight: 500; }}
  .prop-item.wide {{ grid-column: span 2; }}

  .da-list {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }}
  .da-tag {{ font-family: var(--mono); font-size: 11px; font-weight: 500; padding: 2px 7px; border-radius: 4px; border: 1px solid var(--border); }}
  .da-tag.donor {{ background: rgba(251,191,36,0.08); color: var(--amber); border-color: rgba(251,191,36,0.2); }}
  .da-tag.acceptor {{ background: rgba(96,165,250,0.08); color: var(--accent); border-color: rgba(96,165,250,0.2); }}

  .int-table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 11px; }}
  .int-table th {{ text-align: left; font-weight: 600; font-size: 10px; color: var(--text-faint); text-transform: uppercase; padding: 4px 6px; border-bottom: 1px solid var(--border); }}
  .int-table td {{ padding: 5px 6px; font-family: var(--mono); font-size: 11px; border-bottom: 1px solid rgba(39,45,58,0.4); vertical-align: top; }}
  .int-table tr:hover td {{ background: rgba(96,165,250,0.04); }}
  .hb-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--hb-color); margin-right: 4px; }}
  .pi-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--pi-color); margin-right: 4px; }}
  .tag-short {{ color: var(--rose); }} .tag-medium {{ color: var(--amber); }} .tag-long {{ color: var(--green); }}
  .tag-ff {{ color: var(--pi-color); }} .tag-ef {{ color: var(--green); }}
  .empty-msg {{ font-size: 11px; color: var(--text-faint); font-style: italic; padding: 4px 0; }}

  .rdkit-badge {{ display: inline-block; font-size: 8px; font-weight: 600; padding: 1px 5px; border-radius: 3px; margin-left: 6px; background: rgba(96,165,250,0.12); color: var(--accent); border: 1px solid rgba(96,165,250,0.2); font-family: var(--mono); }}

  .dl-grid {{ display: flex; flex-direction: column; gap: 4px; margin-top: 6px; }}
  .dl-btn {{
    display: flex; align-items: center; justify-content: center; gap: 8px;
    padding: 6px 10px; border-radius: 6px;
    background: var(--surface2); border: 1px solid var(--border);
    color: var(--text); font-family: var(--mono); font-size: 11px;
    cursor: pointer; transition: background 0.15s, border-color 0.15s; width: 100%;
  }}
  .dl-btn:hover {{ background: rgba(96,165,250,0.08); border-color: var(--accent-dim); }}
  .dl-name {{ flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: left; }}
  .dl-size {{ font-size: 9px; color: var(--text-faint); white-space: nowrap; }}

  .surf-legend {{ display: flex; align-items: center; gap: 6px; margin-top: 8px; }}
  .surf-bar {{
    flex: 1; height: 10px; border-radius: 5px;
    background: linear-gradient(90deg, #FFFF00, #FFFFFF, #00FFFF);
    border: 1px solid var(--border);
  }}
  .surf-legend-label {{ font-size: 9px; color: var(--text-faint); font-family: var(--mono); white-space: nowrap; }}
</style>
</head>
<body>

<div class="topbar">
  <h1>{title}</h1>
  <span class="badge">3D Conformer</span>
  <div style="flex:1"></div>

  <button class="dl-btn" id="btnBg" style="width:auto; padding:4px 8px; font-size:16px;" title="Toggle Background">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
  </button>
  <button class="dl-btn" id="btnShot" style="width:auto; padding:4px 8px; font-size:16px;" title="Take Screenshot">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
  </button>

  <div class="toggle-group" style="margin-left:8px;">
    <label class="toggle"><input type="checkbox" id="toggleHB" checked><span class="slider"></span></label>
    <span class="toggle-label" style="color:var(--hb-color);">H-bonds</span>
  </div>
  <div class="toggle-group">
    <label class="toggle"><input type="checkbox" id="togglePi" checked><span class="slider"></span></label>
    <span class="toggle-label" style="color:var(--pi-color);">&pi;-stacking</span>
  </div>
  <div class="toggle-group">
    <label class="toggle"><input type="checkbox" id="toggleSurf"><span class="slider"></span></label>
    <span class="toggle-label" style="color:var(--surf-color);">Surface</span>
  </div>

  <div class="slider-container" id="slider-ui" style="display:none;">
    <span style="font-size:11px; color:var(--text-dim); font-family:var(--mono);">Conf:</span>
    <input type="range" id="confSlider" min="0" max="0" value="0">
    <span id="confLabel" style="font-family:var(--mono); font-size:12px; min-width:50px; text-align:right;">1</span>
  </div>
</div>

<div class="main">
  <div class="viewport">
    <div id="viewer"></div>
    <div class="viewport-hint">Drag to rotate &middot; Scroll to zoom &middot; Hover for labels</div>
  </div>
  <div class="panel" id="panel"></div>
</div>

<script>
const SCENE = {scene_json};
const STATS = {stats_json};
const PROPS = {props_json};
const DA    = {da_json};
const OUTPUT_FILES = {files_json};

let viewer, model;
let hbShapes = [];
let piShapes = [];
let showHB = true, showPi = true, showSurf = false;
let currentIdx = 0;
let surfaceObj = null;
let isDarkBg = true;

/* ---- Exposure → colour (Yellow=buried → White → Cyan=exposed) ---- */
function exposureColor(frac) {{
  let r, g, b;
  if (frac <= 0.5) {{
    const t = frac / 0.5;
    r = 255; 
    g = 255; 
    b = Math.round(255 * t);
  }} else {{
    const t = (frac - 0.5) / 0.5;
    r = Math.round(255 * (1 - t)); 
    g = 255; 
    b = 255;
  }}
  return "rgb(" + r + "," + g + "," + b + ")";
}}

function initViewer() {{
  viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "#0A0E13", antialias: true, preserveDrawingBuffer: true }});
  loadConformer(0);
  setupUI();
  updatePanel(0);
}}

function loadConformer(idx) {{
  viewer.removeAllModels();
  viewer.removeAllShapes();
  viewer.removeAllSurfaces();
  hbShapes = [];
  piShapes = [];
  surfaceObj = null;

  model = viewer.addModel();
  const top = SCENE.topology;
  const coords = SCENE.conformers[idx];
  const atomMap = {{}};

  const exp = SCENE.surface_exposure[idx] || [];

  const jsAtoms = top.atoms.map((a, i) => {{
    atomMap[a.id] = i;
    return {{
      elem: a.element, x: coords[i][0], y: coords[i][1], z: coords[i][2],
      atom: a.name, serial: a.id,
      properties: {{ sybylType: a.type, exposure: (exp[i] !== undefined ? exp[i] : 0.5) }},
      color: a.color, bonds: [], bondOrder: []
    }};
  }});

  top.bonds.forEach(b => {{
    const i1 = atomMap[b.from], i2 = atomMap[b.to];
    if (i1 !== undefined && i2 !== undefined) {{
      jsAtoms[i1].bonds.push(i2); jsAtoms[i1].bondOrder.push(1);
      jsAtoms[i2].bonds.push(i1); jsAtoms[i2].bondOrder.push(1);
    }}
  }});

  model.addAtoms(jsAtoms);
  viewer.setStyle({{}}, {{ stick: {{ radius: 0.14, colorscheme: "Jmol" }} }});

  viewer.setHoverable({{}}, true,
    function(atom, v, e, c) {{
      if (!atom.labelObj) {{
        const expVal = atom.properties.exposure !== undefined ? " | exp: " + (atom.properties.exposure * 100).toFixed(0) + "%" : "";
        atom.labelObj = v.addLabel(atom.atom + " [" + (atom.properties.sybylType || "") + "]" + expVal,
          {{ position: {{ x: atom.x, y: atom.y, z: atom.z }}, backgroundColor: "rgba(0,0,0,0.85)", fontColor: "white", fontSize: 12, backgroundOpacity: 0.85, inFront: true, borderColor: "rgba(96,165,250,0.5)", borderThickness: 1 }}
        );
      }}
    }},
    function(atom, v) {{ if (atom.labelObj) {{ v.removeLabel(atom.labelObj); delete atom.labelObj; }} }}
  );

  drawHBonds(idx);
  drawPiStacking(idx);
  if (showSurf && exp.length > 0) drawSurface();

  viewer.zoomTo();
  viewer.render();
}}

function drawHBonds(idx) {{
  const hbs = SCENE.hbonds[idx] || [];
  hbs.forEach(hb => {{
    hbShapes.push(viewer.addCylinder({{ start: {{ x: hb.h_xyz[0], y: hb.h_xyz[1], z: hb.h_xyz[2] }}, end: {{ x: hb.a_xyz[0], y: hb.a_xyz[1], z: hb.a_xyz[2] }}, radius: 0.04, color: "#fbbf24", dashed: true, dashLength: 0.15, gapLength: 0.1, fromCap: 1, toCap: 1, hidden: !showHB }}));
  }});
}}

function drawPiStacking(idx) {{
  const pis = SCENE.pi_stacking[idx] || [];
  pis.forEach(pi => {{
    piShapes.push(viewer.addCylinder({{ start: {{ x: pi.ca_xyz[0], y: pi.ca_xyz[1], z: pi.ca_xyz[2] }}, end: {{ x: pi.cb_xyz[0], y: pi.cb_xyz[1], z: pi.cb_xyz[2] }}, radius: 0.04, color: "#60a5fa", dashed: true, dashLength: 0.2, gapLength: 0.12, fromCap: 1, toCap: 1, hidden: !showPi }}));
  }});
}}

function drawSurface() {{
  surfaceObj = viewer.addSurface($3Dmol.SurfaceType.VDW, {{
    opacity: 0.90,
    colorfunc: function(atom) {{
      const e = (atom && atom.properties && atom.properties.exposure !== undefined) ? atom.properties.exposure : 0.5;
      const col = exposureColor(e);
      const m = col.match(/rgb\((\d+),(\d+),(\d+)\)/);
      if (m) return new $3Dmol.Color(parseInt(m[1]) / 255, parseInt(m[2]) / 255, parseInt(m[3]) / 255);
      return new $3Dmol.Color(0.8, 0.8, 0.8);
    }}
  }});
}}

function setupUI() {{
  const maxIdx = SCENE.conformers.length - 1;
  if (maxIdx > 0) {{
    document.getElementById("slider-ui").style.display = "flex";
    const slider = document.getElementById("confSlider");
    slider.max = maxIdx;
    slider.addEventListener("input", (e) => {{
      currentIdx = parseInt(e.target.value);
      document.getElementById("confLabel").innerText = (currentIdx + 1) + " / " + (maxIdx + 1);
      loadConformer(currentIdx);
      updatePanel(currentIdx);
    }});
  }}

  document.getElementById("toggleHB").addEventListener("change", (e) => {{ showHB = e.target.checked; loadConformer(currentIdx); viewer.render(); }});
  document.getElementById("togglePi").addEventListener("change", (e) => {{ showPi = e.target.checked; loadConformer(currentIdx); viewer.render(); }});
  document.getElementById("toggleSurf").addEventListener("change", (e) => {{ showSurf = e.target.checked; loadConformer(currentIdx); viewer.render(); }});

  document.getElementById("btnBg").addEventListener("click", () => {{
    isDarkBg = !isDarkBg;
    viewer.setBackgroundColor(isDarkBg ? "#0A0E13" : "#FFFFFF");
    viewer.render();
  }});

  document.getElementById("btnShot").addEventListener("click", () => {{
    const uri = viewer.pngURI();
    const a = document.createElement("a");
    a.download = SCENE.names[currentIdx] + "_3D_screenshot.png";
    a.href = uri;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }});
}}

function updatePanel(idx) {{
  const s = STATS, currentName = SCENE.names[idx] || "N/A", hbs = SCENE.hbonds[idx] || [], pis = SCENE.pi_stacking[idx] || [], rg = SCENE.rgyr[idx] || {{}}, psa = SCENE.psa3d[idx] || {{}};
  let html = `<div class="section"><div class="section-title">Molecule</div><div style="font-family:var(--mono); font-size:14px; font-weight:600;">${{currentName}}</div><div style="font-size:12px; color:var(--text-dim); margin-top:3px;">${{s.total_conformers}} conformer(s) &middot; viewing #${{idx+1}}</div></div>`;

  if (PROPS.molecular_formula) {{
    html += `<div class="section"><div class="section-title">Molecular Properties (RDKit)</div><div class="prop-grid">
      <div class="prop-item wide"><span class="prop-label">Formula</span><span class="prop-value">${{PROPS.molecular_formula}}</span></div>
      <div class="prop-item"><span class="prop-label">MW (Da)</span><span class="prop-value">${{PROPS.molecular_weight}}</span></div>
      <div class="prop-item"><span class="prop-label">cLogP</span><span class="prop-value">${{PROPS.logP}}</span></div>
      <div class="prop-item"><span class="prop-label">TPSA (\u00C5\u00B2)</span><span class="prop-value">${{PROPS.TPSA}}</span></div>
      <div class="prop-item"><span class="prop-label">nRotB</span><span class="prop-value">${{PROPS.n_rotatable_bonds}}</span></div>
      <div class="prop-item"><span class="prop-label">HBD</span><span class="prop-value">${{PROPS.n_HBD}}</span></div>
      <div class="prop-item"><span class="prop-label">HBA</span><span class="prop-value">${{PROPS.n_HBA}}</span></div>
      <div class="prop-item"><span class="prop-label">Rings</span><span class="prop-value">${{PROPS.n_rings}} (${{PROPS.n_aromatic_rings}} arom.)</span></div>
    </div></div>`;
  }}

  html += `<div class="section"><div class="section-title">Parsed Counts</div><div class="prop-grid"><div class="prop-item"><span class="prop-label">Total atoms</span><span class="prop-value">${{s.n_atoms}}</span></div><div class="prop-item"><span class="prop-label">Heavy atoms</span><span class="prop-value">${{s.n_heavy_atoms}}</span></div><div class="prop-item"><span class="prop-label">Bonds</span><span class="prop-value">${{s.n_bonds}}</span></div></div></div>`;

  if (rg.rgyr_all !== undefined || psa.psa3d_sasa !== undefined) {{
    html += `<div class="section"><div class="section-title">Shape &amp; Polarity &mdash; conformer #${{idx+1}}</div><div class="prop-grid">`;
    if (rg.rgyr_all !== undefined) html += `<div class="prop-item"><span class="prop-label">Rgyr (all)</span><span class="prop-value">${{rg.rgyr_all}} \u00C5</span></div><div class="prop-item"><span class="prop-label">Rgyr (heavy)</span><span class="prop-value">${{rg.rgyr_heavy}} \u00C5</span></div>`;
    if (psa.psa3d_sasa !== undefined) html += `<div class="prop-item"><span class="prop-label">3D PSA (SASA)</span><span class="prop-value">${{psa.psa3d_sasa}} \u00C5\u00B2</span></div><div class="prop-item"><span class="prop-label">3D PSA (mol)</span><span class="prop-value">${{psa.psa3d_molsurf}} \u00C5\u00B2</span></div><div class="prop-item"><span class="prop-label">SASA total</span><span class="prop-value">${{psa.sasa_total}} \u00C5\u00B2</span></div><div class="prop-item"><span class="prop-label">Molsurf total</span><span class="prop-value">${{psa.molsurf_total}} \u00C5\u00B2</span></div><div class="prop-item"><span class="prop-label">Polar atoms</span><span class="prop-value">${{psa.n_polar_atoms}}</span></div><div class="prop-item"><span class="prop-label">Polar frac (SASA)</span><span class="prop-value">${{psa.sasa_total > 0 ? (100 * psa.psa3d_sasa / psa.sasa_total).toFixed(1) : 0}}%</span></div>`;
    html += `</div><div style="margin-top:10px; font-size:10px; color:var(--text-dim);">Surface toggle colours atom exposure (SASA):</div><div class="surf-legend"><span class="surf-legend-label">Buried</span><div class="surf-bar"></div><span class="surf-legend-label">Exposed</span></div></div>`;
  }}

  if (DA.donors.length > 0) {{ html += `<div class="section"><div class="section-title">H-Bond Donors (${{DA.donors.length}})<span class="rdkit-badge">RDKit</span></div><div class="da-list">`; DA.donors.forEach(d => html += `<span class="da-tag donor" title="${{d.type}}">${{d.name}}</span>`); html += `</div></div>`; }}
  if (DA.acceptors.length > 0) {{ html += `<div class="section"><div class="section-title">H-Bond Acceptors (${{DA.acceptors.length}})<span class="rdkit-badge">RDKit</span></div><div class="da-list">`; DA.acceptors.forEach(a => html += `<span class="da-tag acceptor" title="${{a.type}}">${{a.name}}</span>`); html += `</div></div>`; }}

  html += `<div class="section"><div class="section-title"><span class="hb-dot"></span>Intramolecular H-Bonds &mdash; #${{idx+1}} (${{hbs.length}})</div>`;
  if (hbs.length > 0) {{
    html += `<table class="int-table"><tr><th>D&ndash;H&middot;&middot;&middot;A</th><th>d(H&middot;&middot;&middot;A)</th><th>&ang;DHA</th><th>&ang;XAH</th><th>Sep</th><th>Range</th></tr>`;
    hbs.forEach(hb => html += `<tr><td>${{hb.donor_name}}&ndash;${{hb.hydrogen_name}}&middot;&middot;&middot;${{hb.acceptor_name}}</td><td>${{hb.dist_HA.toFixed(2)}} \u00C5</td><td>${{hb.angle_DHA.toFixed(1)}}&deg;</td><td>${{hb.angle_XAH.toFixed(1)}}&deg;</td><td>${{hb.bond_sep}}</td><td><span class="${{hb.range_class === 'short-range' ? 'tag-short' : hb.range_class === 'medium-range' ? 'tag-medium' : 'tag-long'}}">${{hb.range_class}}</span></td></tr>`);
    html += `</table>`;
  }} else {{ html += `<div class="empty-msg">No H-bonds detected in this conformer.</div>`; }}
  html += `</div>`;

  html += `<div class="section"><div class="section-title"><span class="pi-dot"></span>&pi;&ndash;&pi; Stacking &mdash; #${{idx+1}} (${{pis.length}})</div>`;
  if (pis.length > 0) {{
    html += `<table class="int-table"><tr><th>Patches</th><th>Class</th><th>d(c&middot;&middot;&middot;c)</th><th>&ang;planes</th></tr>`;
    pis.forEach(pi => html += `<tr><td>${{pi.patch_a.replace('ring_system_','RS').replace('.patch_','p')}} &harr; ${{pi.patch_b.replace('ring_system_','RS').replace('.patch_','p')}}</td><td><span class="${{pi.pi_class === 'face-to-face' ? 'tag-ff' : 'tag-ef'}}">${{pi.pi_class}}</span></td><td>${{pi.centroid_dist.toFixed(2)}} \u00C5</td><td>${{pi.plane_angle.toFixed(1)}}&deg;</td></tr>`);
    html += `</table>`;
  }} else {{ html += `<div class="empty-msg">No &pi;&ndash;&pi; stacking detected.</div>`; }}
  html += `</div>`;

  if (OUTPUT_FILES.length > 0) {{
    html += `<div class="section" style="border-bottom:none;"><div class="section-title">Output Files (${{OUTPUT_FILES.length}})</div><div class="dl-grid">`;
    OUTPUT_FILES.forEach((f, i) => html += `<button class="dl-btn" data-file-idx="${{i}}" title="Download ${{f.name}}"><svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2v8m0 0l-3-3m3 3l3-3M3 13h10"/></svg><span class="dl-name">${{f.name}}</span><span class="dl-size">${{(new Blob([f.content]).size / 1024).toFixed(1)}} KB</span></button>`);
    html += `</div></div>`;
  }}
  document.getElementById("panel").innerHTML = html;

  document.querySelectorAll(".dl-btn[data-file-idx]").forEach(btn => {{
    btn.addEventListener("click", () => {{
      const f = OUTPUT_FILES[parseInt(btn.getAttribute("data-file-idx"))];
      if (!f || !f.content) return;
      const url = URL.createObjectURL(new Blob([f.content], {{ type: "text/csv;charset=utf-8" }}));
      const a = document.createElement("a"); a.href = url; a.download = f.name;
      document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
    }});
  }});
}}

document.addEventListener("DOMContentLoaded", initViewer);
</script>
</body>
</html>"""

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate an interactive 3D view of a MOL2 ensemble.")
    p.add_argument("mol2_path", help="Input MOL2 file.")
    p.add_argument("-o", "--output", default=None, help="Output HTML path.")
    p.add_argument("--open", action="store_true", help="Open HTML in browser.")
    args = p.parse_args(argv)

    mol2_path = Path(args.mol2_path).expanduser().resolve()
    blocks = split_mol2_blocks(mol2_path)
    if not blocks: return 1

    # When run as a standalone script, defaults to True for both computations
    scene = build_ensemble_scene(blocks, True, True)
    if not scene: return 1

    atoms = parse_atoms_and_bonds(blocks[0].text)
    stats = build_stats(atoms, len(blocks))
    rdkit_props = compute_rdkit_properties(blocks[0].text)
    da = enumerate_donors_acceptors(atoms, blocks[0].text)

    out_path = Path(args.output) if args.output else mol2_path.with_name(f"{mol2_path.stem}_3D.html")
    out_path.write_text(generate_html(scene, stats, rdkit_props, da, mol2_path.stem), encoding="utf-8")
    print(f"Wrote: {out_path}")
    if args.open: webbrowser.open(str(out_path))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
