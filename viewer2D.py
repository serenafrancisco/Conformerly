#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""viewer2D.py

Molecular 2D topology viewer.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Set, Sequence, Tuple

from mol2_io import split_mol2_blocks, Mol2Block
from imhb import (
    parse_atoms_and_bonds,
    identify_donors_acceptors_rdkit,
    _HAS_RDKIT as _IMHB_HAS_RDKIT,
)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

try:
    from pi import (
        mol_from_mol2_block,
        detect_aromatic_rings,
        AromaticRing,
    )
    _HAS_PI = True
except Exception:
    _HAS_PI = False


_RING_SYSTEM_COLORS = [
    "#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", 
    "#e67e22", "#e84393", "#00cec9", "#6c5ce7", "#fdcb6e", "#00b894",
]

def compute_rdkit_properties(mol) -> dict:
    if mol is None: return {}
    mol_noH = Chem.RemoveHs(mol)
    return {
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol_noH),
        "molecular_weight": round(Descriptors.ExactMolWt(mol_noH), 3),
        "logP": round(Descriptors.MolLogP(mol_noH), 2),
        "TPSA": round(Descriptors.TPSA(mol_noH), 2),
        "n_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol_noH),
        "n_HBD": rdMolDescriptors.CalcNumHBD(mol_noH),
        "n_HBA": rdMolDescriptors.CalcNumHBA(mol_noH),
        "n_rings": rdMolDescriptors.CalcNumRings(mol_noH),
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol_noH),
    }

def enumerate_donors_acceptors(atoms: Dict[int, dict], block_text: str) -> dict:
    donors, acceptors, donor_ids, acceptor_ids = [], [], set(), set()
    if _HAS_RDKIT and _IMHB_HAS_RDKIT:
        try:
            donor_ids, acceptor_ids = identify_donors_acceptors_rdkit(block_text)
        except Exception: pass
    for atom_id, a in sorted(atoms.items()):
        if atom_id in acceptor_ids:
            acceptors.append({"id": atom_id, "name": a["name"], "type": a["type"]})
        if atom_id in donor_ids:
            h_count = sum(1 for nb in a["bonds"] if atoms.get(nb, {}).get("element") == "H")
            donors.append({"id": atom_id, "name": a["name"], "type": a["type"], "n_H": h_count})
    return {"donors": donors, "acceptors": acceptors}

def detect_ring_systems_for_viz(block_text: str) -> List[dict]:
    """Return a list of dicts describing each individual aromatic ring for the 2D viewer."""
    if not (_HAS_RDKIT and _HAS_PI): return []
    try:
        mol = mol_from_mol2_block(block_text)
        rings = detect_aromatic_rings(mol)
        result = []
        for i, r in enumerate(rings):
            color = _RING_SYSTEM_COLORS[i % len(_RING_SYSTEM_COLORS)]
            atom_names = [
                mol.GetAtomWithIdx(int(idx)).GetProp("_TriposAtomName")
                if mol.GetAtomWithIdx(int(idx)).HasProp("_TriposAtomName")
                else mol.GetAtomWithIdx(int(idx)).GetSymbol()
                for idx in r.atom_indices
            ]
            result.append({
                "label":             r.label,
                "atom_indices_0based": list(r.atom_indices),
                "n_atoms":           len(r.atom_indices),
                "n_patches":         1,          # each ring is its own single patch
                "color":             color,
                "atom_names":        atom_names,
                "patches":           [{"label": r.label, "atom_names": atom_names}],
            })
        return result
    except Exception:
        return []

def generate_2d_svg(block_text: str, donor_ids: Set[int] = frozenset(), acceptor_ids: Set[int] = frozenset(), ring_systems: Optional[List[dict]] = None) -> str:
    if not _HAS_RDKIT: return '<div style="color:#FF0D0D; padding:20px; font-family:monospace;">Error: RDKit required.</div>'
    mol = Chem.MolFromMol2Block(block_text, sanitize=True, removeHs=False)
    if mol is None: return '<div style="color:#FF0D0D; padding:20px;">Error parsing MOL2.</div>'

    atom_info, rdkit_to_mol2 = {}, {}
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        mol2_id = int(a.GetProp("_TriposAtomId")) if a.HasProp("_TriposAtomId") else idx + 1
        atom_info[idx] = {"name": a.GetProp("_TriposAtomName") if a.HasProp("_TriposAtomName") else a.GetSymbol(), "sybyl": a.GetProp("_TriposAtomType") if a.HasProp("_TriposAtomType") else a.GetSymbol(), "elem": a.GetSymbol(), "mol2_id": mol2_id}
        rdkit_to_mol2[idx] = mol2_id

    rwmol = Chem.RWMol(mol)
    to_remove = [a.GetIdx() for a in rwmol.GetAtoms() if a.GetAtomicNum() == 1 and a.GetNeighbors() and a.GetNeighbors()[0].GetAtomicNum() == 6]
    old_to_new = {old: new for new, old in enumerate(i for i in range(rwmol.GetNumAtoms()) if i not in set(to_remove))}
    for idx in sorted(to_remove, reverse=True): rwmol.RemoveAtom(idx)
    mol_display = rwmol.GetMol()
    new_atom_info = {new: atom_info[old] for old, new in old_to_new.items()}

    rdDepictor.Compute2DCoords(mol_display)
    canvas_w, canvas_h = 900, 900
    d2d = rdMolDraw2D.MolDraw2DSVG(canvas_w, canvas_h)
    opts = d2d.drawOptions()
    opts.clearBackground = False
    opts.padding = 0.12
    d2d.DrawMolecule(mol_display)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()

    xs, ys, coord_map = [], [], {}
    for atom in mol_display.GetAtoms():
        idx = atom.GetIdx()
        try:
            pos = d2d.GetDrawCoords(idx)
            coord_map[idx] = (pos.x, pos.y)
            xs.append(pos.x); ys.append(pos.y)
        except Exception: pass

    if xs and ys:
        pad = 50
        min_x, min_y, box_w, box_h = min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad
    else: min_x, min_y, box_w, box_h = 0, 0, canvas_w, canvas_h

    ring_fill_group = ['<g id="ring-system-fills" style="pointer-events:none;display:none;">']
    if ring_systems:
        for rs in ring_systems:
            ring_coords = [coord_map[old_to_new[ri]] for ri in rs["atom_indices_0based"] if ri in old_to_new and old_to_new[ri] in coord_map]
            if len(ring_coords) >= 3:
                for (cx, cy) in ring_coords: ring_fill_group.append(f'<circle cx="{cx}" cy="{cy}" r="16" fill="{rs["color"]}" fill-opacity="0.18" stroke="none"/>')
    ring_fill_group.append('</g>')

    halo_group = ['<g id="da-halos" style="pointer-events:none;display:none;">']
    for idx, (cx, cy) in coord_map.items():
        mol2_id = new_atom_info.get(idx, {}).get("mol2_id", -1)
        if mol2_id in donor_ids and mol2_id in acceptor_ids:
            halo_group.append(f'<circle cx="{cx}" cy="{cy}" r="18" fill="rgba(251,191,36,0.28)" stroke="none"/><circle cx="{cx}" cy="{cy}" r="18" fill="rgba(96,165,250,0.22)" stroke="none"/>')
        elif mol2_id in donor_ids: halo_group.append(f'<circle cx="{cx}" cy="{cy}" r="18" fill="rgba(251,191,36,0.28)" stroke="none"/>')
        elif mol2_id in acceptor_ids: halo_group.append(f'<circle cx="{cx}" cy="{cy}" r="18" fill="rgba(96,165,250,0.28)" stroke="none"/>')
    halo_group.append('</g>')

    hitbox_group = ['<g id="atom-hitboxes">']
    for idx, (cx, cy) in coord_map.items():
        info = new_atom_info.get(idx, {})
        role = "donor acceptor" if info.get("mol2_id") in donor_ids and info.get("mol2_id") in acceptor_ids else "donor" if info.get("mol2_id") in donor_ids else "acceptor" if info.get("mol2_id") in acceptor_ids else ""
        hitbox_group.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="transparent" class="atom-hitbox" data-name="{info.get("name","?")}" data-sybyl="{info.get("sybyl","?")}" data-role="{role}" style="cursor:crosshair;"/>')
    hitbox_group.append('</g></svg>')

    svg = svg.replace('</svg>', '\n'.join(ring_fill_group) + '\n' + '\n'.join(halo_group) + '\n' + '\n'.join(hitbox_group))
    return svg.replace(f'width="{canvas_w}px" height="{canvas_h}px"', f'width="100%" height="100%" viewBox="{min_x:.1f} {min_y:.1f} {box_w:.1f} {box_h:.1f}" preserveAspectRatio="xMidYMid meet"')

def build_stats(atoms: Dict[int, dict], block: Mol2Block, total_conformers: int) -> dict:
    elements, sybyl_types = {}, {}
    for a in atoms.values():
        elements[a["element"]] = elements.get(a["element"], 0) + 1
        sybyl_types[a["type"]] = sybyl_types.get(a["type"], 0) + 1
    return {"molecule_name": block.name, "total_conformers": total_conformers, "n_atoms": len(atoms), "n_heavy_atoms": sum(1 for a in atoms.values() if a["element"] != "H"), "elements": dict(sorted(elements.items())), "sybyl_types": dict(sorted(sybyl_types.items()))}

def generate_html(svg_content: str, stats: dict, rdkit_props: dict, donors_acceptors: dict, ring_systems_info: list, title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — 2D Topology Viewer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Source+Sans+3:wght@300;400;500;600;700&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  :root {{ --bg: #0c0f13; --surface: #14181f; --surface2: #1a1f28; --border: #272d3a; --border-light: #333b4d; --text: #e2e8f0; --text-dim: #8892a4; --text-faint: #5a6478; --accent: #60a5fa; --accent-dim: #3b82f6; --green: #34d399; --amber: #fbbf24; --mono: 'IBM Plex Mono', monospace; --sans: 'Source Sans 3', system-ui, sans-serif; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--sans); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
  .topbar {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 10px 20px; display: flex; align-items: center; gap: 14px; }}
  .topbar h1 {{ font-size: 14px; font-weight: 600; letter-spacing: -0.3px; }}
  .badge {{ background: linear-gradient(135deg, var(--accent-dim), var(--accent)); color: #fff; font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 10px; font-family: var(--mono); text-transform: uppercase; letter-spacing: 0.5px; }}
  .main {{ flex: 1; display: flex; overflow: hidden; }}
  .viewport {{ flex: 1; position: relative; background: #ffffff; overflow: hidden; cursor: grab; }}
  .viewport:active {{ cursor: grabbing; }}
  .svg-container {{ width: 100%; height: 100%; transform-origin: 0 0; }}
  .svg-container svg {{ width: 100%; height: 100%; }}
  .panel {{ width: 370px; min-width: 370px; background: var(--surface); border-left: 1px solid var(--border); overflow-y: auto; scrollbar-width: thin; scrollbar-color: var(--border) transparent; }}
  .section {{ padding: 14px 18px; border-bottom: 1px solid var(--border); }}
  .section-title {{ font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; color: var(--text-faint); margin-bottom: 10px; }}
  .prop-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }}
  .prop-item {{ display: flex; flex-direction: column; gap: 1px; padding: 6px 8px; border-radius: 6px; background: var(--surface2); border: 1px solid transparent; }}
  .prop-label {{ font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.4px; }}
  .prop-value {{ font-family: var(--mono); font-size: 13px; font-weight: 500; }}
  .prop-item.wide {{ grid-column: span 2; }}
  .da-list {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }}
  .da-tag {{ font-family: var(--mono); font-size: 11px; font-weight: 500; padding: 2px 7px; border-radius: 4px; border: 1px solid var(--border); }}
  .da-tag.donor {{ background: rgba(251,191,36,0.08); color: var(--amber); border-color: rgba(251,191,36,0.2); }}
  .da-tag.acceptor {{ background: rgba(96,165,250,0.08); color: var(--accent); border-color: rgba(96,165,250,0.2); }}
  .rs-legend {{ display: flex; flex-direction: column; gap: 8px; margin-top: 6px; }}
  .rs-item-block {{ border-radius: 6px; background: var(--surface2); border: 1px solid transparent; overflow: hidden; }}
  .rs-item {{ display: flex; align-items: center; gap: 8px; padding: 5px 8px; }}
  .rs-swatch {{ width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }}
  .rs-label {{ font-family: var(--mono); font-size: 12px; font-weight: 500; }}
  .rs-detail {{ font-size: 10px; color: var(--text-dim); margin-left: auto; white-space: nowrap; }}
  .rs-atoms {{ padding: 4px 8px 6px 30px; font-size: 10px; color: var(--text-dim); display: flex; flex-wrap: wrap; align-items: center; gap: 3px; }}
  .rs-atoms-label {{ font-size: 9px; text-transform: uppercase; color: var(--text-faint); margin-right: 2px; }}
  .rs-atom-tag {{ font-family: var(--mono); font-size: 10px; font-weight: 500; padding: 1px 4px; border-radius: 3px; background: rgba(255,255,255,0.04); border: 1px solid var(--border); color: var(--text-dim); }}
  .rdkit-badge {{ display: inline-block; font-size: 8px; font-weight: 600; padding: 1px 5px; border-radius: 3px; margin-left: 6px; background: rgba(96,165,250,0.12); color: var(--accent); border: 1px solid rgba(96,165,250,0.2); font-family: var(--mono); }}
  .zoom-controls {{ position: absolute; bottom: 14px; left: 14px; display: flex; gap: 4px; z-index: 10; }}
  .zoom-btn {{ width: 32px; height: 32px; border-radius: 6px; border: 1px solid var(--border); background: var(--surface); color: var(--text); font-size: 16px; font-weight: 700; cursor: pointer; display: flex; align-items: center; justify-content: center; font-family: var(--mono); transition: background 0.15s; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }}
  .zoom-btn:hover {{ background: var(--surface2); }}
  #tooltip {{ position: absolute; display: none; background: rgba(12,15,19,0.92); color: #fff; padding: 5px 10px; border-radius: 6px; font-size: 12px; font-family: var(--mono); font-weight: 500; pointer-events: none; z-index: 1000; box-shadow: 0 4px 12px rgba(0,0,0,0.4); border: 1px solid var(--border); white-space: nowrap; }}
  #tooltip .sybyl {{ color: var(--accent); }}
  #tooltip .role-donor {{ color: var(--amber); font-size: 10px; }}
  #tooltip .role-acceptor {{ color: var(--accent); font-size: 10px; }}
  .mode-group {{ display: flex; gap: 0; border-radius: 6px; overflow: hidden; border: 1px solid var(--border); }}
  .mode-btn {{ padding: 4px 12px; font-size: 11px; font-family: var(--mono); font-weight: 500; background: var(--surface2); color: var(--text-dim); border: none; cursor: pointer; border-right: 1px solid var(--border); }}
  .mode-btn:last-child {{ border-right: none; }}
  .mode-btn.active {{ background: var(--accent-dim); color: #fff; }}
</style>
</head>
<body>

<div class="topbar">
  <h1>{title}</h1>
  <span class="badge">2D Topology</span>
  <div style="flex:1"></div>
  <div class="mode-group">
    <button class="mode-btn active" id="modeClean">Clean</button>
    <button class="mode-btn" id="modeDA">D/A</button>
    <button class="mode-btn" id="modeRings">Rings</button>
  </div>
  <span style="font-size:11px; color:var(--text-dim); font-family:var(--mono); margin-left:10px;">Scroll to zoom &middot; Drag to pan</span>
</div>

<div class="main">
  <div class="viewport" id="viewport">
    <div class="svg-container" id="svg-container">{svg_content}</div>
    <div class="zoom-controls">
      <button class="zoom-btn" id="zoomIn" title="Zoom in"><svg width="16" height="16" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg></button>
      <button class="zoom-btn" id="zoomOut" title="Zoom out"><svg width="16" height="16" viewBox="0 0 24 24"><path d="M5 12h14"/></svg></button>
      <button class="zoom-btn" id="zoomReset" title="Reset view"><svg width="14" height="14" viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg></button>
      <button class="zoom-btn" id="btnScreenshot" title="Download PNG Screenshot">
        <svg width="16" height="16" viewBox="0 0 24 24"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
      </button>
    </div>
  </div>
  <div class="panel" id="panel"></div>
</div>
<div id="tooltip"></div>

<script>
const STATS = {json.dumps(stats)};
const PROPS = {json.dumps(rdkit_props)};
const DA    = {json.dumps(donors_acceptors)};
const RING_SYSTEMS = {json.dumps(ring_systems_info)};

document.addEventListener("DOMContentLoaded", () => {{
  const viewport = document.getElementById("viewport");
  const container = document.getElementById("svg-container");
  let scale = 1, panX = 0, panY = 0;
  let isPanning = false, startX = 0, startY = 0, startPanX = 0, startPanY = 0;
  const ZOOM_STEP = 0.15, MIN_SCALE = 0.2, MAX_SCALE = 10;

  function applyTransform() {{ container.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`; }}

  viewport.addEventListener("wheel", (e) => {{
    e.preventDefault();
    const rect = viewport.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const oldScale = scale;
    scale = e.deltaY < 0 ? Math.min(MAX_SCALE, scale * (1 + ZOOM_STEP)) : Math.max(MIN_SCALE, scale / (1 + ZOOM_STEP));
    const ratio = scale / oldScale;
    panX = mx - ratio * (mx - panX); panY = my - ratio * (my - panY);
    applyTransform();
  }}, {{ passive: false }});

  viewport.addEventListener("mousedown", (e) => {{
    if (e.button !== 0) return;
    isPanning = true; startX = e.clientX; startY = e.clientY;
    startPanX = panX; startPanY = panY;
  }});
  window.addEventListener("mousemove", (e) => {{
    if (!isPanning) return;
    panX = startPanX + (e.clientX - startX); panY = startPanY + (e.clientY - startY);
    applyTransform();
  }});
  window.addEventListener("mouseup", () => {{ isPanning = false; }});

  document.getElementById("zoomIn").addEventListener("click", () => {{
    const rect = viewport.getBoundingClientRect();
    const cx = rect.width / 2, cy = rect.height / 2; const oldScale = scale;
    scale = Math.min(MAX_SCALE, scale * (1 + ZOOM_STEP)); const ratio = scale / oldScale;
    panX = cx - ratio * (cx - panX); panY = cy - cy - ratio * (cy - panY); applyTransform();
  }});
  document.getElementById("zoomOut").addEventListener("click", () => {{
    const rect = viewport.getBoundingClientRect();
    const cx = rect.width / 2, cy = rect.height / 2; const oldScale = scale;
    scale = Math.max(MIN_SCALE, scale / (1 + ZOOM_STEP)); const ratio = scale / oldScale;
    panX = cx - ratio * (cx - panX); panY = cy - ratio * (cy - panY); applyTransform();
  }});
  document.getElementById("zoomReset").addEventListener("click", () => {{ scale = 1; panX = 0; panY = 0; applyTransform(); }});

  // Export Screenshot as PNG
  document.getElementById("btnScreenshot").addEventListener("click", () => {{
    const svgEl = document.querySelector("#svg-container svg");
    const vb = svgEl.viewBox.baseVal;
    const w = vb.width || 900;
    const h = vb.height || 900;
    
    let source = new XMLSerializer().serializeToString(svgEl);
    if(!source.match(/^<svg[^>]+xmlns="http\\:\\/\\/www\\.w3\\.org\\/2000\\/svg"/)){{
        source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
    }}
    
    source = source.replace(/width="100%"/, `width="${{w}}px"`);
    source = source.replace(/height="100%"/, `height="${{h}}px"`);

    const blob = new Blob([source], {{type: "image/svg+xml;charset=utf-8"}});
    const url = URL.createObjectURL(blob);
    
    const img = new Image();
    img.onload = () => {{
      const canvas = document.createElement("canvas");
      const outScale = 2; // Output resolution multiplier
      canvas.width = w * outScale;
      canvas.height = h * outScale;
      const ctx = canvas.getContext("2d");
      
      // Draw a white background so PNG is not transparent
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      ctx.scale(outScale, outScale);
      ctx.drawImage(img, 0, 0, w, h);
      
      const a = document.createElement("a");
      a.href = canvas.toDataURL("image/png");
      a.download = STATS.molecule_name + "_2D.png";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }};
    img.src = url;
  }});

  let html = `<div class="section"><div class="section-title">Molecule</div><div style="font-family:var(--mono); font-size:14px; font-weight:600;">${{STATS.molecule_name}}</div><div style="font-size:12px; color:var(--text-dim); margin-top:3px;">${{STATS.total_conformers}} conformer(s) in file</div></div>`;
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
  html += `<div class="section"><div class="section-title">Atom Census</div><div class="prop-grid"><div class="prop-item"><span class="prop-label">Total atoms</span><span class="prop-value">${{STATS.n_atoms}}</span></div><div class="prop-item"><span class="prop-label">Heavy atoms</span><span class="prop-value">${{STATS.n_heavy_atoms}}</span></div></div></div>`;
  if (DA.donors.length > 0) {{
    html += `<div class="section overlay-section da-section" style="display:none;"><div class="section-title">H-Bond Donors (${{DA.donors.length}}) &mdash; <span style="color:var(--amber);">&bull;</span> amber halo<span class="rdkit-badge">RDKit</span></div><div class="da-list">`;
    DA.donors.forEach(d => html += `<span class="da-tag donor" title="${{d.type}} \u00b7 ${{d.n_H}}H">${{d.name}}</span>`);
    html += `</div></div>`;
  }}
  if (DA.acceptors.length > 0) {{
    html += `<div class="section overlay-section da-section" style="display:none;"><div class="section-title">H-Bond Acceptors (${{DA.acceptors.length}}) &mdash; <span style="color:var(--accent);">&bull;</span> blue halo<span class="rdkit-badge">RDKit</span></div><div class="da-list">`;
    DA.acceptors.forEach(a => html += `<span class="da-tag acceptor" title="${{a.type}}">${{a.name}}</span>`);
    html += `</div></div>`;
  }}
  if (RING_SYSTEMS.length > 0) {{
    html += `<div class="section overlay-section rs-section" style="display:none;"><div class="section-title">Aromatic Rings (${{RING_SYSTEMS.length}})<span class="rdkit-badge">RDKit</span></div><div class="rs-legend">`;
    RING_SYSTEMS.forEach(rs => {{
      let atomNamesHtml = rs.atom_names.length ? `<div class="rs-atoms"><span class="rs-atoms-label">Atoms:</span> ` + rs.atom_names.map(n => `<span class="rs-atom-tag">${{n}}</span>`).join(' ') + `</div>` : '';
      html += `<div class="rs-item-block"><div class="rs-item"><div class="rs-swatch" style="background:${{rs.color}};opacity:0.7;"></div><span class="rs-label">${{rs.label}}</span><span class="rs-detail">${{rs.n_atoms}} atoms</span></div>${{atomNamesHtml}}</div>`;
    }});
    html += `</div></div>`;
  }}
  document.getElementById("panel").innerHTML = html;

  function setOverlayMode(mode) {{
    ["modeClean", "modeDA", "modeRings"].forEach(id => document.getElementById(id).classList.remove("active"));
    document.getElementById(mode === "clean" ? "modeClean" : mode === "da" ? "modeDA" : "modeRings").classList.add("active");
    if(document.getElementById("da-halos")) document.getElementById("da-halos").style.display = mode === "da" ? "" : "none";
    if(document.getElementById("ring-system-fills")) document.getElementById("ring-system-fills").style.display = mode === "rings" ? "" : "none";
    document.querySelectorAll(".da-section").forEach(el => el.style.display = mode === "da" ? "" : "none");
    document.querySelectorAll(".rs-section").forEach(el => el.style.display = mode === "rings" ? "" : "none");
  }}
  document.getElementById("modeClean").addEventListener("click", () => setOverlayMode("clean"));
  document.getElementById("modeDA").addEventListener("click", () => setOverlayMode("da"));
  document.getElementById("modeRings").addEventListener("click", () => setOverlayMode("rings"));
  setOverlayMode("clean");

  const tooltip = document.getElementById("tooltip");
  document.querySelectorAll(".atom-hitbox").forEach(box => {{
    box.addEventListener("mouseenter", e => {{
      const name = e.target.getAttribute("data-name"), sybyl = e.target.getAttribute("data-sybyl"), role = e.target.getAttribute("data-role") || "";
      let roleBadge = role.includes("donor") && role.includes("acceptor") ? ' <span class="role-donor">D</span>/<span class="role-acceptor">A</span>' : role.includes("donor") ? ' <span class="role-donor">donor</span>' : role.includes("acceptor") ? ' <span class="role-acceptor">acceptor</span>' : "";
      tooltip.innerHTML = `${{name}} <span class="sybyl">[${{sybyl}}]</span>${{roleBadge}}`;
      tooltip.style.display = "block";
    }});
    box.addEventListener("mouseleave", () => tooltip.style.display = "none");
    box.addEventListener("mousemove", e => {{ tooltip.style.left = (e.pageX + 14) + "px"; tooltip.style.top = (e.pageY - 32) + "px"; }});
  }});
}});
</script>
</body>
</html>"""

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate a 2D topology view of a MOL2 file.")
    p.add_argument("mol2_path", help="Input MOL2 file.")
    p.add_argument("-o", "--output", default=None, help="Output HTML path.")
    p.add_argument("--open", action="store_true", help="Open HTML in browser.")
    args = p.parse_args(argv)

    if not _HAS_RDKIT:
        print("ERROR: RDKit is required for 2D viewing.", file=sys.stderr)
        return 1

    mol2_path = Path(args.mol2_path).expanduser().resolve()
    blocks = split_mol2_blocks(mol2_path)
    if not blocks: return 1

    blk = blocks[0]
    atoms = parse_atoms_and_bonds(blk.text)
    if not atoms: return 1

    stats = build_stats(atoms, blk, len(blocks))
    mol = Chem.MolFromMol2Block(blk.text, sanitize=True, removeHs=False)
    rdkit_props = compute_rdkit_properties(mol) if mol else {}
    da = enumerate_donors_acceptors(atoms, blk.text)
    donor_ids = {d["id"] for d in da["donors"]}
    acceptor_ids = {a["id"] for a in da["acceptors"]}

    ring_systems = detect_ring_systems_for_viz(blk.text)
    ring_systems_info = [{"label": rs["label"], "n_atoms": rs["n_atoms"], "n_patches": rs["n_patches"], "color": rs["color"], "atom_names": rs.get("atom_names", []), "patches": rs.get("patches", [])} for rs in ring_systems]

    svg_content = generate_2d_svg(blk.text, donor_ids, acceptor_ids, ring_systems)
    out_path = Path(args.output) if args.output else mol2_path.with_name(f"{mol2_path.stem}_2D.html")
    out_path.write_text(generate_html(svg_content, stats, rdkit_props, da, ring_systems_info, mol2_path.stem), encoding="utf-8")
    print(f"Wrote: {out_path}")
    if args.open: webbrowser.open(str(out_path))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())