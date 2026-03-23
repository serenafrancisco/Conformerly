"""imhb.py

IMHB (intramolecular hydrogen bond) analysis — v1.8 (RDKit donor/acceptor detection).

Donor and acceptor atoms are now identified using RDKit's pharmacophoric
feature definitions (BaseFeatures.fdef) instead of manual element/Sybyl-type
heuristics.  This gives more chemically accurate results, especially for
nitrogen atoms whose lone-pair availability depends on hybridisation and
conjugation context.

Outputs written by `run_imhb`:
- <prefix>_hbonds.csv         (detailed list of IMHBs, one row per H-bond)
- <prefix>_hbonds_summary.csv (per-conformer counts + labels present)
- <prefix>_hbond_ids.csv      (per-label persistence across conformers)
"""

from __future__ import annotations

import csv
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from rdkit import Chem, RDConfig
    from rdkit.Chem import ChemicalFeatures
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

from mol2_io import Mol2Block


# ========================================================================
# Default parameters (exposed via CLI) — aligned to Maestro thresholds
# ========================================================================

# --- Standard H-bond criteria (Maestro "H-bonds") ---
IMHB_MAX_DIST_HA       = 2.8     # max H···A distance (Å)
IMHB_MIN_ANGLE_DHA     = 120.0   # min donor angle D-H···A
IMHB_MIN_ANGLE_ACCEPTOR = 90.0   # min acceptor angle X-A···H

# --- Strict angle for close-range contacts (intramolecular safeguard) ---
IMHB_MIN_ANGLE_STRICT  = 130.0   # applied when bond sep < threshold
IMHB_MIN_BOND_SEPARATION = 5     # bond-sep cutoff for strict vs normal


# ========================================================================
# RDKit-based donor / acceptor identification
# ========================================================================

_FEATURE_FACTORY = None


def _get_feature_factory():
    """Lazy-load the RDKit chemical feature factory (BaseFeatures.fdef)."""
    global _FEATURE_FACTORY
    if _FEATURE_FACTORY is None:
        if not _HAS_RDKIT:
            raise RuntimeError(
                "RDKit is required for donor/acceptor identification.\n"
                "Typical install: conda install -c conda-forge rdkit"
            )
        fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        _FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(fdef_path)
    return _FEATURE_FACTORY


def identify_donors_acceptors_rdkit(
    block_text: str,
) -> Tuple[Set[int], Set[int]]:
    """Identify H-bond donor and acceptor heavy atoms using RDKit features.

    Returns two sets of **MOL2 atom IDs** (1-based):
      donor_ids   — heavy atoms flagged as pharmacophoric donors
      acceptor_ids — heavy atoms flagged as pharmacophoric acceptors

    The mapping from 0-based RDKit indices to 1-based MOL2 IDs assumes
    that atom order is preserved by ``Chem.MolFromMol2Block`` (which it is
    for well-formed MOL2 files).
    """
    if not _HAS_RDKIT:
        raise RuntimeError("RDKit is required for donor/acceptor identification.")

    mol = Chem.MolFromMol2Block(block_text, sanitize=True, removeHs=False)
    if mol is None:
        raise ValueError("RDKit failed to parse the MOL2 block for donor/acceptor detection.")

    factory = _get_feature_factory()
    feats = factory.GetFeaturesForMol(mol)

    # Build RDKit-index → MOL2-ID mapping.
    # MOL2 atom IDs are read in file order; RDKit preserves that order.
    rdkit_to_mol2: Dict[int, int] = {}
    for atom in mol.GetAtoms():
        rdkit_idx = atom.GetIdx()
        # _TriposAtomId is set by MolFromMol2Block (string of the MOL2 atom ID)
        if atom.HasProp("_TriposAtomId"):
            rdkit_to_mol2[rdkit_idx] = int(atom.GetProp("_TriposAtomId"))
        else:
            rdkit_to_mol2[rdkit_idx] = rdkit_idx + 1  # fallback: assume 1-based sequential

    donor_ids: Set[int] = set()
    acceptor_ids: Set[int] = set()

    for feat in feats:
        family = feat.GetFamily()
        atom_indices = feat.GetAtomIds()  # 0-based RDKit indices
        if family == "Donor":
            for idx in atom_indices:
                mol2_id = rdkit_to_mol2.get(idx, idx + 1)
                donor_ids.add(mol2_id)
        elif family == "Acceptor":
            for idx in atom_indices:
                mol2_id = rdkit_to_mol2.get(idx, idx + 1)
                acceptor_ids.add(mol2_id)

    return donor_ids, acceptor_ids


# ========================================================================
# Geometry helpers
# ========================================================================

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(a - b))


def _angle_dha(coord_d: np.ndarray, coord_h: np.ndarray, coord_a: np.ndarray) -> float:
    """Donor angle D-H···A (degrees).  Vertex at H."""
    v1 = coord_d - coord_h
    v2 = coord_a - coord_h
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _angle_xah(coord_x: np.ndarray, coord_a: np.ndarray, coord_h: np.ndarray) -> float:
    """Acceptor angle X-A···H (degrees).  Vertex at A.

    X is the heavy-atom neighbour of the acceptor A;
    H is the hydrogen.  This angle checks that H approaches A
    from a direction compatible with a lone-pair orbital.
    """
    v1 = coord_x - coord_a
    v2 = coord_h - coord_a
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _best_acceptor_angle(
    atoms: Dict[int, dict], acc_id: int, h_id: int,
) -> float:
    """Return the MAXIMUM X-A···H angle over all heavy-atom neighbours X of A.

    If the acceptor has multiple heavy neighbours, we take the most
    favourable (largest) angle — a single good approach direction is enough.
    Returns 0.0 if A has no heavy neighbours (should not happen for real
    acceptors, but guards against degenerate input).
    """
    acoord = atoms[acc_id]["coords"]
    hcoord = atoms[h_id]["coords"]
    best = 0.0
    for nb_id in atoms[acc_id]["bonds"]:
        nb = atoms.get(nb_id)
        if nb is None or nb["element"] == "H":
            continue  # skip hydrogens — we want heavy-atom X only
        ang = _angle_xah(nb["coords"], acoord, hcoord)
        if ang > best:
            best = ang
    return best


# ========================================================================
# Topology helpers
# ========================================================================

def _bond_separation_bfs(atoms: Dict[int, dict], start: int, end: int, max_depth: int) -> int:
    """Shortest covalent bond count between two atoms (BFS)."""
    if start == end:
        return 0
    visited = {start}
    q: deque[Tuple[int, int]] = deque([(start, 0)])
    while q:
        node, depth = q.popleft()
        if depth >= max_depth:
            continue
        for nb in atoms[node]["bonds"]:
            if nb == end:
                return depth + 1
            if nb not in visited:
                visited.add(nb)
                q.append((nb, depth + 1))
    return max_depth


def _shortest_path_bfs(
    atoms: Dict[int, dict], start: int, end: int, max_depth: int
) -> List[int]:
    """Return shortest covalent path (list of atom IDs) from start to end, inclusive."""
    if start == end:
        return [start]
    visited: Dict[int, Optional[int]] = {start: None}
    q: deque[int] = deque([start])
    depth_map: Dict[int, int] = {start: 0}
    while q:
        node = q.popleft()
        depth = depth_map[node]
        if depth >= max_depth:
            continue
        for nb in atoms[node]["bonds"]:
            if nb == end:
                path = [end, node]
                cur = node
                while visited[cur] is not None:
                    cur = visited[cur]
                    path.append(cur)
                path.reverse()
                return path
            if nb not in visited:
                visited[nb] = node
                depth_map[nb] = depth + 1
                q.append(nb)
    return []


# ========================================================================
# MOL2 parsing: ATOM + BOND sections
# ========================================================================

def parse_atoms_and_bonds(block_text: str) -> Dict[int, dict]:
    """Parse a MOL2 block and return a dict of atom records keyed by atom ID.

    Each atom record contains:
      id, name, coords (np.ndarray), type (Sybyl), element, bonds (list of IDs)
    """
    atoms: Dict[int, dict] = {}
    section: Optional[str] = None

    for raw in block_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@<TRIPOS>ATOM"):
            section = "ATOM"
            continue
        if line.startswith("@<TRIPOS>BOND"):
            section = "BOND"
            continue
        if line.startswith("@<TRIPOS>"):
            section = None
            continue

        if section == "ATOM":
            parts = line.split()
            if len(parts) < 6:
                continue
            atom_id = int(parts[0])
            name = parts[1]
            coords = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float)
            sybyl_type = parts[5]
            element = sybyl_type.split(".")[0]
            atoms[atom_id] = {
                "id": atom_id,
                "name": name,
                "coords": coords,
                "type": sybyl_type,
                "element": element,
                "bonds": [],
            }

        elif section == "BOND":
            parts = line.split()
            if len(parts) < 3:
                continue
            a1 = int(parts[1])
            a2 = int(parts[2])
            if a1 in atoms and a2 in atoms:
                atoms[a1]["bonds"].append(a2)
                atoms[a2]["bonds"].append(a1)

    return atoms


# ========================================================================
# Range classification
# ========================================================================

def classify_hbond_range(bond_separation: int) -> str:
    """Classify H-bond by D···A bond separation.

    - short-range:  N <= 4
    - medium-range: 4 < N <= 7
    - long-range:   N > 7
    """
    if bond_separation <= 4:
        return "short-range"
    elif bond_separation <= 7:
        return "medium-range"
    else:
        return "long-range"


# ========================================================================
# H-bond label
# ========================================================================

def make_hb_label(donor_name: str, hydrogen_name: str, acceptor_name: str) -> str:
    """Unique, conformer-independent label for an H-bond type.

    Format: @<donor_name>@<hydrogen_name>@<acceptor_name>
    Example: @O1@HO1@N3
    """
    return f"@{donor_name}@{hydrogen_name}@{acceptor_name}"


# ========================================================================
# Pseudo-ring helpers
# ========================================================================

def compute_pseudoring_size(bond_separation: int) -> int:
    """Pseudo-ring atom count = bond_separation + 2 (adds D and H)."""
    return bond_separation + 2


def classify_pseudoring_type(ring_size: int) -> str:
    """S(n) label per Etter / Jeffrey & Saenger convention."""
    if ring_size <= 4:
        return f"S({ring_size})-strained"
    else:
        return f"S({ring_size})"


def compute_pseudoring_geometry(
    atoms: Dict[int, dict],
    donor_id: int,
    h_id: int,
    acceptor_id: int,
    cov_path: List[int],
) -> dict:
    """Geometric descriptors for the pseudo-ring (centroid, planarity, perimeter, dist_DA)."""
    ring_ids = list(cov_path) + [h_id]
    coords = np.array([atoms[aid]["coords"] for aid in ring_ids], dtype=float)
    centroid_xyz = coords.mean(axis=0)

    if len(coords) >= 3:
        centered = coords - centroid_xyz
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[-1]
        planarity_rmsd = float(np.sqrt(np.mean((centered @ normal) ** 2)))
    else:
        planarity_rmsd = float("nan")

    n = len(ring_ids)
    perimeter = sum(
        float(np.linalg.norm(
            atoms[ring_ids[i]]["coords"] - atoms[ring_ids[(i + 1) % n]]["coords"]
        ))
        for i in range(n)
    )

    dist_da = _dist(atoms[donor_id]["coords"], atoms[acceptor_id]["coords"])

    return {
        "dist_DA": round(dist_da, 4),
        "pseudoring_atoms_ids": ";".join(str(x) for x in ring_ids),
        "pseudoring_centroid_x": round(float(centroid_xyz[0]), 4),
        "pseudoring_centroid_y": round(float(centroid_xyz[1]), 4),
        "pseudoring_centroid_z": round(float(centroid_xyz[2]), 4),
        "pseudoring_planarity_rmsd": round(planarity_rmsd, 4),
        "pseudoring_perimeter_A": round(perimeter, 4),
    }


# ========================================================================
# IMHB detection (main loop)
# ========================================================================

def find_imhbs(
    atoms: Dict[int, dict],
    max_dist_ha: float,
    min_angle_dha: float,
    min_angle_strict: float,
    min_bond_separation: int,
    min_angle_acceptor: float,
    rdkit_donor_ids: Optional[Set[int]] = None,
    rdkit_acceptor_ids: Optional[Set[int]] = None,
) -> List[dict]:
    """Detect all intramolecular H-bonds in one conformer.

    Donor and acceptor atoms are identified by the RDKit feature factory
    (``rdkit_donor_ids`` and ``rdkit_acceptor_ids``).  If these sets are
    not supplied, a fallback element-based heuristic is used (N/O/S
    donors, N/O/F/S acceptors) for backward compatibility, but the
    preferred path is always RDKit-based.

    Filters applied (in order):
      1. H···A distance  ≤ max_dist_ha
      2. D-H···A donor angle  ≥ min_angle_dha (or min_angle_strict for close-range)
      3. X-A···H acceptor angle  ≥ min_angle_acceptor
    """
    _FALLBACK_DONOR_ELEMENTS = {"N", "O", "S"}
    _FALLBACK_ACCEPTOR_ELEMENTS = {"O", "N", "F", "S"}

    # --- Build lists of potential acceptors and donor-H pairs ---
    acceptors: List[int] = []
    donor_h_pairs: List[Tuple[int, int]] = []

    if rdkit_donor_ids is not None and rdkit_acceptor_ids is not None:
        # RDKit-based: donors and acceptors are pre-identified heavy atoms
        acceptors = [aid for aid in rdkit_acceptor_ids if aid in atoms]

        # Donor-H pairs: for each RDKit donor, find its bonded H atoms
        for donor_id in rdkit_donor_ids:
            if donor_id not in atoms:
                continue
            for nb_id in atoms[donor_id]["bonds"]:
                nb = atoms.get(nb_id)
                if nb and nb["element"] == "H":
                    donor_h_pairs.append((donor_id, nb_id))
    else:
        # Fallback: element-based heuristic (pre-v1.8 behaviour)
        for atom_id, a in atoms.items():
            if a["element"] in _FALLBACK_ACCEPTOR_ELEMENTS:
                acceptors.append(atom_id)
            if a["element"] == "H":
                for nb_id in a["bonds"]:
                    nb = atoms.get(nb_id)
                    if nb and nb["element"] in _FALLBACK_DONOR_ELEMENTS:
                        donor_h_pairs.append((nb_id, atom_id))
                        break

    out: List[dict] = []

    for donor_id, h_id in donor_h_pairs:
        for acc_id in acceptors:
            # --- Skip trivial cases ---
            if donor_id == acc_id:
                continue
            if acc_id in atoms[h_id]["bonds"]:
                continue  # acceptor directly bonded to H → not an H-bond

            # --- Compute geometry ---
            hcoord = atoms[h_id]["coords"]
            acoord = atoms[acc_id]["coords"]

            ha = _dist(hcoord, acoord)
            if ha > max_dist_ha:
                continue

            dcoord = atoms[donor_id]["coords"]
            dha = _angle_dha(dcoord, hcoord, acoord)

            # --- Bond separation + strict-angle safeguard ---
            topo = _bond_separation_bfs(atoms, donor_id, acc_id, max_depth=30)
            donor_cutoff = min_angle_strict if topo < min_bond_separation else min_angle_dha
            if dha < donor_cutoff:
                continue

            # --- Acceptor angle check ---
            xah = _best_acceptor_angle(atoms, acc_id, h_id)
            if xah < min_angle_acceptor:
                continue

            # --- Passed → record the H-bond ---
            cov_path = _shortest_path_bfs(atoms, donor_id, acc_id, max_depth=30)
            pr_size = compute_pseudoring_size(int(topo))
            pr_type = classify_pseudoring_type(pr_size)

            d_name = atoms[donor_id]["name"]
            h_name = atoms[h_id]["name"]
            a_name = atoms[acc_id]["name"]

            if cov_path:
                pr_geom = compute_pseudoring_geometry(atoms, donor_id, h_id, acc_id, cov_path)
            else:
                pr_geom = {
                    "dist_DA": float("nan"),
                    "pseudoring_atoms_ids": "",
                    "pseudoring_centroid_x": float("nan"),
                    "pseudoring_centroid_y": float("nan"),
                    "pseudoring_centroid_z": float("nan"),
                    "pseudoring_planarity_rmsd": float("nan"),
                    "pseudoring_perimeter_A": float("nan"),
                }

            out.append({
                "hb_label": make_hb_label(d_name, h_name, a_name),
                "donor_id": donor_id,
                "donor_name": d_name,
                "donor_type": atoms[donor_id]["type"],
                "hydrogen_id": h_id,
                "hydrogen_name": h_name,
                "acceptor_id": acc_id,
                "acceptor_name": a_name,
                "acceptor_type": atoms[acc_id]["type"],
                "dist_HA": round(ha, 4),
                "dist_DA": pr_geom.pop("dist_DA"),
                "angle_DHA": round(dha, 3),
                "angle_XAH": round(xah, 3),
                "bond_separation_D_A": int(topo),
                "donor_angle_cutoff_used": float(donor_cutoff),
                "distance_range_class": classify_hbond_range(int(topo)),
                "pseudoring_size": pr_size,
                "pseudoring_type": pr_type,
                **pr_geom,
            })

    return out


# ========================================================================
# Output writers
# ========================================================================

def _write_detail_csv(rows: List[dict], out_path: Path) -> None:
    """Write _hbonds.csv (one row per detected H-bond)."""
    fieldnames = [
        "molecule_conformer",
        "molecule_name",
        "donor_id",
        "donor_name",
        "donor_type",
        "hydrogen_id",
        "hydrogen_name",
        "acceptor_id",
        "acceptor_name",
        "acceptor_type",
        "dist_HA",
        "dist_DA",
        "angle_DHA",
        "angle_XAH",
        "bond_separation_D_A",
        "donor_angle_cutoff_used",
        "distance_range_class",
        "pseudoring_size",
        "pseudoring_type",
        "pseudoring_atoms_ids",
        "pseudoring_centroid_x",
        "pseudoring_centroid_y",
        "pseudoring_centroid_z",
        "pseudoring_planarity_rmsd",
        "pseudoring_perimeter_A",
        "hb_label",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_summary_csv(summary_rows: List[dict], out_path: Path) -> None:
    """Write _hbonds_summary.csv (one row per conformer)."""
    fieldnames = [
        "molecule_conformer",
        "molecule_name",
        "hbonds_total",
        "short_range_count",
        "medium_range_count",
        "long_range_count",
        "hb_labels_present",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)


def _write_hbond_ids_csv(
    all_rows: List[dict],
    all_conformer_indices: List[int],
    out_path: Path,
) -> None:
    """Write H-bond label persistence matrix.

    Rows  = unique hb_label values (one per distinct H-bond type).
    Columns = conformer_<index> (1/0 presence flag) + summary counts.
    """
    label_to_conformers: Dict[str, set] = {}
    label_to_range: Dict[str, str] = {}
    label_to_pr_type: Dict[str, str] = {}
    for r in all_rows:
        lbl = r["hb_label"]
        if lbl not in label_to_conformers:
            label_to_conformers[lbl] = set()
            label_to_range[lbl] = r.get("distance_range_class", "")
            label_to_pr_type[lbl] = r.get("pseudoring_type", "")
        label_to_conformers[lbl].add(r["molecule_conformer"])

    if not label_to_conformers:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            f.write("hb_label,n_conformers_present,frequency,distance_range_class,pseudoring_type\n")
        return

    conf_cols = [f"conformer_{c}" for c in sorted(all_conformer_indices)]
    fieldnames = (
        ["hb_label", "n_conformers_present", "frequency",
         "distance_range_class", "pseudoring_type"]
        + conf_cols
    )

    total_conformers = len(all_conformer_indices)
    sorted_indices = sorted(all_conformer_indices)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for lbl in sorted(label_to_conformers, key=lambda x: -len(label_to_conformers[x])):
            present = label_to_conformers[lbl]
            n = len(present)
            freq = round(n / total_conformers, 4) if total_conformers > 0 else 0.0
            row: dict = {
                "hb_label": lbl,
                "n_conformers_present": n,
                "frequency": freq,
                "distance_range_class": label_to_range[lbl],
                "pseudoring_type": label_to_pr_type[lbl],
            }
            for idx in sorted_indices:
                row[f"conformer_{idx}"] = 1 if idx in present else 0
            w.writerow(row)


# ========================================================================
# Public API
# ========================================================================

def run_imhb(
    blocks: List[Mol2Block],
    out_dir: Path,
    out_prefix: str,
    hb_max_dist: float = IMHB_MAX_DIST_HA,
    hb_min_angle: float = IMHB_MIN_ANGLE_DHA,
    hb_min_angle_strict: float = IMHB_MIN_ANGLE_STRICT,
    hb_min_bond_sep: int = IMHB_MIN_BOND_SEPARATION,
    hb_min_angle_acceptor: float = IMHB_MIN_ANGLE_ACCEPTOR,
) -> Tuple[Path, Path, Path]:
    """Run IMHB for all blocks and write: detail CSV + summary CSV + hbond_ids CSV.

    Donor/acceptor atoms are identified once via RDKit (topology is shared
    across conformers in a multi-MOL2 ensemble).
    """
    all_rows: List[dict] = []
    summary_rows: List[dict] = []
    all_conformer_indices: List[int] = []

    # Identify donors and acceptors from the first block (topology-invariant)
    rdkit_donor_ids: Optional[Set[int]] = None
    rdkit_acceptor_ids: Optional[Set[int]] = None
    if blocks and _HAS_RDKIT:
        try:
            rdkit_donor_ids, rdkit_acceptor_ids = identify_donors_acceptors_rdkit(blocks[0].text)
        except Exception:
            pass  # fall back to element heuristic if RDKit fails

    for blk in blocks:
        all_conformer_indices.append(blk.index)
        atoms = parse_atoms_and_bonds(blk.text)

        if not atoms:
            summary_rows.append({
                "molecule_conformer": blk.index,
                "molecule_name": blk.name,
                "hbonds_total": 0,
                "short_range_count": 0,
                "medium_range_count": 0,
                "long_range_count": 0,
                "hb_labels_present": "",
            })
            continue

        rows = find_imhbs(
            atoms=atoms,
            max_dist_ha=float(hb_max_dist),
            min_angle_dha=float(hb_min_angle),
            min_angle_strict=float(hb_min_angle_strict),
            min_bond_separation=int(hb_min_bond_sep),
            min_angle_acceptor=float(hb_min_angle_acceptor),
            rdkit_donor_ids=rdkit_donor_ids,
            rdkit_acceptor_ids=rdkit_acceptor_ids,
        )

        for r in rows:
            r["molecule_conformer"] = blk.index
            r["molecule_name"] = blk.name
            all_rows.append(r)

        short_count  = sum(1 for r in rows if r["distance_range_class"] == "short-range")
        medium_count = sum(1 for r in rows if r["distance_range_class"] == "medium-range")
        long_count   = sum(1 for r in rows if r["distance_range_class"] == "long-range")
        labels_present = ";".join(sorted({r["hb_label"] for r in rows}))

        summary_rows.append({
            "molecule_conformer": blk.index,
            "molecule_name": blk.name,
            "hbonds_total": len(rows),
            "short_range_count": short_count,
            "medium_range_count": medium_count,
            "long_range_count": long_count,
            "hb_labels_present": labels_present,
        })

    detail_path    = out_dir / f"{out_prefix}_hbonds.csv"
    summary_path   = out_dir / f"{out_prefix}_hbonds_summary.csv"
    hbond_ids_path = out_dir / f"{out_prefix}_hbond_ids.csv"

    _write_detail_csv(all_rows, detail_path)
    _write_summary_csv(summary_rows, summary_path)
    _write_hbond_ids_csv(all_rows, all_conformer_indices, hbond_ids_path)

    return detail_path, summary_path, hbond_ids_path
