"""pi.py

Aromatic ring detection + pi-pi stacking analysis (RDKit-based) -- v2.1.

Works on **individual aromatic rings** rather than on a two-level
Moiety / RingPatch hierarchy.

Rationale for removing ring-system grouping
-------------------------------------------
The previous version grouped fused rings (e.g. the two rings of
naphthalene) into a single "ring system" (Moiety) and called individual
rings within it "patches".  Pairs belonging to the *same* ring system
were never evaluated.  This exclusion is now handled entirely by the
bond-separation filter:

  * Fused rings share atoms  ->  min bond distance = 0  ->  excluded by
    ``0 < min_bond_sep`` (default 2).
  * Tricyclic end-rings (e.g. anthracene ring-1 vs ring-3) have min bond
    distance = 2 and therefore pass the filter, but the geometric
    criteria (centroid distance > 4.4 A at ~0 deg inter-plane angle for
    any planar fused system) exclude them anyway.

For all practical drug-like aromatic systems the bond-separation + geometry
double filter is equivalent to the old ring-system grouping, with
considerably simpler code.

Bug fix vs v1.9 (carried forward)
----------------------------------
The original code used ``<=`` with a default of 1, meaning rings at
distance 1 (biphenyl-type) were excluded but rings at distance 2 were
not -- so Ph-X-Ph one-atom-bridged pairs were always detected, with no
way to suppress them.  The default has been raised to 2 so that, with
``<=``, both distance-1 and distance-2 pairs are excluded by default.
Only pairs with three or more bonds of separation (flexible two-atom+
linkers) are evaluated geometrically.

Outputs written by `run_pi`:
- <prefix>_ar_rings.csv          (individual aromatic rings -- topology only)
- <prefix>_pi_stacking.csv       (detailed pi-pi classification per ring pair per conformer)
- <prefix>_pi_summary.csv        (per-conformer aggregate counts + pi_labels_present)
- <prefix>_pi_label_ids.csv      (per-label persistence across conformers)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd          # type: ignore
    from rdkit import Chem       # type: ignore
    _HAS_RDKIT = True
except Exception:
    pd = None                    # type: ignore
    Chem = None                  # type: ignore
    _HAS_RDKIT = False

from mol2_io import Mol2Block


# ========================================================================
# Data classes
# ========================================================================

@dataclass(frozen=True)
class AromaticRing:
    """A single fully aromatic ring."""
    label: str                     # e.g. "ring_1"
    atom_indices: Tuple[int, ...]  # RDKit 0-based atom indices, sorted


# ========================================================================
# RDKit availability check
# ========================================================================

def assert_rdkit_available() -> None:
    if not _HAS_RDKIT:
        raise RuntimeError(
            "π–π analysis requires RDKit + pandas.\n"
            "Typical install: conda install -c conda-forge rdkit pandas"
        )


# ========================================================================
# Geometry helpers
# ========================================================================

def mol_from_mol2_block(block_text: str):
    """Parse a MOL2 text block into an RDKit Mol object."""
    assert_rdkit_available()
    mol = Chem.MolFromMol2Block(block_text, sanitize=True, removeHs=False)  # type: ignore[attr-defined]
    if mol is None:
        raise ValueError("RDKit failed to parse a MOL2 block.")
    return mol


def coords_for_atoms(mol, atom_indices: Sequence[int]) -> np.ndarray:
    """Return Nx3 coordinate array for the given atom indices."""
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(int(i))) for i in atom_indices], dtype=float)


def centroid(xyz: np.ndarray) -> np.ndarray:
    """Mean position of an Nx3 array; returns [nan,nan,nan] if empty."""
    return xyz.mean(axis=0) if xyz.size else np.array([np.nan, np.nan, np.nan], dtype=float)


def unit_vector(v: np.ndarray) -> np.ndarray:
    """Normalise a vector; returns [nan,nan,nan] for zero-length input."""
    n = float(np.linalg.norm(v))
    return (v / n) if n != 0.0 else np.array([np.nan, np.nan, np.nan], dtype=float)


def fit_plane_normal(xyz: np.ndarray) -> np.ndarray:
    """Best-fit plane normal via SVD.  Needs >= 3 points."""
    if xyz.shape[0] < 3:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    centered = xyz - xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return unit_vector(vt[-1, :])


def plane_angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    """Angle between two plane normals (0–90°, symmetry-aware)."""
    if np.any(np.isnan(n1)) or np.any(np.isnan(n2)):
        return float("nan")
    dot = abs(float(np.dot(n1, n2)))
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(np.arccos(dot)))


# ========================================================================
# Bond-separation exclusion
# ========================================================================

def _bridge_is_sp2(mol, atom_a_idx: int, atom_b_idx: int) -> bool:
    """Return True if the unique bridging atom between two atoms 2 bonds apart
    is sp2- or sp-hybridised (indicating a conjugated, rotation-restricted link).

    For atoms A and B separated by exactly 2 bonds, the bridging atom X is the
    one that is a direct neighbour of both A and B.  If X is sp2 (e.g. a vinyl,
    imine, or carbonyl carbon; an aromatic atom) the A–X–B segment is part of a
    conjugated pi-system and the apparent geometry between the two rings is
    structurally forced rather than driven by a non-covalent stacking force.

    Returns False if no sp2/sp bridge is found or if mol is None.
    """
    if mol is None:
        return False
    nbrs_a = {n.GetIdx() for n in mol.GetAtomWithIdx(int(atom_a_idx)).GetNeighbors()}
    nbrs_b = {n.GetIdx() for n in mol.GetAtomWithIdx(int(atom_b_idx)).GetNeighbors()}
    for bridge_idx in nbrs_a & nbrs_b:
        hyb = mol.GetAtomWithIdx(bridge_idx).GetHybridization()
        if hyb in (
            Chem.rdchem.HybridizationType.SP2,   # type: ignore[attr-defined]
            Chem.rdchem.HybridizationType.SP,    # type: ignore[attr-defined]
        ):
            return True
    return False


def exclude_pair_by_bonds(
    dist_mat: np.ndarray,
    ra: "AromaticRing",
    rb: "AromaticRing",
    min_bond_sep: int,
    mol=None,
) -> bool:
    """Return True if the ring pair should be excluded based on bond topology.

    Two independent criteria are applied:

    1. **Bond-separation filter** (primary):
       Exclude if the minimum topological distance between any atom of ring A
       and any atom of ring B is less than or equal to ``min_bond_sep``.
       ``min_bond_sep`` therefore means: "exclude pairs up to and including
       this many bonds of separation."

         min_bond_sep = 2  (default)
           * dist = 0  fused rings (shared atoms)          -> excluded  (0 <= 2)
           * dist = 1  directly bonded (biphenyl-type)     -> excluded  (1 <= 2)
           * dist = 2  one bridging atom (Ph-X-Ph)         -> excluded  (2 <= 2)
           * dist >= 3 two+ bridging atoms, flexible linker-> allowed

    2. **sp2-bridge filter** (secondary, applied when dist == min_bond_sep + 1):
       Even when the bond-separation filter passes a pair, exclude it if the
       single bridging atom is sp2- or sp-hybridised.  Such bridges create an
       extended conjugated pi-system that locks the rings toward coplanarity;
       any apparent stacking geometry is a structural artefact.  This filter
       is mainly relevant when the user lowers min_bond_sep below the default.

    Parameters
    ----------
    dist_mat : np.ndarray
        All-pairs topological distance matrix from ``Chem.GetDistanceMatrix``.
    ra, rb : AromaticRing
        The two rings to evaluate.
    min_bond_sep : int
        Pairs at this distance or closer are excluded (default 2).
    mol : RDKit Mol, optional
        Required for the sp2-bridge filter.  If None the filter is skipped.
    """
    if min_bond_sep <= 0:
        return False

    ra_atoms = list(ra.atom_indices)
    rb_atoms = list(rb.atom_indices)
    sub = dist_mat[np.ix_(ra_atoms, rb_atoms)]
    min_dist = int(np.min(sub))

    # Primary: bond-separation — exclude if at or within the threshold
    if min_dist <= min_bond_sep:
        return True

    # Secondary: sp2-bridge check for the next distance tier (min_bond_sep + 1).
    # Catches conjugated linkers that survive the primary filter when the user
    # has deliberately lowered min_bond_sep.
    if min_dist == min_bond_sep + 1 and mol is not None:
        positions = np.argwhere(sub == min_dist)
        for i_loc, j_loc in positions:
            if _bridge_is_sp2(mol, ra_atoms[int(i_loc)], rb_atoms[int(j_loc)]):
                return True

    return False


# ========================================================================
# Aromatic ring detection (individual rings)
# ========================================================================

def detect_aromatic_rings(mol) -> List[AromaticRing]:
    """Return all fully aromatic rings as individual AromaticRing objects.

    Only rings where *every* atom is flagged as aromatic by RDKit are
    included.  Non-aromatic cycles and partially aromatic rings are
    silently skipped.

    Labels are assigned in the order RDKit enumerates rings (deterministic
    for a given molecule) as ``ring_1``, ``ring_2``, ...
    """
    rings = [tuple(int(a) for a in r) for r in mol.GetRingInfo().AtomRings()]
    result: List[AromaticRing] = []
    ring_id = 1
    for ring in rings:
        if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            result.append(AromaticRing(
                label=f"ring_{ring_id}",
                atom_indices=tuple(sorted(ring)),
            ))
            ring_id += 1
    return result





# ========================================================================
# Ring annotation rows for ar_rings.csv
# ========================================================================

def _sybyl_type_for_rdkit_atom(atom) -> str:
    """Return the Sybyl atom type stored by MOL2 import, falling back to element."""
    if atom.HasProp("_TriposAtomType"):
        return atom.GetProp("_TriposAtomType")
    return atom.GetSymbol()


def ring_annotation_rows(
    mol,
    mol_name: str,
    rings: List[AromaticRing],
) -> List[Dict[str, object]]:
    """Build CSV-ready rows describing each individual aromatic ring.

    One row per ring.  Columns: molecule_name, ring_label, n_atoms,
    atom_types (Sybyl), atom_names, centroid_x/y/z.
    """
    rows: List[Dict[str, object]] = []
    for r in rings:
        atoms = [mol.GetAtomWithIdx(int(i)) for i in r.atom_indices]
        c = centroid(coords_for_atoms(mol, r.atom_indices))
        rows.append({
            "molecule_name": mol_name,
            "ring_label":    r.label,
            "n_atoms":       len(r.atom_indices),
            "atom_types":    ",".join(_sybyl_type_for_rdkit_atom(a) for a in atoms),
            "atom_names":    ",".join(
                a.GetProp("_TriposAtomName") if a.HasProp("_TriposAtomName") else ""
                for a in atoms
            ),
            "centroid_x": round(float(c[0]), 4),
            "centroid_y": round(float(c[1]), 4),
            "centroid_z": round(float(c[2]), 4),
        })
    return rows


# ========================================================================
# π–π classification (Maestro-aligned cutoffs)
# ========================================================================

def classify_pi(
    c1: np.ndarray, n1: np.ndarray,
    c2: np.ndarray, n2: np.ndarray,
    params: Dict[str, float],
) -> Tuple[str, float]:
    """Classify a patch pair as face-to-face / edge-to-face / none.

    Maestro logic: centroid distance + inter-plane angle only.

      - Face-to-face:  dist ≤ 4.4 Å  AND  angle ≤ 30°
      - Edge-to-face:  dist ≤ 5.5 Å  AND  angle ≥ 60°

    Returns (classification, angle).
    """
    d = float(np.linalg.norm(c1 - c2))

    prefilter = max(float(params["pi_ff_max_dist"]), float(params["pi_ef_max_dist"]))
    if d > prefilter:
        return "none", float("nan")

    ang = plane_angle_deg(n1, n2)
    if np.isnan(ang):
        return "none", float("nan")

    # Face-to-face (parallel)
    if d <= float(params["pi_ff_max_dist"]) and ang <= float(params["pi_parallel_angle"]):
        return "face-to-face", ang

    # Edge-to-face (T-shaped)
    if d <= float(params["pi_ef_max_dist"]) and ang >= float(params["pi_t_angle_min"]):
        return "edge-to-face", ang

    return "none", ang


# ========================================================================
# pi-pi classification for a single ring pair
# ========================================================================

def classify_ring_pair(
    mol,
    ra: AromaticRing,
    rb: AromaticRing,
    params: Dict[str, float],
) -> Tuple[str, float, float]:
    """Classify a pair of individual rings as face-to-face / edge-to-face / none.

    Returns (pi_class, plane_angle_deg, centroid_dist_A).
    """
    ca = centroid(coords_for_atoms(mol, ra.atom_indices))
    na = fit_plane_normal(coords_for_atoms(mol, ra.atom_indices))
    cb = centroid(coords_for_atoms(mol, rb.atom_indices))
    nb = fit_plane_normal(coords_for_atoms(mol, rb.atom_indices))
    dist_ = float(np.linalg.norm(ca - cb))
    pi_class, ang = classify_pi(ca, na, cb, nb, params)
    return pi_class, ang, dist_


# ========================================================================
# Pair-level computation for one conformer
# ========================================================================

def compute_pair_outputs(
    mol,
    rings: List[AromaticRing],
    min_bond_sep: int,
    use_pi_criteria: bool,
    pi_params: Dict[str, float],
) -> Tuple[List[Dict[str, object]], int, int]:
    """Evaluate all ring pairs and return pi-stacking rows, pair count, excluded count."""
    topo = Chem.GetDistanceMatrix(mol)  # type: ignore[attr-defined]

    prefilter = max(float(pi_params["pi_ff_max_dist"]), float(pi_params["pi_ef_max_dist"]))

    pi_rows: List[Dict[str, object]] = []
    excluded = 0
    n_pairs = 0

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            ra, rb = rings[i], rings[j]

            if exclude_pair_by_bonds(topo, ra, rb, min_bond_sep, mol=mol):
                excluded += 1
                continue

            n_pairs += 1

            if use_pi_criteria:
                pi_class, ang, dist_ = classify_ring_pair(mol, ra, rb, pi_params)
                if pi_class != "none":
                    pi_rows.append({
                        "ring_a":                 ra.label,
                        "ring_b":                 rb.label,
                        # keep patch_a/patch_b as aliases so plots_ids.py
                        # and build_pi_label_ids continue to work unchanged
                        "patch_a":                ra.label,
                        "patch_b":                rb.label,
                        "pi_class":               pi_class,
                        "patch_centroid_dist_A":  round(dist_, 4),
                        "plane_angle_deg":        round(ang, 4),
                    })
            else:
                ca = centroid(coords_for_atoms(mol, ra.atom_indices))
                cb = centroid(coords_for_atoms(mol, rb.atom_indices))
                d_centroid = float(np.linalg.norm(ca - cb))
                if d_centroid <= prefilter:
                    pi_rows.append({
                        "ring_a":                ra.label,
                        "ring_b":                rb.label,
                        "patch_a":               ra.label,
                        "patch_b":               rb.label,
                        "pi_class":              "centroid_threshold",
                        "patch_centroid_dist_A": round(d_centroid, 4),
                        "plane_angle_deg":       float("nan"),
                    })

    return pi_rows, n_pairs, excluded


# ========================================================================
# π–π label persistence (analogous to IMHB hbond_ids)
# ========================================================================

def build_pi_label_ids(
    pi_rows_all: List[Dict],
    n_conformers_total: int,
) -> Tuple[List[Dict], Dict[Tuple[str, str], str]]:
    """Assign persistent PI_N labels to unique patch-pair contacts and build
    a per-conformer binary presence table, mirroring the hbond_ids.csv format.

    Output columns (one row per label):
        pi_label              — e.g. PI_1
        n_conformers_present  — how many conformers the stacking appears in
        frequency             — n_present / n_total (4 d.p.)
        pi_class              — dominant classification (face-to-face / edge-to-face)
        patch_a               — canonical patch label (lexicographically smaller)
        patch_b               — canonical patch label (lexicographically larger)
        conformer_1 … conformer_N — 1 if the stacking is present, 0 otherwise

    Parameters
    ----------
    pi_rows_all         : All π-stacking contact rows collected across all
                          conformers (output of ``compute_pair_outputs``).
    n_conformers_total  : Total number of conformers processed.

    Returns
    -------
    label_id_rows : List[Dict]
        One row per unique PI label, ready to write to CSV.
    pair_to_label : Dict[Tuple[str, str], str]
        Mapping from canonical (patch_a, patch_b) → PI_N label string.
    """
    pair_to_label: Dict[Tuple[str, str], str] = {}
    label_counter = 1
    label_to_conformers: Dict[str, set] = {}
    label_to_classes: Dict[str, Dict[str, int]] = {}

    for row in pi_rows_all:
        pa = str(row.get("patch_a", ""))
        pb = str(row.get("patch_b", ""))
        key: Tuple[str, str] = (min(pa, pb), max(pa, pb))

        if key not in pair_to_label:
            label = f"PI_{label_counter}"
            pair_to_label[key] = label
            label_counter += 1
            label_to_conformers[label] = set()
            label_to_classes[label] = {}

        label = pair_to_label[key]
        label_to_conformers[label].add(int(row["molecule_conformer"]))
        pi_class = str(row.get("pi_class", ""))
        label_to_classes[label][pi_class] = (
            label_to_classes[label].get(pi_class, 0) + 1
        )

    # All conformer indices seen (1-based, sorted)
    all_conformers = list(range(1, n_conformers_total + 1))

    label_id_rows: List[Dict] = []
    for key, label in sorted(
        pair_to_label.items(),
        key=lambda kv: int(kv[1].split("_")[1]),
    ):
        pa, pb = key
        present_set = label_to_conformers[label]
        n_present = len(present_set)
        class_counts = label_to_classes[label]
        dominant_class = max(class_counts, key=class_counts.get) if class_counts else ""

        row: Dict = {
            "pi_label":             label,
            "n_conformers_present": n_present,
            "frequency":            round(n_present / n_conformers_total, 4)
                                    if n_conformers_total > 0 else 0.0,
            "pi_class":             dominant_class,
            "patch_a":              pa,
            "patch_b":              pb,
        }
        # Binary presence vector — one column per conformer
        for ci in all_conformers:
            row[f"conformer_{ci}"] = 1 if ci in present_set else 0

        label_id_rows.append(row)

    return label_id_rows, pair_to_label


def _write_pi_label_ids_csv(rows: List[Dict], out_path: Path) -> None:
    """Write ``<prefix>_pi_label_ids.csv`` — mirrors hbond_ids.csv layout."""
    import csv as _csv
    if not rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "pi_label,n_conformers_present,frequency,pi_class,patch_a,patch_b\n",
            encoding="utf-8",
        )
        return
    fieldnames = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ========================================================================
# Public API
# ========================================================================

def run_pi(
    blocks: List[Mol2Block],
    out_dir: Path,
    out_prefix: str,
    min_bond_sep: int = 2,
    use_pi_criteria: bool = True,
    # --- Maestro-aligned defaults (distance + angle only) ---
    pi_ff_max_dist: float = 4.4,       # face-to-face max centroid dist (A)
    pi_ef_max_dist: float = 5.5,       # edge-to-face max centroid dist (A)
    pi_parallel_angle: float = 30.0,   # max angle for face-to-face (deg)
    pi_t_angle_min: float = 60.0,      # min angle for edge-to-face (deg)
) -> List[Path]:
    """Run aromatic pi-pi analysis on all conformer blocks and write CSVs.

    Returns paths to four CSV files:
      [ar_rings.csv, pi_stacking.csv, pi_summary.csv, pi_label_ids.csv]
    """
    assert_rdkit_available()

    pi_params = {
        "pi_ff_max_dist":    float(pi_ff_max_dist),
        "pi_ef_max_dist":    float(pi_ef_max_dist),
        "pi_parallel_angle": float(pi_parallel_angle),
        "pi_t_angle_min":    float(pi_t_angle_min),
    }

    ring_rows: List[Dict] = []
    pi_rows_all: List[Dict] = []
    pi_summary_rows: List[Dict] = []
    rings_captured = False

    for blk in blocks:
        mol = mol_from_mol2_block(blk.text)
        if mol.GetNumConformers() == 0:
            raise ValueError(f"Block {blk.index} has no 3D coordinates (no conformer).")

        rings = detect_aromatic_rings(mol)

        # Ring topology is conformer-independent: report once from the first block
        if not rings_captured:
            ring_rows.extend(ring_annotation_rows(mol, blk.name, rings))
            rings_captured = True

        pi_rows, n_pairs, excluded_pairs = compute_pair_outputs(
            mol=mol,
            rings=rings,
            min_bond_sep=int(min_bond_sep),
            use_pi_criteria=bool(use_pi_criteria),
            pi_params=pi_params,
        )

        for r in pi_rows:
            r["molecule_conformer"] = blk.index
            r["molecule_name"] = blk.name
        pi_rows_all.extend(pi_rows)

        # pi summary: per-conformer counts
        if use_pi_criteria:
            ff_count = sum(1 for r in pi_rows if r.get("pi_class") == "face-to-face")
            ef_count = sum(1 for r in pi_rows if r.get("pi_class") == "edge-to-face")
            total    = ff_count + ef_count
            pi_summary_rows.append({
                "molecule_conformer":      blk.index,
                "molecule_name":           blk.name,
                "n_aromatic_rings":        len(rings),
                "n_pairs_evaluated":       n_pairs,
                "n_pairs_excluded_by_bonds": excluded_pairs,
                "mode":                    "pi_criteria",
                "pi_face_to_face_count":   ff_count,
                "pi_edge_to_face_count":   ef_count,
                "pi_total_count":          total,
            })
        else:
            pi_summary_rows.append({
                "molecule_conformer":      blk.index,
                "molecule_name":           blk.name,
                "n_aromatic_rings":        len(rings),
                "n_pairs_evaluated":       n_pairs,
                "n_pairs_excluded_by_bonds": excluded_pairs,
                "mode":                    "centroid_threshold",
                "pi_face_to_face_count":   0,
                "pi_edge_to_face_count":   0,
                "pi_total_count":          len(pi_rows),
            })

    # --- Assign persistent PI labels ---
    label_id_rows, pair_to_label = build_pi_label_ids(
        pi_rows_all=pi_rows_all,
        n_conformers_total=len(blocks),
    )

    # Stamp pi_label onto every contact row
    for row in pi_rows_all:
        pa = str(row.get("patch_a", ""))
        pb = str(row.get("patch_b", ""))
        key = (min(pa, pb), max(pa, pb))
        row["pi_label"] = pair_to_label.get(key, "")

    # Add pi_labels_present to per-conformer summary rows
    conf_to_labels: Dict[int, List[str]] = {}
    for row in pi_rows_all:
        ci = int(row["molecule_conformer"])
        lbl = row.get("pi_label", "")
        if lbl:
            conf_to_labels.setdefault(ci, [])
            if lbl not in conf_to_labels[ci]:
                conf_to_labels[ci].append(lbl)

    for srow in pi_summary_rows:
        ci = int(srow["molecule_conformer"])
        lbls = sorted(conf_to_labels.get(ci, []),
                      key=lambda s: int(s.split("_")[1]) if "_" in s else 0)
        srow["pi_labels_present"] = ",".join(lbls)

    # --- Build DataFrames and write CSVs ---

    # Individual rings (topology only)
    rings_col_order = [
        "molecule_name", "ring_label", "n_atoms",
        "atom_types", "atom_names",
        "centroid_x", "centroid_y", "centroid_z",
    ]
    df_rings = (
        pd.DataFrame(ring_rows)[rings_col_order]
        if ring_rows else pd.DataFrame(columns=rings_col_order)
    )

    # pi summary
    sum_cols = [
        "molecule_conformer", "molecule_name", "n_aromatic_rings",
        "n_pairs_evaluated", "n_pairs_excluded_by_bonds", "mode",
        "pi_face_to_face_count", "pi_edge_to_face_count", "pi_total_count",
        "pi_labels_present",
    ]
    df_pi_sum = (
        pd.DataFrame(pi_summary_rows).sort_values(["molecule_conformer"])
        if pi_summary_rows else pd.DataFrame(columns=sum_cols)
    )

    # pi stacking detail
    _PI_COL_ORDER = [
        "molecule_conformer", "molecule_name",
        "ring_a", "ring_b",
        "pi_class",
        "patch_centroid_dist_A", "plane_angle_deg",
        "pi_label",
    ]
    if pi_rows_all:
        df_pi = pd.DataFrame(pi_rows_all)
        known = set(_PI_COL_ORDER)
        extras = [c for c in df_pi.columns if c not in known]
        col_order = _PI_COL_ORDER[:-1] + extras + ["pi_label"]
        col_order = [c for c in col_order if c in df_pi.columns]
        df_pi = df_pi[col_order].sort_values(
            ["molecule_conformer", "ring_a", "ring_b"]
        )
    else:
        df_pi = pd.DataFrame(columns=_PI_COL_ORDER)

    rings_path     = out_dir / f"{out_prefix}_ar_rings.csv"
    pi_path        = out_dir / f"{out_prefix}_pi_stacking.csv"
    pi_sum_path    = out_dir / f"{out_prefix}_pi_summary.csv"
    pi_labels_path = out_dir / f"{out_prefix}_pi_label_ids.csv"

    df_rings.to_csv(rings_path, index=False)
    df_pi.to_csv(pi_path, index=False)
    df_pi_sum.to_csv(pi_sum_path, index=False)
    _write_pi_label_ids_csv(label_id_rows, pi_labels_path)

    return [rings_path, pi_path, pi_sum_path, pi_labels_path]