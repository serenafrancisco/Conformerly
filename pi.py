"""pi.py

Aromatic ring detection + π–π stacking analysis (RDKit-based) — v1.9.

Only fully aromatic ring systems are detected. Non-aromatic cycles and
alkyl clusters are intentionally excluded.

Outputs written by `run_pi`:
- <prefix>_ar_ring_systems.csv   (aromatic ring systems with patch detail — topology only)
- <prefix>_pi_stacking.csv       (detailed π–π classification per patch pair per conformer,
                                  now includes pi_label column)
- <prefix>_pi_summary.csv        (per-conformer aggregate counts + pi_labels_present)
- <prefix>_pi_label_ids.csv      (per-label persistence across conformers, analogous to
                                  hbond_ids.csv in imhb.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
class Moiety:
    """A fully aromatic ring system treated as a π unit."""
    label: str
    atom_indices: Tuple[int, ...]


@dataclass(frozen=True)
class RingPatch:
    """A single aromatic ring patch inside a ring system."""
    label: str
    atom_indices: Tuple[int, ...]


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
# Graph helper: connected components
# ========================================================================

def connected_components(n: int, edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
    """Return connected components given n nodes and an edge list."""
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = [False] * n
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(sorted(comp))
    return comps


# ========================================================================
# Aromatic ring system detection
# ========================================================================

def detect_aromatic_ring_systems(mol) -> Tuple[List[Moiety], Dict[str, List[RingPatch]]]:
    """Detect fully aromatic ring systems only.

    A ring system is included only if every atom in the merged system is
    aromatic. Non-aromatic cycles are silently skipped.
    """
    rings = [tuple(int(a) for a in r) for r in mol.GetRingInfo().AtomRings()]
    if not rings:
        return [], {}

    ring_sets = [set(r) for r in rings]

    # Keep only rings where every atom is aromatic
    aromatic_ring_indices = [
        i for i, r in enumerate(ring_sets)
        if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r)
    ]
    if not aromatic_ring_indices:
        return [], {}

    # Merge fused/spiro aromatic rings into systems
    edges: List[Tuple[int, int]] = []
    for ii in range(len(aromatic_ring_indices)):
        for jj in range(ii + 1, len(aromatic_ring_indices)):
            ri = aromatic_ring_indices[ii]
            rj = aromatic_ring_indices[jj]
            if ring_sets[ri].intersection(ring_sets[rj]):
                edges.append((ii, jj))

    comps = connected_components(len(aromatic_ring_indices), edges)

    moieties: List[Moiety] = []
    patches_by_system: Dict[str, List[RingPatch]] = {}

    for sys_id, comp in enumerate(comps, start=1):
        union_atoms: set[int] = set()
        for local_idx in comp:
            union_atoms.update(ring_sets[aromatic_ring_indices[local_idx]])

        sys_atoms = tuple(sorted(union_atoms))
        sys_label = f"ring_system_{sys_id}"
        moieties.append(Moiety(label=sys_label, atom_indices=sys_atoms))

        patches: List[RingPatch] = []
        for pid, local_idx in enumerate(comp, start=1):
            ring_atoms = tuple(sorted(rings[aromatic_ring_indices[local_idx]]))
            patches.append(RingPatch(label=f"{sys_label}.patch_{pid}", atom_indices=ring_atoms))
        patches_by_system[sys_label] = patches

    return moieties, patches_by_system


# ========================================================================
# Bond-separation exclusion for ring-system pairs
# ========================================================================

def exclude_pair_by_bonds(dist_mat: np.ndarray, ma: Moiety, mb: Moiety, min_bond_sep: int) -> bool:
    """Return True if the closest atoms of two ring systems are within min_bond_sep bonds."""
    if min_bond_sep <= 0:
        return False
    sub = dist_mat[np.ix_(list(ma.atom_indices), list(mb.atom_indices))]
    return int(np.min(sub)) <= min_bond_sep


# ========================================================================
# Moiety annotation rows (with patch-level detail)
# ========================================================================

def _sybyl_type_for_rdkit_atom(atom) -> str:
    """Return the Sybyl atom type stored by MOL2 import, falling back to element."""
    if atom.HasProp("_TriposAtomType"):
        return atom.GetProp("_TriposAtomType")
    return atom.GetSymbol()


def moiety_annotation_rows(
    mol,
    mol_name: str,
    moieties: Sequence[Moiety],
    patches_by_system: Dict[str, List[RingPatch]],
) -> List[Dict[str, object]]:
    """Build CSV-ready rows describing each aromatic ring system.

    One row per patch (expanded format).  For single-ring systems there
    is one row.  For fused systems (e.g. naphthalene) the system-level
    columns are repeated on each row and the patch-level columns differ.

    Column changes vs v1.6:
      - ``atom_symbols``  → ``atom_types``  (Sybyl types, not element letters)
      - ``patch_atom_symbols`` → ``patch_atom_types`` (Sybyl types)
      - ``atom_indices`` and ``patch_atom_indices`` removed
      - ``n_patches`` moved immediately after ``n_atoms``
      - Expanded: one row per patch instead of semicolon-separated
    """
    rows: List[Dict[str, object]] = []
    for m in moieties:
        c = centroid(coords_for_atoms(mol, m.atom_indices))
        sys_atoms = [mol.GetAtomWithIdx(int(i)) for i in m.atom_indices]
        sys_types = [_sybyl_type_for_rdkit_atom(a) for a in sys_atoms]
        sys_names = [
            a.GetProp("_TriposAtomName") if a.HasProp("_TriposAtomName") else ""
            for a in sys_atoms
        ]

        patches = patches_by_system.get(m.label, [])
        n_patches = len(patches)

        # System-level fields (repeated on every row)
        sys_base: Dict[str, object] = {
            "molecule_name": mol_name,
            "moiety_label": m.label,
            "n_atoms": len(m.atom_indices),
            "n_patches": n_patches,
            "atom_types": ",".join(sys_types),
            "atom_names": ",".join(sys_names),
            "centroid_x": round(float(c[0]), 4),
            "centroid_y": round(float(c[1]), 4),
            "centroid_z": round(float(c[2]), 4),
        }

        for p in patches:
            p_atoms = [mol.GetAtomWithIdx(int(i)) for i in p.atom_indices]
            pc = centroid(coords_for_atoms(mol, p.atom_indices))
            rows.append({
                **sys_base,
                "patch_labels": p.label,
                "patch_n_atoms": len(p.atom_indices),
                "patch_atom_types": ",".join(_sybyl_type_for_rdkit_atom(a) for a in p_atoms),
                "patch_atom_names": ",".join(
                    a.GetProp("_TriposAtomName") if a.HasProp("_TriposAtomName") else ""
                    for a in p_atoms
                ),
                "patch_centroid_x": round(float(pc[0]), 4),
                "patch_centroid_y": round(float(pc[1]), 4),
                "patch_centroid_z": round(float(pc[2]), 4),
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
# All π–π contacts between two ring systems (patch-level search)
# ========================================================================

def pi_all_for_system_pair(
    mol,
    sys_a: Moiety,
    sys_b: Moiety,
    patches_a: Sequence[RingPatch],
    patches_b: Sequence[RingPatch],
    params: Dict[str, float],
) -> List[Dict[str, object]]:
    """Test all patch pairs and return ALL π–π contacts found.

    Each passing patch pair generates its own row.  If no contacts are
    found, an empty list is returned (non-contacts are not reported,
    analogous to _hbonds.csv which only lists detected H-bonds).
    """
    contacts: List[Dict[str, object]] = []

    for pa in patches_a:
        ca = centroid(coords_for_atoms(mol, pa.atom_indices))
        na = fit_plane_normal(coords_for_atoms(mol, pa.atom_indices))
        for pb in patches_b:
            cb = centroid(coords_for_atoms(mol, pb.atom_indices))
            nb = fit_plane_normal(coords_for_atoms(mol, pb.atom_indices))
            dist_ = float(np.linalg.norm(ca - cb))
            pi_class, ang = classify_pi(ca, na, cb, nb, params)
            if pi_class != "none":
                contacts.append({
                    "moiety_a": sys_a.label,
                    "moiety_b": sys_b.label,
                    "patch_a": pa.label,
                    "patch_b": pb.label,
                    "pi_class": pi_class,
                    "patch_centroid_dist_A": round(dist_, 4),
                    "plane_angle_deg": round(ang, 4),
                })

    return contacts


# ========================================================================
# Pair-level computation for one conformer
# ========================================================================

def compute_pair_outputs(
    mol,
    moieties: Sequence[Moiety],
    patches_by_system: Dict[str, List[RingPatch]],
    min_bond_sep: int,
    use_pi_criteria: bool,
    pi_params: Dict[str, float],
) -> Tuple[List[Dict[str, object]], int, int]:
    """Return π-stacking rows, evaluated-pair count, excluded-pair count.

    Distance computation removed in v1.7 — centroid distances are already
    reported per-contact in _pi_stacking.csv.
    """
    topo = Chem.GetDistanceMatrix(mol)  # type: ignore[attr-defined]

    moiety_centroids: Dict[str, np.ndarray] = {
        m.label: centroid(coords_for_atoms(mol, m.atom_indices)) for m in moieties
    }
    labels = sorted(m.label for m in moieties)
    label_to_m = {m.label: m for m in moieties}

    # Pre-filter distance for non-criteria mode: use max of the two Maestro limits
    prefilter = max(float(pi_params["pi_ff_max_dist"]), float(pi_params["pi_ef_max_dist"]))

    pi_rows: List[Dict[str, object]] = []
    excluded = 0
    n_pairs = 0

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            ma, mb = label_to_m[a], label_to_m[b]

            if exclude_pair_by_bonds(topo, ma, mb, min_bond_sep):
                excluded += 1
                continue

            n_pairs += 1

            # --- π–π stacking ---
            if use_pi_criteria:
                contact_rows = pi_all_for_system_pair(
                    mol, ma, mb,
                    patches_by_system.get(a, []),
                    patches_by_system.get(b, []),
                    pi_params,
                )
                pi_rows.extend(contact_rows)
            else:
                # Simple threshold mode: only report if within pre-filter distance
                d_centroid = float(np.linalg.norm(moiety_centroids[a] - moiety_centroids[b]))
                if d_centroid <= prefilter:
                    pi_rows.append({
                        "moiety_a": a,
                        "moiety_b": b,
                        "patch_a": "",
                        "patch_b": "",
                        "pi_class": "centroid_threshold",
                        "patch_centroid_dist_A": d_centroid,
                        "plane_angle_deg": float("nan"),
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
    min_bond_sep: int = 1,
    use_pi_criteria: bool = True,
    # --- Maestro-aligned defaults (distance + angle only) ---
    pi_ff_max_dist: float = 4.4,       # face-to-face max centroid dist (Å)
    pi_ef_max_dist: float = 5.5,       # edge-to-face max centroid dist (Å)
    pi_parallel_angle: float = 30.0,   # max angle for face-to-face (°)
    pi_t_angle_min: float = 60.0,      # min angle for edge-to-face (°)
) -> List[Path]:
    """Run aromatic π–π analysis on all conformer blocks and write CSVs.

    Returns paths to four CSV files:
      [ar_ring_systems.csv, pi_stacking.csv, pi_summary.csv, pi_label_ids.csv]
    """
    assert_rdkit_available()

    pi_params = {
        "pi_ff_max_dist":    float(pi_ff_max_dist),
        "pi_ef_max_dist":    float(pi_ef_max_dist),
        "pi_parallel_angle": float(pi_parallel_angle),
        "pi_t_angle_min":    float(pi_t_angle_min),
    }

    moiety_rows: List[Dict] = []
    pi_rows_all: List[Dict] = []
    pi_summary_rows: List[Dict] = []
    moieties_captured = False

    for blk in blocks:
        mol = mol_from_mol2_block(blk.text)
        if mol.GetNumConformers() == 0:
            raise ValueError(f"Block {blk.index} has no 3D coordinates (no conformer).")

        moieties, patches_by_system = detect_aromatic_ring_systems(mol)

        # Aromatic moieties are topology-derived: report once from the first conformer
        if not moieties_captured:
            moiety_rows.extend(moiety_annotation_rows(mol, blk.name, moieties, patches_by_system))
            moieties_captured = True

        pi_rows, n_pairs, excluded_pairs = compute_pair_outputs(
            mol=mol,
            moieties=moieties,
            patches_by_system=patches_by_system,
            min_bond_sep=int(min_bond_sep),
            use_pi_criteria=bool(use_pi_criteria),
            pi_params=pi_params,
        )

        # Stamp conformer info onto pi rows
        for r in pi_rows:
            r["molecule_conformer"] = blk.index
            r["molecule_name"] = blk.name
        pi_rows_all.extend(pi_rows)

        # π summary: per-conformer counts
        if use_pi_criteria:
            ff_count = sum(1 for r in pi_rows if r.get("pi_class") == "face-to-face")
            ef_count = sum(1 for r in pi_rows if r.get("pi_class") == "edge-to-face")
            total    = ff_count + ef_count
            pi_summary_rows.append({
                "molecule_conformer": blk.index,
                "molecule_name": blk.name,
                "n_aromatic_systems": len(moieties),
                "n_pairs_evaluated": n_pairs,
                "n_pairs_excluded_by_bonds": excluded_pairs,
                "mode": "pi_criteria",
                "pi_face_to_face_count": ff_count,
                "pi_edge_to_face_count": ef_count,
                "pi_total_count": total,
            })
        else:
            arom_contacts = len(pi_rows)
            pi_summary_rows.append({
                "molecule_conformer": blk.index,
                "molecule_name": blk.name,
                "n_aromatic_systems": len(moieties),
                "n_pairs_evaluated": n_pairs,
                "n_pairs_excluded_by_bonds": excluded_pairs,
                "mode": "centroid_threshold",
                "pi_face_to_face_count": 0,
                "pi_edge_to_face_count": 0,
                "pi_total_count": arom_contacts,
            })

    # --- Assign persistent PI labels and stamp onto contact rows ---

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
    # Build a quick mapping: conformer_index -> sorted list of PI labels
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

    # Moieties: topology-only, expanded (one row per patch)
    moiety_col_order = [
        "molecule_name", "moiety_label", "n_atoms", "n_patches",
        "atom_types", "atom_names",
        "centroid_x", "centroid_y", "centroid_z",
        "patch_labels", "patch_n_atoms",
        "patch_atom_types", "patch_atom_names",
        "patch_centroid_x", "patch_centroid_y", "patch_centroid_z",
    ]
    if moiety_rows:
        df_moieties = pd.DataFrame(moiety_rows)[moiety_col_order]  # type: ignore[union-attr]
        df_moieties = df_moieties.sort_values(["moiety_label", "patch_labels"])
    else:
        df_moieties = pd.DataFrame(columns=moiety_col_order)  # type: ignore[union-attr]

    # Pi summary: per-conformer counts + labels present
    sum_cols = [
        "molecule_conformer", "molecule_name", "n_aromatic_systems",
        "n_pairs_evaluated", "n_pairs_excluded_by_bonds", "mode",
        "pi_face_to_face_count", "pi_edge_to_face_count", "pi_total_count",
        "pi_labels_present",
    ]
    df_pi_sum = pd.DataFrame(pi_summary_rows).sort_values(["molecule_conformer"]) if pi_summary_rows else \
        pd.DataFrame(columns=sum_cols)  # type: ignore[union-attr]

    # Pi stacking detail: one row per patch-pair contact.
    # Column order mirrors hbonds.csv: identity first, geometry middle,
    # pi_label last — so the label column is always the rightmost column.
    _PI_COL_ORDER = [
        "molecule_conformer", "molecule_name",
        "moiety_a", "moiety_b",
        "patch_a", "patch_b",
        "pi_class",
        "patch_centroid_dist_A", "plane_angle_deg",
        "pi_label",
    ]
    if pi_rows_all:
        df_pi = pd.DataFrame(pi_rows_all)  # type: ignore[union-attr]
        # Keep any extra columns that might be present, insert before pi_label
        known = set(_PI_COL_ORDER)
        extras = [c for c in df_pi.columns if c not in known]
        col_order = _PI_COL_ORDER[:-1] + extras + ["pi_label"]
        col_order = [c for c in col_order if c in df_pi.columns]
        df_pi = df_pi[col_order].sort_values(
            ["molecule_conformer", "moiety_a", "moiety_b", "patch_a", "patch_b"]
        )
    else:
        df_pi = pd.DataFrame(columns=_PI_COL_ORDER)  # type: ignore[union-attr]

    moieties_path  = out_dir / f"{out_prefix}_ar_ring_systems.csv"
    pi_path        = out_dir / f"{out_prefix}_pi_stacking.csv"
    pi_sum_path    = out_dir / f"{out_prefix}_pi_summary.csv"
    pi_labels_path = out_dir / f"{out_prefix}_pi_label_ids.csv"

    df_moieties.to_csv(moieties_path, index=False)   # type: ignore[union-attr]
    df_pi.to_csv(pi_path, index=False)                # type: ignore[union-attr]
    df_pi_sum.to_csv(pi_sum_path, index=False)        # type: ignore[union-attr]
    _write_pi_label_ids_csv(label_id_rows, pi_labels_path)

    return [moieties_path, pi_path, pi_sum_path, pi_labels_path]