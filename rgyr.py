"""rgyr.py

Radius of gyration (Rgyr) calculation — Chameleons v1.0.

Computes the mass-weighted radius of gyration for each conformer in a
multi-MOL2 ensemble.  The radius of gyration quantifies molecular
compactness: lower values indicate a more compact (folded) geometry,
higher values a more extended one.

    Rgyr = sqrt( Σ mᵢ·|rᵢ − r_cm|² / Σ mᵢ )

where mᵢ is the atomic mass, rᵢ the atom position, and r_cm the
centre of mass.

Outputs written by `run_rgyr`:
- <prefix>_rgyr.csv   (one row per conformer with Rgyr and auxiliary data)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mol2_io import Mol2Block


# ========================================================================
# Standard atomic masses (a subset covering typical organic / drug-like
# molecules; unknown elements fall back to 12.0 Da)
# ========================================================================

_ATOMIC_MASS: Dict[str, float] = {
    "H":  1.008,
    "He": 4.003,
    "Li": 6.941,
    "B":  10.81,
    "C":  12.011,
    "N":  14.007,
    "O":  15.999,
    "F":  18.998,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.086,
    "P":  30.974,
    "S":  32.065,
    "Cl": 35.453,
    "K":  39.098,
    "Ca": 40.078,
    "Fe": 55.845,
    "Zn": 65.380,
    "Se": 78.960,
    "Br": 79.904,
    "I":  126.904,
}

_DEFAULT_MASS: float = 12.0  # fallback for exotic elements


def _element_mass(element: str) -> float:
    """Return atomic mass for *element* (case-insensitive first match)."""
    return _ATOMIC_MASS.get(element, _ATOMIC_MASS.get(element.capitalize(), _DEFAULT_MASS))


# ========================================================================
# MOL2 ATOM parser (lightweight — only needs coords + element)
# ========================================================================

def _parse_atoms(block_text: str) -> List[dict]:
    """Parse a MOL2 ATOM section.

    Returns a list of dicts with keys: id, name, coords (np.ndarray),
    element, mass.
    """
    atoms: List[dict] = []
    in_atom_section = False

    for raw in block_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom_section = True
            continue
        if line.startswith("@<TRIPOS>"):
            if in_atom_section:
                break  # finished ATOM section
            continue

        if in_atom_section:
            parts = line.split()
            if len(parts) < 6:
                continue
            atom_id = int(parts[0])
            name = parts[1]
            coords = np.array(
                [float(parts[2]), float(parts[3]), float(parts[4])], dtype=float,
            )
            sybyl_type = parts[5]
            element = sybyl_type.split(".")[0]
            mass = _element_mass(element)
            atoms.append({
                "id": atom_id,
                "name": name,
                "coords": coords,
                "element": element,
                "mass": mass,
            })

    return atoms


# ========================================================================
# Radius of gyration
# ========================================================================

def compute_rgyr(
    atoms: List[dict],
    heavy_only: bool = False,
) -> Tuple[float, np.ndarray, float]:
    """Compute mass-weighted radius of gyration.

    Parameters
    ----------
    atoms : list of atom dicts (must contain 'coords', 'mass', 'element')
    heavy_only : if True, exclude hydrogen atoms from the calculation

    Returns
    -------
    rgyr      : radius of gyration (Å)
    centre    : centre of mass (3-array, Å)
    total_mass: sum of masses used (Da)
    """
    selected = atoms if not heavy_only else [a for a in atoms if a["element"] != "H"]
    if not selected:
        return 0.0, np.zeros(3), 0.0

    coords = np.array([a["coords"] for a in selected])  # (N, 3)
    masses = np.array([a["mass"] for a in selected])     # (N,)

    total_mass = float(masses.sum())
    if total_mass == 0.0:
        return 0.0, np.zeros(3), 0.0

    # Centre of mass
    centre = (masses[:, None] * coords).sum(axis=0) / total_mass  # (3,)

    # Rgyr = sqrt( Σ mᵢ |rᵢ - r_cm|² / M )
    displacements = coords - centre  # (N, 3)
    sq_dists = (displacements ** 2).sum(axis=1)  # (N,)
    rgyr = float(np.sqrt((masses * sq_dists).sum() / total_mass))

    return rgyr, centre, total_mass


# ========================================================================
# Output writer
# ========================================================================

def _write_rgyr_csv(rows: List[dict], out_path: Path) -> None:
    """Write _rgyr.csv (one row per conformer)."""
    fieldnames = [
        "molecule_conformer",
        "molecule_name",
        "rgyr_all_atoms",
        "rgyr_heavy_atoms",
        "n_atoms_total",
        "n_heavy_atoms",
        "total_mass_Da",
        "com_x",
        "com_y",
        "com_z",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ========================================================================
# Public API
# ========================================================================

def run_rgyr(
    blocks: List[Mol2Block],
    out_dir: Path,
    out_prefix: str,
) -> Path:
    """Compute radius of gyration for every conformer and write a CSV.

    Parameters
    ----------
    blocks     : list of Mol2Block instances (from mol2_io.split_mol2_blocks)
    out_dir    : output directory
    out_prefix : filename prefix (typically the MOL2 stem)

    Returns
    -------
    Path to the written ``<prefix>_rgyr.csv`` file.
    """
    rows: List[dict] = []

    for blk in blocks:
        atoms = _parse_atoms(blk.text)
        if not atoms:
            rows.append({
                "molecule_conformer": blk.index,
                "molecule_name": blk.name,
                "rgyr_all_atoms": "",
                "rgyr_heavy_atoms": "",
                "n_atoms_total": 0,
                "n_heavy_atoms": 0,
                "total_mass_Da": 0.0,
                "com_x": "",
                "com_y": "",
                "com_z": "",
            })
            continue

        # All-atom Rgyr
        rgyr_all, com_all, mass_all = compute_rgyr(atoms, heavy_only=False)

        # Heavy-atom-only Rgyr
        heavy_atoms = [a for a in atoms if a["element"] != "H"]
        rgyr_heavy, _, _ = compute_rgyr(atoms, heavy_only=True)

        rows.append({
            "molecule_conformer": blk.index,
            "molecule_name": blk.name,
            "rgyr_all_atoms": round(rgyr_all, 4),
            "rgyr_heavy_atoms": round(rgyr_heavy, 4),
            "n_atoms_total": len(atoms),
            "n_heavy_atoms": len(heavy_atoms),
            "total_mass_Da": round(mass_all, 3),
            "com_x": round(float(com_all[0]), 4),
            "com_y": round(float(com_all[1]), 4),
            "com_z": round(float(com_all[2]), 4),
        })

    out_path = out_dir / f"{out_prefix}_rgyr.csv"
    _write_rgyr_csv(rows, out_path)
    return out_path
