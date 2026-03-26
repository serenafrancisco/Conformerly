"""psa3d.py

3D Polar Surface Area — Chameleons v1.0.

Computes the three-dimensional polar surface area (3D PSA) for each
conformer in a multi-MOL2 ensemble using the Shrake–Rupley algorithm
(numerical sphere-point method).

Two variants are reported per conformer:

  * **Solvent-exposed 3D PSA** (probe radius = 1.4 Å)
    The polar contribution to the solvent-accessible surface area
    (SASA).  This represents how much polar surface the solvent can
    reach once a 1.4 Å water-sized probe is rolled over the molecule.

  * **Molecular 3D PSA** (probe radius = 0.0 Å)
    The polar contribution to the van der Waals (molecular) surface.
    This is the intrinsic polar surface dictated purely by atomic
    radii, independent of solvent probe size.

Polar atoms are defined following the VegaZZ convention:
  N, O, S, P (polar heavy atoms) and any H NOT covalently bonded to C.
  Apolar atoms are C and H bonded to C.

The algorithm distributes test points uniformly over each atom's
expanded sphere (van der Waals radius + probe radius) using a
Fibonacci lattice.  A point is "exposed" if it does not fall inside
the expanded sphere of any other atom.  The exposed fraction times the
sphere area gives that atom's contribution.

Outputs written by `run_psa3d`:
- <prefix>_3dpsa.csv   (one row per conformer)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# --- Numba JIT Compilation for C-level speeds ---
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

from mol2_io import Mol2Block

# ========================================================================
# Default parameters (exposed via CLI)
# ========================================================================

PSA3D_PROBE_SASA: float = 1.4      # solvent probe radius (Å)
PSA3D_PROBE_MOL:  float = 0.0      # molecular surface (no probe)
PSA3D_N_SPHERE_POINTS: int = 960    # Fibonacci lattice density


# ========================================================================
# Bondi van der Waals radii (Å)
# ========================================================================

_VDW_RADII: Dict[str, float] = {
    "H":  1.20, "He": 1.40, "Li": 1.82, "B":  1.92,
    "C":  1.70, "N":  1.55, "O":  1.52, "F":  1.47,
    "Ne": 1.54, "Na": 2.27, "Mg": 1.73, "Al": 1.84,
    "Si": 2.10, "P":  1.80, "S":  1.80, "Cl": 1.75,
    "Ar": 1.88, "K":  2.75, "Ca": 2.31, "Fe": 2.04,
    "Zn": 2.01, "Se": 1.90, "Br": 1.85, "I":  1.98,
}

_DEFAULT_VDW: float = 1.70  # fallback (carbon-like)


def _vdw_radius(element: str) -> float:
    """Return Bondi van der Waals radius for *element*."""
    return _VDW_RADII.get(element, _VDW_RADII.get(element.capitalize(), _DEFAULT_VDW))


# ========================================================================
# Fibonacci lattice — uniform sphere points
# ========================================================================

def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate *n* approximately uniformly distributed points on a unit sphere."""
    indices = np.arange(n, dtype=float)
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / n)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * indices      # golden angle
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


# ========================================================================
# MOL2 parser (atoms + bonds — needed to identify polar H)
# ========================================================================

def _parse_atoms_and_bonds(block_text: str) -> List[dict]:
    atoms_by_id: Dict[int, dict] = {}
    section: Optional[str] = None

    for raw in block_text.splitlines():
        line = raw.strip()
        if not line: continue
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
            if len(parts) < 6: continue
            atom_id = int(parts[0])
            name = parts[1]
            coords = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float)
            sybyl_type = parts[5]
            element = sybyl_type.split(".")[0]
            atoms_by_id[atom_id] = {
                "id": atom_id, "name": name, "coords": coords,
                "element": element, "vdw_radius": _vdw_radius(element), "bonds": []
            }
        elif section == "BOND":
            parts = line.split()
            if len(parts) < 3: continue
            a1, a2 = int(parts[1]), int(parts[2])
            if a1 in atoms_by_id and a2 in atoms_by_id:
                atoms_by_id[a1]["bonds"].append(a2)
                atoms_by_id[a2]["bonds"].append(a1)

    return list(atoms_by_id.values())


# ========================================================================
# Polar atom identification — VegaZZ convention
#
# Polar  : N, O, S, P  +  H not bonded to C
# Apolar : C           +  H bonded to C
# ========================================================================

_POLAR_HEAVY: Set[str] = {"N", "O", "S", "P"}

def _identify_polar_indices(atoms: List[dict]) -> Set[int]:
    """Return the set of atom indices considered polar under the VegaZZ convention.

    Polar atoms:
      - Heavy atoms whose element is N, O, S, or P.
      - Hydrogen atoms that are NOT bonded to a carbon atom.
        (Equivalently, any H bonded exclusively to N, O, S, P, or other
        heteroatoms is polar; H bonded to C is apolar.)
    """
    id_to_idx = {a["id"]: i for i, a in enumerate(atoms)}
    polar: Set[int] = set()

    for i, a in enumerate(atoms):
        if a["element"] in _POLAR_HEAVY:
            polar.add(i)
        elif a["element"] == "H":
            # H is polar unless at least one of its bonds leads to a carbon
            bonded_to_carbon = any(
                atoms[id_to_idx[nb_id]]["element"] == "C"
                for nb_id in a["bonds"]
                if nb_id in id_to_idx
            )
            if not bonded_to_carbon:
                polar.add(i)

    return polar


# ========================================================================
# Shrake–Rupley surface area (JIT Compiled Core)
# ========================================================================

if _HAS_NUMBA:
    @njit(fastmath=True)
    def _shrake_rupley_numba(coords, expanded, sphere_pts):
        n_atoms = len(coords)
        n_pts = len(sphere_pts)
        sasa = np.zeros(n_atoms)
        
        # Pre-allocate memory for fast lookup
        neighbor_indices = np.zeros(n_atoms, dtype=np.int32)
        neighbor_radii_sq = np.zeros(n_atoms, dtype=np.float64)
        
        for i in range(n_atoms):
            r_i = expanded[i]
            n_neighbors = 0
            
            for j in range(n_atoms):
                if i == j: 
                    continue
                r_j = expanded[j]
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                dz = coords[i, 2] - coords[j, 2]
                dist_sq = dx*dx + dy*dy + dz*dz
                
                if dist_sq < (r_i + r_j)**2:
                    neighbor_indices[n_neighbors] = j
                    neighbor_radii_sq[n_neighbors] = r_j * r_j
                    n_neighbors += 1
                    
            if n_neighbors == 0:
                sasa[i] = 4.0 * np.pi * r_i * r_i
                continue
                
            exposed_count = 0
            for p in range(n_pts):
                px = coords[i, 0] + r_i * sphere_pts[p, 0]
                py = coords[i, 1] + r_i * sphere_pts[p, 1]
                pz = coords[i, 2] + r_i * sphere_pts[p, 2]
                
                is_buried = False
                for n_idx in range(n_neighbors):
                    j = neighbor_indices[n_idx]
                    r_j_sq = neighbor_radii_sq[n_idx]
                    dx = px - coords[j, 0]
                    dy = py - coords[j, 1]
                    dz = pz - coords[j, 2]
                    
                    if dx*dx + dy*dy + dz*dz < r_j_sq:
                        is_buried = True
                        break
                        
                if not is_buried:
                    exposed_count += 1
                    
            sasa[i] = 4.0 * np.pi * r_i * r_i * (exposed_count / n_pts)
            
        return sasa

def _shrake_rupley(
    coords: np.ndarray,
    radii: np.ndarray,
    probe_radius: float,
    n_sphere_points: int = PSA3D_N_SPHERE_POINTS,
) -> np.ndarray:
    
    n_atoms = len(coords)
    if n_atoms == 0:
        return np.zeros(0)

    sphere_pts = _fibonacci_sphere(n_sphere_points)
    expanded = radii + probe_radius

    # Use compiled C code if Numba is installed
    if _HAS_NUMBA:
        return _shrake_rupley_numba(coords, expanded, sphere_pts)

    # Slow Python/Numpy fallback
    sasa = np.zeros(n_atoms)
    if n_atoms > 1:
        diff = coords[:, None, :] - coords[None, :, :]
        dist_sq = (diff ** 2).sum(axis=2)
    else:
        dist_sq = np.zeros((1, 1))

    for i in range(n_atoms):
        r_i = expanded[i]
        if n_atoms > 1:
            cutoff_sq = (r_i + expanded) ** 2
            mask = (dist_sq[i] < cutoff_sq)
            mask[i] = False
            nb_idx = np.where(mask)[0]
        else:
            nb_idx = np.array([], dtype=int)

        if len(nb_idx) == 0:
            sasa[i] = 4.0 * np.pi * r_i * r_i
            continue

        test_pts = coords[i] + r_i * sphere_pts
        nb_coords = coords[nb_idx]
        nb_radii  = expanded[nb_idx]

        dp = test_pts[:, None, :] - nb_coords[None, :, :]
        d2 = (dp ** 2).sum(axis=2)
        buried = (d2 < nb_radii[None, :] ** 2).any(axis=1)

        exposed_frac = float((~buried).sum()) / n_sphere_points
        sasa[i] = 4.0 * np.pi * r_i * r_i * exposed_frac

    return sasa


# ========================================================================
# High-level API calls
# ========================================================================

def compute_3dpsa(
    atoms: List[dict],
    probe_radius: float,
    n_sphere_points: int = PSA3D_N_SPHERE_POINTS,
) -> Tuple[float, float, int, int]:
    
    if not atoms:
        return 0.0, 0.0, 0, 0

    coords = np.array([a["coords"] for a in atoms])
    radii  = np.array([a["vdw_radius"] for a in atoms])
    sasa = _shrake_rupley(coords, radii, probe_radius, n_sphere_points)
    
    polar_idx = _identify_polar_indices(atoms)
    psa = float(sasa[list(polar_idx)].sum()) if polar_idx else 0.0

    return float(sasa.sum()), psa, len(atoms), len(polar_idx)


def compute_per_atom_exposure(
    atoms: List[dict],
    probe_radius: float = PSA3D_PROBE_SASA,
    n_sphere_points: int = 480,
) -> Tuple[List[float], List[bool]]:
    
    if not atoms:
        return [], []

    coords = np.array([a["coords"] for a in atoms])
    radii  = np.array([a["vdw_radius"] for a in atoms])

    sasa = _shrake_rupley(coords, radii, probe_radius, n_sphere_points)
    max_area = 4.0 * np.pi * (radii + probe_radius) ** 2
    exposure = np.where(max_area > 0, sasa / max_area, 0.0)

    polar_set = _identify_polar_indices(atoms)
    is_polar = [i in polar_set for i in range(len(atoms))]

    return [round(float(e), 3) for e in exposure], is_polar


def _write_3dpsa_csv(rows: List[dict], out_path: Path) -> None:
    fieldnames = [
        "molecule_conformer", "molecule_name", "psa3d_sasa", "psa3d_molsurf",
        "sasa_total", "molsurf_total", "n_atoms_total", "n_polar_atoms",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_psa3d(
    blocks: List[Mol2Block],
    out_dir: Path,
    out_prefix: str,
    probe_sasa: float = PSA3D_PROBE_SASA,
    probe_mol: float = PSA3D_PROBE_MOL,
    n_sphere_points: int = PSA3D_N_SPHERE_POINTS,
) -> Path:
    
    rows: List[dict] = []

    for blk in blocks:
        atoms = _parse_atoms_and_bonds(blk.text)
        if not atoms:
            rows.append({
                "molecule_conformer": blk.index, "molecule_name": blk.name,
                "psa3d_sasa": "", "psa3d_molsurf": "", "sasa_total": "",
                "molsurf_total": "", "n_atoms_total": 0, "n_polar_atoms": 0,
            })
            continue

        sasa_tot, psa_sasa, n_all, n_polar = compute_3dpsa(atoms, probe_sasa, n_sphere_points)
        mol_tot, psa_mol, _, _ = compute_3dpsa(atoms, probe_mol, n_sphere_points)

        rows.append({
            "molecule_conformer": blk.index,
            "molecule_name": blk.name,
            "psa3d_sasa": round(psa_sasa, 2),
            "psa3d_molsurf": round(psa_mol, 2),
            "sasa_total": round(sasa_tot, 2),
            "molsurf_total": round(mol_tot, 2),
            "n_atoms_total": n_all,
            "n_polar_atoms": n_polar,
        })

    out_path = out_dir / f"{out_prefix}_3dpsa.csv"
    _write_3dpsa_csv(rows, out_path)
    return out_path