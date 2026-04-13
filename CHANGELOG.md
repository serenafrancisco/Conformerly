# Conformerly - Automated conformational analysis of bRo5 molecules

### 2024-04-13 - Version 1.4.0

#### Fix & Refactor: pi.py — individual aromatic rings, bond-separation semantics, sp2-bridge exclusion
#### Fix: imhb.py — ring-path filter for geometrically forced H-bond contacts

---

**pi.py** (v1.9 → v2.1)

* **Bond-separation default raised from 1 to 2**: the original default only excluded directly bonded ring pairs (distance 1), leaving one-atom-bridged rings (Ph–X–Ph, distance 2) always detected. The default has been raised to 2 so that pairs at distances 0, 1, and 2 are all excluded by default; only rings with 3+ bonds of separation reach the geometric classifier.
* **Bond-separation operator `<=`**: `exclude_pair_by_bonds` uses `<= min_bond_sep` — pairs at or within the threshold are excluded. `min_bond_sep` means the maximum separation still considered too close to evaluate.
* **New sp2-bridge filter**: for pairs one step beyond the threshold (`min_dist == min_bond_sep + 1`), the bridging atom is inspected via RDKit. If it is sp2- or sp-hybridised (imine, vinyl, enamine, aromatic linker), the pair is excluded even though it passed the primary filter. Such bridges create extended conjugated π-systems that lock rings toward coplanarity, making any apparent stacking a structural artefact.
* **Removed ring-system / patch hierarchy**: `Moiety`, `RingPatch`, `detect_aromatic_ring_systems`, `moiety_annotation_rows`, `pi_all_for_system_pair`, `connected_components` removed. Replaced by flat individual `AromaticRing` objects with `detect_aromatic_rings`, `ring_annotation_rows`, `classify_ring_pair`, `_bridge_is_sp2`.
* **Output renamed**: `_ar_ring_systems.csv` → `_ar_rings.csv`; `moiety_a`/`moiety_b` columns replaced by `ring_a`/`ring_b` in `_pi_stacking.csv`.

**imhb.py**

* **New ring-path filter**: contacts whose shortest covalent D→A path traverses any ring bond are now excluded. When the D→A path crosses a ring bond, the donor–acceptor proximity is geometrically locked by the rigid ring framework (e.g. substituents on adjacent atoms of an aromatic ring, or on atoms belonging to a fused bicyclic system). Such contacts are structural artefacts, not conformationally driven IMHBs.
* Two new helper functions added: `_bond_in_ring(atoms, u, v)` (BFS-based cycle detection) and `_path_contains_ring_bond(atoms, path)`.
* Filter is applied inside `find_imhbs` after all geometric checks pass, using the `cov_path` already computed for pseudoring geometry — no additional BFS calls.
* `find_imhbs` docstring updated to list filter 4.

**analysis.py**
* Default `pi_min_bond_sep` updated from 1 to 2; tooltip updated.
* ZIP download list: `ar_ring_systems.csv` → `ar_rings.csv`.

**main.py**
* `--min-bond-sep` default updated from 1 to 2; help text updated.

**results_manager.py**
* File-suffix recognition: `ar_ring_systems` → `ar_rings`.

**viewer2D.py**
* Import and `detect_ring_systems_for_viz` updated for individual rings; JS label formatter fixed.

**viewer3D.py**
* Import, pi scene block, and JS table updated for individual rings; `min_bond_sep` corrected to 2.

**about.py**
* IMHB feature card updated to describe the ring-path filter.
* π–π feature card updated to describe individual-ring detection and dual exclusion logic.

---

### 2024-03-26 - Version 1.3.0

#### NEW: Output now includes frequency matrices for IMHBs and π-π stackings.

* Updated analysis.py
* Updated viewer3D.py
* Updated viewer2D.py

### 2024-03-25 - Version 1.2.0

* Updated analysis.py
* Updated viewer3D.py
* Updated viewer2D.py

### 2024-03-24 - Version 1.1.0

#### Fix: removed forced analysis of IMHB and stackings when not requested.
* Updated analysis.py
* Updated viewer3D.py
* Updated viewer2D.py

### 2024-03-23 - Version 1.0.0

#### First version available on GitHub and Streamlit.
