# Conformerly - Automated conformational analysis of bRo5 molecules

### 2024-04-15 - Version 1.5.2

#### Fix: imhb.py — ring-path filter now uses proximity check to avoid false exclusions in PROTACs and multi-ring molecules

**Root cause**: `_path_crosses_small_ring_bond()` (v1.5.0–v1.5.1) returned True as soon as
the D→A path crossed any bond belonging to a small ring (≤ 8 atoms), regardless of where
the acceptor was relative to that ring. This incorrectly excluded IMHBs where the donor
is *in* or *attached to* a small aromatic ring but the acceptor is in a remote part of the
molecule.

**Concrete failure case**: in a PROTAC, N53 (N.pl3 with H, in a 5-membered pyrazole ring)
forms an IMHB with O9 (carbonyl, 23 bonds away through the linker). The first bond on the
D→A path (N53–C42) belongs to the pyrazole ring (size 5 ≤ 8), so the old filter excluded
the contact. But O9 is 22 bonds from C42 — entirely in a different structural unit. The
ring does not force the D···A proximity.

**Fix**: a proximity check is now applied to every small ring bond found on the path. For
the ring bond at path index `i`:

    min_D = i                  (bonds from D to the near ring atom)
    min_A = len(path) − 2 − i (bonds from A to the far ring atom)

A ring bond triggers exclusion only if **both** min_D ≤ 2 **and** min_A ≤ 2. This limits
exclusion to cases where D and A are both within 2 bonds of adjacent ring atoms — the
canonical geometrically-forced situation (ortho-disubstituted aromatic, fused bicyclic).

**All previous test cases still pass**:
* BI-3663 N47–H···O55 (crosses 6-membered + 5-membered ring, min_D=1/2, min_A=2/1): still EXCLUDED ✓
* Macrocycle trans-annular IMHBs (ring bond size > 8): still KEPT ✓
* Open-chain IMHBs (no ring bonds): still KEPT ✓
* PROTAC N53–H···O9 (ring bond at path start, acceptor 22 bonds away, min_A=22): now correctly KEPT ✓

**Files changed**: `imhb.py` only.

---

### 2024-04-15 - Version 1.5.1

#### Fix: imhb.py — tertiary aliphatic nitrogen atoms (sp3, no H, neutral) now correctly detected as H-bond acceptors

RDKit's `BaseFeatures.fdef` NAcceptor SMARTS (`[N&v3;H0;$(Nc)]`) requires bonding to
aromatic carbon. Tertiary amines bonded only to aliphatic carbons were silently excluded.
Fix: supplementary scan in `identify_donors_acceptors_rdkit()` adds sp3 N with no H and
formal charge 0.

---

### 2024-04-15 - Version 1.5.0

#### Macrocycle support: ring-path filter made size-aware; about.py and analysis.py updated

`_path_contains_ring_bond()` replaced by `_path_crosses_small_ring_bond()`. Bonds in rings
of > 8 atoms (macrocycle backbones) no longer trigger exclusion. New helper
`_smallest_ring_size()`. Threshold `_MAX_RIGID_RING_SIZE = 8`.

---

### 2024-04-13 - Version 1.4.0

pi.py (v2.1): bond-separation default 1→2, operator <=, sp2-bridge filter, flat AromaticRing
architecture, _ar_ring_systems.csv → _ar_rings.csv. imhb.py: initial ring-path filter
(superseded by v1.5.x). Companion files updated.

---

### 2024-03-26 - Version 1.3.0
NEW: frequency matrices for IMHBs and π–π stackings.

### 2024-03-25 - Version 1.2.0 / 2024-03-24 - Version 1.1.0 / 2024-03-23 - Version 1.0.0
See repository history.
