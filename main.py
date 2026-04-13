#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chameleons v1.1 — Conformational and Hydrogen-bond Analysis via Modular
Evaluation of Local Environments, Orbital geometry, and Noncovalent Stacking.

CLI mode:
  python3 main.py <molecule_name>
  python3 main.py <molecule_name> --only-hbonds
  python3 main.py --batch

Help (shows all options):
  python3 main.py -h

Print the output manual:
  python3 main.py --print-manual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from manual import print_manual
from mol2_io import split_mol2_blocks
from imhb import (
    IMHB_MAX_DIST_HA,
    IMHB_MIN_ANGLE_DHA,
    IMHB_MIN_ANGLE_ACCEPTOR,
    IMHB_MIN_ANGLE_STRICT,
    IMHB_MIN_BOND_SEPARATION,
    run_imhb,
)
from pi import run_pi
from rgyr import run_rgyr
from psa3d import (
    PSA3D_N_SPHERE_POINTS,
    run_psa3d,
)
from results_manager import generate_tsv_summary


# =====================================================================
# ARGUMENT PARSER
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Chameleons v1.1 — Run IMHB, aromatic π–π stacking, radius of gyration, "
                    "and 3D polar surface area analysis from the ./input directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--print-manual", action="store_true",
                   help="Print output interpretation manual and exit.")
    p.add_argument("--view-2d", action="store_true",
                   help="Generate an interactive 2D HTML topology viewer (requires RDKit).")
    p.add_argument("--view-3d", action="store_true",
                   help="Generate an interactive 3D HTML ensemble viewer with a conformer slider.")

    p.add_argument("molecule_name", nargs="?", default=None,
                   help="Prefix of the molecule to analyze (e.g., ARV-110). "
                        "Script will process all matching files in ./input/.")
    p.add_argument("--batch", action="store_true",
                   help="Run analysis on all molecules found in the ./input/ directory.")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-hbonds", action="store_true",
                   help="Run ONLY intramolecular H-bond (IMHB) analysis.")
    g.add_argument("--only-pi", action="store_true",
                   help="Run ONLY aromatic π–π stacking analysis.")
    g.add_argument("--only-rgyr", action="store_true",
                   help="Run ONLY radius of gyration analysis.")
    g.add_argument("--only-psa3d", action="store_true",
                   help="Run ONLY 3D polar surface area analysis.")

    # -----------------------------------------------------------------
    # IMHB parameters
    # -----------------------------------------------------------------
    p.add_argument("--hb-max-dist", type=float, default=IMHB_MAX_DIST_HA,
                   help="Maximum H···A distance (Å) for H-bonds.")
    p.add_argument("--hb-min-angle", type=float, default=IMHB_MIN_ANGLE_DHA,
                   help="Minimum D–H···A donor angle (°) for H-bonds.")
    p.add_argument("--hb-min-angle-acceptor", type=float, default=IMHB_MIN_ANGLE_ACCEPTOR,
                   help="Minimum X–A···H acceptor angle (°).")
    p.add_argument("--hb-min-angle-strict", type=float, default=IMHB_MIN_ANGLE_STRICT,
                   help="Stricter D–H···A angle (°) applied when D and A are close.")
    p.add_argument("--hb-min-bond-sep", type=int, default=IMHB_MIN_BOND_SEPARATION,
                   help="Bond-separation threshold: pairs below this use --hb-min-angle-strict.")

    # -----------------------------------------------------------------
    # π–π stacking parameters
    # -----------------------------------------------------------------
    p.add_argument("--use-pi-criteria", dest="use_pi_criteria", action="store_true", default=True,
                   help="Enable full geometric π–π criteria (default: ON).")
    p.add_argument("--no-pi-criteria", dest="use_pi_criteria", action="store_false",
                   help="Disable geometric π–π criteria; use centroid distance threshold only.")
    p.add_argument("--pi-ff-max-dist", type=float, default=4.4,
                   help="Max centroid distance (Å) for face-to-face π–π.")
    p.add_argument("--pi-ef-max-dist", type=float, default=5.5,
                   help="Max centroid distance (Å) for edge-to-face π–π.")
    p.add_argument("--pi-parallel-angle", type=float, default=30.0,
                   help="Max inter-plane angle (°) for face-to-face stacking.")
    p.add_argument("--pi-t-angle-min", type=float, default=60.0,
                   help="Min inter-plane angle (°) for edge-to-face stacking.")
    p.add_argument("--min-bond-sep", type=int, default=2,
                   help="Exclude ring pairs whose bond separation is at or below this value. "
                        "Default 2: excludes fused (dist 0), directly bonded (dist 1), and "
                        "one-atom-bridged rings Ph-X-Ph (dist 2). Only 3+ bond separated "
                        "rings are evaluated geometrically.")

    # -----------------------------------------------------------------
    # 3D PSA parameters
    # -----------------------------------------------------------------
    p.add_argument("--psa3d-n-points", type=int, default=PSA3D_N_SPHERE_POINTS,
                   help="Number of Fibonacci lattice points per atom sphere for 3D PSA.")

    p.add_argument("--quiet", action="store_true", help="Suppress terminal output.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip automatic plot generation after the TSV is written.")
    return p


# =====================================================================
# SINGLE-FILE PROCESSING
# =====================================================================

def _process_single_file(
    in_path: Path,
    out_dir: Path,
    out_prefix: str,
    args: argparse.Namespace,
    run_hbonds: bool,
    run_hydro: bool,
    run_gyration: bool,
    run_psa: bool,
    quiet: bool = False,
) -> None:
    """Process one MOL2 file and write all outputs to the molecule's specific directory."""

    blocks = split_mol2_blocks(in_path)
    if not blocks:
        if not quiet:
            print(f"  WARNING: no molecule blocks found in {in_path.name}", file=sys.stderr)
        return

    # Collect output file paths for the 3D viewer download panel
    generated_files: list[Path] = []

    # --- H-bond analysis ---
    if run_hbonds:
        try:
            detail, summary, hbond_ids = run_imhb(
                blocks=blocks,
                out_dir=out_dir,
                out_prefix=out_prefix,
                hb_max_dist=float(args.hb_max_dist),
                hb_min_angle=float(args.hb_min_angle),
                hb_min_angle_strict=float(args.hb_min_angle_strict),
                hb_min_bond_sep=int(args.hb_min_bond_sep),
                hb_min_angle_acceptor=float(args.hb_min_angle_acceptor),
            )
            generated_files.extend([detail, summary, hbond_ids])
            if not quiet:
                print(f"  [hbonds] wrote: {detail.name}")
                print(f"  [hbonds] wrote: {summary.name}")
                print(f"  [hbonds] wrote: {hbond_ids.name}")
        except Exception as ex:
            if not quiet:
                print(f"  WARNING: H-bond analysis failed for {in_path.name}: {ex}", file=sys.stderr)

    # --- Aromatic π–π analysis ---
    if run_hydro:
        try:
            outs = run_pi(
                blocks=blocks,
                out_dir=out_dir,
                out_prefix=out_prefix,
                min_bond_sep=int(args.min_bond_sep),
                use_pi_criteria=bool(args.use_pi_criteria),
                pi_ff_max_dist=float(args.pi_ff_max_dist),
                pi_ef_max_dist=float(args.pi_ef_max_dist),
                pi_parallel_angle=float(args.pi_parallel_angle),
                pi_t_angle_min=float(args.pi_t_angle_min),
            )
            generated_files.extend(outs)
            if not quiet:
                for pth in outs:
                    print(f"  [pi] wrote: {pth.name}")
        except Exception as ex:
            if not quiet:
                print(f"  WARNING: π–π analysis failed for {in_path.name}: {ex}", file=sys.stderr)

    # --- Radius of gyration ---
    if run_gyration:
        try:
            rgyr_path = run_rgyr(
                blocks=blocks,
                out_dir=out_dir,
                out_prefix=out_prefix,
            )
            generated_files.append(rgyr_path)
            if not quiet:
                print(f"  [rgyr] wrote: {rgyr_path.name}")
        except Exception as ex:
            if not quiet:
                print(f"  WARNING: Rgyr analysis failed for {in_path.name}: {ex}", file=sys.stderr)

    # --- 3D Polar Surface Area ---
    if run_psa:
        try:
            psa3d_path = run_psa3d(
                blocks=blocks,
                out_dir=out_dir,
                out_prefix=out_prefix,
                n_sphere_points=int(args.psa3d_n_points),
            )
            generated_files.append(psa3d_path)
            if not quiet:
                print(f"  [psa3d] wrote: {psa3d_path.name}")
        except Exception as ex:
            if not quiet:
                print(f"  WARNING: 3D PSA analysis failed for {in_path.name}: {ex}", file=sys.stderr)

    # --- 3D Structure viewer ---
    if getattr(args, "view_3d", False):
        try:
            from viewer3D import build_ensemble_scene, build_stats as build_stats_3d
            from viewer3D import generate_html as gen_html_3d
            from viewer3D import compute_rdkit_properties as rdkit_props_3d
            from viewer3D import enumerate_donors_acceptors as enum_da_3d
            from imhb import parse_atoms_and_bonds as _parse
            scene = build_ensemble_scene(blocks)
            if scene and blocks:
                atoms = _parse(blocks[0].text)
                stats = build_stats_3d(atoms, len(blocks))
                rdkit_props = rdkit_props_3d(blocks[0].text)
                da = enum_da_3d(atoms, blocks[0].text)
                # Collect generated CSV/output file contents for download panel
                output_files_data = []
                for fp in generated_files:
                    try:
                        content = fp.read_text(encoding="utf-8")
                        output_files_data.append({"name": fp.name, "content": content})
                    except Exception:
                        pass
                html = gen_html_3d(scene, stats, rdkit_props, da, out_prefix,
                                   output_files=output_files_data)
                html_path = out_dir / f"{out_prefix}_3D_view.html"
                html_path.write_text(html, encoding="utf-8")
                if not quiet:
                    print(f"  [viewer3D] wrote: {html_path.name}")
        except Exception as ex:
            if not quiet:
                print(f"  WARNING: 3D Viewer generation failed for {in_path.name}: {ex}", file=sys.stderr)


# =====================================================================
# MAIN RUN LOGIC
# =====================================================================

def _run(args: argparse.Namespace) -> int:
    if args.print_manual:
        print_manual()
        return 0

    if not args.molecule_name and not args.batch:
        print("ERROR: You must provide a molecule name or use the --batch flag.", file=sys.stderr)
        return 2

    input_dir = Path("./input").resolve()
    results_dir = Path("./results").resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: Input directory not found at {input_dir}. Please create it and add MOL2 files.", file=sys.stderr)
        return 2

    results_dir.mkdir(parents=True, exist_ok=True)

    only_h = bool(args.only_hbonds)
    only_p = bool(args.only_pi)
    only_r = bool(args.only_rgyr)
    only_s = bool(args.only_psa3d)

    run_hbonds   = not (only_p or only_r or only_s)
    run_hydro    = not (only_h or only_r or only_s)
    run_gyration = not (only_h or only_p or only_s)
    run_psa      = not (only_h or only_p or only_r)

    # Determine which files to process
    target_files = []
    if args.batch:
        target_files = sorted(input_dir.glob("*.mol2"))
    else:
        target_files = sorted(input_dir.glob(f"{args.molecule_name}_*.mol2"))

    if not target_files:
        print(f"ERROR: No matching MOL2 files found in {input_dir}", file=sys.stderr)
        return 3

    # Group files by molecule name prefix (Format: <molecule>_<solvent>.mol2)
    molecules = {}
    for f in target_files:
        stem = f.stem
        if "_" not in stem:
            if not args.quiet:
                print(f"Skipping {f.name}: Does not match <molecule>_<solvent>.mol2 format.")
            continue
            
        # Split by last underscore to separate molecule name and solvent
        parts = stem.rsplit("_", 1)
        mol_name = parts[0]
        solvent = parts[1]
        
        if mol_name not in molecules:
            molecules[mol_name] = []
        molecules[mol_name].append((f, solvent))

    if not args.quiet:
        print(f"[input] directory: {input_dir}")
        print(f"[found] {len(target_files)} valid MOL2 files across {len(molecules)} molecule(s)")
        print(f"[out]   {results_dir}")
        print()

    # Process each molecule
    for mol_name, files in molecules.items():
        if not args.quiet:
            print(f"--- Processing Molecule: {mol_name} ---")
            
        mol_out_dir = results_dir / mol_name
        mol_out_dir.mkdir(parents=True, exist_ok=True)
        
        for mol2_file, solvent in files:
            if not args.quiet:
                print(f"  Solvent: {solvent}")
                
            out_prefix = f"{mol_name}_{solvent}"
            _process_single_file(
                in_path=mol2_file,
                out_dir=mol_out_dir,
                out_prefix=out_prefix,
                args=args,
                run_hbonds=run_hbonds,
                run_hydro=run_hydro,
                run_gyration=run_gyration,
                run_psa=run_psa,
                quiet=args.quiet,
            )
            
        # --- 2D Structure viewer (once per molecule, topology is solvent-independent) ---
        if getattr(args, "view_2d", False) and files:
            first_mol2 = files[0][0]   # Path of the first MOL2 file for this molecule
            try:
                from viewer2D import generate_2d_svg, build_stats as build_stats_2d, generate_html as gen_html_2d
                from viewer2D import compute_rdkit_properties as rdkit_props_2d, enumerate_donors_acceptors as enum_da_2d
                from viewer2D import detect_ring_systems_for_viz, _HAS_RDKIT
                if not _HAS_RDKIT:
                    print(f"  WARNING: RDKit required for --view-2d. Skipping 2D viewer.", file=sys.stderr)
                else:
                    from rdkit import Chem as _Chem
                    from imhb import parse_atoms_and_bonds as _parse
                    from mol2_io import split_mol2_blocks as _split
                    _blocks = _split(first_mol2)
                    if _blocks:
                        blk = _blocks[0]
                        atoms = _parse(blk.text)
                        if atoms:
                            stats = build_stats_2d(atoms, blk, len(_blocks))
                            mol   = _Chem.MolFromMol2Block(blk.text, sanitize=True, removeHs=False)
                            rdkit_props  = rdkit_props_2d(mol) if mol else {}
                            da           = enum_da_2d(atoms, blk.text)
                            ring_systems = detect_ring_systems_for_viz(blk.text)
                            rs_info = [{"label": rs["label"], "n_atoms": rs["n_atoms"],
                                        "n_patches": rs["n_patches"], "color": rs["color"],
                                        "atom_names": rs.get("atom_names", []),
                                        "patches": rs.get("patches", [])}
                                       for rs in ring_systems]
                            svg       = generate_2d_svg(blk.text,
                                            {d["id"] for d in da["donors"]},
                                            {a["id"] for a in da["acceptors"]},
                                            ring_systems)
                            html      = gen_html_2d(svg, stats, rdkit_props, da, rs_info, mol_name)
                            html_path = mol_out_dir / f"{mol_name}_2D_view.html"
                            html_path.write_text(html, encoding="utf-8")
                            if not args.quiet:
                                print(f"  [viewer2D] wrote: {html_path.name}")
            except Exception as ex:
                if not args.quiet:
                    print(f"  WARNING: 2D Viewer generation failed for {mol_name}: {ex}", file=sys.stderr)

        # Generate the unified TSV for this molecule
        tsv_path = None
        try:
            tsv_path = generate_tsv_summary(mol_name, mol_out_dir, results_dir)
            if not args.quiet:
                print(f"  [summary] generated: {tsv_path.name}")
        except Exception as ex:
            print(f"  WARNING: TSV summary generation failed for {mol_name}: {ex}", file=sys.stderr)

        # Generate plots from the TSV (skipped with --no-plots or if TSV failed)
        if tsv_path is not None and not getattr(args, "no_plots", False):
            try:
                from plots import run_plots
                run_plots(tsv_path=tsv_path, out_dir=mol_out_dir, quiet=args.quiet)
            except Exception as ex:
                if not args.quiet:
                    print(f"  WARNING: plot generation failed for {mol_name}: {ex}", file=sys.stderr)

        if not args.quiet:
            print()

    return 0


# =====================================================================
# ENTRY POINT
# =====================================================================

def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))