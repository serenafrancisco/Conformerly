"""results_manager.py

Chameleons v1.1 — Results Aggregator.
Compiles individual CSV outputs into a master TSV summary per molecule.
"""

import csv
from pathlib import Path
from collections import defaultdict
import os

def generate_tsv_summary(molecule_name: str, mol_results_dir: Path, out_dir: Path) -> Path:
    """Read CSVs for a given molecule and compile a summary TSV."""
    
    # Structure: data[solvent][conformer_id] = { ...metrics... }
    data = defaultdict(lambda: defaultdict(dict))
    solvents_found = set()
    
    # 1. Gather all files in the molecule's result directory
    for file_path in mol_results_dir.glob(f"{molecule_name}_*.csv"):
        # Expected format: <molecule>_<solvent>_<analysis>.csv
        filename = file_path.name
        prefix = filename.replace(".csv", "")
        
        # Extract solvent and analysis type
        parts = prefix.split("_")
        if len(parts) < 3:
            continue
            
        # We know molecule_name might have underscores, so strip it out first
        remainder = prefix[len(molecule_name)+1:] # +1 for the underscore
        
        # remainder is something like "H2O_hbonds_summary"
        # Let's find the analysis type by checking known suffixes
        analysis_types = [
            "hbonds_summary", "hbonds", "hbond_ids",
            "pi_summary", "pi_stacking", "ar_ring_systems",
            "rgyr", "3dpsa"
        ]
        
        analysis = None
        solvent = None
        for a_type in analysis_types:
            if remainder.endswith(a_type):
                analysis = a_type
                # Extract solvent by removing the analysis suffix
                solvent = remainder[:-(len(a_type)+1)] # +1 for the underscore
                break
                
        if not solvent or not analysis:
            continue
            
        solvents_found.add(solvent)
        
        # Parse the specific CSVs we need for the TSV
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                conf_id = int(row.get("molecule_conformer", 0))
                if conf_id == 0:
                    continue
                
                mol_name_in_file = row.get("molecule_name", molecule_name)
                data[solvent][conf_id]["Molecule"] = mol_name_in_file
                data[solvent][conf_id]["Solvent"] = solvent
                
                if analysis == "hbonds_summary":
                    data[solvent][conf_id]["IMHB_Tot"] = row.get("hbonds_total", "")
                    data[solvent][conf_id]["IMHB_SR"] = row.get("short_range_count", "")
                    data[solvent][conf_id]["IMHB_MR"] = row.get("medium_range_count", "")
                    data[solvent][conf_id]["IMHB_LR"] = row.get("long_range_count", "")
                
                elif analysis == "pi_summary":
                    data[solvent][conf_id]["Pi_Tot"] = row.get("pi_total_count", "")
                    data[solvent][conf_id]["Pi_FF"] = row.get("pi_face_to_face_count", "")
                    data[solvent][conf_id]["Pi_EF"] = row.get("pi_edge_to_face_count", "")
                    
                elif analysis == "rgyr":
                    data[solvent][conf_id]["Rgyr_AA"] = row.get("rgyr_all_atoms", "")
                    
                elif analysis == "3dpsa":
                    data[solvent][conf_id]["3D-PSA"] = row.get("psa3d_molsurf", "")
                    data[solvent][conf_id]["3D-PSA(SA)"] = row.get("psa3d_sasa", "")

    # 2. Write TSV
    tsv_path = out_dir / f"{molecule_name}.tsv"
    fieldnames = [
        "Conformer", "Molecule", "Solvent", "IMHB_Tot", "IMHB_SR", 
        "IMHB_MR", "IMHB_LR", "Pi_Tot", "Pi_FF", "Pi_EF", 
        "Rgyr_AA", "3D-PSA", "3D-PSA(SA)"
    ]
    
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        
        for solvent in sorted(solvents_found):
            for conf_id in sorted(data[solvent].keys()):
                row_data = data[solvent][conf_id]
                row_data["Conformer"] = conf_id
                
                # Fill missing fields with empty strings to ensure clean TSV
                for field in fieldnames:
                    if field not in row_data:
                        row_data[field] = ""
                        
                writer.writerow(row_data)
                
    return tsv_path