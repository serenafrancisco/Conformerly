"""mol2_io.py

Shared MOL2 utilities:
- split a single MOL2 file into molecule blocks (supports multi-MOL2)
- extract a block name

Kept separate so both analyzers can reuse it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


_MOL2_MOLECULE_SPLIT_RE = re.compile(r"(?=@<TRIPOS>MOLECULE)", re.MULTILINE)

@dataclass(frozen=True)
class Mol2Block:
    """A single molecule block from a MOL2 file."""

    index: int    # 1-based conformer index
    name: str     # molecule name from the MOL2 header
    text: str     # raw block text, including the MOLECULE header


def extract_mol2_name(block_text: str) -> str:
    """Return the MOL2 name (line after '@<TRIPOS>MOLECULE'), if present."""
    lines = block_text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("@<TRIPOS>MOLECULE"):
            return lines[i + 1].strip() if i + 1 < len(lines) else ""
    return ""


def split_mol2_blocks(path: Path) -> List[Mol2Block]:
    """Read MOL2 and split into blocks (single or multi-MOL2)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks_raw = [b for b in _MOL2_MOLECULE_SPLIT_RE.split(text) if b.strip()]

    blocks: List[Mol2Block] = []
    for i, blk in enumerate(blocks_raw, start=1):
        name = extract_mol2_name(blk) or f"mol_{i}"
        blocks.append(Mol2Block(index=i, name=name, text=blk))
    return blocks
