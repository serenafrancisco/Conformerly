#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pages/about.py — About Conformerly
Landing / introduction page. Fill in the sections marked with TODO.
"""

import streamlit as st

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from components.footer import show_footer

# ── Cover banner ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
.cover-banner {
    width: 100%;
    height: 190px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1rem;
    overflow: hidden;
    background: var(--background-color, #0a1628);
    background-image:
        radial-gradient(ellipse 55% 60% at 15% 55%,
            rgba(96,165,250,0.08) 0%, transparent 70%),
        radial-gradient(ellipse 40% 50% at 82% 38%,
            rgba(167,139,250,0.07) 0%, transparent 70%);
    box-shadow: 0 6px 30px rgba(0,0,0,0.18);
}
.cover-wordmark {
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    opacity: 0.07;
    user-select: none;
}
</style>
<div class="cover-banner">
  <span class="cover-wordmark">CONFORMERLY</span>
</div>
""", unsafe_allow_html=True)

# TODO: replace the banner above with:
#   st.image("cover.png", use_container_width=True)
# once you have a cover image.

st.title("About Conformerly")
st.caption("Conformational ensemble analysis · Rgyr · 3D-PSA · IMHB · π–π stacking")
st.markdown("<br><br>", unsafe_allow_html=True)



# ── What is Conformerly? ──────────────────────────────────────────────────────

st.header("What is Conformerly?")

# TODO: replace the placeholder text below with your own description.
st.markdown("""
*Placeholder — describe Conformerly here.*

Conformerly is an interactive analysis suite for conformational ensembles
of drug-like molecules ...
""")

# ── Method overview ───────────────────────────────────────────────────────────

st.header("Methods")

# TODO: add method descriptions, equations, or diagrams.
st.markdown("""
*Placeholder — describe the computational methods here.*

| Module | Description |
|--------|-------------|
| **Rgyr** | Radius of gyration |
| **3D-PSA** | Three-dimensional polar surface area (Shrake–Rupley) |
| **IMHB** | Intramolecular hydrogen bonds |
| **π–π** | Aromatic π–π stacking interactions |
""")

# ── How to use ────────────────────────────────────────────────────────────────

st.header("How to Use")

# TODO: expand with screenshots or a step-by-step guide.
st.markdown("""
1. Go to the **Analysis** page from the sidebar.
2. Upload one or more multi-conformer **MOL2** files named
   `<molecule_name>_<solvent>.mol2` (*e.g.* `ARV110_H2O.mol2`).
3. Configure analysis parameters in the sidebar.
4. Click **:material/rocket_launch: Run Analysis** and explore the results.
""")

# ── Call to action ────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Start Your Analysis", type="primary", use_container_width=False):
    st.switch_page("pages/analysis.py")

st.markdown("<br><br>", unsafe_allow_html=True)

# ── Authors / citation ────────────────────────────────────────────────────────

st.header("Authors")

# TODO: add author list and citation.
st.markdown("""
**Conformerly** was developed by the MedChemBeyond Lab at the Dept. of Molecular Biotechnology and Health Sciences 
(University of Turin, Italy).
""")


show_footer(image="figures/footer.png", caption="")
