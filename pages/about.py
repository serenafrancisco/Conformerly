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

st.image("figures/logo-banner.png", use_container_width=True)

# ↓ Add this to kill the gap
st.markdown("""
    <style>
        [data-testid="stImage"] { margin-bottom: -12rem; }
        h1:first-of-type { margin-top: 0; }
    </style>
""", unsafe_allow_html=True)

st.title("Welcome to Conformerly!")
st.caption("Conformational ensemble analysis · Rgyr · 3D-PSA · IMHB · π–π stacking")
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br>", unsafe_allow_html=True)



# ── What is Conformerly? ──────────────────────────────────────────────────────

st.header(":material/help: What is Conformerly?")

st.markdown("""
**Beyond Rule-of-Five** (bRo5) compounds — including **macrocycles**, **PROTACs**, and other 
large flexible molecules — can exhibit *chameleonic* properties, adapting their shape and 
physicochemical profile to different environments. To capture these conformational changes 
quantitatively, we developed **Conformerly**, a web tool for fast and automated analysis of 
conformational ensembles of bRo5 compounds.

Conformerly is designed to handle the full spectrum of bRo5 chemical space: from rigid 
aromatic scaffolds to highly flexible macrocyclic backbones and long PROTAC linkers.
""")

st.markdown("<br>", unsafe_allow_html=True)

# ── Method overview ───────────────────────────────────────────────────────────

st.header(":material/interests: Features")

st.markdown("""
This tool works on **multi-conformer MOL2 files** returned by conformational sampling 
experiments performed in at least two different solvents. It supports any bRo5 molecule 
class — **macrocycles**, **PROTACs**, cyclic peptides, and other large flexible 
scaffolds — and returns the following output parameters:
""")

st.markdown("""
<style>
.param-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1rem 0 1.5rem 0;
}
.param-card {
    border-radius: 12px;
    padding: 1.2rem 1rem;
    border: 1px solid rgba(0, 121, 176, 0.25);
    background: rgba(0, 121, 176, 0.05);
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
}
.param-card .badge {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.2rem 0.5rem;
    border-radius: 20px;
    width: fit-content;
    background: rgba(0, 121, 176, 0.15);
    color: #0079b0;
}
.param-card .badge.optional {
    background: rgba(150, 150, 150, 0.15);
    color: #888;
}
.param-card .param-name {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0079b0;
    margin-top: 0.2rem;
}
.param-card .param-full {
    font-size: 0.82rem;
    opacity: 0.75;
    line-height: 1.4;
}
</style>

<div class="param-grid">
    <div class="param-card">
        <div class="badge">Default</div>
        <div class="param-name">Rgyr</div>
        <div class="param-full">Radius of gyration</div>
    </div>
    <div class="param-card">
        <div class="badge">Default</div>
        <div class="param-name">3D-PSA</div>
        <div class="param-full">Three-dimensional polar surface area (Shrake–Rupley)</div>
    </div>
    <div class="param-card">
        <div class="badge optional">Optional</div>
        <div class="param-name">IMHB</div>
        <div class="param-full">Intramolecular hydrogen bonds</div>
    </div>
    <div class="param-card">
        <div class="badge optional">Optional</div>
        <div class="param-name">π–π</div>
        <div class="param-full">Aromatic π–π stacking interactions</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
While the Rgyr and 3D-PSA are computed by default, the user can find out more about the drivers of 
chameleonicity by ticking the **:material/check_box_outline_blank: Intramolecular H-bonds (IMHB)** and 
**:material/check_box_outline_blank: Aromatic π–π Stacking** boxes in the **Run Analysis** sidebar.
""")



st.markdown("<br>", unsafe_allow_html=True)


# ── How to use ────────────────────────────────────────────────────────────────

st.header(":material/rocket_launch: Let's Get Started")

# TODO: expand with screenshots or a step-by-step guide.
st.markdown("""
1. Go to the **:material/science: Run Analysis** page from the sidebar.
2. Upload at least two multi-conformer **MOL2** files, one for each different solvent
(format: **`<molecule_name>_<solvent>.mol2`** — *e.g.* `ARV110_H2O.mol2`).
3. Configure analysis parameters in the sidebar.
4. Click **:material/rocket_launch: Run Analysis** and explore the results.
""")


# ── Call to action ────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .stButton > button {
        background-color: #0079b0 !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #005f8a !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

if st.button(":material/play_arrow: Start Your Analysis", use_container_width=False):
    st.switch_page("pages/analysis.py")

st.markdown("<br><br>", unsafe_allow_html=True)




show_footer(image="figures/footer.png", caption="")
