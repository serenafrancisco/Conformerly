#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pages/links.py — Conformerly
Useful Links page. Add your papers, website, and resources below.
"""

import streamlit as st
from pathlib import Path
import base64
import sys
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


# Resolves to the project root (where streamlit_app.py lives)
ROOT = Path(__file__).parent.parent
IMAGES = ROOT / "figures"

st.title("Useful Links")
st.caption("Papers, resources, and website.")

st.markdown("<br><br>", unsafe_allow_html=True)


# ── Website ───────────────────────────────────────────────────────────────────

st.header("🌐 Check our Website")

st.markdown("##### [MedChemBeyond Lab — Official Website](https://www.cassmedchem.unito.it/)")

st.markdown("<br><br>", unsafe_allow_html=True)

# ── Publications ──────────────────────────────────────────────────────────────

st.header("📄 Related Papers")
st.markdown("<br>", unsafe_allow_html=True)

publications = [
    {
        "title": "Molecular properties, including chameleonicity, as essential tools for designing the next generation of oral beyond rule of five drugs",
        "authors": "Garcia Jimenez, D <em>et al</em>.",
        "year": 2024,
        "journal": "*ADMET DMPK*",
        "doi": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11542721/",
        "image": IMAGES / "ADMET-DMPK.png",
        "caption": ""
    },
    {
        "title": "IMHB-Mediated Chameleonicity in Drug Design: A Focus on Structurally Related PROTACs",
        "authors": "Garcia Jimenez, D <em>et al</em>.",
        "year": 2024,
        "journal": "*Journal of Medicinal Chemistry*",
        "doi": "https://pubs.acs.org/doi/10.1021/acs.jmedchem.4c01200",
        "image": IMAGES / "IMHBs.png",
        "caption": ""
    },
    {
        "title": "Chamelogk: A Chromatographic Chameleonicity Quantifier to Design Orally Bioavailable Beyond-Rule-of-5 Drugs",
        "authors": "Garcia Jimenez, D <em>et al</em>.",
        "year": 2023,
        "journal": "*Journal of Medicinal Chemistry*",
        "doi": "https://pubs.acs.org/doi/10.1021/acs.jmedchem.3c00823",
        "image": IMAGES / "chamelogk.png",
        "caption": ""
    },
    {
        "title": "Conformational Sampling Deciphers the Chameleonic Properties of a VHL-Based Degrader",
        "authors": "Ermondi, G <em>et al</em>.",
        "year": 2023,
        "journal": "*Pharmaceutics*",
        "doi": "https://www.mdpi.com/1999-4923/15/1/272",
        "image": IMAGES / "pharmaceutics.png",
        "caption": ""
    },
    {
        "title": "Refinement of Computational Access to Molecular Physicochemical Properties: From Ro5 to bRo5",
        "authors": "Rossi Sebastiano, M <em>et al</em>.",
        "year": 2022,
        "journal": "*Journal of Medicinal Chemistry*",
        "doi": "https://pubs.acs.org/doi/10.1021/acs.jmedchem.2c00774",
        "image": IMAGES / "mrs_jmedchem.png",
        "caption": ""
    },
    {
        "title": "Impact of Dynamically Exposed Polarity on Permeability and Solubility of Chameleonic Drugs Beyond the Rule of 5",
        "authors": "Rossi Sebastiano, M <em>et al</em>.",
        "year": 2018,
        "journal": "*Journal of Medicinal Chemistry*",
        "doi": "https://pubs.acs.org/doi/10.1021/acs.jmedchem.8b00347",
        "image": IMAGES / "mrs_jmedchem_2.png",
        "caption": ""
    },
]

def img_to_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

for pub in publications:
    b64 = img_to_b64(pub["image"]) if pub.get("image") else None
    img_html = (
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:220px; height:160px; object-fit:cover; border-radius:8px; flex-shrink:0;">'
        if b64 else ""
    )

    card = f"""
    <div style="
        display: flex;
        gap: 20px;
        align-items: center;
        background: #f4f6f9;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        border-left: 5px solid #2e86ab;
    ">
        {img_html}
        <div>
            <p style="font-size:1.05em; font-weight:700; margin:0 0 6px 0; color:#1a1a2e;">{pub['title']}</p>
            <p style="margin:0 0 4px 0; color:#555; font-size:0.9em;">{pub['authors']} ({pub['year']})</p>
            <p style="margin:0 0 10px 0; color:#888; font-size:0.85em; font-style:italic;">{pub['journal'].replace('*','')}</p>
            <a href="{pub['doi']}" target="_blank" style="
                padding: 5px 14px;
                background: #2e86ab;
                color: white;
                border-radius: 20px;
                text-decoration: none;
                font-size: 0.82em;
            ">DOI →</a>
        </div>
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ── Related tools & databases ─────────────────────────────────────────────────

st.header("🔗 Related Tools & Databases")

st.markdown("""
- *Placeholder — add related tools or databases here.*
""")

st.markdown("<br><br>", unsafe_allow_html=True)

# ── Contact ───────────────────────────────────────────────────────────────────

st.header("✉️ Contact")

st.markdown("""
- *Placeholder — add contact information here.*
""")


show_footer(image="figures/footer.png", caption="")