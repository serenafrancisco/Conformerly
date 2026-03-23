#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streamlit_app.py — Conformerly
Multi-page entry point.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
from PIL import Image

icon = Image.open("figures/logo-favicon.png")

st.set_page_config(
    page_title="Conformerly",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Page definitions ──────────────────────────────────────────────────────────

about_page = st.Page(
    "pages/about.py",
    title="About Conformerly",
    icon=":material/info:",
    default=True,
)

analysis_page = st.Page(
    "pages/analysis.py",
    title="Run Analysis",
    icon=":material/science:",
)

links_page = st.Page(
    "pages/links.py",
    title="Useful Links",
    icon=":material/link:",
)

# ── Navigation ────────────────────────────────────────────────────────────────

pg = st.navigation([about_page, analysis_page, links_page])

pg.run()