from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent.parent

def show_footer(image: str = None, caption: str = ""):
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    st.markdown(
        """
        <p style="text-align:center; color:#888; font-size:0.8em; line-height:1.6;">
            <strong>Conformerly</strong> is developed and maintained by the <strong>MedChemBeyond Lab</strong>.<br>
            <a href="https://www.dmbhs.unito.it/do/home.pl" target="_blank" style="color:#888;">Dept. of Molecular Biotechnology and Health Sciences</a>.<br>
            <a href="https://www.unito.it" target="_blank" style="color:#888;">University of Turin</a>, Italy.
        </p>
        """,
        unsafe_allow_html=True
    )
    if image:
        col1, col2, col3 = st.columns([1, 2, 1])  # center the image
        with col2:
            st.image(ROOT / image, caption=caption, use_container_width=True)