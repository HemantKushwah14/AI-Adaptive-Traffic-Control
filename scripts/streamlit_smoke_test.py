"""Minimal Streamlit page to prove the web stack works.
Run:  streamlit run scripts\\streamlit_smoke_test.py"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Smoke test", page_icon="🚦", layout="wide")
st.title("🚦 Streamlit + Plotly smoke test")
st.write("If you can see this page and the chart below, Phase 1 web setup works.")
df = pd.DataFrame({"t": np.arange(60), "value": np.random.default_rng(0).normal(size=60).cumsum()})
st.plotly_chart(px.line(df, x="t", y="value", template="plotly_dark"), use_container_width=True)