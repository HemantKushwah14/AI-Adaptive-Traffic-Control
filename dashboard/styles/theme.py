"""Visual theme: colour tokens and the custom CSS (dark navy, electric blue, cyan, green)."""
from __future__ import annotations

import streamlit as st

COLORS = {
    "bg": "#070f24", "bg2": "#0b1735", "card": "rgba(15, 28, 63, 0.78)",
    "border": "rgba(96, 140, 255, 0.20)", "blue": "#2f7bff", "cyan": "#22d3ee",
    "green": "#22e58a", "amber": "#fbbf24", "red": "#ff4d5e", "purple": "#a78bfa",
    "text": "#e6edff", "muted": "#8da2cf", "grid": "rgba(141,162,207,0.14)", "plot_bg": "#060d1f",
}

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter','Segoe UI',system-ui,sans-serif; }
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1100px 560px at 88% -8%, rgba(47,123,255,.20), transparent 60%),
              radial-gradient(900px 480px at -8% 8%, rgba(34,211,238,.10), transparent 55%), var(--bg);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1560px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#081230 0%,#060d22 100%); border-right: 1px solid var(--border); }
h1,h2,h3,h4 { color: var(--text); letter-spacing: -0.01em; }

.brand { display:flex; align-items:center; gap:12px; padding: 4px 4px 14px 4px; }
.brand-logo { width:42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  font-size:22px; background: linear-gradient(135deg,#2f7bff,#22d3ee); box-shadow: 0 6px 22px rgba(47,123,255,.45); }
.brand-name { font-weight:800; font-size:16px; line-height:1.1; color:#fff; }
.brand-sub { font-size:11px; color: var(--muted); letter-spacing:.08em; text-transform:uppercase; }

[data-testid="stSidebar"] div[role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] div[role="radiogroup"] > label { background: transparent; border:1px solid transparent;
  border-radius: 12px; padding: 9px 12px; cursor:pointer; transition: all .15s ease; width:100%; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(47,123,255,.12); border-color: var(--border); }
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display:none; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
  background: linear-gradient(90deg, rgba(47,123,255,.30), rgba(34,211,238,.12)); border-color: rgba(34,211,238,.45); }
[data-testid="stSidebar"] div[role="radiogroup"] p { font-weight:600; font-size:14px; }

.page-header { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; margin: 0 0 18px 0; padding-bottom: 14px;
  border-bottom: 1px solid var(--border); }
.ph-title { font-size: 28px; font-weight: 800; color:#fff; letter-spacing:-0.02em; }
.ph-icon { margin-right: 10px; }
.ph-sub { color: var(--muted); font-size: 14px; margin-top: 2px; max-width: 880px; }

.kpi { position:relative; background: var(--card); border:1px solid var(--border); border-radius:18px; padding:15px 18px 13px;
  margin-bottom:14px; overflow:hidden; min-height:116px; backdrop-filter: blur(6px); transition: transform .15s ease, border-color .15s ease; }
.kpi:hover { transform: translateY(-2px); border-color: rgba(34,211,238,.5); }
.kpi::before { content:""; position:absolute; left:0; top:0; right:0; height:3px; background: var(--accent); }
.accent-cyan { --accent: linear-gradient(90deg,#22d3ee,#2f7bff); }
.accent-blue { --accent: linear-gradient(90deg,#2f7bff,#a78bfa); }
.accent-green { --accent: linear-gradient(90deg,#22e58a,#22d3ee); }
.accent-amber { --accent: linear-gradient(90deg,#fbbf24,#fb923c); }
.accent-red { --accent: linear-gradient(90deg,#ff4d5e,#fb923c); }
.accent-purple { --accent: linear-gradient(90deg,#a78bfa,#2f7bff); }
.kpi-top { display:flex; justify-content:space-between; align-items:center; }
.kpi-label { color: var(--muted); font-size:12px; font-weight:600; letter-spacing:.07em; text-transform:uppercase; }
.kpi-icon { font-size:18px; opacity:.9; }
.kpi-value { font-size:30px; font-weight:800; color:#fff; margin-top:6px; line-height:1.1; letter-spacing:-0.02em; }
.kpi-sub { color: var(--muted); font-size:12.5px; margin-top:6px; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(255,77,94,.55);} 70%{box-shadow:0 0 0 14px rgba(255,77,94,0);} 100%{box-shadow:0 0 0 0 rgba(255,77,94,0);} }
.pulse { animation: pulse 1.6s infinite; border-color: rgba(255,77,94,.7); }

.card { background: var(--card); border:1px solid var(--border); border-radius:18px; padding:16px 18px; margin-bottom:14px; }
.card-title { color: var(--muted); font-size:12px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-bottom:10px; }
.big-phase { font-size:26px; font-weight:800; color:#fff; letter-spacing:-0.01em; }
.big-count { font-size:44px; font-weight:800; line-height:1; color: var(--cyan); }
.muted { color: var(--muted); font-size:12.5px; }

.pill { display:inline-block; padding:3px 11px; border-radius:999px; font-size:12px; font-weight:700; letter-spacing:.03em; border:1px solid transparent; }
.pill-green { color:#22e58a; background:rgba(34,229,138,.12); border-color:rgba(34,229,138,.35); }
.pill-amber { color:#fbbf24; background:rgba(251,191,36,.12); border-color:rgba(251,191,36,.35); }
.pill-red { color:#ff4d5e; background:rgba(255,77,94,.13); border-color:rgba(255,77,94,.4); }
.pill-blue { color:#7aa7ff; background:rgba(47,123,255,.15); border-color:rgba(47,123,255,.4); }
.pill-cyan { color:#22d3ee; background:rgba(34,211,238,.12); border-color:rgba(34,211,238,.35); }
.pill-grey { color:#a9b8dc; background:rgba(141,162,207,.12); border-color:rgba(141,162,207,.28); }

.chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:6px; }
.chip { padding:6px 12px; border-radius:10px; font-size:12px; font-weight:700; color:#a9b8dc; border:1px solid var(--border); background: rgba(20,36,80,.55); }
.chip.active { color:#04101f; background: linear-gradient(135deg,#22e58a,#22d3ee); border-color:transparent; }
.chip.target { color:#fbbf24; border-color:rgba(251,191,36,.6); }
.flow { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:12px; }
.flow .node { padding:7px 12px; border-radius:10px; font-size:12.5px; font-weight:700; background:rgba(47,123,255,.14); border:1px solid rgba(47,123,255,.35); color:#cfe0ff; }
.flow .arrow { color: var(--cyan); font-weight:800; }

.tbl { width:100%; border-collapse:collapse; font-size:13px; }
.tbl th { text-align:left; color: var(--muted); font-size:11px; letter-spacing:.07em; text-transform:uppercase; padding:6px 8px; border-bottom:1px solid var(--border); }
.tbl td { padding:7px 8px; border-bottom:1px solid rgba(96,140,255,.09); color: var(--text); }
.tbl tr:last-child td { border-bottom:none; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; }
.dot.ok { background:#22e58a; box-shadow:0 0 8px #22e58a; } .dot.warn { background:#fbbf24; } .dot.bad { background:#ff4d5e; } .dot.off { background:#4a5578; }

.alert { border-radius:14px; padding:12px 16px; margin: 8px 0 12px 0; font-size:13.5px; border:1px solid; }
.alert b { display:block; margin-bottom:2px; }
.alert-red { background:rgba(255,77,94,.10); border-color:rgba(255,77,94,.45); color:#ffc2c8; }
.alert-amber { background:rgba(251,191,36,.09); border-color:rgba(251,191,36,.40); color:#fde7a6; }
.alert-blue { background:rgba(47,123,255,.10); border-color:rgba(47,123,255,.40); color:#cfe0ff; }
.alert-green { background:rgba(34,229,138,.09); border-color:rgba(34,229,138,.40); color:#b6f5d4; }
.alert pre { white-space:pre-wrap; margin:6px 0 0 0; font-size:12px; color:inherit; background:transparent; }
.logline { font-size:12px; color:#b9c8ee; padding:3px 0; border-bottom:1px dashed rgba(96,140,255,.12); }

.stButton > button, .stDownloadButton > button { border-radius:12px; border:1px solid var(--border); background: rgba(20,36,80,.8);
  color: var(--text); font-weight:600; transition: all .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--cyan); color:#fff;
  box-shadow: 0 0 0 1px var(--cyan), 0 6px 22px rgba(34,211,238,.18); }
.stButton > button[kind="primary"] { background: linear-gradient(135deg,#2f7bff,#22d3ee); border:none; color:#04101f; font-weight:800; }
.stButton > button:disabled { opacity:.38; }
[data-testid="stPlotlyChart"] { border-radius:16px; overflow:hidden; border:1px solid var(--border); background: rgba(6,13,31,.55); }
"""


def inject_css() -> None:
    root = ":root{" + "".join(f"--{k}:{v};" for k, v in COLORS.items()) + "}"
    st.markdown(f"<style>{root}{_CSS}</style>", unsafe_allow_html=True)