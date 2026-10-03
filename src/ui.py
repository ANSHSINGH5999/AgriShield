"""Styling and small UI helpers for the Streamlit app (custom CSS + Plotly theme)."""
import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

FOREST, LEAF, SPROUT, CREAM, INK, MUTED, AMBER, RED = "#17432b", "#3f8a55", "#cfe8c4", "#f7f6f0", "#1d2a22", "#5d6b62", "#c98a1b", "#b5452f"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=DM+Sans:wght@400;500;700&display=swap');
:root {{ --forest:{FOREST}; --leaf:{LEAF}; --sprout:{SPROUT}; --cream:{CREAM}; --ink:{INK}; --muted:{MUTED}; }}
html, body, [class*="css"], .stApp {{ font-family: 'DM Sans', 'Segoe UI', system-ui, sans-serif; color: var(--ink); }}
.stApp {{ background: var(--cream); }}
h1, h2, h3 {{ font-family: 'Fraunces', Georgia, serif !important; color: var(--forest) !important; letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background: var(--forest); }}
[data-testid="stSidebar"] * {{ color: #eef5ec !important; }}
[data-testid="stSidebar"] .stRadio label {{ padding: 6px 4px; border-radius: 10px; }}
[data-testid="stSidebar"] .stRadio label:hover {{ background: rgba(255,255,255,0.08); }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
.hero {{ background: linear-gradient(135deg, {FOREST} 0%, #24603d 100%); color: #f3f8f1; border-radius: 22px; padding: 28px 32px; margin-bottom: 22px; }}
.hero h1 {{ color: #ffffff !important; margin: 0 0 6px 0; font-size: 2.1rem; }}
.hero p {{ color: #d7ead2; margin: 0; max-width: 70ch; font-size: 1.02rem; }}
.eyebrow {{ text-transform: uppercase; letter-spacing: .12em; font-size: .74rem; font-weight: 700; color: {SPROUT}; margin-bottom: 8px; }}
.card {{ background: #ffffff; border: 1px solid #e3e6dc; border-radius: 18px; padding: 18px 20px; height: 100%; box-shadow: 0 1px 2px rgba(23,67,43,.04); }}
.card .label {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 700; }}
.card .value {{ font-family: 'Fraunces', Georgia, serif; font-size: 2rem; color: var(--forest); line-height: 1.15; margin-top: 4px; }}
.card .note {{ font-size: .84rem; color: var(--muted); margin-top: 4px; }}
.section-tag {{ display:inline-block; background:{SPROUT}; color:{FOREST}; border-radius:999px; padding:3px 10px; font-size:.75rem; font-weight:700; margin-bottom:8px; }}
.banner {{ border-radius: 14px; padding: 14px 18px; margin: 10px 0; font-size: .97rem; }}
.banner.ok {{ background: #e8f4e3; border: 1px solid #b9dcae; color: #1d4d2c; }}
.banner.warn {{ background: #fdf3e1; border: 1px solid #efd29a; color: #6b4a0c; }}
.banner.bad {{ background: #fbe9e5; border: 1px solid #ebbcb1; color: #74291b; }}
.pill {{ display:inline-block; border-radius:999px; padding:2px 10px; font-size:.8rem; font-weight:700; }}
.pill.ok {{ background:#dff0d8; color:#22562f; }} .pill.warn {{ background:#fbe7c5; color:#7a520b; }}
.small {{ font-size:.85rem; color: var(--muted); }}
.disclaimer {{ font-size:.8rem; opacity:.85; border-top:1px solid rgba(255,255,255,.2); padding-top:12px; margin-top:16px; }}
div[data-testid="stFileUploader"] section {{ border-radius: 16px; border: 2px dashed #b9cdb2; background: #fbfcf8; }}
.stButton>button, .stDownloadButton>button {{ background: var(--forest); color: #fff; border-radius: 12px; border: 0; padding: .55rem 1.1rem; font-weight: 700; }}
.stButton>button:hover {{ background: #22573a; color:#fff; }}
img {{ border-radius: 14px; }}
.step {{ display:flex; align-items:center; gap:10px; margin: 30px 0 12px; }}
.step .num {{ width:30px; height:30px; border-radius:50%; background:var(--forest); color:#fff; display:flex; align-items:center;
             justify-content:center; font-weight:700; font-size:.9rem; flex:none; }}
.step h2 {{ margin:0 !important; font-size:1.55rem !important; }}
.big {{ background:#fff; border:1px solid #e3e6dc; border-radius:20px; padding:22px 24px; height:100%; }}
.big .label {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); font-weight:700; }}
.big .value {{ font-family:'Fraunces', Georgia, serif; font-size:2.1rem; line-height:1.15; color:var(--forest); margin-top:6px; }}
.big .note {{ font-size:.85rem; color:var(--muted); margin-top:6px; }}
.status-ok {{ color:#22562f !important; }} .status-warn {{ color:#8a4b0c !important; }}
.robust {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid #e3e6dc; border-radius:16px; overflow:hidden; }}
.robust th, .robust td {{ padding:10px 14px; border-bottom:1px solid #eef0ea; text-align:left; font-size:.95rem; }}
.robust th {{ font-size:.75rem; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); background:#fbfcf8; }}
.robust tr:last-child td {{ border-bottom:0; }}
.final {{ background:linear-gradient(135deg, {FOREST} 0%, #24603d 100%); color:#eef5ec; border-radius:22px; padding:24px 28px; }}
.final .row {{ display:flex; justify-content:space-between; gap:16px; padding:8px 0; border-bottom:1px solid rgba(255,255,255,.14); }}
.final .row:last-child {{ border-bottom:0; }}
.final .k {{ color:{SPROUT}; font-size:.9rem; }} .final .v {{ font-weight:700; text-align:right; }}
</style>
"""

pio.templates["agrishield"] = go.layout.Template(layout=dict(
    font=dict(family="DM Sans, Segoe UI, sans-serif", color=INK, size=13),
    colorway=[FOREST, LEAF, AMBER, RED, "#5b7c99", "#8a6fb0", "#9cc58a"],
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
    xaxis=dict(gridcolor="#eef0ea", zerolinecolor="#dfe3da"), yaxis=dict(gridcolor="#eef0ea", zerolinecolor="#dfe3da"),
    margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2)))
pio.templates.default = "agrishield"


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, text: str) -> None:
    st.markdown(f'<div class="hero"><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1>'
                f'<p>{html.escape(text)}</p></div>', unsafe_allow_html=True)


def card(label: str, value: str, note: str = "") -> None:
    st.markdown(f'<div class="card"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div>'
                f'<div class="note">{html.escape(note)}</div></div>', unsafe_allow_html=True)


def banner(kind: str, text: str) -> None:
    st.markdown(f'<div class="banner {kind}">{text}</div>', unsafe_allow_html=True)


def step(num: int, title: str) -> None:
    st.markdown(f'<div class="step"><div class="num">{num}</div><h2>{html.escape(title)}</h2></div>', unsafe_allow_html=True)


def big(label: str, value: str, note: str = "", css: str = "") -> None:
    st.markdown(f'<div class="big"><div class="label">{html.escape(label)}</div><div class="value {css}">{html.escape(value)}</div>'
                f'<div class="note">{html.escape(note)}</div></div>', unsafe_allow_html=True)


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"
