"""Reusable HTML/Streamlit UI building blocks. All dynamic text is HTML-escaped by the callers
that pass user/model text; numbers are formatted here."""
from __future__ import annotations

import html as _html
from typing import Iterable, List, Optional, Sequence

import streamlit as st


def esc(x) -> str:
    return _html.escape(str(x))


def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def fmt(x: Optional[float], digits: int = 1, suffix: str = "", none: str = "—") -> str:
    if x is None or x != x:
        return none
    return f"{x:,.{digits}f}{suffix}"


def page_header(title: str, subtitle: str = "", icon: str = "", right: str = "") -> None:
    md(f'<div class="page-header"><div><div class="ph-title"><span class="ph-icon">{icon}</span>{esc(title)}</div>'
       f'<div class="ph-sub">{esc(subtitle)}</div></div><div>{right}</div></div>')


def pill(text: str, tone: str = "blue") -> str:
    return f'<span class="pill pill-{tone}">{esc(text)}</span>'


def kpi_card(label: str, value: str, sub: str = "", accent: str = "cyan", icon: str = "",
             pulse: bool = False) -> str:
    """`value` and `sub` are trusted HTML (numbers / pills built by the caller)."""
    cls = f"kpi accent-{accent}" + (" pulse" if pulse else "")
    return (f'<div class="{cls}"><div class="kpi-top"><span class="kpi-label">{esc(label)}</span>'
            f'<span class="kpi-icon">{icon}</span></div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def kpi_grid(items: Sequence[dict], cols: int = 4) -> None:
    """items: dicts of kpi_card keyword arguments."""
    for i in range(0, len(items), cols):
        columns = st.columns(cols)
        for col, item in zip(columns, items[i:i + cols]):
            with col:
                md(kpi_card(**item))


def card(title: str, body_html: str) -> str:
    return f'<div class="card"><div class="card-title">{esc(title)}</div>{body_html}</div>'


def alert(text: str, tone: str = "amber", title: str = "", preformatted: bool = False) -> None:
    head = f"<b>{esc(title)}</b>" if title else ""
    body = f"<pre>{esc(text)}</pre>" if preformatted else esc(text)
    md(f'<div class="alert alert-{tone}">{head}{body}</div>')


def html_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """Cells are trusted HTML."""
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="tbl"><tr>{th}</tr>{body}</table>'


def phase_chips(names: Sequence[str], active: int, target: Optional[int] = None,
                transitioning: bool = False) -> str:
    chips: List[str] = []
    for i, n in enumerate(names):
        cls = "chip"
        if i == active and not transitioning:
            cls += " active"
        elif target is not None and i == target and (transitioning or target != active):
            cls += " target"
        chips.append(f'<span class="{cls}">{esc(n)}</span>')
    return '<div class="chips">' + "".join(chips) + "</div>"


def flow(nodes: Sequence[str]) -> str:
    parts = []
    for i, n in enumerate(nodes):
        if i:
            parts.append('<span class="arrow">→</span>')
        parts.append(f'<span class="node">{esc(n)}</span>')
    return '<div class="flow">' + "".join(parts) + "</div>"


def sumo_error(message: str) -> None:
    """Readable error box; SUMO-not-found errors carry the full setup instructions."""
    if "SUMO could not be found" in message or "not importable" in message:
        alert(message, "red", "SUMO is not installed or not configured", preformatted=True)
    else:
        alert(message, "red", "Simulation could not start", preformatted=True)