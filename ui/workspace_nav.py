"""
ui/workspace_nav.py — Six-Workspace Navigation Layer (V9 Phase 10)

Per the V9 spec (§48): "Eventually reduce the user experience to
DISCOVER / MARKET / OPPORTUNITIES / RESEARCH / PORTFOLIO / ALPHA LAB...
The existing 23-tab functionality can remain underneath while this
becomes the primary navigation."

Scoping note, read before extending this file: Streamlit's stable public
API (this app pins streamlit>=1.35.0) has no supported way to
programmatically select an already-rendered st.tabs() tab. The only way
to force-switch a tab is a DOM-click JS hack against internal
`.stTabs button` element ordering — version-fragile by construction, and
directly at odds with this codebase's own dashboard.py regression
discipline (verified `with tabs[N]:` block integrity, caution about
manual edits to an already-10,000+-line file). So this is deliberately a
WAY-FINDING layer, not a hard router: it shows what each workspace
contains and a one-line live summary, and the person clicks the
matching tab in the bar below — the 23 tabs remain completely untouched,
exactly as the spec allows. If a future Streamlit version ships a
supported programmatic tab-select API, upgrading this to true one-click
switching is a contained change to this file alone.
"""
from __future__ import annotations
import streamlit as st

# Maps each of the six workspaces to the real tab labels underneath —
# must exactly match the `tabs = st.tabs([...])` list in dashboard.py.
# If a tab is renamed there, update its label here too.
WORKSPACES = {
    "🔭 Discover": {
        "description": "Find stocks.",
        "tabs": ["🏆 Leaderboard", "🔍 Stock Deep Dive", "📊 Scan Delta", "📖 Guide"],
    },
    "🌎 Market": {
        "description": "Understand the market.",
        "tabs": ["🌍 Theme Heatmap", "🔄 Sector Rotation", "🤖 AI Briefing", "🧠 Interpretation"],
    },
    "🎯 Opportunities": {
        "description": "Actionable setups.",
        "tabs": ["👁 Setup Monitor", "✅ Pre-Buy Checklist", "🎯 Options Flow",
                "🕵️ Insider Tracker", "⏱ Backtester"],
    },
    "🧠 Research": {
        "description": "Fundamentals, valuation, Early Alpha.",
        "tabs": ["🏛 Long-Term Investing", "📊 Dividend Calculator", "⚖️ Risk Calculator",
                "📅 Earnings Calendar"],
    },
    "💼 Portfolio": {
        "description": "Positions, watchlists, journal, risk.",
        "tabs": ["💼 Portfolio Tracker", "📋 Watchlists", "📓 Trade Journal",
                "🔔 Alert Settings", "📡 Discovery Tracker"],
    },
    "🧪 Alpha Lab": {
        "description": "Prove what works.",
        "tabs": ["📡 Discovery Tracker"],  # Alpha Lab is embedded inside this tab
        "note": "Lives inside 📡 Discovery Tracker below, at the bottom of the page.",
    },
}


def render_workspace_nav():
    """Renders the six-workspace selector strip. Purely additive — never
    raises up into dashboard.py, never touches the real tab bar or any
    `with tabs[N]:` content below it."""
    try:
        st.markdown("##### Workspaces")
        names = list(WORKSPACES.keys())
        selected = st.radio("Workspace", names, horizontal=True, label_visibility="collapsed",
                            key="workspace_nav_selection")
        info = WORKSPACES[selected]
        tabs_txt = " · ".join(info["tabs"])
        st.caption(f"{info['description']} Open one of these tabs below: **{tabs_txt}**"
                  + (f" — {info['note']}" if "note" in info else ""))
    except Exception:
        # Way-finding only — a failure here must never block access to the
        # real tabs underneath.
        pass
