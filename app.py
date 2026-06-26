"""Streamlit app: probabilités par tranche pour les marchés Polymarket '# of tweets' d'Elon Musk.

Lancer :  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tweetanalyst import backtest as BT  # noqa: E402
from tweetanalyst import data as D  # noqa: E402
from tweetanalyst import model as M  # noqa: E402
from tweetanalyst import pipeline as P  # noqa: E402

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:  # graceful: live mode just disabled
    _HAS_AUTOREFRESH = False

DAYS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

st.set_page_config(page_title="Elon Tweet Tracker", layout="wide")
st.title("📊 Elon Musk — probabilités par tranche (Polymarket)")
st.caption(
    "Modèle: intensité saisonnière jour×heure (ET) + processus auto-excitant de Hawkes "
    "(bursts) + Monte-Carlo de la fin de semaine. Données: xtracker.polymarket.com (source "
    "de résolution) + Gamma API (tranches & prix live)."
)


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=900)
def list_active_markets(handle: str) -> list[tuple[str, str]]:
    out = []
    for tw in D.get_trackings(handle):
        if tw.is_active and tw.market_link:
            out.append((tw.title, D.slug_from_url(tw.market_link)))
    return out


@st.cache_data(show_spinner=True, ttl=600)
def cached_run(slug: str, handle: str, now_iso: str | None, n_sims: int,
               half_life: float, fit_days: float, gamma: float,
               refresh_token: int = 0) -> dict:
    # refresh_token busts the cache on each live tick (live mode); 0 = normal 10-min cache.
    now = dt.datetime.fromisoformat(now_iso) if now_iso else None
    run = P.run_forecast(slug, handle=handle, now=now, n_sims=n_sims, gamma=gamma)
    conf = M.confidence_report(run.table, run.forecast.samples,
                               run.forecast.summary()["hours_remaining"])
    # repackage to a cacheable dict (avoid caching heavy objects with live handles)
    return {
        "confidence": conf,
        "title": run.market.title,
        "window_start": run.window_start,
        "window_end": run.window_end,
        "now": run.now,
        "table": run.table,
        "samples": run.forecast.samples,
        "summary": run.forecast.summary(),
        "heatmap": run.fit.intensity.heatmap(),
        "alpha": run.fit.hawkes.alpha,
        "beta": run.fit.hawkes.beta,
        "burst_h": run.fit.hawkes.burst_timescale_h,
        "mean_level": run.fit.intensity.mean_level,
        "weekly_totals": run.fit.intensity.weekly_totals,
    }


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.header("Paramètres")
handle = st.sidebar.text_input("Compte (handle)", value="elonmusk")

try:
    markets = list_active_markets(handle)
except Exception as e:  # noqa: BLE001
    markets = []
    st.sidebar.warning(f"Marchés actifs indisponibles: {e}")

market_labels = [m[0] for m in markets]
mode = st.sidebar.radio("Marché", ["Marchés actifs", "URL / slug manuel"], index=0)
if mode == "Marchés actifs" and markets:
    pick = st.sidebar.selectbox("Choisir", market_labels)
    slug = dict(zip(market_labels, [m[1] for m in markets]))[pick]
else:
    url = st.sidebar.text_input(
        "URL Polymarket ou slug",
        value="elon-musk-of-tweets-june-26-july-3",
    )
    slug = D.slug_from_url(url)

n_sims = st.sidebar.select_slider("Simulations Monte-Carlo", [4000, 8000, 20000, 40000], value=20000)
half_life = st.sidebar.slider("Demi-vie récence (jours)", 7, 60, 28)
fit_days = st.sidebar.slider("Fenêtre fit Hawkes (jours)", 30, 180, 90)
gamma = st.sidebar.slider(
    "Recalibrage (sharpening γ)", 1.0, 3.0, float(P.CALIBRATED_GAMMA), 0.05,
    help="γ>1 resserre la distribution pour corriger la sous-confiance mesurée au backtest. "
         "1.0 = probas brutes du modèle. Valeur par défaut calibrée sur 16 semaines.",
)

override = st.sidebar.checkbox("Forcer une date 'as of' (backtest manuel)")
now_iso = None
if override:
    d = st.sidebar.date_input("Date", value=dt.date(2026, 6, 23))
    t = st.sidebar.time_input("Heure (UTC)", value=dt.time(16, 0))
    now_iso = dt.datetime.combine(d, t).replace(tzinfo=dt.timezone.utc).isoformat()

# ---- Live mode: auto re-pull (incremental) + recompute on an interval ----
st.sidebar.markdown("---")
live = st.sidebar.checkbox(
    "🔴 Mode live (auto-refresh)", value=False, disabled=(override or not _HAS_AUTOREFRESH),
    help="Re-tire les nouveaux tweets (incrémental, ~5 min de latence XTracker) et recalcule "
         "automatiquement. À activer surtout en fin de semaine près de la clôture.",
)
if override:
    st.sidebar.caption("Mode live indisponible avec une date forcée.")
elif not _HAS_AUTOREFRESH:
    st.sidebar.caption("Installer `streamlit-autorefresh` pour activer le mode live.")

refresh_token = 0
if live:
    interval = st.sidebar.select_slider("Intervalle live", [30, 60, 120, 300], value=60,
                                        format_func=lambda x: f"{x}s")
    tick = st_autorefresh(interval=interval * 1000, key="live_tick")
    refresh_token = int(tick)  # changes each tick -> busts cached_run -> fresh pull + recompute

if st.sidebar.button("🔄 Rafraîchir les données"):
    cached_run.clear()
    list_active_markets.clear()

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
try:
    R = cached_run(slug, handle, now_iso, n_sims, half_life, fit_days, gamma, refresh_token)
except Exception as e:  # noqa: BLE001
    st.error(f"Erreur: {e}")
    st.stop()

if live:
    st.caption(f"🔴 Live — dernière mise à jour {dt.datetime.now().strftime('%H:%M:%S')} "
               f"(re-tirage incrémental toutes les {interval}s)")

s = R["summary"]
remaining_h = s["hours_remaining"]
settled = remaining_h <= 0

st.subheader(R["title"])
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tweets observés", s["n_obs"])
c2.metric("Heures restantes", f"{remaining_h:.0f}" if not settled else "réglé")
c3.metric("Total final (médiane)", f"{s['median']:.0f}")
c4.metric("Intervalle 90%", f"{s['p5']:.0f}–{s['p95']:.0f}")
c5.metric("Niveau hebdo moyen", f"{R['mean_level']:.0f}")

# --------------------------------------------------------------------------- #
# Bracket table with edge
# --------------------------------------------------------------------------- #
st.markdown("### Probabilités par tranche")
# ---- Confidence panel (built on the distance-to-edge insight) ----
conf = R["confidence"]
saf = conf["safety"]
if saf >= 1.5:
    badge, color = "🟢 CONFIANT", "rgba(0,160,0,0.18)"
elif conf["young_week"]:
    badge, color = "⚪️ TROP TÔT", "rgba(150,150,150,0.18)"
elif saf <= 0.7:
    badge, color = "🟠 BORD DE TRANCHE", "rgba(230,150,0,0.20)"
else:
    badge, color = "🟡 INTERMÉDIAIRE", "rgba(220,200,0,0.15)"

st.markdown(
    f"<div style='background:{color};padding:10px 14px;border-radius:8px'>"
    f"<b>Indice de confiance — {badge}</b> &nbsp;|&nbsp; "
    f"Tranche la plus probable : <b>{conf['top_label']}</b> à <b>{conf['top_prob']:.0%}</b> "
    f"&nbsp;|&nbsp; Total projeté ≈ <b>{conf['proj_total']:.0f}</b>, "
    f"marge au bord de tranche : <b>{conf['margin']:.0f} tweets</b> "
    f"(soit <b>{conf['safety']:.1f}×</b> l'incertitude restante)<br>"
    f"<span style='font-size:0.9em'>→ {conf['regime']}</span></div>",
    unsafe_allow_html=True,
)
st.caption(
    "La confiance = proba sur la tranche leader. La *marge au bord* est le levier clé : un total "
    "loin d'un bord de 20 = modèle sûr (peu d'edge) ; collé à un bord = un burst peut tout faire "
    "basculer (incertitude = opportunité)."
)

df = pd.DataFrame(R["table"])
df_disp = pd.DataFrame(
    {
        "Tranche": df["label"],
        "Proba modèle": df["model_prob"],
        "Prix OUI": df["yes_price"],
        "Prix NON": df["no_price"],
        "Action": df["best_side"],
        "Edge du pari": df["best_edge"],
    }
)


def _hl_action(row):
    side = row["Action"]
    e = row["Edge du pari"]
    if side == "OUI" and pd.notna(e) and e > 0.03:
        return ["background-color: rgba(0,180,0,0.22)"] * len(row)
    if side == "NON" and pd.notna(e) and e > 0.03:
        return ["background-color: rgba(80,140,230,0.22)"] * len(row)
    return [""] * len(row)


styled = (
    df_disp.style.format(
        {"Proba modèle": "{:.1%}", "Prix OUI": "{:.2f}", "Prix NON": "{:.2f}",
         "Edge du pari": "{:+.1%}"},
        na_rep="—",
    )
    .apply(_hl_action, axis=1)
)
st.dataframe(styled, use_container_width=True, hide_index=True, height=min(680, 38 * (len(df) + 1)))
st.caption(
    "**Action** = côté recommandé. 🟢 Vert = acheter **OUI** (tranche sous-cotée). "
    "🔵 Bleu = acheter **NON** (tranche surcotée). *Edge du pari* = avantage estimé du côté "
    "recommandé, après recalibrage γ. Seuls les paris à edge > 3 pts sont surlignés."
)

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("#### Distribution simulée du total final")
    samples = R["samples"]
    fig = go.Figure()
    fig.add_histogram(x=samples, nbinsx=60, marker_color="#4C9BE8", name="simulé")
    fig.add_vline(x=s["n_obs"], line_dash="dot", line_color="gray",
                  annotation_text=f"observés ({s['n_obs']})")
    fig.add_vline(x=s["median"], line_color="#E8704C", annotation_text="médiane")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="total tweets sur la semaine", yaxis_title="fréquence")
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.markdown("#### Modèle vs marché par tranche")
    fig2 = go.Figure()
    fig2.add_bar(x=df["label"], y=df["model_prob"], name="Modèle", marker_color="#4C9BE8")
    if df["market_price"].notna().any():
        fig2.add_bar(x=df["label"], y=df["market_price"], name="Marché", marker_color="#E8B04C")
    fig2.update_layout(height=360, barmode="group", margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="tranche", yaxis_title="probabilité",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("#### Rythme de tweets — intensité par jour × heure (heure ET, pondérée récence)")
hm = R["heatmap"]  # (7, 24) tweets/hour
figh = go.Figure(
    go.Heatmap(z=hm, x=[f"{h:02d}h" for h in range(24)], y=DAYS,
               colorscale="YlOrRd", colorbar=dict(title="tw/h"))
)
figh.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(figh, use_container_width=True)
st.caption(
    f"Hawkes: α={R['alpha']:.2f} (part de tweets déclenchés par 'burst'), "
    f"échelle de burst ≈ {R['burst_h']*60:.0f} min. "
    f"Totaux hebdo récents: {[int(x) for x in R['weekly_totals'][-8:]]}"
)

# --------------------------------------------------------------------------- #
# Backtest (optional, heavier)
# --------------------------------------------------------------------------- #
st.markdown("---")
with st.expander("🔬 Backtest de calibration (replay des semaines passées)"):
    n_weeks = st.slider("Semaines à rejouer", 4, 24, 10)
    bt_sims = st.select_slider("Sims/backtest", [2000, 4000, 8000], value=4000)
    if st.button("Lancer le backtest"):
        with st.spinner("Replay en cours…"):
            posts = D.load_posts(handle, start=R["now"] - dt.timedelta(days=260), end=R["now"])
            res = BT.run_backtest(posts, R["window_end"], n_weeks=n_weeks, n_sims=bt_sims)
        g_opt, ll0, ll1 = BT.fit_sharpening(res.prob_matrix, res.true_idx)
        m1, m2, m3 = st.columns(3)
        m1.metric("Log-loss (↓ mieux)", f"{ll1:.3f}", delta=f"{ll1 - ll0:+.3f} vs brut")
        m2.metric("Brier multiclasse (↓ mieux)", f"{res.brier:.3f}")
        m3.metric("γ optimal (sharpening)", f"{g_opt:.2f}")
        st.caption(
            f"γ ajusté sur ces {n_weeks} semaines = **{g_opt:.2f}** (in-sample, donc légèrement "
            f"optimiste). Reporte cette valeur dans le curseur 'Recalibrage' à gauche si tu veux "
            f"l'appliquer aux prévisions."
        )
        rel = res.reliability
        figr = go.Figure()
        figr.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash",
                         line_color="gray", name="parfait")
        figr.add_scatter(x=rel["mean_pred"], y=rel["hit_rate"], mode="markers+lines",
                         marker_size=8, name="modèle")
        figr.update_layout(height=380, xaxis_title="probabilité prédite",
                           yaxis_title="fréquence réalisée", title="Courbe de fiabilité")
        st.plotly_chart(figr, use_container_width=True)
        st.dataframe(res.records, use_container_width=True, hide_index=True)
