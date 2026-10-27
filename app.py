import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from factsheet_export import render_factsheet_section, normalize_index

st.set_page_config(page_title="Portfolio vs Benchmark", layout="wide")
st.title("📊 Portfolio vs Benchmark composite")

# =====================
# Paramètres
# =====================
allocation = {
    "TTE.PA": 0.05,
    "MC.PA": 0.05,
    "INGA.AS": 0.05,
    "SAP.DE": 0.04,
    "ACLN.SW": 0.05,
    "THEON.AS": 0.04,
    "BOI.PA": 0.05,
    "EOAN.DE": 0.05,
    "GOOGL": 0.03,
    "META": 0.02,
    "HWM": 0.03,
    "AMZN": 0.03,
    "0P0000ZWX4.F": 0.08,
    "0P0001861S.F": 0.08,
    "0P00000M6C.F": 0.09,
    "0P00008ESK.F": 0.08,
    "0P0000A6ZG.F": 0.05,
    "0P0000WHLW.F": 0.08,
}

BENCHMARK_WEIGHTS = {"IEV": 0.35, "SPY": 0.20, "TLT": 0.25, "VNQ": 0.10, "EEM": 0.05}

ticker_names = {
    "TTE.PA": "TotalEnergies", "MC.PA": "LVMH", "INGA.AS": "ING Groep",
    "SAP.DE": "SAP", "ACLN.SW": "ACLN", "THEON.AS": "Theon Intl",
    "BOI.PA": "Boiron", "EOAN.DE": "E.ON", "GOOGL": "Alphabet",
    "META": "Meta", "HWM": "Howmet", "AMZN": "Amazon",
    "0P0000ZWX4.F": "Helium Fund Perf A EUR",
    "0P0001861S.F": "Eleva Abs Ret Eurp S EUR",
    "0P00000M6C.F": "R-co Conviction Credit Euro",
    "0P00008ESK.F": "AXAIMFIIS US SD HY EUR H",
    "0P0000A6ZG.F": "Immobilier 21 AC",
    "0P0000WHLW.F": "GemEquity R",
}

USD_TICKERS = ["GOOGL", "META", "HWM", "AMZN"]

# Tickers de secours essayés dans l'ordre si le ticker principal ne renvoie rien.
# Pour un fonds .F qui échoue, ajouter ici une autre cotation (ex. "0P0000XXXX.PA",
# l'ISIN du fonds ou un ETF proxy), par exemple :
#   "0P0000A6ZG.F": ["FR0010541557.PA"],
FALLBACK_TICKERS = {
    "TTE.PA": ["FP.PA"],
}

MIN_OBS = 30          # nombre minimal de cours pour qu'une ligne soit utilisable
STALE_DAYS = 10       # au-delà, un cours est jugé figé (fonds non coté, ticker mort)
LATE_START_DAYS = 30  # historique démarrant plus tard que la date de début

tickers = list(allocation.keys())
start = st.sidebar.date_input("Start date", datetime(2020, 1, 1))
start_iso = pd.Timestamp(start).strftime("%Y-%m-%d")


def pct(x):
    """Rendement simple sans remplissage implicite (compatible pandas 2.x et 3.x)."""
    return x / x.shift(1) - 1


# =====================
# Chargement des prix (robuste)
# =====================
def _extract_close(df):
    """Extrait une série de cours exploitable d'un DataFrame yfinance, ou None."""
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    for col in ("Adj Close", "Close"):
        if col in df.columns:
            s = df[col]
            if isinstance(s, pd.DataFrame):      # colonnes dupliquées
                s = s.iloc[:, 0]
            s = pd.to_numeric(s, errors="coerce").dropna()
            s = s[s > 0]
            if len(s):
                return normalize_index(s)        # supprime le fuseau horaire
    return None


def _try_symbol(symbol, start_iso, retries=2):
    """Plusieurs méthodes yfinance, plusieurs tentatives. Retourne une Series ou None."""
    methods = (
        lambda: yf.download(symbol, start=start_iso, auto_adjust=False,
                            progress=False, threads=False),
        lambda: yf.Ticker(symbol).history(start=start_iso, auto_adjust=False),
    )
    for attempt in range(retries):
        for method in methods:
            try:
                s = _extract_close(method())
                if s is not None:
                    return s
            except Exception:
                continue
        time.sleep(0.4 * (attempt + 1))
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_close_cached(symbol, start_iso):
    """Mise en cache des SUCCÈS uniquement : une exception n'est jamais mise en cache,
    un échec transitoire de Yahoo n'est donc pas figé pendant 1 h."""
    s = _try_symbol(symbol, start_iso)
    if s is None:
        raise ValueError(f"Aucune donnée pour {symbol}")
    return s


def download_single_ticker(ticker, start_iso):
    """Retourne (Series de cours normalisée, symbole utilisé) ou (None, None)."""
    for symbol in [ticker] + FALLBACK_TICKERS.get(ticker, []):
        try:
            return _fetch_close_cached(symbol, start_iso), symbol
        except Exception:
            continue
    return None, None


def load_prices(tickers, start_iso):
    """Télécharge chaque ticker. Retourne (DataFrame des cours valides, rapport par ticker)."""
    series, report = {}, {}
    for t in tickers:
        s, used = download_single_ticker(t, start_iso)
        if s is None:
            report[t] = dict(status="failed", source="-", first=None, last=None,
                             note="Aucune donnée renvoyée par Yahoo Finance")
        elif len(s) < MIN_OBS:
            report[t] = dict(status="failed", source=used, first=s.index[0], last=s.index[-1],
                             note=f"Historique trop court ({len(s)} cours)")
        else:
            series[t] = s
            report[t] = dict(status="ok", source=used, first=s.index[0], last=s.index[-1], note="")
    if not series:
        return pd.DataFrame(), report
    prices = pd.concat(series, axis=1).sort_index().ffill()
    return prices, report


def load_fx(start_iso):
    """EUR/USD (USD pour 1 EUR). Repli sur EUR=X (EUR pour 1 USD) inversé."""
    s, _ = download_single_ticker("EURUSD=X", start_iso)
    if s is not None and len(s) >= MIN_OBS:
        return s
    inv, _ = download_single_ticker("EUR=X", start_iso)
    if inv is not None and len(inv) >= MIN_OBS:
        return 1.0 / inv
    return None


with st.spinner("Téléchargement des cours..."):
    prices, report = load_prices(tickers, start_iso)

if prices.empty:
    st.error("Aucune donnée de prix n'a pu être téléchargée. Vérifiez votre connexion ou les tickers.")
    st.stop()

# ---- EUR/USD : sans taux de change, les lignes USD sont exclues proprement
fx_series = load_fx(start_iso)
if fx_series is None:
    st.warning("⚠️ Taux EUR/USD indisponible : les lignes cotées en USD sont exclues du portefeuille.")
    for t in USD_TICKERS:
        if t in prices.columns:
            prices = prices.drop(columns=t)
            report[t].update(status="failed", note="Taux EUR/USD indisponible")

# ---- contrôles qualité sur les lignes retenues
last_global = prices.index[-1]
for t in prices.columns:
    r = report[t]
    if (last_global - r["last"]).days > STALE_DAYS:
        r.update(status="stale",
                 note=f"Cours figé depuis le {r['last']:%d/%m/%Y} (dernier cours reporté)")
    elif (r["first"] - pd.Timestamp(start_iso)).days > LATE_START_DAYS:
        r.update(status="partial",
                 note=f"Historique depuis le {r['first']:%d/%m/%Y} (rendement 0 % avant)")

# ---- poids effectifs : lignes exclues réparties au prorata, poids cash conservé
excluded = [t for t in tickers if t not in prices.columns]
kept = [t for t in tickers if t in prices.columns]
total_target = sum(allocation.values())
w_kept = pd.Series({t: allocation[t] for t in kept})
weights = w_kept * (total_target / w_kept.sum())
allocation_eff = weights.to_dict()
excluded_names = [ticker_names.get(t, t) for t in excluded]

if excluded:
    lost = sum(allocation[t] for t in excluded) * 100
    st.warning(
        f"⚠️ {len(excluded)} ligne(s) exclue(s) faute de données exploitables : "
        f"{', '.join(excluded_names)} ({lost:.1f}% du poids cible). "
        "Leur poids est réparti au prorata sur les autres lignes ; "
        "la mention figure dans la factsheet."
    )
issues = [t for t in kept if report[t]["status"] in ("stale", "partial")]
if issues:
    st.info("ℹ️ Lignes à surveiller : " + " ; ".join(
        f"{ticker_names.get(t, t)} ({report[t]['note']})" for t in issues))
if not excluded and not issues:
    st.success("Tous les actifs ont été téléchargés avec succès !")

with st.expander("🔍 Qualité des données et poids effectifs du portefeuille"):
    icons = {"ok": "✅ Disponible", "partial": "⚠️ Historique partiel",
             "stale": "⚠️ Cours figé", "failed": "❌ Exclu"}
    df_weights = pd.DataFrame({
        "Actif": [ticker_names.get(t, t) for t in tickers],
        "Ticker": tickers,
        "Source": [report[t]["source"] for t in tickers],
        "Statut": [icons[report[t]["status"]] for t in tickers],
        "Poids cible": [f"{allocation[t] * 100:.1f}%" for t in tickers],
        "Poids effectif": [f"{weights[t] * 100:.1f}%" if t in weights.index else "0.0%" for t in tickers],
        "Détail": [report[t]["note"] for t in tickers],
    })
    st.dataframe(df_weights, use_container_width=True, hide_index=True)
    st.caption(f"Total investi : {weights.sum() * 100:.1f}% "
               f"(cible {total_target * 100:.1f}%, le solde est conservé en liquidités).")

# =====================
# NAV portefeuille
# =====================
# FX aligné sur le calendrier des cours par propagation du dernier taux connu
# (les jours fériés US / EU diffèrent : une jointure "inner" supprimait des dates).
fx_aligned = None
if fx_series is not None:
    fx_aligned = (fx_series.reindex(prices.index.union(fx_series.index))
                  .ffill().reindex(prices.index).ffill().bfill())

usd_ok = [t for t in USD_TICKERS if t in prices.columns]
prices_eur = prices.copy()
for t in usd_ok:
    prices_eur[t] = prices[t] / fx_aligned
prices_eur = prices_eur.ffill()

returns = pct(prices_eur).fillna(0)[weights.index]
portfolio_returns = (returns * weights).sum(axis=1)
portfolio_index = (1 + portfolio_returns).cumprod()

hedged_returns = returns.copy()
if usd_ok:
    fx_ret = pct(fx_aligned).fillna(0)
    for t in usd_ok:
        asset_ret_usd = pct(prices[t]).fillna(0)
        hedged_returns[t] = asset_ret_usd - fx_ret
portfolio_returns_hedged = (hedged_returns * weights).sum(axis=1)
portfolio_index_hedged = (1 + portfolio_returns_hedged).cumprod()

# =====================
# Benchmark
# =====================
def load_benchmark_composite(start_iso):
    """Retourne (indice base 1, poids effectifs, tickers en échec)."""
    series, failed = {}, []
    for tk in BENCHMARK_WEIGHTS:
        s, _ = download_single_ticker(tk, start_iso)
        if s is None or len(s) < MIN_OBS:
            failed.append(tk)
        else:
            series[tk] = s
    if not series:
        return None, {}, failed
    px = pd.concat(series, axis=1).sort_index().ffill()
    w = pd.Series({k: BENCHMARK_WEIGHTS[k] for k in px.columns})
    w = w * (sum(BENCHMARK_WEIGHTS.values()) / w.sum())   # le poids total (cash inclus) est conservé
    ret = pct(px).fillna(0)
    return (1 + (ret * w).sum(axis=1)).cumprod(), w.to_dict(), failed


bench_index, bench_weights_eff, bench_failed = load_benchmark_composite(start_iso)
if bench_index is None:
    st.error("Aucune donnée de benchmark n'a pu être téléchargée.")
    st.stop()
if bench_failed:
    st.warning("⚠️ Composantes du benchmark indisponibles et exclues : "
               f"{', '.join(bench_failed)}. Les autres poids sont réajustés au prorata.")

# ---- index temporels normalisés (sans fuseau) avant toute intersection
portfolio_index = normalize_index(portfolio_index)
portfolio_index_hedged = normalize_index(portfolio_index_hedged)
bench_index = normalize_index(bench_index)

common_index = portfolio_index.index.intersection(bench_index.index)
if len(common_index) < MIN_OBS:
    st.error("Historique commun portefeuille / benchmark insuffisant "
             f"({len(common_index)} dates). Reculez la date de début.")
    st.stop()

# rebasage à 1 sur la fenêtre commune
portfolio_index = portfolio_index.loc[common_index] / portfolio_index.loc[common_index].iloc[0]
portfolio_index_hedged = portfolio_index_hedged.loc[common_index] / portfolio_index_hedged.loc[common_index].iloc[0]
bench_index = bench_index.loc[common_index] / bench_index.loc[common_index].iloc[0]

# =====================
# FONCTIONS DE CALCUL
# =====================
def calculate_performance(index_series, days):
    if len(index_series) < 2:
        return 0.0
    start_date = index_series.index[-1] - timedelta(days=days)
    if start_date < index_series.index[0]:
        start_date = index_series.index[0]
    start_value = index_series[index_series.index >= start_date].iloc[0]
    end_value = index_series.iloc[-1]
    return (end_value / start_value - 1) * 100

def calculate_volatility(returns_series, periods=252):
    return returns_series.std() * np.sqrt(periods)

def calculate_sharpe_ratio(returns_series, risk_free_rate=0.04, periods=252):
    excess_returns = returns_series - (risk_free_rate / periods)
    return (excess_returns.mean() * periods) / (returns_series.std() * np.sqrt(periods))

def calculate_sortino_ratio(returns_series, risk_free_rate=0.04, periods=252):
    excess_returns = returns_series - (risk_free_rate / periods)
    downside_returns = returns_series[returns_series < 0]
    downside_std = downside_returns.std() * np.sqrt(periods)
    if downside_std == 0:
        return 0
    return (excess_returns.mean() * periods) / downside_std

def calculate_max_drawdown(index_series):
    cummax = index_series.cummax()
    drawdown = (index_series - cummax) / cummax
    return drawdown.min() * 100

def calculate_calmar_ratio(index_series, returns_series, periods=252):
    annual_return = (index_series.iloc[-1] / index_series.iloc[0] - 1) * (periods / len(index_series))
    max_dd = abs(calculate_max_drawdown(index_series)) / 100
    if max_dd == 0:
        return 0
    return annual_return / max_dd

def calculate_beta(portfolio_returns, benchmark_returns):
    covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
    variance = np.var(benchmark_returns)
    return covariance / variance if variance != 0 else 0

def calculate_alpha(portfolio_return, benchmark_return, beta, risk_free_rate=0.04):
    return portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))

# =====================
# CALCUL DES METRIQUES
# =====================
port_ret = pct(portfolio_index).dropna()
port_ret_hedged = pct(portfolio_index_hedged).dropna()
bench_ret = pct(bench_index).dropna()

ytd_days = (datetime.now() - datetime(datetime.now().year, 1, 1)).days
perf_1y_port = calculate_performance(portfolio_index, 365)
perf_1y_bench = calculate_performance(bench_index, 365)
perf_3y_port = calculate_performance(portfolio_index, 3 * 365)
perf_3y_bench = calculate_performance(bench_index, 3 * 365)
perf_ytd_port = calculate_performance(portfolio_index, ytd_days)
perf_ytd_bench = calculate_performance(bench_index, ytd_days)

vol_port = calculate_volatility(port_ret)
vol_bench = calculate_volatility(bench_ret)
vol_port_hedged = calculate_volatility(port_ret_hedged)

sharpe_port = calculate_sharpe_ratio(port_ret)
sharpe_bench = calculate_sharpe_ratio(bench_ret)
sharpe_hedged = calculate_sharpe_ratio(port_ret_hedged)

sortino_port = calculate_sortino_ratio(port_ret)
sortino_bench = calculate_sortino_ratio(bench_ret)

mdd_port = calculate_max_drawdown(portfolio_index)
mdd_bench = calculate_max_drawdown(bench_index)
mdd_hedged = calculate_max_drawdown(portfolio_index_hedged)

common_dates = port_ret.index.intersection(bench_ret.index)
beta = calculate_beta(port_ret[common_dates].values, bench_ret[common_dates].values)
alpha = calculate_alpha(perf_1y_port / 100, perf_1y_bench / 100, beta) * 100

calmar_port = calculate_calmar_ratio(portfolio_index, port_ret)
calmar_bench = calculate_calmar_ratio(bench_index, bench_ret)

# =====================
# Affichage des graphiques
# =====================
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_index.index, y=portfolio_index, name="Portfolio", line=dict(width=3, color='#1f3a93')))
fig.add_trace(go.Scatter(x=bench_index.index, y=bench_index, name="Benchmark composite", line=dict(width=3, color='#d32f2f')))
fig.add_trace(go.Scatter(x=portfolio_index_hedged.index, y=portfolio_index_hedged, name="Portfolio hedgé USD", line=dict(width=3, dash="dot", color="#2ecc71")))
fig.update_layout(height=600, template="plotly_white", title="Performance cumulée", hovermode='x unified')
st.plotly_chart(fig, use_container_width=True)

# Top 5
st.subheader("🏆 Les 5 lignes les plus performantes sur le dernier mois")
month_start = prices_eur.index[-1] - timedelta(days=30)
prices_month = prices_eur[prices_eur.index >= month_start].ffill()
if len(prices_month) >= 2:
    monthly_perf = ((prices_month.iloc[-1] / prices_month.iloc[0] - 1) * 100).dropna()
    monthly_perf = monthly_perf.sort_values(ascending=False)
    top5 = monthly_perf.head(5)
    col_bar, col_line = st.columns(2)
    top5_labels = [ticker_names.get(t, t) for t in top5.index]
    with col_bar:
        bar_colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in top5.values]
        fig_bar = go.Figure(go.Bar(
            x=top5_labels,
            y=top5.values,
            marker_color=bar_colors,
            text=[f"{v:.2f}%" for v in top5.values],
            textposition='outside'
        ))
        fig_bar.update_layout(
            title="Performance mensuelle (%)",
            template="plotly_white",
            yaxis_title="%",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_line:
        fig_line = go.Figure()
        for ticker in top5.index:
            series = prices_month[ticker].dropna()
            if len(series) >= 2:
                normalized = (series / series.iloc[0]) * 100
                fig_line.add_trace(go.Scatter(
                    x=normalized.index,
                    y=normalized,
                    name=ticker_names.get(ticker, ticker),
                    mode='lines',
                    line=dict(width=2)
                ))
        fig_line.update_layout(
            title="Évolution normalisée sur le mois (base 100)",
            template="plotly_white",
            yaxis_title="Base 100",
            height=400
        )
        st.plotly_chart(fig_line, use_container_width=True)

st.subheader("📈 Performances")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Portefeuille**")
    st.metric("Perf YTD", f"{perf_ytd_port:.2f}%")
    st.metric("Perf 1 an", f"{perf_1y_port:.2f}%")
    st.metric("Perf 3 ans", f"{perf_3y_port:.2f}%")
with col2:
    st.markdown("**Portefeuille Hedgé USD**")
    st.metric("Volatilité", f"{vol_port_hedged:.2f}%")
    st.metric("Sharpe", f"{sharpe_hedged:.2f}")
    st.metric("Max Drawdown", f"{mdd_hedged:.2f}%")
with col3:
    st.markdown("**Benchmark**")
    st.metric("Perf YTD", f"{perf_ytd_bench:.2f}%")
    st.metric("Perf 1 an", f"{perf_1y_bench:.2f}%")
    st.metric("Perf 3 ans", f"{perf_3y_bench:.2f}%")

st.subheader("📊 Statistiques globales")
col1, col2, col3 = st.columns(3)
col1.metric("Perf depuis le début", f"{(portfolio_index.iloc[-1] - 1) * 100:.2f}%")
col2.metric("Volatilité (annualisée)", f"{vol_port:.2f}%")
col3.metric("Sharpe Ratio", f"{sharpe_port:.2f}")

st.caption("Mise à jour automatique toutes les heures")

# =====================
# FACTSHEET PDF (FR + EN, 1 clic, données temps réel)
# =====================
render_factsheet_section(
    portfolio_index_hedged, bench_index, port_ret_hedged,
    allocation_eff, ticker_names, weights, prices_eur,
    bench_weights=bench_weights_eff,
    excluded=excluded_names,
)
