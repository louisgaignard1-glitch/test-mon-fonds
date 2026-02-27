import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Portfolio vs Benchmark", layout="wide")
st.title("📊 Portfolio vs Benchmark composite")

# =====================
# Allocation portefeuille
# =====================
allocation = {
    "TTE.PA": 0.05,
    "MC.PA": 0.05,
    "INGA.AS": 0.05,
    "SAP.DE": 0.04,
    "ACLN.SW": 0.05,
    "UBER": 0.04,
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
    "0P0000WHLW.F": 0.08
}

tickers = list(allocation.keys())
start = st.sidebar.date_input("Start date", datetime(2020, 1, 1))

# =====================
# Chargement prix
# =====================
@st.cache_data(ttl=3600)
def load_prices(tickers, start):
    prices = pd.DataFrame()
    failed_tickers = []
    for t in tickers:
        try:
            tmp = yf.download(t, start=start)
            if not tmp.empty:
                if "Adj Close" in tmp.columns:
                    prices[t] = tmp["Adj Close"]
                elif "Close" in tmp.columns:
                    prices[t] = tmp["Close"]
                else:
                    st.write(f"Aucune colonne 'Adj Close' ou 'Close' trouvée pour {t}")
                    failed_tickers.append(t)
            else:
                failed_tickers.append(t)
        except Exception as e:
            st.write(f"Erreur lors du téléchargement des données pour {t}: {e}")
            failed_tickers.append(t)
    if failed_tickers:
        st.warning(f"Les tickers suivants n'ont pas pu être téléchargés : {', '.join(failed_tickers)}. Vérifiez leur validité.")
    return prices

prices = load_prices(tickers, start)
if prices.empty:
    st.error("Aucune donnée de prix n'a pu être téléchargée. Vérifiez les tickers ou votre connexion Internet.")
    st.stop()

# =====================
# Construction portefeuille
# =====================
weights = pd.Series(allocation)
weights = weights[weights.index.isin(prices.columns)]
weights = weights / weights.sum()   
if weights.empty:
    st.error("Aucun poids valide n'a pu être calculé. Vérifiez les tickers et les allocations.")
    st.stop()

usd_tickers = ["UBER", "GOOGL", "META", "HWM", "AMZN"]

# =====================
# Télécharger EUR/USD
# =====================
fx = yf.download("EURUSD=X", start=start)
if fx.empty:
    st.error("Impossible de télécharger les données EUR/USD.")
    st.stop()

if "Adj Close" in fx.columns:
    fx_series = fx["Adj Close"]
elif "Close" in fx.columns:
    fx_series = fx["Close"]
else:
    fx_series = fx.iloc[:, 0]
fx_series.index = pd.to_datetime(fx_series.index)

# =====================
# NAV portefeuille non hedgé
# =====================
prices_eur = prices.copy()
for t in usd_tickers:
    if t in prices.columns:
        combined = pd.concat([prices[t], fx_series], axis=1, join='inner').ffill()
        if len(combined.columns) == 2:
            combined.columns = ['price', 'fx']
            prices_eur[t] = combined['price'] * (1 / combined['fx'])
        else:
            st.warning(f"La structure des données pour {t} n'est pas celle attendue. Vérifiez les colonnes.")

if prices_eur.empty:
    st.error("Aucune donnée de prix en EUR n'a pu être calculée.")
    st.stop()

returns = prices_eur.pct_change().fillna(0)
portfolio_returns = (returns * weights).sum(axis=1)
portfolio_index = (1 + portfolio_returns).cumprod()
if portfolio_index.empty:
    st.error("Impossible de calculer l'indice du portefeuille.")
    st.stop()

# =====================
# NAV portefeuille hedgé
# =====================
hedged_prices = prices.copy()
for t in usd_tickers:
    if t in prices.columns:
        combined = pd.concat([prices[t], fx_series], axis=1, join='outer').ffill()
        if len(combined.columns) == 2:
            combined.columns = ['price', 'fx']
            hedged_prices[t] = combined['price']
        else:
            st.warning(f"La structure des données pour {t} n'est pas celle attendue. Vérifiez les colonnes.")

if hedged_prices.empty:
    st.error("Aucune donnée de prix hedgé n'a pu être calculée.")
    st.stop()

hedged_returns = hedged_prices.pct_change().fillna(0)
portfolio_returns_hedged = (hedged_returns * weights).sum(axis=1)
portfolio_index_hedged = (1 + portfolio_returns_hedged).cumprod()
if portfolio_index_hedged.empty:
    st.error("Impossible de calculer l'indice du portefeuille hedgé.")
    st.stop()

# =====================
# Benchmark composite
# =====================
@st.cache_data(ttl=3600)
def load_benchmark_composite(start):
    benchmark_weights = {
        "IEV": 0.35,
        "SPY": 0.20,
        "TLT": 0.25,
        "VNQ": 0.10,
        "EEM": 0.05
    }
    prices = pd.DataFrame()
    for ticker in benchmark_weights.keys():
        tmp = yf.download(ticker, start=start)
        if not tmp.empty:
            if "Adj Close" in tmp.columns:
                prices[ticker] = tmp["Adj Close"]
            elif "Close" in tmp.columns:
                prices[ticker] = tmp["Close"]
    if prices.empty:
        st.error("Aucune donnée de benchmark n'a pu être téléchargée.")
        st.stop()
    prices = prices.fillna(method="ffill")
    weights = pd.Series(benchmark_weights)
    returns = prices.pct_change().fillna(0)
    bench_returns = (returns * weights).sum(axis=1)
    return (1 + bench_returns).cumprod()

bench_index = load_benchmark_composite(start)
if bench_index.empty:
    st.error("Impossible de calculer le benchmark.")
    st.stop()

common_index = portfolio_index.index.intersection(bench_index.index)

portfolio_index = portfolio_index.loc[common_index]
portfolio_index_hedged = portfolio_index_hedged.loc[common_index]
bench_index = bench_index.loc[common_index]
# =====================
# Graphique
# =====================
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=portfolio_index.index,
    y=portfolio_index,
    name="Portfolio",
    line=dict(width=3, color='blue')
))
fig.add_trace(go.Scatter(
    x=bench_index.index,
    y=bench_index,
    name="Benchmark composite",
    line=dict(width=3, color='red')
))
fig.add_trace(go.Scatter(
    x=portfolio_index_hedged.index,
    y=portfolio_index_hedged,
    name="Portfolio hedgé USD",
    line=dict(width=3, dash="dot", color="green")
))
fig.update_layout(
    height=600,
    template="plotly_white",
    title="Performance cumulée"
)
st.plotly_chart(fig, use_container_width=True)

# =====================
# Texte explicatif 
# =====================
st.subheader("📊 Composition du benchmark")
st.markdown("""
Le benchmark composite reflète la structure multi-actifs du portefeuille :
• 35% MSCI Europe Index (IEV) → actions européennes
• 20% S&P 500 → actions américaines
• 25% Obligations américaines à long terme → obligations
• 10% Immobilier américain → immobilier
• 5% MSCI Emerging Markets → actions émergentes
Ce benchmark permet une comparaison plus réaliste qu’un indice actions pur.
""")
st.subheader("💱 Couverture FX USD")

st.markdown("""
Cette simulation couvre le risque de change des actions américaines (ex: UBER, GOOGL) en utilisant un **contrat forward** pour figer le taux EUR/USD.

**Formule appliquée :**
Return hedgé = Return en USD − Variation du taux EUR/USD

→ Cela neutralise l’impact des fluctuations du change, comme si vous aviez verrouillé le taux de change initial.
*(Simplification : pas de coût de couverture inclus.)*
""")


# =====================
# Calcul des performances
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

portfolio_perf_yesterday = calculate_performance(portfolio_index, 1)
portfolio_perf_1y = calculate_performance(portfolio_index, 365)
portfolio_perf_3y = calculate_performance(portfolio_index, 3*365)

portfolio_hedged_perf_yesterday = calculate_performance(portfolio_index_hedged, 1)
portfolio_hedged_perf_1y = calculate_performance(portfolio_index_hedged, 365)
portfolio_hedged_perf_3y = calculate_performance(portfolio_index_hedged, 3*365)

benchmark_perf_yesterday = calculate_performance(bench_index, 1)
benchmark_perf_1y = calculate_performance(bench_index, 365)
benchmark_perf_3y = calculate_performance(bench_index, 3*365)

# =====================
# Affichage des performances
# =====================
st.subheader("📈 Performances")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Portefeuille**")
    st.metric("Perf de la veille", f"{portfolio_perf_yesterday:.2f}%")
    st.metric("Perf sur 1 an", f"{portfolio_perf_1y:.2f}%")
    st.metric("Perf sur 3 ans", f"{portfolio_perf_3y:.2f}%")
with col2:
    st.markdown("**Portefeuille Hedgé USD**")
    st.metric("Perf de la veille", f"{portfolio_hedged_perf_yesterday:.2f}%")
    st.metric("Perf sur 1 an", f"{portfolio_hedged_perf_1y:.2f}%")
    st.metric("Perf sur 3 ans", f"{portfolio_hedged_perf_3y:.2f}%")
with col3:
    st.markdown("**Benchmark**")
    st.metric("Perf de la veille", f"{benchmark_perf_yesterday:.2f}%")
    st.metric("Perf sur 1 an", f"{benchmark_perf_1y:.2f}%")
    st.metric("Perf sur 3 ans", f"{benchmark_perf_3y:.2f}%")

# =====================
# Metrics globales
# =====================
st.subheader("📊 Statistiques globales")
col1, col2, col3 = st.columns(3)
col1.metric("Perf portefeuille (depuis le début)", f"{(portfolio_index.iloc[-1]-1)*100:.2f}%")
col2.metric("Perf portefeuille hedgé (depuis le début)", f"{(portfolio_index_hedged.iloc[-1]-1)*100:.2f}%")
col3.metric("Perf benchmark (depuis le début)", f"{(bench_index.iloc[-1]-1)*100:.2f}%")

st.caption("Mise à jour automatique toutes les heures")
