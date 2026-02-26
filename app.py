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
    "LU0912261970": 0.08,
    "LU1331974276": 0.08,
    "FR0007008750": 0.09,
    "LU0292585626": 0.08,
    "FR0010541821": 0.05,
    "FR0011268705": 0.08
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
        st.warning(f"Impossible de récupérer les prix pour : {', '.join(failed_tickers)}")

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

returns = prices.pct_change().fillna(0)

# NAV non hedgée
portfolio_returns = (returns * weights).sum(axis=1)
portfolio_index = (1 + portfolio_returns).cumprod()

# =====================
# Hedge FX USD via forwards (simulation robuste)
# =====================
usd_tickers = ["UBER", "GOOGL", "META", "HWM", "AMZN"]

# téléchargement FX
fx = yf.download("EURUSD=X", start=start, auto_adjust=True)

hedged_returns = returns.copy()

if not fx.empty:

    # S'assurer qu'on récupère une vraie Series
    if isinstance(fx, pd.DataFrame):
        fx_series = fx.iloc[:, 0]
    else:
        fx_series = fx

    # Forcer index datetime
    fx_series.index = pd.to_datetime(fx_series.index)
    returns.index = pd.to_datetime(returns.index)

    # Alignement STRICT sur index portefeuille
    fx_series = fx_series.reindex(returns.index)

    # Remplissage sécurisé
    fx_series = fx_series.ffill().bfill()

    # Calcul return FX
    fx_returns = fx_series.pct_change()
    fx_returns = fx_returns.fillna(0)

    # Conversion explicite en numpy array (anti bug pandas 3.13)
    fx_array = fx_returns.to_numpy()

    # Hedge ticker par ticker
    for t in usd_tickers:
        if t in hedged_returns.columns:
            hedged_returns[t] = hedged_returns[t].to_numpy() - fx_array

# NAV hedgée
portfolio_returns_hedged = (hedged_returns * weights).sum(axis=1)
portfolio_index_hedged = (1 + portfolio_returns_hedged).cumprod()
# =====================
# Benchmark composite
# =====================
@st.cache_data(ttl=3600)
def load_benchmark_composite(start):
    benchmark_weights = {
        "IEV": 0.35,    # ETF MSCI Europe Index (alternative à STOXX Europe 600)
        "SPY": 0.20,     # S&P 500
        "TLT": 0.25,     # Obligations américaines à long terme
        "VNQ": 0.10,     # Immobilier américain
        "EEM": 0.05,     # MSCI Emerging Markets
    }

    try:
        prices = pd.DataFrame()
        for ticker, weight in benchmark_weights.items():
            try:
                tmp = yf.download(ticker, start=start)
                if not tmp.empty:
                    if "Adj Close" in tmp.columns:
                        prices[ticker] = tmp["Adj Close"]
                    elif "Close" in tmp.columns:
                        prices[ticker] = tmp["Close"]
                    else:
                        st.warning(f"Aucune colonne 'Adj Close' ou 'Close' trouvée pour {ticker}")
                else:
                    st.warning(f"Aucune donnée trouvée pour {ticker}")
            except Exception as e:
                st.warning(f"Erreur lors du téléchargement des données pour {ticker}: {e}")

        if prices.empty:
            st.error("Aucune donnée disponible pour le benchmark. Vérifiez les tickers ou la date.")
            return pd.Series([1.0], index=[pd.to_datetime("today")])

        prices = prices.fillna(method="ffill")
        weights = pd.Series(benchmark_weights)

        returns = prices.pct_change().fillna(0)
        bench_returns = (returns * weights).sum(axis=1)
        bench_index = (1 + bench_returns).cumprod()

        return bench_index

    except Exception as e:
        st.error(f"Erreur lors du chargement du benchmark : {e}")
        return pd.Series([1.0], index=[pd.to_datetime("today")])

# Charge les données du benchmark
bench_index = load_benchmark_composite(start)

# Vérifie que bench_index est valide
if bench_index is None or bench_index.empty or not hasattr(bench_index, 'index'):
    st.error("Erreur : Impossible de calculer le benchmark. Vérifiez les données.")
    st.stop()

# =====================
# Graphique
# =====================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=portfolio_index.index,
    y=portfolio_index,
    name="Portfolio",
    line=dict(width=3)
))

fig.add_trace(go.Scatter(
    x=bench_index.index,
    y=bench_index,
    name="Benchmark composite",
    line=dict(width=3)  # Ligne continue
))
fig.add_trace(go.Scatter(
    x=portfolio_index_hedged.index,
    y=portfolio_index_hedged,
    name="Portfolio hedgé USD",
    line=dict(width=3, dash="dot")
))
fig.update_layout(
    height=600,
    template="plotly_white",
    title="Performance cumulée"
)

st.plotly_chart(fig, use_container_width=True)

# =====================
# Texte explicatif benchmark
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
Une simulation de couverture du risque dollar est appliquée via des contrats à terme FX (forwards).

Les actions américaines sont couvertes en neutralisant la variation EUR/USD :

Return hedgé ≈ Return action USD − Return EURUSD

Cette approche simule un hedge forward à 100% sans coût de carry.
""")
# =====================
# Calcul des performances
# =====================

# Fonction pour calculer la performance sur une période donnée
def calculate_performance(index_series, days):
    if len(index_series) < 2:
        return 0.0
    if days == 1:  # Performance de la veille
        if len(index_series) >= 2:
            return (index_series.iloc[-1] / index_series.iloc[-2] - 1) * 100
        else:
            return 0.0
    else:
        start_date = index_series.index[-1] - timedelta(days=days)
        if start_date < index_series.index[0]:
            start_date = index_series.index[0]
        start_value = index_series[index_series.index >= start_date].iloc[0]
        end_value = index_series.iloc[-1]
        return (end_value / start_value - 1) * 100

# Calcul des performances pour le portefeuille
portfolio_perf_yesterday = calculate_performance(portfolio_index, 1)
portfolio_perf_1y = calculate_performance(portfolio_index, 365)
portfolio_perf_3y = calculate_performance(portfolio_index, 3*365)

# Calcul des performances pour le benchmark
benchmark_perf_yesterday = calculate_performance(bench_index, 1)
benchmark_perf_1y = calculate_performance(bench_index, 365)
benchmark_perf_3y = calculate_performance(bench_index, 3*365)

# =====================
# Affichage des performances
# =====================
st.subheader("📈 Performances")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Portefeuille**")
    st.metric("Perf de la veille", f"{portfolio_perf_yesterday:.2f}%")
    st.metric("Perf sur 1 an", f"{portfolio_perf_1y:.2f}%")
    st.metric("Perf sur 3 ans", f"{portfolio_perf_3y:.2f}%")

with col2:
    st.markdown("**Benchmark**")
    st.metric("Perf de la veille", f"{benchmark_perf_yesterday:.2f}%")
    st.metric("Perf sur 1 an", f"{benchmark_perf_1y:.2f}%")
    st.metric("Perf sur 3 ans", f"{benchmark_perf_3y:.2f}%")

# =====================
# Metrics globales
# =====================
st.subheader("📊 Statistiques globales")

col1, col2 = st.columns(2)

col1.metric("Perf portefeuille (depuis début)", f"{(portfolio_index.iloc[-1]-1)*100:.2f}%")
col2.metric("Perf benchmark (depuis début)", f"{(bench_index.iloc[-1]-1)*100:.2f}%")

st.caption("Mise à jour automatique toutes les heures")
