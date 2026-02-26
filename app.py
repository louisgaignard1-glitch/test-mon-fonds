import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

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

# Convertir les clés du dictionnaire en liste pour éviter les problèmes de hash
tickers = list(allocation.keys())

start = st.sidebar.date_input("Start date", datetime(2020,1,1))

# =====================
# Chargement prix
# =====================
@st.cache_data(ttl=3600)
def load_prices(tickers, start):
    prices = pd.DataFrame()

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
        except Exception as e:
            st.write(f"Erreur lors du téléchargement des données pour {t}: {e}")

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
portfolio_returns = (returns * weights).sum(axis=1)
portfolio_index = (1 + portfolio_returns).cumprod()

# =====================
# Benchmark composite
# =====================
@st.cache_data(ttl=3600)
def load_benchmark_composite(start):
    benchmark_weights = {
    "EXSA.DE": 0.35,  # STOXX Europe 600
    "SPY": 0.20,      # S&P 500
    "AGG": 0.25,      # Bloomberg US Aggregate Bond (remplace AGGG.L)
    "EPRE.AS": 0.10,  # FTSE EPRA NAREIT Europe
    "EEM": 0.05,      # MSCI Emerging Markets
    # Supprime "EUR=X" ou remplace par un actif liquide (ex: "BIL" pour les liquidités US)
}


    prices = yf.download(list(benchmark_weights.keys()), start=start)["Adj Close"]
    prices = prices.fillna(method="ffill")

    weights = pd.Series(benchmark_weights)

    returns = prices.pct_change().fillna(0)
    bench_returns = (returns * weights).sum(axis=1)

    bench_index = (1 + bench_returns).cumprod()

    return bench_index

bench_index = load_benchmark_composite(start)

# =====================
# Texte explicatif benchmark
# =====================
st.subheader("📊 Composition du benchmark")

st.markdown("""
Le benchmark composite reflète la structure multi-actifs du portefeuille :

• 35% STOXX Europe 600 → actions européennes
• 20% S&P 500 → actions américaines
• 25% Bloomberg Global Aggregate → obligations globales
• 10% FTSE EPRA NAREIT Europe → immobilier coté
• 5% MSCI Emerging Markets → actions émergentes
• 5% Cash proxy → liquidités

Ce benchmark permet une comparaison plus réaliste qu’un indice actions pur.
""")

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
    line=dict(width=3, dash="dash")
))

fig.update_layout(
    height=600,
    template="plotly_white",
    title="Performance cumulée"
)

st.plotly_chart(fig, use_container_width=True)

# =====================
# Metrics
# =====================
st.subheader("📈 Statistiques")

col1, col2 = st.columns(2)

col1.metric("Perf portefeuille", f"{(portfolio_index.iloc[-1]-1)*100:.2f}%")
col2.metric("Perf benchmark", f"{(bench_index.iloc[-1]-1)*100:.2f}%")

st.caption("Mise à jour automatique toutes les heures")
