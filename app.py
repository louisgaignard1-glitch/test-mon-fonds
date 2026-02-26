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
portfolio_returns = (returns * weights).sum(axis=1)
portfolio_index = (1 + portfolio_returns).cumprod()

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

        prices = prices.fillna(method="ffill
