import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
import base64

# =====================
# LIBRAIRIES PDF
# =====================
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
import io

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
    "0P0000WHLW.F": 0.08
}

tickers = list(allocation.keys())
start = st.sidebar.date_input("Start date", datetime(2020, 1, 1))

# =====================
# Chargement des prix (avec solutions de repli)
# =====================
@st.cache_data(ttl=3600)
def download_single_ticker(ticker, start, max_retries=3):
    """Télécharge un ticker avec plusieurs tentatives et solutions de repli."""
    if ticker == "TTE.PA":
        for attempt in range(max_retries):
            try:
                tmp = yf.download(ticker, start=start, auto_adjust=False, progress=False)
                if not tmp.empty:
                    return tmp
            except:
                pass
            try:
                ticker_obj = yf.Ticker(ticker)
                tmp = ticker_obj.history(start=start)
                if not tmp.empty:
                    return tmp
            except:
                pass
            try:
                 tmp = yf.download("FP.PA", start=start, auto_adjust=False, progress=False)
                 if not tmp.empty:
                     return tmp
            except:
                pass
                
    else:
        for attempt in range(max_retries):
            try:
                tmp = yf.download(ticker, start=start, auto_adjust=False, progress=False)
                if not tmp.empty:
                    return tmp
            except:
                pass
            try:
                ticker_obj = yf.Ticker(ticker)
                tmp = ticker_obj.history(start=start)
                if not tmp.empty:
                    return tmp
            except:
                pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_prices(tickers, start):
    prices = pd.DataFrame()
    failed_tickers = []
    for t in tickers:
        tmp = download_single_ticker(t, start)
        if tmp.empty:
            failed_tickers.append(t)
            prices[t] = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        else:
            if isinstance(tmp.columns, pd.MultiIndex):
                tmp.columns = tmp.columns.get_level_values(0)
            if "Adj Close" in tmp.columns:
                prices[t] = tmp["Adj Close"]
            elif "Close" in tmp.columns:
                prices[t] = tmp["Close"]
            prices[t].index = pd.to_datetime(prices[t].index)
    prices = prices.ffill()
    if not failed_tickers:
        st.success("Tous les actifs ont été téléchargés avec succès !")
    else:
        st.warning(f"Les tickers suivants n'ont pas pu être téléchargés : {', '.join(failed_tickers)}.")
    return prices, failed_tickers

prices, failed_tickers = load_prices(tickers, start)

if prices.empty:
    st.error("Aucune donnée de prix n'a pu être téléchargée.")
    st.stop()

# =====================
# Construction portefeuille
# =====================
weights = pd.Series(allocation)

# =====================
# Tableau de contrôle des poids
# =====================
with st.expander("🔍 Vérification des poids effectifs du portefeuille"):
    ticker_names_display = {
        "TTE.PA": "TotalEnergies", "MC.PA": "LVMH", "INGA.AS": "ING Groep",
        "SAP.DE": "SAP", "ACLN.SW": "ACLN", "THEON.AS": "Theon Intl",
        "BOI.PA": "Boiron", "EOAN.DE": "E.ON", "GOOGL": "Alphabet",
        "META": "Meta", "HWM": "Howmet", "AMZN": "Amazon",
        "0P0000ZWX4.F": "Helium Fund Perf A EUR",
        "0P0001861S.F": "Eleva Abs Ret Eurp S EUR",
        "0P00000M6C.F": "R-co Conviction Credit Euro",
        "0P00008ESK.F": "AXAIMFIIS US Short Dur HY",
        "0P0000A6ZG.F": "Immobilier 21 AC",
        "0P0000WHLW.F": "GemEquity R",
    }
    df_weights = pd.DataFrame({
        "Actif": [ticker_names_display.get(t, t) for t in weights.index],
        "Ticker": weights.index,
        "Poids cible": [f"{weights[t]*100:.1f}%" for t in weights.index],
        "Statut": ["✅ Disponible" if t not in failed_tickers else "❌ Non téléchargé" for t in weights.index],
    }).reset_index(drop=True)
    st.dataframe(df_weights, use_container_width=True)
    total_effectif = weights.sum() * 100
    st.caption(f"Total poids effectif : {total_effectif:.1f}% (identique aux poids cibles)")

# =====================
# Télécharger EUR/USD
# =====================
fx = yf.download("EURUSD=X", start=start, auto_adjust=False, progress=False)
if fx.empty:
    st.error("Impossible de télécharger les données EUR/USD.")
    st.stop()
if isinstance(fx.columns, pd.MultiIndex):
    fx.columns = fx.columns.get_level_values(0)
if "Adj Close" in fx.columns:
    fx_series = fx["Adj Close"]
elif "Close" in fx.columns:
    fx_series = fx["Close"]
else:
    fx_series = fx.iloc[:, 0]
fx_series.index = pd.to_datetime(fx_series.index)
fx_series = fx_series.sort_index().drop_duplicates()

# =====================
# NAV portefeuille non hedgé
# =====================
prices_eur = prices.copy()
usd_tickers = ["GOOGL", "META", "HWM", "AMZN"]
for t in usd_tickers:
    if t in prices.columns and t not in failed_tickers:
        prices_t = prices[t].copy()
        prices_t.index = pd.to_datetime(prices_t.index)
        prices_t = prices_t.sort_index().drop_duplicates()
        combined = pd.concat(
            [prices_t, fx_series],
            axis=1,
            join='inner'
        ).ffill()
        if len(combined.columns) == 2:
            combined.columns = ['price', 'fx']
            prices_eur[t] = combined['price'] * (1 / combined['fx'])
prices_eur = prices_eur.ffill()
returns = prices_eur.pct_change().fillna(0)
portfolio_returns = (returns * weights).sum(axis=1)
portfolio_index = (1 + portfolio_returns).cumprod()

# =====================
# NAV portefeuille hedgé
# =====================
hedged_returns = returns.copy()
for t in usd_tickers:
    if t in prices.columns and t not in failed_tickers:
        asset_ret_usd = prices[t].pct_change().fillna(0)
        fx_ret = fx_series.pct_change().fillna(0)
        combined_ret = pd.DataFrame({'asset': asset_ret_usd, 'fx': fx_ret}).ffill().fillna(0)
        hedged_returns[t] = combined_ret['asset'] - combined_ret['fx']
portfolio_returns_hedged = (hedged_returns * weights).sum(axis=1)
portfolio_index_hedged = (1 + portfolio_returns_hedged).cumprod()

# =====================
# Benchmark composite
# =====================
@st.cache_data(ttl=3600)
def load_benchmark_composite(start):
    benchmark_weights = {
        "IEV": 0.35, "SPY": 0.20, "TLT": 0.25, "VNQ": 0.10, "EEM": 0.05
    }
    prices = pd.DataFrame()
    for ticker in benchmark_weights.keys():
        tmp = yf.download(ticker, start=start, auto_adjust=False, progress=False)
        if not tmp.empty:
            if isinstance(tmp.columns, pd.MultiIndex):
                tmp.columns = tmp.columns.get_level_values(0)
            if "Adj Close" in tmp.columns:
                prices[ticker] = tmp["Adj Close"]
            elif "Close" in tmp.columns:
                prices[ticker] = tmp["Close"]
            prices[ticker].index = pd.to_datetime(prices[ticker].index)
    if prices.empty:
        st.error("Aucune donnée de benchmark n'a pu être téléchargée.")
        st.stop()
    prices = prices.ffill()
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
# FUNCTIONS DE CALCUL DE METRIQUES
# =====================

def calculate_performance(index_series, days):
    """Calcule la performance sur une période donnée"""
    if len(index_series) < 2:
        return 0.0
    start_date = index_series.index[-1] - timedelta(days=days)
    if start_date < index_series.index[0]:
        start_date = index_series.index[0]
    start_value = index_series[index_series.index >= start_date].iloc[0]
    end_value = index_series.iloc[-1]
    return (end_value / start_value - 1) * 100

def calculate_volatility(returns_series, periods=252):
    """Volatilité annualisée"""
    return returns_series.std() * np.sqrt(periods)

def calculate_sharpe_ratio(returns_series, risk_free_rate=0.04, periods=252):
    """Ratio de Sharpe (annualisé)"""
    excess_returns = returns_series - (risk_free_rate / periods)
    return (excess_returns.mean() * periods) / (returns_series.std() * np.sqrt(periods))

def calculate_sortino_ratio(returns_series, risk_free_rate=0.04, periods=252):
    """Ratio de Sortino (annualisé)"""
    excess_returns = returns_series - (risk_free_rate / periods)
    downside_returns = returns_series[returns_series < 0]
    downside_std = downside_returns.std() * np.sqrt(periods)
    if downside_std == 0:
        return 0
    return (excess_returns.mean() * periods) / downside_std

def calculate_max_drawdown(index_series):
    """Drawdown maximum"""
    cummax = index_series.cummax()
    drawdown = (index_series - cummax) / cummax
    return drawdown.min() * 100

def calculate_calmar_ratio(index_series, returns_series, periods=252):
    """Ratio de Calmar"""
    annual_return = (index_series.iloc[-1] / index_series.iloc[0] - 1) * (periods / len(index_series))
    max_dd = abs(calculate_max_drawdown(index_series)) / 100
    if max_dd == 0:
        return 0
    return annual_return / max_dd

def calculate_beta(portfolio_returns, benchmark_returns):
    """Beta du portefeuille par rapport au benchmark"""
    covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
    variance = np.var(benchmark_returns)
    return covariance / variance if variance != 0 else 0

def calculate_alpha(portfolio_return, benchmark_return, beta, risk_free_rate=0.04):
    """Alpha de Jensen"""
    return portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))

# =====================
# CALCUL DES METRIQUES
# =====================

# Returns pour métriques
port_ret = portfolio_returns.dropna()
port_ret_hedged = portfolio_returns_hedged.dropna()
bench_ret = (bench_index.pct_change()).dropna()

# Performances
perf_1y_port = calculate_performance(portfolio_index, 365)
perf_1y_bench = calculate_performance(bench_index, 365)
perf_3y_port = calculate_performance(portfolio_index, 3*365)
perf_3y_bench = calculate_performance(bench_index, 3*365)
perf_ytd_port = calculate_performance(portfolio_index, 
    (datetime.now() - datetime(datetime.now().year, 1, 1)).days)
perf_ytd_bench = calculate_performance(bench_index, 
    (datetime.now() - datetime(datetime.now().year, 1, 1)).days)

# Volatilités
vol_port = calculate_volatility(port_ret)
vol_bench = calculate_volatility(bench_ret)
vol_port_hedged = calculate_volatility(port_ret_hedged)

# Ratios
sharpe_port = calculate_sharpe_ratio(port_ret)
sharpe_bench = calculate_sharpe_ratio(bench_ret)
sharpe_hedged = calculate_sharpe_ratio(port_ret_hedged)

sortino_port = calculate_sortino_ratio(port_ret)
sortino_bench = calculate_sortino_ratio(bench_ret)

# Drawdown
mdd_port = calculate_max_drawdown(portfolio_index)
mdd_bench = calculate_max_drawdown(bench_index)
mdd_hedged = calculate_max_drawdown(portfolio_index_hedged)

# Beta & Alpha (align les index)
common_dates = port_ret.index.intersection(bench_ret.index)
beta = calculate_beta(port_ret[common_dates].values, bench_ret[common_dates].values)
alpha = calculate_alpha(perf_1y_port/100, perf_1y_bench/100, beta) * 100

# Calmar
calmar_port = calculate_calmar_ratio(portfolio_index, port_ret)
calmar_bench = calculate_calmar_ratio(bench_index, bench_ret)

# =====================
# Graphique principal
# =====================
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_index.index, y=portfolio_index, name="Portfolio", line=dict(width=3, color='blue')))
fig.add_trace(go.Scatter(x=bench_index.index, y=bench_index, name="Benchmark composite", line=dict(width=3, color='red')))
fig.add_trace(go.Scatter(x=portfolio_index_hedged.index, y=portfolio_index_hedged, name="Portfolio hedgé USD", line=dict(width=3, dash="dot", color="green")))
fig.update_layout(height=600, template="plotly_white", title="Performance cumulée")
st.plotly_chart(fig, use_container_width=True)

# =====================
# Top 5 performants sur le dernier mois
# =====================
st.subheader("🏆 Les 5 lignes les plus performantes sur le dernier mois")
ticker_names = {
    "TTE.PA": "TotalEnergies", "MC.PA": "LVMH", "INGA.AS": "ING Group",
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
month_start = prices_eur.index[-1] - timedelta(days=30)
prices_month = prices_eur[prices_eur.index >= month_start].ffill()
if len(prices_month) >= 2:
    monthly_perf = {}
    for t in prices_month.columns:
        if t in failed_tickers:
            monthly_perf[t] = 0.0
        else:
            monthly_perf[t] = (prices_month[t].iloc[-1] / prices_month[t].iloc[0] - 1) * 100
    monthly_perf = pd.Series(monthly_perf).sort_values(ascending=False)
    top5 = monthly_perf.head(5)
    col_bar, col_line = st.columns(2)
    top5_labels = [ticker_names.get(t, t) for t in top5.index]
    with col_bar:
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in top5.values]
        fig_bar = go.Figure(go.Bar(
            x=top5_labels,
            y=top5.values,
            marker_color=colors,
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
            if ticker in prices_month.columns:
                series = prices_month[ticker]
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
    st.markdown("**Détail du Top 5**")
    df_top5 = pd.DataFrame({
        "Actif": top5_labels,
        "Ticker": top5.index,
        "Performance mois": [f"{v:.2f}%" for v in top5.values],
        "Poids dans le portefeuille": [f"{weights[t]*100:.1f}%" for t in top5.index],
        "Statut": ["✅ Disponible" if t not in failed_tickers else "❌ Non téléchargé" for t in top5.index]
    }).reset_index(drop=True)
    st.dataframe(df_top5, use_container_width=True)
else:
    st.warning("Pas assez de données disponibles pour calculer le top 5 mensuel.")

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
""")
st.subheader("💱 Couverture FX USD")
st.markdown("""
Cette simulation couvre le risque de change des actions américaines en utilisant un **contrat forward** pour figer le taux EUR/USD.
**Formule appliquée :**
Return hedgé = Return en USD − Variation du taux EUR/USD
→ Cela neutralise l'impact des fluctuations du change.
*(Simplification : pas de coût de couverture inclus.)*
""")

# =====================
# Affichage des performances
# =====================
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
col1.metric("Perf depuis le début", f"{(portfolio_index.iloc[-1]-1)*100:.2f}%")
col2.metric("Volatilité (annualisée)", f"{vol_port:.2f}%")
col3.metric("Sharpe Ratio", f"{sharpe_port:.2f}")

st.caption("Mise à jour automatique toutes les heures")

# =====================
# GENERATEUR DE FACTSHEET PDF
# =====================

def generate_factsheet_pdf(portfolio_index, bench_index, portfolio_index_hedged,
                         portfolio_returns, benchmark_returns, portfolio_returns_hedged,
                         weights, allocation, prices_eur, start_date):
    """Génère une factsheet PDF profesionnelle"""
    
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, rightMargin=0.5*inch, 
                           leftMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f3a93'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1f3a93'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold',
        borderColor=colors.HexColor('#1f3a93'),
        borderPadding=4,
        borderWidth=0.5,
        borderRadius=2
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=9,
        alignment=TA_LEFT
    )
    
    # En-tête
    date_str = datetime.now().strftime("%d %B %Y")
    story.append(Paragraph("FUND FACTSHEET", title_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(f"Performance Report - {date_str}", 
                          ParagraphStyle('subtitle', parent=styles['Normal'], 
                                       fontSize=10, alignment=TA_CENTER, 
                                       textColor=colors.grey)))
    story.append(Spacer(1, 0.15*inch))
    
    # =========== PERFORMANCE SECTION ===========
    story.append(Paragraph("PERFORMANCE OVERVIEW", heading_style))
    
    # Tableau performance
    perf_data = [
        ['Period', 'Portfolio', 'Portfolio Hedged', 'Benchmark', 'Portfolio vs Bench'],
        ['YTD', f"{perf_ytd_port:.2f}%", f"{calculate_performance(portfolio_index_hedged, (datetime.now() - datetime(datetime.now().year, 1, 1)).days):.2f}%", 
         f"{perf_ytd_bench:.2f}%", f"{perf_ytd_port - perf_ytd_bench:.2f}%"],
        ['1 Year', f"{perf_1y_port:.2f}%", f"{calculate_performance(portfolio_index_hedged, 365):.2f}%", 
         f"{perf_1y_bench:.2f}%", f"{perf_1y_port - perf_1y_bench:.2f}%"],
        ['3 Years', f"{perf_3y_port:.2f}%", f"{calculate_performance(portfolio_index_hedged, 3*365):.2f}%", 
         f"{perf_3y_bench:.2f}%", f"{perf_3y_port - perf_3y_bench:.2f}%"],
        ['Since Inception', f"{(portfolio_index.iloc[-1]-1)*100:.2f}%", 
         f"{(portfolio_index_hedged.iloc[-1]-1)*100:.2f}%", 
         f"{(bench_index.iloc[-1]-1)*100:.2f}%", 
         f"{(portfolio_index.iloc[-1]-1)*100 - (bench_index.iloc[-1]-1)*100:.2f}%"],
    ]
    
    perf_table = Table(perf_data, colWidths=[1.2*inch, 1*inch, 1.2*inch, 1*inch, 1.2*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f3a93')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 0.15*inch))
    
    # =========== RISK METRICS ===========
    story.append(Paragraph("RISK METRICS", heading_style))
    
    risk_data = [
        ['Metric', 'Portfolio', 'Portfolio Hedged', 'Benchmark'],
        ['Volatility (p.a.)', f"{vol_port:.2f}%", f"{vol_port_hedged:.2f}%", f"{vol_bench:.2f}%"],
        ['Sharpe Ratio', f"{sharpe_port:.2f}", f"{sharpe_hedged:.2f}", f"{sharpe_bench:.2f}"],
        ['Sortino Ratio', f"{sortino_port:.2f}", f"{sortino_port:.2f}", f"{sortino_bench:.2f}"],
        ['Max Drawdown', f"{mdd_port:.2f}%", f"{mdd_hedged:.2f}%", f"{mdd_bench:.2f}%"],
        ['Calmar Ratio', f"{calmar_port:.2f}", f"{calmar_port:.2f}", f"{calmar_bench:.2f}"],
        ['Beta vs Bench', f"{beta:.2f}", "N/A", "1.00"],
    ]
    
    risk_table = Table(risk_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f3a93')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 0.15*inch))
    
    # =========== PORTFOLIO COMPOSITION ===========
    story.append(Paragraph("PORTFOLIO COMPOSITION", heading_style))
    
    allocation_data = [['Asset', 'Weight', 'Category']]
    ticker_names_dict = {
        "TTE.PA": "TotalEnergies", "MC.PA": "LVMH", "INGA.AS": "ING Groep",
        "SAP.DE": "SAP", "ACLN.SW": "ACLN", "THEON.AS": "Theon Intl",
        "BOI.PA": "Boiron", "EOAN.DE": "E.ON", "GOOGL": "Alphabet",
        "META": "Meta", "HWM": "Howmet", "AMZN": "Amazon",
        "0P0000ZWX4.F": "Helium Fund Perf A EUR",
        "0P0001861S.F": "Eleva Abs Ret Eurp S EUR",
        "0P00000M6C.F": "R-co Conviction Credit Euro",
        "0P00008ESK.F": "AXAIMFIIS US Short Dur HY",
        "0P0000A6ZG.F": "Immobilier 21 AC",
        "0P0000WHLW.F": "GemEquity R",
    }
    
    asset_categories = {
        "TTE.PA": "Energy", "MC.PA": "Luxury", "INGA.AS": "Banking",
        "SAP.DE": "Tech", "ACLN.SW": "Healthcare", "THEON.AS": "Healthcare",
        "BOI.PA": "Healthcare", "EOAN.DE": "Energy", "GOOGL": "Tech",
        "META": "Tech", "HWM": "Industrials", "AMZN": "Tech",
        "0P0000ZWX4.F": "Absolute Return", "0P0001861S.F": "Absolute Return",
        "0P00000M6C.F": "Credit", "0P00008ESK.F": "Fixed Income",
        "0P0000A6ZG.F": "Real Estate", "0P0000WHLW.F": "Equity",
    }
    
    sorted_alloc = sorted(allocation.items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_alloc:
        allocation_data.append([
            ticker_names_dict.get(ticker, ticker),
            f"{weight*100:.1f}%",
            asset_categories.get(ticker, "Other")
        ])
    
    alloc_table = Table(allocation_data, colWidths=[3*inch, 1*inch, 1.5*inch])
    alloc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f3a93')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    story.append(alloc_table)
    story.append(Spacer(1, 0.15*inch))
    
    # =========== FOOTER ===========
    footer_text = f"<i>Portfolio Inception: {start_date.strftime('%B %d, %Y')} | Data as of: {datetime.now().strftime('%B %d, %Y')}</i>"
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(footer_text, ParagraphStyle(
        'footer', parent=styles['Normal'], fontSize=8, 
        alignment=TA_CENTER, textColor=colors.grey)))
    
    story.append(Spacer(1, 0.05*inch))
    disclaimer = """<b>Disclaimer:</b> This factsheet is for informational purposes only and does not constitute 
    investment advice. Performance data is calculated from real market data and actual portfolio compositions. 
    Past performance is not indicative of future results. Investors should carefully review this information before making 
    investment decisions."""
    story.append(Paragraph(disclaimer, ParagraphStyle(
        'disclaimer', parent=styles['Normal'], fontSize=7, 
        alignment=TA_LEFT, textColor=colors.grey, leading=10)))
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer

# =====================
# BOUTON TELECHARGEMENT PDF
# =====================
st.subheader("📥 Télécharger la Factsheet")

if st.button("📄 Générer et télécharger la Factsheet PDF", key="generate_pdf"):
    with st.spinner("Génération de la factsheet en cours..."):
        pdf = generate_factsheet_pdf(
            portfolio_index, bench_index, portfolio_index_hedged,
            portfolio_returns, (bench_index.pct_change()),
            portfolio_returns_hedged,
            weights, allocation, prices_eur, start
        )
        
        # Affichage du bouton de téléchargement
        st.download_button(
            label="💾 Télécharger Factsheet PDF",
            data=pdf,
            file_name=f"Portfolio_Factsheet_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            key="download_pdf_button"
        )
        st.success("✅ Factsheet générée avec succès !")
