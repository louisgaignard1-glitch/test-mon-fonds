"""
factsheet_export.py - v8 (bilingue FR / EN)
-------------------------------------------
Factsheet PDF 3 pages (style asset manager) generee a partir des donnees
temps reel du portefeuille (prices_eur, indices, rendements) calculees par app.py.

Nouveautes v7
  - parametre lang="fr" | "en" : tous les libelles statiques sont traduits
    (titres, entetes de tableaux, SRI, mentions legales, graphiques, formats
    numeriques et dates)
  - normalisation des index temporels (suppression des fuseaux horaires) avant
    toute jointure / intersection : normalize_index()
  - parametre excluded : liste des lignes exclues faute de donnees, mentionnee
    dans la factsheet
  - deux boutons de telechargement (FR / EN) dans render_factsheet_section()
  - v8 : calendriers alignes par union + ffill (aucune date supprimee), parametre
    rebal (frequence de rebalancement, frais de transaction) affiche dans la
    factsheet ; RF est la source unique du taux sans risque (importee par app.py)

Usage dans app.py :
    from factsheet_export import render_factsheet_section, normalize_index
    render_factsheet_section(portfolio_index_hedged, bench_index,
                             portfolio_returns_hedged, allocation,
                             ticker_names, weights, prices_eur,
                             bench_weights=bench_weights_eff,
                             excluded=["Nom de la ligne exclue", ...],
                             rebal=dict(freq="M", bps=10.0))
"""

import io
import datetime
import threading

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader, simpleSplit

# ═════════════════════════════════════════════════════════
# PARAMETRES EDITABLES
# ═════════════════════════════════════════════════════════
RF = 0.03          # taux sans risque annuel utilise pour Sharpe / alpha
ANN = 252

# (classe, zone) de chaque composante du benchmark
BENCH_META = {
    "IEV": ("Actions", "Europe"), "SPY": ("Actions", "Amerique du Nord"),
    "TLT": ("Credit & Obligations", "Amerique du Nord"),
    "VNQ": ("Immobilier", "Amerique du Nord"),
    "EEM": ("Actions", "Marches emergents"),
}

# Metadonnees des lignes en direct : (zone, secteur, devise)
STOCK_META = {
    "TTE.PA":  ("Europe", "Energie", "EUR"),
    "MC.PA":   ("Europe", "Conso. discretionnaire", "EUR"),
    "INGA.AS": ("Europe", "Finance", "EUR"),
    "SAP.DE":  ("Europe", "Technologie", "EUR"),
    "ACLN.SW": ("Europe", "Industrie", "CHF"),
    "THEON.AS": ("Europe", "Industrie", "EUR"),
    "BOI.PA":  ("Europe", "Sante", "EUR"),
    "EOAN.DE": ("Europe", "Services publics", "EUR"),
    "GOOGL":   ("Amerique du Nord", "Communication", "USD"),
    "META":    ("Amerique du Nord", "Communication", "USD"),
    "HWM":     ("Amerique du Nord", "Industrie", "USD"),
    "AMZN":    ("Amerique du Nord", "Conso. discretionnaire", "USD"),
}
# Metadonnees des fonds : (classe, zone, devise)  -> a ajuster selon vos fonds
FUND_META = {
    "0P0000ZWX4.F": ("Alternatif", "Europe", "EUR"),
    "0P0001861S.F": ("Alternatif", "Europe", "EUR"),
    "0P00000M6C.F": ("Credit & Obligations", "Europe", "EUR"),
    "0P00008ESK.F": ("Credit & Obligations", "Amerique du Nord", "EUR (couvert)"),
    "0P0000A6ZG.F": ("Immobilier", "Europe", "EUR"),
    "0P0000WHLW.F": ("Actions", "Marches emergents", "EUR"),
}

CLS_ORDER = ["Actions", "Credit & Obligations", "Alternatif", "Immobilier", "Liquidites"]
REG_ORDER = ["Europe", "Amerique du Nord", "Marches emergents", "Liquidites"]

USD_TICKERS = [t for t, m in STOCK_META.items() if m[2] == "USD"]


def line_meta(t: str) -> dict:
    """Retourne classe / zone / secteur / devise / type d'une ligne."""
    if t in STOCK_META:
        z, s, d = STOCK_META[t]
        return dict(cls="Actions", region=z, sector=s, ccy=d, kind="Direct")
    if t in FUND_META:
        c, z, d = FUND_META[t]
        return dict(cls=c, region=z, sector="Fonds", ccy=d, kind="Fonds")
    return dict(cls="Actions", region="Europe", sector="Autre", ccy="EUR", kind="Direct")


# ═════════════════════════════════════════════════════════
# INTERNATIONALISATION (FR / EN)
# ═════════════════════════════════════════════════════════
# La langue courante est stockee par thread : chaque session Streamlit tourne
# dans son propre thread, deux utilisateurs ne se melangent donc pas.
_L = threading.local()


def _lang() -> str:
    return getattr(_L, "lang", "fr")


def _is_fr() -> bool:
    return _lang() == "fr"


# cle -> (francais, anglais)
TX = {
    # en-tete / pied de page
    "fund_title": ("PORTEFEUILLE MULTI-ACTIFS", "MULTI-ASSET PORTFOLIO"),
    "fund_sub": ("Gestion diversifiée - Europe & États-Unis - Couverture USD/EUR",
                 "Diversified management - Europe & United States - USD/EUR hedge"),
    "hdr_disc": ("Document à usage informatif - Les performances passées ne préjugent pas des performances futures",
                 "For information purposes only - Past performance is not a reliable indicator of future results"),
    "as_of": ("Au {d}", "As of {d}"),
    "hdr_page": ("Factsheet mensuelle - Page {p} / 3", "Monthly factsheet - Page {p} / 3"),
    "ftr1": ("Données : Yahoo Finance (cours ajustés), calculs en EUR. Couverture de change simulée sans coût. "
             "Frais de transaction simulés (cf. informations), hors frais de gestion.",
             "Data: Yahoo Finance (adjusted prices), calculations in EUR. FX hedge simulated at no cost. "
             "Simulated transaction costs (see fund information), excluding management fees."),
    "ftr2": ("Généré le {d}. Ceci n'est pas un conseil en investissement.",
             "Generated on {d}. This is not investment advice."),
    "no_data": ("Données insuffisantes", "Insufficient data"),

    # objectif
    "objective": (
        "Portefeuille multi-actifs diversifié combinant des actions européennes et américaines en direct et une "
        "poche de fonds (alternatif, crédit, immobilier, marchés frontières). L'objectif est de surperformer un "
        "benchmark composite sur un horizon de placement de 5 ans, avec une couverture du risque de change "
        "USD/EUR simulée sur la poche américaine.",
        "Diversified multi-asset portfolio combining directly held European and US equities with a bucket of funds "
        "(alternative, credit, real estate, frontier markets). The objective is to outperform a composite benchmark "
        "over a 5-year investment horizon, with a simulated USD/EUR currency hedge on the US bucket."),

    # page 1 - informations
    "sec_info": ("INFORMATIONS", "FUND INFORMATION"),
    "i_incep": ("Date de lancement", "Inception date"),
    "i_ccy": ("Devise", "Currency"),
    "i_univ": ("Univers", "Universe"),
    "i_univ_v": ("Europe / USA / Fonds", "Europe / US / Funds"),
    "i_val": ("Valorisation", "Valuation"),
    "i_val_v": ("Quotidienne", "Daily"),
    "i_hor": ("Horizon recommandé", "Recommended horizon"),
    "i_hor_v": ("5 ans", "5 years"),
    "i_hedge": ("Couverture USD", "USD hedge"),
    "i_hedge_v": ("Simulée (forward)", "Simulated (forward)"),
    "i_rebal": ("Rebalancement", "Rebalancing"),
    "freq_D": ("Quotidien (théorique)", "Daily (theoretical)"),
    "freq_M": ("Mensuel", "Monthly"),
    "freq_Q": ("Trimestriel", "Quarterly"),
    "i_fees": ("Frais de transaction", "Transaction costs"),
    "i_fees_v": ("{b} bps / mouvement", "{b} bps per trade"),
    "i_nl": ("Nombre de lignes", "Number of holdings"),
    "i_nav": ("VL (base 100)", "NAV (base 100)"),
    # SRI
    "sec_sri": ("INDICATEUR DE RISQUE (SRI)", "RISK INDICATOR (SRI)"),
    "sri_low": ("Risque le plus faible", "Lower risk"),
    "sri_high": ("Risque le plus élevé", "Higher risk"),
    "sri_note": ("Classe estimée à partir de la volatilité annualisée depuis l'origine ({v}). "
                 "Indicatif, hors méthode PRIIPs.",
                 "Class estimated from annualised volatility since inception ({v}). "
                 "Indicative only, not PRIIPs methodology."),
    # indicateurs cles
    "sec_key": ("INDICATEURS CLÉS (ORIGINE)", "KEY FIGURES (SINCE INCEPTION)"),
    "k_ann": ("Rendement annualisé", "Annualised return"),
    "k_vol": ("Volatilité", "Volatility"),
    "k_sharpe": ("Ratio de Sharpe", "Sharpe ratio"),
    "k_sortino": ("Ratio de Sortino", "Sortino ratio"),
    "k_mdd": ("Max drawdown", "Max drawdown"),
    "k_beta": ("Bêta", "Beta"),
    "k_te": ("Tracking error", "Tracking error"),
    # benchmark
    "sec_bench": ("BENCHMARK COMPOSITE", "COMPOSITE BENCHMARK"),
    "bl_IEV": ("Europe (IEV)", "Europe (IEV)"),
    "bl_SPY": ("S&P 500 (SPY)", "S&P 500 (SPY)"),
    "bl_TLT": ("Oblig. US LT (TLT)", "US long-term bonds (TLT)"),
    "bl_VNQ": ("Immobilier US (VNQ)", "US real estate (VNQ)"),
    "bl_EEM": ("Émergents (EEM)", "Emerging markets (EEM)"),
    # colonne droite
    "sec_obj": ("ORIENTATION DE GESTION", "INVESTMENT OBJECTIVE"),
    "sec_perf": ("ÉVOLUTION DE LA PERFORMANCE (BASE 100)", "PERFORMANCE EVOLUTION (BASE 100)"),
    "sec_cal": ("PERFORMANCES ANNUELLES CALENDAIRES", "CALENDAR YEAR PERFORMANCE"),
    "cal_note": ("* année en cours", "* year to date"),
    "leg_port": ("Portefeuille (hedgé USD)", "Portfolio (USD hedged)"),
    "leg_port_s": ("Portefeuille", "Portfolio"),
    "leg_bench": ("Benchmark composite", "Composite benchmark"),
    "leg_bench_s": ("Benchmark", "Benchmark"),
    # performances glissantes
    "sec_roll": ("PERFORMANCES GLISSANTES", "ROLLING PERFORMANCE"),
    "p_1M": ("1 mois", "1 month"), "p_3M": ("3 mois", "3 months"), "p_YTD": ("YTD", "YTD"),
    "p_1Y": ("1 an", "1 year"), "p_3Y": ("3 ans", "3 years"), "p_5Y": ("5 ans", "5 years"),
    "p_ITD": ("Origine", "Inception"),
    "r_cum_p": ("Cumulée portefeuille", "Portfolio cumulative"),
    "r_cum_b": ("Cumulée benchmark", "Benchmark cumulative"),
    "r_gap": ("Écart cumulé", "Cumulative gap"),
    "r_ann_p": ("Annualisée portefeuille", "Portfolio annualised"),
    "r_ann_b": ("Annualisée benchmark", "Benchmark annualised"),
    # statistiques
    "sec_stats": ("STATISTIQUES (PÉRIODES GLISSANTES)", "STATISTICS (ROLLING PERIODS)"),
    "s_vol_p": ("Volatilité portefeuille", "Portfolio volatility"),
    "s_vol_b": ("Volatilité benchmark", "Benchmark volatility"),
    "s_te": ("Tracking error", "Tracking error"),
    "s_sh_p": ("Sharpe portefeuille", "Portfolio Sharpe"),
    "s_sh_b": ("Sharpe benchmark", "Benchmark Sharpe"),
    "s_ir": ("Ratio d'information", "Information ratio"),
    "s_alpha": ("Alpha annualisé", "Annualised alpha"),
    "s_beta": ("Bêta", "Beta"),
    "s_r2": ("R2", "R2"),
    "s_corr": ("Coefficient de corrélation", "Correlation coefficient"),
    # risque extreme
    "sec_ext": ("RISQUE EXTRÊME", "EXTREME RISK"),
    "e_pos": ("% mois positifs", "% positive months"),
    "e_best": ("Meilleur mois", "Best month"),
    "e_worst": ("Pire mois", "Worst month"),
    "e_dd_p": ("Max DD portef.", "Max DD portfolio"),
    "e_dd_b": ("Max DD bench.", "Max DD benchmark"),
    "e_rec_p": ("Recouvrement portef.", "Recovery portfolio"),
    "e_rec_b": ("Recouvrement bench.", "Recovery benchmark"),
    "e_var": ("VaR 95% 1 jour", "1-day 95% VaR"),
    "e_trough": ("Date du point bas", "Trough date"),
    "e_sortino": ("Sortino", "Sortino"),
    "ongoing": ("en cours", "ongoing"),
    "days": ("{n} j", "{n} d"),

    # page 2
    "sec_cls": ("PAR CLASSE D'ACTIFS (% DE L'ACTIF NET)", "BY ASSET CLASS (% OF NET ASSETS)"),
    "sec_reg": ("PAR ZONE GÉOGRAPHIQUE (% DE L'ACTIF NET)", "BY REGION (% OF NET ASSETS)"),
    "sec_sect": ("SECTEURS (POCHE ACTIONS EN DIRECT)", "SECTORS (DIRECT EQUITY BUCKET)"),
    "sec_conc": ("STRUCTURE ET CONCENTRATION", "STRUCTURE AND CONCENTRATION"),
    "c_direct": ("Poche actions en direct", "Direct equity bucket"),
    "c_funds": ("Poche fonds", "Funds bucket"),
    "c_cash": ("Liquidités", "Cash"),
    "c_usd": ("Exposition USD (couverte)", "USD exposure (hedged)"),
    "c_top5": ("5 premières lignes", "Top 5 holdings"),
    "c_top10": ("10 premières lignes", "Top 10 holdings"),
    "c_max": ("Poids max d'une ligne", "Largest holding"),
    "c_eff": ("Nombre de lignes effectif (1/HHI)", "Effective number of holdings (1/HHI)"),
    "sec_top10": ("10 PRINCIPALES POSITIONS", "TOP 10 HOLDINGS"),
    "top_right": ("Nombre total de lignes : {n}", "Total number of holdings: {n}"),
    "h_line": ("Ligne", "Holding"),
    "h_class": ("Classe", "Asset class"),
    "h_sector": ("Secteur", "Sector"),
    "h_region": ("Zone", "Region"),
    "h_weight": ("Poids", "Weight"),
    "h_1m_perf": ("Perf. 1 mois", "1M perf."),
    "top10_total": ("Total 10 premières", "Top 10 total"),
    "sec_contrib": ("CONTRIBUTION À LA PERFORMANCE", "PERFORMANCE CONTRIBUTION"),
    "period": ("Période : {p}", "Period: {p}"),
    "best5": ("5 meilleures contributions", "Top 5 contributors"),
    "worst5": ("5 contributions les plus négatives", "Top 5 detractors"),
    "h_contrib": ("Contrib.", "Contrib."),
    "contrib_note": ("Contribution = poids x performance de la ligne sur 30 jours (en EUR). "
                     "Approximation : ne tient pas compte de la dérive des poids sur la période.",
                     "Contribution = weight x 30-day holding performance (in EUR). "
                     "Approximation: does not account for weight drift over the period."),
    "to": (" au ", " to "),

    # page 3
    "sec_comp": ("COMPOSITION DU PORTEFEUILLE", "PORTFOLIO COMPOSITION"),
    "h_ticker": ("Ticker", "Ticker"),
    "h_ccy": ("Devise", "Currency"),
    "h_1m": ("1 mois", "1M"),
    "h_ytd": ("YTD", "YTD"),
    "total": ("TOTAL", "TOTAL"),
    "hedged_abbr": (" h.", " hdg"),
    "sec_monthly": ("HISTORIQUE DE PERFORMANCE MENSUELLE (%)", "MONTHLY PERFORMANCE HISTORY (%)"),
    "h_year": ("Année", "Year"),
    "sec_dd": ("DRAWDOWN", "DRAWDOWN"),
    "sec_rv": ("VOLATILITÉ GLISSANTE 3 MOIS (ANN.)", "ROLLING 3-MONTH VOLATILITY (ANN.)"),
    "sec_risk": ("PRINCIPAUX RISQUES ET MENTIONS LÉGALES", "KEY RISKS AND LEGAL NOTICE"),
    "risks": (
        "Risque actions : la valeur des actions varie selon des facteurs propres aux sociétés et des facteurs "
        "économiques ou politiques ; des pertes importantes sont possibles. Risque de change : les titres "
        "libellés en USD ou CHF exposent le portefeuille aux variations de change ; la couverture USD présentée "
        "est simulée et sans coût. Risque de taux et de crédit : la poche fonds de crédit est sensible aux "
        "variations de taux et à la qualité des émetteurs. Risque de liquidité : certains fonds (immobilier, "
        "marchés frontières) peuvent être moins liquides. Risque de concentration : le portefeuille compte un "
        "nombre limité de lignes.",
        "Equity risk: share prices fluctuate with company-specific, economic and political factors, and "
        "significant losses are possible. Currency risk: securities denominated in USD or CHF expose the "
        "portfolio to exchange-rate movements; the USD hedge shown is simulated at no cost. Interest-rate and "
        "credit risk: the credit funds are sensitive to changes in rates and to issuer quality. Liquidity risk: "
        "some funds (real estate, frontier markets) may be less liquid. Concentration risk: the portfolio holds "
        "a limited number of positions."),
    "legal": (
        "Méthodologie : performances calculées à partir des cours ajustés (dividendes réinvestis) convertis en "
        "EUR, {rebal}. Benchmark composé d'ETF cotés en USD "
        "(IEV, SPY, TLT, VNQ, EEM), non convertis en EUR. Sharpe et alpha calculés avec un taux sans risque de "
        "{rf}. Ce document est communiqué à titre informatif, ne constitue ni un conseil en investissement ni "
        "une offre d'achat ou de vente. Les performances passées ne sont pas un indicateur fiable des "
        "performances futures. Tout investissement comporte un risque de perte en capital.",
        "Methodology: performance is computed from adjusted prices (dividends reinvested) converted to EUR, "
        "{rebal}. The benchmark is made of USD-listed ETFs (IEV, SPY, "
        "TLT, VNQ, EEM) and is not converted to EUR. Sharpe ratio and alpha use a risk-free rate of {rf}. This "
        "document is provided for information only and is neither investment advice nor an offer to buy or "
        "sell. Past performance is not a reliable indicator of future results. All investments carry a risk of "
        "capital loss."),
    "legal_rebal": ("avec rebalancement {freq} vers les poids cibles (dérive des poids entre deux dates) "
                    "et frais de transaction de {bps} bps par montant échangé",
                    "with {freq} rebalancing to target weights (weight drift between dates) "
                    "and transaction costs of {bps} bps on traded amounts"),
    "legal_rebal_none": ("avec des poids constants (rebalancement quotidien implicite)",
                         "using constant weights (implicit daily rebalancing)"),
    "excl_note": (
        "Lignes exclues faute de données exploitables : {names}. Leur poids a été réparti au prorata sur les "
        "autres lignes.",
        "Holdings excluded for lack of usable data: {names}. Their weight was reallocated pro rata to the "
        "remaining holdings."),
    "months": (["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"],
               ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]),

    # interface Streamlit
    "ui_title": ("Factsheet mensuelle PDF", "Monthly factsheet PDF"),
    "ui_btn_fr": ("Télécharger la factsheet (PDF, 3 pages)", "Télécharger la factsheet (PDF, 3 pages)"),
    "ui_btn_en": ("Download factsheet (PDF, 3 pages)", "Download factsheet (PDF, 3 pages)"),
    "ui_caption": ("Données au {d} - régénérée automatiquement à chaque actualisation des cours.",
                   "Data as of {d} - regenerated automatically at each price refresh."),
}

# libelles des categories (cles internes -> francais, anglais)
DISP = {
    "Actions": ("Actions", "Equities"),
    "Credit & Obligations": ("Crédit & Obligations", "Credit & Bonds"),
    "Alternatif": ("Alternatif", "Alternative"),
    "Immobilier": ("Immobilier", "Real estate"),
    "Liquidites": ("Liquidités", "Cash"),
    "Europe": ("Europe", "Europe"),
    "Amerique du Nord": ("Amérique du Nord", "North America"),
    "Marches emergents": ("Marchés émergents", "Emerging markets"),
    "Energie": ("Énergie", "Energy"),
    "Conso. discretionnaire": ("Conso. discrétionnaire", "Consumer discr."),
    "Finance": ("Finance", "Financials"),
    "Technologie": ("Technologie", "Technology"),
    "Industrie": ("Industrie", "Industrials"),
    "Sante": ("Santé", "Health care"),
    "Services publics": ("Services publics", "Utilities"),
    "Communication": ("Communication", "Communication"),
    "Fonds": ("Fonds", "Funds"),
    "Autre": ("Autre", "Other"),
}


def T(key: str, **kw):
    pair = TX.get(key)
    if pair is None:
        return key
    s = pair[0] if _is_fr() else pair[1]
    return s.format(**kw) if kw else s


def DL(key: str) -> str:
    """Libelle d'une categorie (classe, zone, secteur) dans la langue courante."""
    pair = DISP.get(key)
    if pair is None:
        return key
    return pair[0] if _is_fr() else pair[1]


def _ccy(code: str) -> str:
    return code.replace(" (couvert)", T("hedged_abbr"))


def _date(ts, with_time=False) -> str:
    if _is_fr():
        return ts.strftime("%d/%m/%Y %H:%M" if with_time else "%d/%m/%Y")
    return ts.strftime("%d %b %Y %H:%M" if with_time else "%d %b %Y")


# ═════════════════════════════════════════════════════════
# PALETTE & GEOMETRIE
# ═════════════════════════════════════════════════════════
P_NAVY = colors.HexColor("#0D2B55")
P_BLUE2 = colors.HexColor("#2E6DA4")
P_BLUE3 = colors.HexColor("#5B9BD5")
P_BLUE_L = colors.HexColor("#D6E4F0")
P_BLUE_XL = colors.HexColor("#EBF3FB")
P_STRIPE = colors.HexColor("#F4F8FC")
P_BORDER = colors.HexColor("#B8CDE4")
P_TEXT = colors.HexColor("#0D2B55")
P_MUTED = colors.HexColor("#5A7490")
P_POS = colors.HexColor("#1A6B3C")
P_NEG = colors.HexColor("#B03030")
P_WHITE = colors.white

C_PORT = "#0D2B55"
C_BENCH = "#8FA3B8"

W, H = A4
ML = MR = 15 * mm
CW = W - ML - MR
HDR_H = 24 * mm
FTR_H = 12 * mm
SAFE_Y = FTR_H + 4 * mm
ST_H = 6.5 * mm
TH_H = 4.8 * mm
TR_H = 4.2 * mm
GAP = 4 * mm
GAP_S = 2 * mm


# ═════════════════════════════════════════════════════════
# FORMATAGE
# ═════════════════════════════════════════════════════════
def _nan(v):
    return v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))


def _num(s: str) -> str:
    """Separateur decimal selon la langue (virgule en FR, point en EN)."""
    return s.replace(".", ",") if _is_fr() else s


def fp(v, d=2, sign=True):
    if _nan(v):
        return "n/a"
    return _num(("+" if (sign and v > 0) else "") + f"{v:.{d}f}%")


def fraw(v, d=2):
    return "n/a" if _nan(v) else _num(f"{v:.{d}f}")


def fpl(v, d=2):
    """Pourcentage sans signe force (poids, volatilite)."""
    return fp(v, d, sign=False)


# ═════════════════════════════════════════════════════════
# NORMALISATION DES INDEX TEMPORELS
# ═════════════════════════════════════════════════════════
def normalize_index(obj):
    """
    Index temporel naif (sans fuseau), a minuit, trie, sans doublons.
    Fonctionne pour Series et DataFrame.
    - tz-aware : tz_localize(None) conserve la date murale locale de la bourse
      (pas de decalage d'un jour comme avec une conversion UTC)
    - indices a fuseaux mixtes : repli sur UTC puis suppression du fuseau
    """
    if obj is None:
        return None
    out = obj.copy()
    try:
        idx = pd.DatetimeIndex(out.index)
    except (ValueError, TypeError):
        idx = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _pct(s):
    """Rendement simple sans fill implicite (compatible pandas 2.x et 3.x)."""
    return s / s.shift(1) - 1


# ═════════════════════════════════════════════════════════
# CALCULS (100 % issus des series temps reel)
# ═════════════════════════════════════════════════════════
def _ret_since(s: pd.Series, days: int) -> float:
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    cut = s.index[-1] - pd.Timedelta(days=days)
    prev = s[s.index <= cut]
    if prev.empty:
        return np.nan
    return (s.iloc[-1] / prev.iloc[-1] - 1) * 100


def _ret_ytd(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    y0 = pd.Timestamp(s.index[-1].year, 1, 1)
    prev = s[s.index < y0]
    base = prev.iloc[-1] if not prev.empty else s.iloc[0]
    return (s.iloc[-1] / base - 1) * 100


def _ret_total(s: pd.Series) -> float:
    s = s.dropna()
    return (s.iloc[-1] / s.iloc[0] - 1) * 100 if len(s) > 1 else np.nan


def _annualize(cum_pct: float, days: float) -> float:
    if _nan(cum_pct) or days < 365:
        return np.nan
    return ((1 + cum_pct / 100) ** (365.0 / days) - 1) * 100


def _win_stats(rp: pd.Series, rb: pd.Series, days) -> dict:
    keys = ["vol", "vol_b", "te", "sharpe", "sharpe_b", "ir", "alpha",
            "beta", "r2", "corr"]
    out = {k: np.nan for k in keys}
    idx = rp.index.intersection(rb.index)
    if len(idx) < 30:
        return out
    rp, rb = rp.loc[idx], rb.loc[idx]
    if days:
        cut = idx[-1] - pd.Timedelta(days=days)
        if idx[0] > cut + pd.Timedelta(days=10):
            return out
        rp, rb = rp[rp.index > cut], rb[rb.index > cut]
    if len(rp) < 30 or rp.std() == 0 or rb.std() == 0:
        return out
    sq = np.sqrt(ANN)
    diff = rp - rb
    var_b = np.var(rb, ddof=1)
    beta = np.cov(rp, rb)[0, 1] / var_b if var_b > 0 else np.nan
    corr = np.corrcoef(rp, rb)[0, 1]
    out.update(
        vol=rp.std() * sq * 100,
        vol_b=rb.std() * sq * 100,
        te=diff.std() * sq * 100,
        sharpe=(rp.mean() * ANN - RF) / (rp.std() * sq),
        sharpe_b=(rb.mean() * ANN - RF) / (rb.std() * sq),
        ir=(diff.mean() * ANN) / (diff.std() * sq) if diff.std() > 0 else np.nan,
        beta=beta,
        alpha=((rp.mean() - RF / ANN) - beta * (rb.mean() - RF / ANN)) * ANN * 100,
        r2=corr ** 2,
        corr=corr,
    )
    return out


def _drawdown(s: pd.Series):
    s = s.dropna()
    dd = (s / s.cummax() - 1) * 100
    trough = dd.idxmin()
    peak_date = s.loc[:trough].idxmax()
    after = s.loc[trough:]
    rec = after[after >= s.loc[peak_date]]
    rec_days = (rec.index[0] - trough).days if len(rec) else None
    return dd, dd.min(), trough, rec_days


def _monthly(r: pd.Series) -> pd.Series:
    return ((1 + r).groupby([r.index.year, r.index.month]).prod() - 1) * 100


def _calendar(s: pd.Series, n=8) -> pd.Series:
    s = s.dropna()
    ye = s.groupby(s.index.year).last()
    prev = ye.shift(1)
    prev.iloc[0] = s.iloc[0]
    return ((ye / prev - 1) * 100).tail(n)


def _sri(vol_pct: float) -> int:
    """Classe de risque 1-7 estimee a partir de la volatilite annualisee."""
    if _nan(vol_pct):
        return 4
    v = vol_pct
    for i, lim in enumerate([0.5, 5, 12, 20, 30, 80], start=1):
        if v < lim:
            return i
    return 7


PERIODS = [("1M", 30), ("3M", 91), ("YTD", None), ("1Y", 365),
           ("3Y", 3 * 365), ("5Y", 5 * 365), ("ITD", "tot")]
STAT_WINDOWS = [("1Y", 365), ("3Y", 3 * 365), ("5Y", 5 * 365), ("ITD", None)]


def build_context(port_idx, bench_idx, prices_eur, allocation, ticker_names,
                  weights, bench_weights, port_ret=None, excluded=None, rebal=None):
    # --- normalisation des index (tz, doublons, ordre) AVANT toute intersection
    port_idx = normalize_index(port_idx)
    bench_idx = normalize_index(bench_idx)
    prices_eur = normalize_index(prices_eur) if prices_eur is not None else None

    # Calendrier unique : union des dates puis ffill PAR SERIE, avant toute intersection.
    # Les jours fériés propres à un marché (US / UE) ne suppriment donc aucune date.
    calendar = port_idx.index.union(bench_idx.index)
    port_idx = port_idx.reindex(calendar).ffill().dropna()
    bench_idx = bench_idx.reindex(calendar).ffill().dropna()
    if prices_eur is not None:
        prices_eur = prices_eur.reindex(prices_eur.index.union(calendar)).ffill()

    common = port_idx.index.intersection(bench_idx.index)
    if len(common) < 3:
        raise ValueError("Historique commun portefeuille / benchmark insuffisant")
    port_idx, bench_idx = port_idx.loc[common], bench_idx.loc[common]
    rp = _pct(port_idx).dropna()
    rb = _pct(bench_idx).dropna()

    first, last = port_idx.index[0], port_idx.index[-1]
    days_total = (last - first).days

    D = dict(port=port_idx, bench=bench_idx, rp=rp, rb=rb, first=first, last=last,
             days_total=days_total, prices=prices_eur, allocation=allocation,
             names=ticker_names, weights=weights, bench_weights=bench_weights or {},
             excluded=list(excluded or []), rebal=rebal)
    D["as_of"] = _date(last)
    D["today"] = _date(pd.Timestamp(datetime.datetime.now()), with_time=True)
    D["inception"] = _date(first)

    # performances glissantes
    rows = []
    for key, d in PERIODS:
        if d is None:
            p, b, nd = _ret_ytd(port_idx), _ret_ytd(bench_idx), None
        elif d == "tot":
            p, b, nd = _ret_total(port_idx), _ret_total(bench_idx), days_total
        else:
            p, b, nd = _ret_since(port_idx, d), _ret_since(bench_idx, d), d
        rows.append(dict(key=key, p=p, b=b,
                         pa=_annualize(p, nd) if nd else np.nan,
                         ba=_annualize(b, nd) if nd else np.nan))
    D["perf_rows"] = rows

    # stats par fenetre
    D["stats"] = {k: _win_stats(rp, rb, d) for k, d in STAT_WINDOWS}
    full = D["stats"]["ITD"]
    D["vol"] = full["vol"]
    D["sri"] = _sri(full["vol"])

    # drawdown
    D["dd"], D["maxdd"], D["dd_date"], D["dd_rec"] = _drawdown(port_idx)
    D["dd_b"], D["maxdd_b"], _, D["dd_rec_b"] = _drawdown(bench_idx)

    # mensuel
    m = _monthly(rp)
    D["monthly"] = m
    D["pos_months"] = (m > 0).mean() * 100 if len(m) else np.nan
    D["best_m"], D["worst_m"] = (m.max(), m.min()) if len(m) else (np.nan, np.nan)
    down = rp[rp < 0].std() * np.sqrt(ANN)
    D["sortino"] = ((rp.mean() * ANN - RF) / down) if down and down > 0 else np.nan
    D["ann_since"] = _annualize(_ret_total(port_idx), days_total)
    D["var95"] = np.percentile(rp, 5) * 100 if len(rp) > 30 else np.nan

    D["cal_p"] = _calendar(port_idx)
    D["cal_b"] = _calendar(bench_idx)

    # lignes
    nav_w = {t: float(w) for t, w in weights.items()}
    cash = max(0.0, 1.0 - sum(nav_w.values()))
    D["cash"] = cash
    lines = []
    for t, w in nav_w.items():
        meta = line_meta(t)
        s = prices_eur[t].dropna() if (prices_eur is not None and t in prices_eur.columns) else pd.Series(dtype=float)
        p1m = _ret_since(s, 30) if len(s) > 1 else np.nan
        pytd = _ret_ytd(s) if len(s) > 1 else np.nan
        lines.append(dict(t=t, name=ticker_names.get(t, t), w=w * 100,
                          p1m=p1m, pytd=pytd,
                          contrib=(w * p1m) if not _nan(p1m) else np.nan, **meta))
    D["lines"] = sorted(lines, key=lambda x: -x["w"])
    D["n_lines"] = len(lines) + (1 if cash > 0 else 0)

    # allocation par classe / zone (portefeuille et benchmark)
    cls_p, reg_p = {k: 0.0 for k in CLS_ORDER}, {k: 0.0 for k in REG_ORDER}
    for L in lines:
        cls_p[L["cls"]] = cls_p.get(L["cls"], 0.0) + L["w"]
        reg_p[L["region"]] = reg_p.get(L["region"], 0.0) + L["w"]
    cls_p["Liquidites"] += cash * 100
    reg_p["Liquidites"] += cash * 100
    cls_b, reg_b = {k: 0.0 for k in CLS_ORDER}, {k: 0.0 for k in REG_ORDER}
    bw = {k: v * 100 for k, v in (bench_weights or {}).items()}
    for k, v in bw.items():
        c_, z_ = BENCH_META.get(k, ("Actions", "Europe"))
        cls_b[c_] += v
        reg_b[z_] += v
    rest = max(0.0, 100 - sum(bw.values()))
    cls_b["Liquidites"] += rest
    reg_b["Liquidites"] += rest
    D.update(cls_p=cls_p, reg_p=reg_p, cls_b=cls_b, reg_b=reg_b)

    sect = {}
    for L in lines:
        if L["kind"] == "Direct":
            sect[L["sector"]] = sect.get(L["sector"], 0.0) + L["w"]
    D["sectors"] = dict(sorted(sect.items(), key=lambda x: -x[1]))

    D["period_1m"] = ""
    if prices_eur is not None and not prices_eur.empty:
        e = prices_eur.index[-1]
        D["period_1m"] = f"{_date(e - pd.Timedelta(days=30))}{T('to')}{_date(e)}"
    return D


# ═════════════════════════════════════════════════════════
# GRAPHIQUES MATPLOTLIB (dimensions exactes en mm -> pas de deformation)
# ═════════════════════════════════════════════════════════
_RC = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#B8CDE4", "axes.linewidth": 0.7,
    "xtick.color": "#5A7490", "ytick.color": "#5A7490",
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "grid.color": "#DCE8F3", "grid.linewidth": 0.5,
    "legend.fontsize": 6.5, "legend.framealpha": 0.95,
    "legend.edgecolor": "#B8CDE4", "font.family": "DejaVu Sans",
    "axes.axisbelow": True,
}


def _fig(w_pt, h_pt):
    matplotlib.rcParams.update(_RC)
    return plt.subplots(figsize=(w_pt / 72.0, h_pt / 72.0))


def _png(fig, dpi=220) -> bytes:
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _empty(w, h, msg=None):
    fig, ax = _fig(w, h)
    ax.text(0.5, 0.5, msg or T("no_data"), ha="center", va="center", color="#5A7490", fontsize=8)
    ax.axis("off")
    return _png(fig)


def _date_axis(ax, span_days):
    if span_days > 3 * 365:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def g_perf(D, w, h):
    p, b = D["port"], D["bench"]
    if len(p) < 3:
        return _empty(w, h)
    p100, b100 = p / p.iloc[0] * 100, b / b.iloc[0] * 100
    fig, ax = _fig(w, h)
    ax.fill_between(p100.index, p100, 100, where=(p100 >= 100), color=C_PORT, alpha=0.08)
    ax.plot(b100.index, b100, color=C_BENCH, lw=1.2, label=T("leg_bench"))
    ax.plot(p100.index, p100, color=C_PORT, lw=1.7, label=T("leg_port"))
    ax.axhline(100, color="#B8CDE4", lw=0.6)
    for s, col in ((p100, C_PORT), (b100, "#5A7490")):
        ax.annotate(f"{s.iloc[-1]:.0f}", xy=(s.index[-1], s.iloc[-1]), xytext=(3, 0),
                    textcoords="offset points", fontsize=6.5, color=col,
                    fontweight="bold", va="center")
    _date_axis(ax, D["days_total"])
    ax.grid(True, axis="y", linestyle="--")
    ax.legend(loc="upper left", frameon=True, handlelength=1.8)
    ax.set_xlim(p100.index[0], p100.index[-1] + (p100.index[-1] - p100.index[0]) * 0.04)
    return _png(fig)


def g_calendar(D, w, h):
    cp, cb = D["cal_p"], D["cal_b"]
    if cp.empty:
        return _empty(w, h)
    years = list(cp.index)
    fig, ax = _fig(w, h)
    x = np.arange(len(years))
    bw = 0.38
    vp = cp.values
    vb = [cb.get(y, np.nan) for y in years]
    ax.bar(x - bw / 2, vp, bw, color=C_PORT, label=T("leg_port_s"))
    ax.bar(x + bw / 2, vb, bw, color=C_BENCH, label=T("leg_bench_s"))
    for xi, v in zip(x - bw / 2, vp):
        if not _nan(v):
            ax.text(xi, v + (0.6 if v >= 0 else -0.6), _num(f"{v:.1f}"),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=5.5, color=C_PORT)
    for xi, v in zip(x + bw / 2, vb):
        if not _nan(v):
            ax.text(xi, v + (0.6 if v >= 0 else -0.6), _num(f"{v:.1f}"),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=5.5, color="#5A7490")
    last_y = D["last"].year
    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}*" if y == last_y else str(y) for y in years])
    ax.axhline(0, color="#B8CDE4", lw=0.7)
    ax.grid(True, axis="y", linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper left", ncol=2, frameon=True, handlelength=1.2)
    lo, hi = np.nanmin(list(vp) + vb), np.nanmax(list(vp) + vb)
    ax.set_ylim(min(lo * 1.3, -2), hi * 1.5 + 4)   # marge haute : la legende ne masque aucune etiquette
    return _png(fig)


def g_compare(pd_, bd_, order, w, h):
    fig, ax = _fig(w, h)
    keys = [k for k in order if pd_.get(k, 0) > 0.05 or bd_.get(k, 0) > 0.05]
    if not keys:
        plt.close(fig)
        return _empty(w, h)
    y = np.arange(len(keys))
    bh = 0.36
    pv = [pd_.get(k, 0) for k in keys]
    bv = [bd_.get(k, 0) for k in keys]
    ax.barh(y - bh / 2, pv, bh, color=C_PORT, label=T("leg_port_s"))
    ax.barh(y + bh / 2, bv, bh, color=C_BENCH, label=T("leg_bench_s"))
    mx = max(pv + bv) or 1
    for yi, v in zip(y - bh / 2, pv):
        ax.text(v + mx * 0.015, yi, _num(f"{v:.1f}%"), va="center", fontsize=6, color=C_PORT)
    for yi, v in zip(y + bh / 2, bv):
        ax.text(v + mx * 0.015, yi, _num(f"{v:.1f}%"), va="center", fontsize=6, color="#5A7490")
    ax.set_yticks(y)
    ax.set_yticklabels([DL(k) for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, mx * 1.22)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, axis="x", linestyle="--")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.legend(loc="lower right", frameon=True, handlelength=1.2)
    return _png(fig)


def g_sectors(D, w, h):
    s = D["sectors"]
    if not s:
        return _empty(w, h)
    fig, ax = _fig(w, h)
    names, vals = list(s.keys()), list(s.values())
    shades = ["#0D2B55", "#1B3A6B", "#2E6DA4", "#4A8DC0", "#5B9BD5", "#7AAFD8", "#96C2E0", "#B2D4EA"]
    cols = [shades[min(i, len(shades) - 1)] for i in range(len(names))]
    ax.barh(range(len(names)), vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.02, i, _num(f"{v:.1f}%"), va="center", fontsize=6.5, color=C_PORT)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([DL(n) for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, axis="x", linestyle="--")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    return _png(fig)


def g_drawdown(D, w, h):
    dd, ddb = D["dd"], D["dd_b"]
    fig, ax = _fig(w, h)
    ax.fill_between(dd.index, dd, 0, color=C_PORT, alpha=0.14)
    ax.plot(dd.index, dd, color=C_PORT, lw=1.1, label=T("leg_port_s"))
    ax.plot(ddb.index, ddb, color=C_BENCH, lw=1.0, label=T("leg_bench_s"))
    # annotation placee a gauche du point bas s'il est proche du bord droit (evite le rognage)
    span = (dd.index[-1] - dd.index[0]).total_seconds() or 1
    frac = (D["dd_date"] - dd.index[0]).total_seconds() / span
    dx, ha = ((-16, "right") if frac > 0.7 else (14, "left"))
    ax.annotate(_num(f"{D['maxdd']:.1f}%"), xy=(D["dd_date"], D["maxdd"]),
                xytext=(dx, 8), textcoords="offset points", fontsize=6.5, color=C_PORT, ha=ha,
                arrowprops=dict(arrowstyle="->", color=C_PORT, lw=0.6))
    _date_axis(ax, D["days_total"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, axis="y", linestyle="--")
    ax.legend(loc="lower left", frameon=True, handlelength=1.5)
    return _png(fig)


def g_rollvol(D, w, h):
    rp, rb = D["rp"], D["rb"]
    win = 63
    if len(rp) < win + 10:
        return _empty(w, h)
    vp = rp.rolling(win).std() * np.sqrt(ANN) * 100
    vb = rb.rolling(win).std() * np.sqrt(ANN) * 100
    fig, ax = _fig(w, h)
    ax.plot(vb.index, vb, color=C_BENCH, lw=1.0, label=T("leg_bench_s"))
    ax.plot(vp.index, vp, color=C_PORT, lw=1.3, label=T("leg_port_s"))
    _date_axis(ax, D["days_total"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, axis="y", linestyle="--")
    ax.legend(loc="upper right", frameon=True, handlelength=1.5)
    return _png(fig)


# ═════════════════════════════════════════════════════════
# PRIMITIVES CANVAS
# ═════════════════════════════════════════════════════════
def _rect(c, x, y, w, h, fill=None, stroke=None, sw=0.3):
    c.saveState()
    if fill is not None:
        c.setFillColor(fill)
    if stroke is not None:
        c.setStrokeColor(stroke)
        c.setLineWidth(sw)
    c.rect(x, y, w, h, fill=1 if fill is not None else 0, stroke=1 if stroke is not None else 0)
    c.restoreState()


def _text(c, x, y, txt, font="Helvetica", size=7, color=P_TEXT, align="left"):
    c.saveState()
    c.setFont(font, size)
    c.setFillColor(color)
    s = str(txt)
    if align == "right":
        c.drawRightString(x, y, s)
    elif align == "center":
        c.drawCentredString(x, y, s)
    else:
        c.drawString(x, y, s)
    c.restoreState()


def _hline(c, x1, x2, y, color=P_BLUE2, lw=0.7):
    c.saveState()
    c.setStrokeColor(color)
    c.setLineWidth(lw)
    c.line(x1, y, x2, y)
    c.restoreState()


def _img(c, png, x, y, w, h):
    c.drawImage(ImageReader(io.BytesIO(png)), x, y, width=w, height=h, mask="auto")


def _para(c, x, y, txt, width, size=6.8, leading=8.4, color=P_TEXT, font="Helvetica"):
    """Texte a la ligne, retourne le y final."""
    for ln in simpleSplit(txt, font, size, width):
        _text(c, x, y - size, ln, font, size, color)
        y -= leading
    return y


def _sec(c, x, y, title, bw=None, right=None):
    bw = CW if bw is None else bw
    _rect(c, x, y - ST_H, bw, ST_H, fill=P_NAVY)
    _rect(c, x, y - ST_H, 2.2 * mm, ST_H, fill=P_BLUE2)
    _text(c, x + 4 * mm, y - ST_H + (ST_H - 7) / 2 + 0.5, title, "Helvetica-Bold", 7.2, P_WHITE)
    if right:
        _text(c, x + bw - 2 * mm, y - ST_H + (ST_H - 7) / 2 + 0.5, right,
              "Helvetica", 6, colors.HexColor("#A0C4E0"), "right")
    return y - ST_H - GAP_S


def _thead(c, x, y, cols, widths, aligns=None):
    tw = sum(widths)
    _rect(c, x, y - TH_H, tw, TH_H, fill=P_BLUE2)
    cx = x
    for i, (col, cw) in enumerate(zip(cols, widths)):
        al = (aligns or ["center"] * len(widths))[i]
        ty = y - TH_H + (TH_H - 6) / 2 + 0.3
        if al == "left":
            _text(c, cx + 1.8 * mm, ty, col, "Helvetica-Bold", 6.4, P_WHITE)
        elif al == "right":
            _text(c, cx + cw - 1.8 * mm, ty, col, "Helvetica-Bold", 6.4, P_WHITE, "right")
        else:
            _text(c, cx + cw / 2, ty, col, "Helvetica-Bold", 6.4, P_WHITE, "center")
        cx += cw
    return y - TH_H


def _trows(c, x, y, rows, widths, aligns=None, rh=TR_H, bold_last=False,
           signed=True, size=6.6):
    aligns = aligns or ["left"] * len(widths)
    tw = sum(widths)
    for i, row in enumerate(rows):
        if y - rh < SAFE_Y:
            break
        bg = P_STRIPE if i % 2 == 0 else P_WHITE
        _rect(c, x, y - rh, tw, rh, fill=bg, stroke=P_BORDER, sw=0.15)
        cx = x
        ty = y - rh + (rh - size) / 2 + 0.4
        for j, (cell, cw) in enumerate(zip(row, widths)):
            s = str(cell)
            fnt = "Helvetica-Bold" if (bold_last and i == len(rows) - 1) else "Helvetica"
            col = P_TEXT
            if signed and j > 0:
                if s.startswith("+"):
                    col = P_POS
                elif s.startswith("-") and len(s) > 1:
                    col = P_NEG
            pad = 1.8 * mm
            if aligns[j] == "left":
                _text(c, cx + pad, ty, s, fnt, size, col)
            elif aligns[j] == "right":
                _text(c, cx + cw - pad, ty, s, fnt, size, col, "right")
            else:
                _text(c, cx + cw / 2, ty, s, fnt, size, col, "center")
            cx += cw
        y -= rh
    return y


def _header(c, D, page):
    _rect(c, 0, H - HDR_H, W, HDR_H, fill=P_NAVY)
    _rect(c, 0, H - HDR_H, 4 * mm, HDR_H, fill=P_BLUE2)
    _hline(c, 0, W, H - HDR_H, color=P_BLUE3, lw=1.2)
    _text(c, ML + 1 * mm, H - 10.5 * mm, T("fund_title"), "Helvetica-Bold", 14, P_WHITE)
    _text(c, ML + 1 * mm, H - 16.5 * mm, T("fund_sub"), "Helvetica", 7.5, colors.HexColor("#8AAFD0"))
    _text(c, ML + 1 * mm, H - 21 * mm, T("hdr_disc"),
          "Helvetica-Oblique", 5.5, colors.HexColor("#6A90B0"))
    _text(c, W - MR, H - 10.5 * mm, T("as_of", d=D["as_of"]), "Helvetica-Bold", 10,
          colors.HexColor("#A0C4E0"), "right")
    _text(c, W - MR, H - 16.5 * mm, T("hdr_page", p=page), "Helvetica", 7,
          colors.HexColor("#8AAFD0"), "right")
    return H - HDR_H - GAP


def _footer(c, D):
    _rect(c, 0, 0, W, FTR_H, fill=P_NAVY)
    _hline(c, 0, W, FTR_H, color=P_BLUE3, lw=0.6)
    _text(c, ML, 6.2 * mm, T("ftr1"), "Helvetica-Oblique", 5.2, colors.HexColor("#8AAFD0"))
    _text(c, ML, 3.2 * mm, T("ftr2", d=D["today"]), "Helvetica-Oblique", 5.2, colors.HexColor("#8AAFD0"))


# ═════════════════════════════════════════════════════════
# PAGE 1
# ═════════════════════════════════════════════════════════
def _page1(c, D):
    top = _header(c, D, 1)
    _footer(c, D)
    LW = 58 * mm
    GP = 5 * mm
    RW = CW - LW - GP
    RX = ML + LW + GP

    # ---- colonne gauche
    yl = _sec(c, ML, top, T("sec_info"), LW)
    info = [
        [T("i_incep"), D["inception"]],
        [T("i_ccy"), "EUR"],
        [T("i_univ"), T("i_univ_v")],
        [T("i_val"), T("i_val_v")],
        [T("i_hor"), T("i_hor_v")],
        [T("i_hedge"), T("i_hedge_v")],
        [T("i_rebal"), T("freq_" + D["rebal"]["freq"]) if D["rebal"] else T("freq_D")],
        [T("i_fees"), T("i_fees_v", b=f"{D['rebal']['bps']:g}") if D["rebal"] else "n/a"],
        [T("i_nl"), str(D["n_lines"])],
        [T("i_nav"), fraw(D["port"].iloc[-1] / D["port"].iloc[0] * 100)],
    ]
    yl = _trows(c, ML, yl, info, [24 * mm, LW - 24 * mm], rh=4.4 * mm, signed=False)
    yl -= GAP

    yl = _sec(c, ML, yl, T("sec_sri"), LW)
    bw7 = LW / 7
    for i in range(1, 8):
        act = (i == D["sri"])
        _rect(c, ML + (i - 1) * bw7, yl - 7 * mm, bw7, 7 * mm,
              fill=P_NAVY if act else P_BLUE_XL, stroke=P_BORDER, sw=0.3)
        _text(c, ML + (i - 1) * bw7 + bw7 / 2, yl - 4.7 * mm, str(i), "Helvetica-Bold", 9,
              P_WHITE if act else P_MUTED, "center")
    yl -= 7 * mm + 1.2 * mm
    _text(c, ML, yl - 5, T("sri_low"), "Helvetica", 5.2, P_MUTED)
    _text(c, ML + LW, yl - 5, T("sri_high"), "Helvetica", 5.2, P_MUTED, "right")
    yl -= 3.5 * mm
    yl = _para(c, ML, yl, T("sri_note", v=fpl(D["vol"], 1)), LW, 5.4, 6.6, P_MUTED)
    yl -= GAP

    yl = _sec(c, ML, yl, T("sec_key"), LW)
    m = D["stats"]["ITD"]
    key = [
        [T("k_ann"), fp(D["ann_since"])],
        [T("k_vol"), fpl(m["vol"])],
        [T("k_sharpe"), fraw(m["sharpe"])],
        [T("k_sortino"), fraw(D["sortino"])],
        [T("k_mdd"), fp(D["maxdd"])],
        [T("k_beta"), fraw(m["beta"])],
        [T("k_te"), fpl(m["te"])],
    ]
    yl = _trows(c, ML, yl, key, [32 * mm, LW - 32 * mm], aligns=["left", "right"], rh=4.2 * mm)
    yl -= GAP

    yl = _sec(c, ML, yl, T("sec_bench"), LW)
    brow = [[T("bl_" + k) if ("bl_" + k) in TX else k, fpl(v * 100, 0)]
            for k, v in D["bench_weights"].items()]
    rest = 100 - sum(v * 100 for v in D["bench_weights"].values())
    if rest > 0.5:
        brow.append([DL("Liquidites"), fpl(rest, 0)])
    yl = _trows(c, ML, yl, brow, [LW - 16 * mm, 16 * mm], aligns=["left", "right"], rh=4.0 * mm, signed=False)

    # ---- colonne droite
    yr = _sec(c, RX, top, T("sec_obj"), RW)
    yr = _para(c, RX + 0.5 * mm, yr, T("objective"), RW - 1 * mm, 6.6, 8.2)
    yr -= 1.5 * mm
    yr = _sec(c, RX, yr, T("sec_perf"), RW)
    ch = 56 * mm
    _img(c, g_perf(D, RW, ch), RX, yr - ch, RW, ch)
    yr -= ch + GAP
    yr = _sec(c, RX, yr, T("sec_cal"), RW, right=T("cal_note"))
    ch2 = 42 * mm
    _img(c, g_calendar(D, RW, ch2), RX, yr - ch2, RW, ch2)
    yr -= ch2

    y = min(yl, yr) - GAP

    # ---- performances glissantes
    y = _sec(c, ML, y, T("sec_roll"), right=T("as_of", d=D["as_of"]))
    labels = [T("p_" + r["key"]) for r in D["perf_rows"]]
    wl = 38 * mm
    wc = (CW - wl) / len(labels)
    y = _thead(c, ML, y, [""] + labels, [wl] + [wc] * len(labels))
    rows = [
        [T("r_cum_p")] + [fp(r["p"]) for r in D["perf_rows"]],
        [T("r_cum_b")] + [fp(r["b"]) for r in D["perf_rows"]],
        [T("r_gap")] + [fp(r["p"] - r["b"]) if not (_nan(r["p"]) or _nan(r["b"])) else "n/a"
                        for r in D["perf_rows"]],
        [T("r_ann_p")] + [fp(r["pa"]) for r in D["perf_rows"]],
        [T("r_ann_b")] + [fp(r["ba"]) for r in D["perf_rows"]],
    ]
    y = _trows(c, ML, y, rows, [wl] + [wc] * len(labels),
               aligns=["left"] + ["right"] * len(labels), rh=4.3 * mm)
    y -= GAP

    # ---- statistiques
    sw = 118 * mm
    sx = ML + sw + 4 * mm
    sw2 = CW - sw - 4 * mm
    ys = _sec(c, ML, y, T("sec_stats"), sw)
    per = [k for k, _ in STAT_WINDOWS]
    w0 = 46 * mm
    wp = (sw - w0) / 4
    ys = _thead(c, ML, ys, [""] + [T("p_" + k) for k in per], [w0] + [wp] * 4)
    S = D["stats"]

    def srow(lbl, k, f):
        return [lbl] + [f(S[p][k]) for p in per]
    rows = [
        srow(T("s_vol_p"), "vol", fpl),
        srow(T("s_vol_b"), "vol_b", fpl),
        srow(T("s_te"), "te", fpl),
        srow(T("s_sh_p"), "sharpe", fraw),
        srow(T("s_sh_b"), "sharpe_b", fraw),
        srow(T("s_ir"), "ir", fraw),
        srow(T("s_alpha"), "alpha", lambda v: fp(v)),
        srow(T("s_beta"), "beta", fraw),
        srow(T("s_r2"), "r2", fraw),
        srow(T("s_corr"), "corr", fraw),
    ]
    ys = _trows(c, ML, ys, rows, [w0] + [wp] * 4, aligns=["left"] + ["right"] * 4, rh=4.1 * mm)

    yb = _sec(c, sx, y, T("sec_ext"), sw2)
    rec = lambda d: T("ongoing") if d is None else T("days", n=d)
    rows = [
        [T("e_pos"), fpl(D["pos_months"], 1)],
        [T("e_best"), fp(D["best_m"])],
        [T("e_worst"), fp(D["worst_m"])],
        [T("e_dd_p"), fp(D["maxdd"])],
        [T("e_dd_b"), fp(D["maxdd_b"])],
        [T("e_rec_p"), rec(D["dd_rec"])],
        [T("e_rec_b"), rec(D["dd_rec_b"])],
        [T("e_var"), fp(D["var95"])],
        [T("e_trough"), _date(D["dd_date"])],
        [T("e_sortino"), fraw(D["sortino"])],
    ]
    _trows(c, sx, yb, rows, [sw2 - 21 * mm, 21 * mm], aligns=["left", "right"], rh=4.1 * mm)


# ═════════════════════════════════════════════════════════
# PAGE 2
# ═════════════════════════════════════════════════════════
def _page2(c, D):
    y = _header(c, D, 2)
    _footer(c, D)
    HW = (CW - 5 * mm) / 2
    X2 = ML + HW + 5 * mm

    # ligne 1 : classe d'actifs / zone
    y1 = _sec(c, ML, y, T("sec_cls"), HW)
    _sec(c, X2, y, T("sec_reg"), HW)
    ch = 46 * mm
    _img(c, g_compare(D["cls_p"], D["cls_b"], CLS_ORDER, HW, ch), ML, y1 - ch, HW, ch)
    _img(c, g_compare(D["reg_p"], D["reg_b"], REG_ORDER, HW, ch), X2, y1 - ch, HW, ch)
    y = y1 - ch - GAP

    # ligne 2 : secteurs / concentration
    y2 = _sec(c, ML, y, T("sec_sect"), HW)
    ch2 = 44 * mm
    _img(c, g_sectors(D, HW, ch2), ML, y2 - ch2, HW, ch2)
    yc = _sec(c, X2, y, T("sec_conc"), HW)
    L = D["lines"]
    direct = sum(x["w"] for x in L if x["kind"] == "Direct")
    funds = sum(x["w"] for x in L if x["kind"] == "Fonds")
    usd = sum(x["w"] for x in L if x["ccy"] == "USD")
    top5 = sum(x["w"] for x in L[:5])
    top10 = sum(x["w"] for x in L[:10])
    hhi = sum((x["w"] / 100) ** 2 for x in L)
    rows = [
        [T("c_direct"), fpl(direct, 1)],
        [T("c_funds"), fpl(funds, 1)],
        [T("c_cash"), fpl(D["cash"] * 100, 1)],
        [T("c_usd"), fpl(usd, 1)],
        [T("c_top5"), fpl(top5, 1)],
        [T("c_top10"), fpl(top10, 1)],
        [T("c_max"), fpl(L[0]["w"], 1) if L else "n/a"],
        [T("c_eff"), fraw(1 / hhi, 1) if hhi > 0 else "n/a"],
    ]
    _trows(c, X2, yc, rows, [HW - 20 * mm, 20 * mm], aligns=["left", "right"], rh=4.5 * mm, signed=False)
    y = y2 - ch2 - GAP

    # principales positions
    y = _sec(c, ML, y, T("sec_top10"), right=T("top_right", n=len(L)))
    wcol = [50 * mm, 34 * mm, 36 * mm, 28 * mm, 14 * mm, 18 * mm]
    y = _thead(c, ML, y, [T("h_line"), T("h_class"), T("h_sector"), T("h_region"),
                          T("h_weight"), T("h_1m_perf")], wcol,
               ["left", "left", "left", "left", "right", "right"])
    rows = [[x["name"][:30], DL(x["cls"]), DL(x["sector"]), DL(x["region"]),
             fpl(x["w"], 1), fp(x["p1m"])] for x in L[:10]]
    rows.append([T("top10_total"), "", "", "", fpl(top10, 1), ""])
    y = _trows(c, ML, y, rows, wcol, aligns=["left"] * 4 + ["right", "right"],
               rh=4.1 * mm, bold_last=True)
    y -= GAP

    # contributions
    y = _sec(c, ML, y, T("sec_contrib"), right=T("period", p=D["period_1m"]))
    valid = [x for x in L if not _nan(x["contrib"])]
    best = sorted(valid, key=lambda x: -x["contrib"])[:5]
    worst = sorted(valid, key=lambda x: x["contrib"])[:5]
    wc3 = [HW - 36 * mm, 15 * mm, 21 * mm]
    _text(c, ML, y - 7, T("best5"), "Helvetica-Bold", 6.8, P_NAVY)
    _text(c, X2, y - 7, T("worst5"), "Helvetica-Bold", 6.8, P_NAVY)
    yy = y - 3.2 * mm
    hd = [T("h_line"), T("h_weight"), T("h_contrib")]
    ya = _thead(c, ML, yy, hd, wc3, ["left", "right", "right"])
    yb = _thead(c, X2, yy, hd, wc3, ["left", "right", "right"])
    fmt = lambda r: [[x["name"][:26], fpl(x["w"], 1), fp(x["contrib"])] for x in r]
    ya = _trows(c, ML, ya, fmt(best), wc3, aligns=["left", "right", "right"], rh=4.1 * mm)
    yb = _trows(c, X2, yb, fmt(worst), wc3, aligns=["left", "right", "right"], rh=4.1 * mm)
    y = min(ya, yb) - 1.5 * mm
    _para(c, ML, y, T("contrib_note"), CW, 5.2, 6.4, P_MUTED, "Helvetica-Oblique")


# ═════════════════════════════════════════════════════════
# PAGE 3
# ═════════════════════════════════════════════════════════
def _page3(c, D):
    y = _header(c, D, 3)
    _footer(c, D)

    # composition complete
    y = _sec(c, ML, y, T("sec_comp"), right=T("as_of", d=D["as_of"]))
    wcol = [46 * mm, 26 * mm, 27 * mm, 27 * mm, 15 * mm, 13 * mm, 13 * mm, 13 * mm]
    y = _thead(c, ML, y, [T("h_line"), T("h_ticker"), T("h_class"), T("h_region"), T("h_ccy"),
                          T("h_weight"), T("h_1m"), T("h_ytd")],
               wcol, ["left", "left", "left", "left", "left", "right", "right", "right"])
    rows = [[x["name"][:28], x["t"], DL(x["cls"])[:20], DL(x["region"])[:20], _ccy(x["ccy"]),
             fpl(x["w"], 1), fp(x["p1m"], 1), fp(x["pytd"], 1)] for x in D["lines"]]
    if D["cash"] > 0:
        rows.append([DL("Liquidites"), "-", DL("Liquidites"), "-", "EUR", fpl(D["cash"] * 100, 1), "", ""])
    rows.append([T("total"), "", "", "", "", fpl(100, 1), "", ""])
    y = _trows(c, ML, y, rows, wcol, aligns=["left"] * 5 + ["right"] * 3, rh=3.85 * mm,
               bold_last=True, size=6.3)
    y -= GAP

    # historique mensuel
    y = _sec(c, ML, y, T("sec_monthly"))
    mo = D["monthly"]
    years = sorted({k[0] for k in mo.index})[-6:]
    names = T("months")
    wy, wa = 15 * mm, 21 * mm
    wm = (CW - wy - wa) / 12
    y = _thead(c, ML, y, [T("h_year")] + names + [T("h_year")], [wy] + [wm] * 12 + [wa])
    rh = 4.4 * mm
    cal = D["cal_p"]
    for yr in years:
        if y - rh < SAFE_Y:
            break
        _rect(c, ML, y - rh, CW, rh, stroke=P_BORDER, sw=0.15)
        _text(c, ML + wy / 2, y - rh + 1.3 * mm, str(yr), "Helvetica-Bold", 6.4, P_NAVY, "center")
        for i in range(12):
            v = mo.get((yr, i + 1), np.nan)
            x0 = ML + wy + i * wm
            if _nan(v):
                _text(c, x0 + wm / 2, y - rh + 1.3 * mm, "-", "Helvetica", 6.2, P_MUTED, "center")
                continue
            a = min(abs(v) / 6.0, 1.0) * 0.75 + 0.1
            base = (0.10, 0.55, 0.30) if v >= 0 else (0.80, 0.25, 0.25)
            col = colors.Color(1 - a * (1 - base[0]), 1 - a * (1 - base[1]), 1 - a * (1 - base[2]))
            _rect(c, x0, y - rh, wm, rh, fill=col, stroke=P_WHITE, sw=0.4)
            _text(c, x0 + wm / 2, y - rh + 1.3 * mm, _num(f"{v:.1f}"), "Helvetica", 6.2,
                  P_WHITE if a > 0.55 else P_TEXT, "center")
        ya_v = cal.get(yr, np.nan)
        _rect(c, ML + wy + 12 * wm, y - rh, wa, rh, fill=P_BLUE_L, stroke=P_WHITE, sw=0.4)
        _text(c, ML + wy + 12 * wm + wa / 2, y - rh + 1.3 * mm, fp(ya_v, 1, sign=False),
              "Helvetica-Bold", 6.4, P_NAVY, "center")
        y -= rh
    y -= GAP

    # drawdown / volatilite
    HW = (CW - 5 * mm) / 2
    X2 = ML + HW + 5 * mm
    y1 = _sec(c, ML, y, T("sec_dd"), HW)
    _sec(c, X2, y, T("sec_rv"), HW)
    ch = 38 * mm
    _img(c, g_drawdown(D, HW, ch), ML, y1 - ch, HW, ch)
    _img(c, g_rollvol(D, HW, ch), X2, y1 - ch, HW, ch)
    y = y1 - ch - GAP

    # risques & mentions
    y = _sec(c, ML, y, T("sec_risk"))
    y = _para(c, ML, y, T("risks"), CW, 5.8, 7.2)
    y -= 1.5 * mm
    if D["rebal"]:
        rebal_txt = T("legal_rebal", freq=T("freq_" + D["rebal"]["freq"]).lower(),
                      bps=f"{D['rebal']['bps']:g}")
    else:
        rebal_txt = T("legal_rebal_none")
    y = _para(c, ML, y, T("legal", rf=fpl(RF * 100, 1), rebal=rebal_txt), CW, 5.4, 6.8,
              P_MUTED, "Helvetica-Oblique")
    if D["excluded"]:
        y -= 1.2 * mm
        _para(c, ML, y, T("excl_note", names=", ".join(D["excluded"])), CW, 5.4, 6.8,
              P_NEG, "Helvetica-Oblique")


# ═════════════════════════════════════════════════════════
# ENTREE PRINCIPALE
# ═════════════════════════════════════════════════════════
def generate_factsheet_pdf(port_idx, bench_idx, port_ret, allocation, ticker_names,
                           weights, prices_eur=None, bench_weights=None,
                           lang="fr", excluded=None, rebal=None) -> bytes:
    """Genere la factsheet. lang : "fr" (defaut) ou "en"."""
    lang = "en" if str(lang).lower().startswith("en") else "fr"
    previous = getattr(_L, "lang", None)
    _L.lang = lang
    try:
        D = build_context(port_idx, bench_idx, prices_eur, allocation, ticker_names,
                          weights, bench_weights, port_ret, excluded, rebal)
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf, pagesize=A4)
        c.setTitle(f"Factsheet - {D['as_of']}")
        c.setAuthor(T("fund_title"))
        _page1(c, D)
        c.showPage()
        _page2(c, D)
        c.showPage()
        _page3(c, D)
        c.showPage()
        c.save()
        return buf.getvalue()
    finally:
        _L.lang = previous or "fr"


@st.cache_data(ttl=900, show_spinner=False)
def _build_cached(cache_key: str, lang: str, _payload: dict) -> bytes:
    return generate_factsheet_pdf(lang=lang, **_payload)


def render_factsheet_section(port_idx, bench_idx, port_ret, allocation, ticker_names,
                             weights, prices_eur=None, bench_weights=None, excluded=None,
                             rebal=None):
    """Deux boutons de telechargement (FR / EN). PDF generes en 1 clic, mis en cache 15 min."""
    st.divider()
    st.subheader(f"{TX['ui_title'][0]} / {TX['ui_title'][1]}")
    payload = dict(port_idx=port_idx, bench_idx=bench_idx, port_ret=port_ret,
                   allocation=allocation, ticker_names=ticker_names, weights=weights,
                   prices_eur=prices_eur, bench_weights=bench_weights, excluded=excluded,
                   rebal=rebal)
    w_sig = ",".join(f"{t}:{float(w):.6f}" for t, w in weights.items())
    base_key = (f"{port_idx.index[-1]:%Y%m%d}-{float(port_idx.iloc[-1]):.6f}-{len(port_idx)}-"
                f"{hash(w_sig)}-{'|'.join(excluded or [])}-"
                f"{sorted((rebal or {}).items())}")
    today = datetime.date.today()
    col_fr, col_en = st.columns(2)
    for col, lang, label, bkey in (
        (col_fr, "fr", TX["ui_btn_fr"][0], "dl_factsheet_fr"),
        (col_en, "en", TX["ui_btn_en"][0], "dl_factsheet_en"),
    ):
        with col:
            try:
                pdf = _build_cached(f"{lang}-{base_key}", lang, payload)
                st.download_button(
                    label,
                    data=pdf,
                    file_name=f"factsheet_{lang.upper()}_{today:%Y%m%d}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                    key=bkey,
                )
            except Exception as e:
                st.error(f"[{lang.upper()}] Erreur lors de la génération de la factsheet : {e}")
    st.caption(f"Données au {port_idx.index[-1]:%d/%m/%Y} / Data as of "
               f"{port_idx.index[-1]:%d %b %Y} - régénérées automatiquement à chaque actualisation des cours.")
