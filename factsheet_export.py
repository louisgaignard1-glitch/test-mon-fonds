"""
factsheet_export.py - v6
------------------------
Factsheet PDF 3 pages (style asset manager) generee a partir des donnees
temps reel du portefeuille (prices_eur, indices, rendements) calculees par app.py.

Page 1 : informations, SRI, performance base 100, annees calendaires,
         performances glissantes, statistiques de risque
Page 2 : repartition classe d'actifs / zone vs benchmark, secteurs,
         concentration, principales positions, contributions du mois
Page 3 : composition complete, historique mensuel, drawdown, volatilite
         glissante, risques et mentions legales

Usage dans app.py :
    from factsheet_export import render_factsheet_section
    render_factsheet_section(portfolio_index_hedged, bench_index,
                             portfolio_returns_hedged, allocation,
                             ticker_names, weights, prices_eur,
                             bench_weights=BENCHMARK_WEIGHTS)
"""

import io
import datetime

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
FUND_TITLE = "PORTEFEUILLE MULTI-ACTIFS"
FUND_SUBTITLE = "Gestion diversifiee - Europe & Etats-Unis - Couverture USD/EUR"
OBJECTIVE = (
    "Portefeuille multi-actifs diversifie combinant des actions europeennes et "
    "americaines en direct et une poche de fonds (alternatif, credit, immobilier, "
    "marches frontieres). L'objectif est de surperformer un benchmark composite "
    "sur un horizon de placement de 5 ans, avec une couverture du risque de change "
    "USD/EUR simulee sur la poche americaine."
)
RF = 0.03          # taux sans risque annuel utilise pour Sharpe / alpha
ANN = 252

BENCH_LABELS = {
    "IEV": "Europe (IEV)", "SPY": "S&P 500 (SPY)", "TLT": "Oblig. US LT (TLT)",
    "VNQ": "Immobilier US (VNQ)", "EEM": "Emergents (EEM)",
}
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
C_ACC = "#5B9BD5"

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


def _fr(s: str) -> str:
    return s.replace(".", ",")


def fp(v, d=2, sign=True):
    if _nan(v):
        return "n/a"
    return _fr(("+" if (sign and v > 0) else "") + f"{v:.{d}f}%")


def fraw(v, d=2):
    return "n/a" if _nan(v) else _fr(f"{v:.{d}f}")


def fpl(v, d=2):
    """Pourcentage sans signe force (poids, volatilite)."""
    return fp(v, d, sign=False)


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


def build_context(port_idx, bench_idx, prices_eur, allocation, ticker_names,
                  weights, bench_weights, port_ret=None):
    port_idx = port_idx.dropna()
    bench_idx = bench_idx.dropna()
    common = port_idx.index.intersection(bench_idx.index)
    port_idx, bench_idx = port_idx.loc[common], bench_idx.loc[common]
    rp = port_idx.pct_change().dropna()
    rb = bench_idx.pct_change().dropna()

    first, last = port_idx.index[0], port_idx.index[-1]
    days_total = (last - first).days

    D = dict(port=port_idx, bench=bench_idx, rp=rp, rb=rb, first=first, last=last,
             days_total=days_total, prices=prices_eur, allocation=allocation,
             names=ticker_names, weights=weights, bench_weights=bench_weights or {})
    D["as_of"] = last.strftime("%d/%m/%Y")
    D["today"] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    D["inception"] = first.strftime("%d/%m/%Y")

    # performances glissantes
    periods = [("1 mois", 30), ("3 mois", 91), ("YTD", None), ("1 an", 365),
               ("3 ans", 3 * 365), ("5 ans", 5 * 365), ("Origine", "tot")]
    rows = []
    for lbl, d in periods:
        if d is None:
            p, b, nd = _ret_ytd(port_idx), _ret_ytd(bench_idx), None
        elif d == "tot":
            p, b, nd = _ret_total(port_idx), _ret_total(bench_idx), days_total
        else:
            p, b, nd = _ret_since(port_idx, d), _ret_since(bench_idx, d), d
        rows.append(dict(label=lbl, p=p, b=b,
                         pa=_annualize(p, nd) if nd else np.nan,
                         ba=_annualize(b, nd) if nd else np.nan))
    D["perf_rows"] = rows

    # stats par fenetre
    D["stats"] = {k: _win_stats(rp, rb, d) for k, d in
                  [("1 an", 365), ("3 ans", 3 * 365), ("5 ans", 5 * 365), ("Origine", None)]}
    full = D["stats"]["Origine"]
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
        cls_p[L["cls"]] += L["w"]
        reg_p[L["region"]] += L["w"]
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
        D["period_1m"] = f"{(e - pd.Timedelta(days=30)).strftime('%d/%m/%Y')} au {e.strftime('%d/%m/%Y')}"
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


def _empty(w, h, msg="Donnees insuffisantes"):
    fig, ax = _fig(w, h)
    ax.text(0.5, 0.5, msg, ha="center", va="center", color="#5A7490", fontsize=8)
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
    ax.plot(b100.index, b100, color=C_BENCH, lw=1.2, label="Benchmark composite")
    ax.plot(p100.index, p100, color=C_PORT, lw=1.7, label="Portefeuille (hedge USD)")
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
    ax.bar(x - bw / 2, vp, bw, color=C_PORT, label="Portefeuille")
    ax.bar(x + bw / 2, vb, bw, color=C_BENCH, label="Benchmark")
    for xi, v in zip(x - bw / 2, vp):
        if not _nan(v):
            ax.text(xi, v + (0.6 if v >= 0 else -0.6), f"{v:.1f}".replace(".", ","),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=5.5, color=C_PORT)
    for xi, v in zip(x + bw / 2, vb):
        if not _nan(v):
            ax.text(xi, v + (0.6 if v >= 0 else -0.6), f"{v:.1f}".replace(".", ","),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=5.5, color="#5A7490")
    last_y = D["last"].year
    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}*" if y == last_y else str(y) for y in years])
    ax.axhline(0, color="#B8CDE4", lw=0.7)
    ax.grid(True, axis="y", linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper left", ncol=2, frameon=True, handlelength=1.2)
    lo, hi = np.nanmin(list(vp) + vb), np.nanmax(list(vp) + vb)
    ax.set_ylim(min(lo * 1.3, -2), hi * 1.3 + 2)
    return _png(fig)


def g_compare(pd_, bd_, order, w, h):
    fig, ax = _fig(w, h)
    keys = [k for k in order if pd_.get(k, 0) > 0.05 or bd_.get(k, 0) > 0.05]
    if not keys:
        return _empty(w, h)
    y = np.arange(len(keys))
    bh = 0.36
    pv = [pd_.get(k, 0) for k in keys]
    bv = [bd_.get(k, 0) for k in keys]
    ax.barh(y - bh / 2, pv, bh, color=C_PORT, label="Portefeuille")
    ax.barh(y + bh / 2, bv, bh, color=C_BENCH, label="Benchmark")
    mx = max(pv + bv) or 1
    for yi, v in zip(y - bh / 2, pv):
        ax.text(v + mx * 0.015, yi, f"{v:.1f}%".replace(".", ","), va="center", fontsize=6, color=C_PORT)
    for yi, v in zip(y + bh / 2, bv):
        ax.text(v + mx * 0.015, yi, f"{v:.1f}%".replace(".", ","), va="center", fontsize=6, color="#5A7490")
    ax.set_yticks(y)
    ax.set_yticklabels(keys)
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
        ax.text(v + max(vals) * 0.02, i, f"{v:.1f}%".replace(".", ","), va="center", fontsize=6.5, color=C_PORT)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
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
    ax.plot(dd.index, dd, color=C_PORT, lw=1.1, label="Portefeuille")
    ax.plot(ddb.index, ddb, color=C_BENCH, lw=1.0, label="Benchmark")
    ax.annotate(f"{D['maxdd']:.1f}%".replace(".", ","), xy=(D["dd_date"], D["maxdd"]),
                xytext=(14, 8), textcoords="offset points", fontsize=6.5, color=C_PORT,
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
    ax.plot(vb.index, vb, color=C_BENCH, lw=1.0, label="Benchmark")
    ax.plot(vp.index, vp, color=C_PORT, lw=1.3, label="Portefeuille")
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
    _text(c, ML + 1 * mm, H - 10.5 * mm, FUND_TITLE, "Helvetica-Bold", 14, P_WHITE)
    _text(c, ML + 1 * mm, H - 16.5 * mm, FUND_SUBTITLE, "Helvetica", 7.5, colors.HexColor("#8AAFD0"))
    _text(c, ML + 1 * mm, H - 21 * mm, "Document a usage informatif - Les performances passees ne presagent pas des performances futures",
          "Helvetica-Oblique", 5.5, colors.HexColor("#6A90B0"))
    _text(c, W - MR, H - 10.5 * mm, f"Au {D['as_of']}", "Helvetica-Bold", 10, colors.HexColor("#A0C4E0"), "right")
    _text(c, W - MR, H - 16.5 * mm, f"Factsheet mensuelle - Page {page} / 3", "Helvetica", 7,
          colors.HexColor("#8AAFD0"), "right")
    return H - HDR_H - GAP


def _footer(c, D):
    _rect(c, 0, 0, W, FTR_H, fill=P_NAVY)
    _hline(c, 0, W, FTR_H, color=P_BLUE3, lw=0.6)
    _text(c, ML, 6.2 * mm,
          "Donnees : Yahoo Finance (cours ajustes), calculs en EUR. Couverture de change simulee sans cout. "
          "Performances nettes de frais de gestion non incluses.",
          "Helvetica-Oblique", 5.2, colors.HexColor("#8AAFD0"))
    _text(c, ML, 3.2 * mm, f"Genere le {D['today']}. Ceci n'est pas un conseil en investissement.",
          "Helvetica-Oblique", 5.2, colors.HexColor("#8AAFD0"))


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
    yl = _sec(c, ML, top, "INFORMATIONS", LW)
    info = [
        ["Date de lancement", D["inception"]],
        ["Devise", "EUR"],
        ["Univers", "Europe / USA / Fonds"],
        ["Valorisation", "Quotidienne"],
        ["Horizon recommande", "5 ans"],
        ["Couverture USD", "Simulee (forward)"],
        ["Nombre de lignes", str(D["n_lines"])],
        ["VL (base 100)", fraw(D["port"].iloc[-1] / D["port"].iloc[0] * 100)],
    ]
    yl = _trows(c, ML, yl, info, [26 * mm, LW - 26 * mm], rh=4.4 * mm, signed=False)
    yl -= GAP

    yl = _sec(c, ML, yl, "INDICATEUR DE RISQUE (SRI)", LW)
    bw7 = LW / 7
    for i in range(1, 8):
        act = (i == D["sri"])
        _rect(c, ML + (i - 1) * bw7, yl - 7 * mm, bw7, 7 * mm,
              fill=P_NAVY if act else P_BLUE_XL, stroke=P_BORDER, sw=0.3)
        _text(c, ML + (i - 1) * bw7 + bw7 / 2, yl - 4.7 * mm, str(i), "Helvetica-Bold", 9,
              P_WHITE if act else P_MUTED, "center")
    yl -= 7 * mm + 1.2 * mm
    _text(c, ML, yl - 5, "Risque le plus faible", "Helvetica", 5.2, P_MUTED)
    _text(c, ML + LW, yl - 5, "Risque le plus eleve", "Helvetica", 5.2, P_MUTED, "right")
    yl -= 3.5 * mm
    yl = _para(c, ML, yl, f"Classe estimee a partir de la volatilite annualisee depuis l'origine "
               f"({fpl(D['vol'], 1)}). Indicatif, hors methode PRIIPs.", LW, 5.4, 6.6, P_MUTED)
    yl -= GAP

    yl = _sec(c, ML, yl, "INDICATEURS CLES (ORIGINE)", LW)
    m = D["stats"]["Origine"]
    key = [
        ["Rendement annualise", fp(D["ann_since"])],
        ["Volatilite", fpl(m["vol"])],
        ["Ratio de Sharpe", fraw(m["sharpe"])],
        ["Ratio de Sortino", fraw(D["sortino"])],
        ["Max drawdown", fp(D["maxdd"])],
        ["Beta", fraw(m["beta"])],
        ["Tracking error", fpl(m["te"])],
    ]
    yl = _trows(c, ML, yl, key, [32 * mm, LW - 32 * mm], aligns=["left", "right"], rh=4.2 * mm)
    yl -= GAP

    yl = _sec(c, ML, yl, "BENCHMARK COMPOSITE", LW)
    brow = [[BENCH_LABELS.get(k, k), fpl(v * 100, 0)] for k, v in D["bench_weights"].items()]
    rest = 100 - sum(v * 100 for v in D["bench_weights"].values())
    if rest > 0.5:
        brow.append(["Liquidites", fpl(rest, 0)])
    yl = _trows(c, ML, yl, brow, [LW - 16 * mm, 16 * mm], aligns=["left", "right"], rh=4.0 * mm, signed=False)

    # ---- colonne droite
    yr = _sec(c, RX, top, "ORIENTATION DE GESTION", RW)
    yr = _para(c, RX + 0.5 * mm, yr, OBJECTIVE, RW - 1 * mm, 6.6, 8.2)
    yr -= 1.5 * mm
    yr = _sec(c, RX, yr, "EVOLUTION DE LA PERFORMANCE (BASE 100)", RW)
    ch = 56 * mm
    _img(c, g_perf(D, RW, ch), RX, yr - ch, RW, ch)
    yr -= ch + GAP
    yr = _sec(c, RX, yr, "PERFORMANCES ANNUELLES CALENDAIRES", RW, right="* annee en cours")
    ch2 = 42 * mm
    _img(c, g_calendar(D, RW, ch2), RX, yr - ch2, RW, ch2)
    yr -= ch2

    y = min(yl, yr) - GAP

    # ---- performances glissantes
    y = _sec(c, ML, y, "PERFORMANCES GLISSANTES", right=f"Au {D['as_of']}")
    labels = [r["label"] for r in D["perf_rows"]]
    wl = 38 * mm
    wc = (CW - wl) / len(labels)
    y = _thead(c, ML, y, [""] + labels, [wl] + [wc] * len(labels))
    rows = [
        ["Cumulee portefeuille"] + [fp(r["p"]) for r in D["perf_rows"]],
        ["Cumulee benchmark"] + [fp(r["b"]) for r in D["perf_rows"]],
        ["Ecart cumule"] + [fp(r["p"] - r["b"]) if not (_nan(r["p"]) or _nan(r["b"])) else "n/a"
                            for r in D["perf_rows"]],
        ["Annualisee portefeuille"] + [fp(r["pa"]) for r in D["perf_rows"]],
        ["Annualisee benchmark"] + [fp(r["ba"]) for r in D["perf_rows"]],
    ]
    y = _trows(c, ML, y, rows, [wl] + [wc] * len(labels),
               aligns=["left"] + ["right"] * len(labels), rh=4.3 * mm)
    y -= GAP

    # ---- statistiques
    sw = 118 * mm
    sx = ML + sw + 4 * mm
    sw2 = CW - sw - 4 * mm
    ys = _sec(c, ML, y, "STATISTIQUES (PERIODES GLISSANTES)", sw)
    per = ["1 an", "3 ans", "5 ans", "Origine"]
    w0 = 46 * mm
    wp = (sw - w0) / 4
    ys = _thead(c, ML, ys, [""] + per, [w0] + [wp] * 4)
    S = D["stats"]

    def srow(lbl, k, f):
        return [lbl] + [f(S[p][k]) for p in per]
    rows = [
        srow("Volatilite portefeuille", "vol", fpl),
        srow("Volatilite benchmark", "vol_b", fpl),
        srow("Tracking error", "te", fpl),
        srow("Sharpe portefeuille", "sharpe", fraw),
        srow("Sharpe benchmark", "sharpe_b", fraw),
        srow("Ratio d'information", "ir", fraw),
        srow("Alpha annualise", "alpha", lambda v: fp(v)),
        srow("Beta", "beta", fraw),
        srow("R2", "r2", fraw),
        srow("Coefficient de correlation", "corr", fraw),
    ]
    ys = _trows(c, ML, ys, rows, [w0] + [wp] * 4, aligns=["left"] + ["right"] * 4, rh=4.1 * mm)

    yb = _sec(c, sx, y, "RISQUE EXTREME", sw2)
    rec = lambda d: "en cours" if d is None else f"{d} j"
    rows = [
        ["% mois positifs", fpl(D["pos_months"], 1)],
        ["Meilleur mois", fp(D["best_m"])],
        ["Pire mois", fp(D["worst_m"])],
        ["Max DD portef.", fp(D["maxdd"])],
        ["Max DD bench.", fp(D["maxdd_b"])],
        ["Recouvrement portef.", rec(D["dd_rec"])],
        ["Recouvrement bench.", rec(D["dd_rec_b"])],
        ["VaR 95% 1 jour", fp(D["var95"])],
        ["Date du point bas", D["dd_date"].strftime("%d/%m/%Y")],
        ["Sortino", fraw(D["sortino"])],
    ]
    _trows(c, sx, yb, rows, [sw2 - 19 * mm, 19 * mm], aligns=["left", "right"], rh=4.1 * mm)


# ═════════════════════════════════════════════════════════
# PAGE 2
# ═════════════════════════════════════════════════════════
def _page2(c, D):
    y = _header(c, D, 2)
    _footer(c, D)
    HW = (CW - 5 * mm) / 2
    X2 = ML + HW + 5 * mm

    # ligne 1 : classe d'actifs / zone
    y1 = _sec(c, ML, y, "PAR CLASSE D'ACTIFS (% DE L'ACTIF NET)", HW)
    _sec(c, X2, y, "PAR ZONE GEOGRAPHIQUE (% DE L'ACTIF NET)", HW)
    ch = 46 * mm
    _img(c, g_compare(D["cls_p"], D["cls_b"], CLS_ORDER, HW, ch), ML, y1 - ch, HW, ch)
    _img(c, g_compare(D["reg_p"], D["reg_b"], REG_ORDER, HW, ch), X2, y1 - ch, HW, ch)
    y = y1 - ch - GAP

    # ligne 2 : secteurs / concentration
    y2 = _sec(c, ML, y, "SECTEURS (POCHE ACTIONS EN DIRECT)", HW)
    ch2 = 44 * mm
    _img(c, g_sectors(D, HW, ch2), ML, y2 - ch2, HW, ch2)
    yc = _sec(c, X2, y, "STRUCTURE ET CONCENTRATION", HW)
    L = D["lines"]
    direct = sum(x["w"] for x in L if x["kind"] == "Direct")
    funds = sum(x["w"] for x in L if x["kind"] == "Fonds")
    usd = sum(x["w"] for x in L if x["ccy"] == "USD")
    top5 = sum(x["w"] for x in L[:5])
    top10 = sum(x["w"] for x in L[:10])
    hhi = sum((x["w"] / 100) ** 2 for x in L)
    rows = [
        ["Poche actions en direct", fpl(direct, 1)],
        ["Poche fonds", fpl(funds, 1)],
        ["Liquidites", fpl(D["cash"] * 100, 1)],
        ["Exposition USD (hedgee)", fpl(usd, 1)],
        ["5 premieres lignes", fpl(top5, 1)],
        ["10 premieres lignes", fpl(top10, 1)],
        ["Poids max d'une ligne", fpl(L[0]["w"], 1) if L else "n/a"],
        ["Nombre de lignes effectif (1/HHI)", fraw(1 / hhi, 1) if hhi > 0 else "n/a"],
    ]
    _trows(c, X2, yc, rows, [HW - 20 * mm, 20 * mm], aligns=["left", "right"], rh=4.5 * mm, signed=False)
    y = y2 - ch2 - GAP

    # principales positions
    y = _sec(c, ML, y, "10 PRINCIPALES POSITIONS", right=f"Nombre total de lignes : {len(L)}")
    wcol = [50 * mm, 34 * mm, 36 * mm, 28 * mm, 14 * mm, 18 * mm]
    y = _thead(c, ML, y, ["Ligne", "Classe", "Secteur", "Zone", "Poids", "Perf. 1 mois"], wcol,
               ["left", "left", "left", "left", "right", "right"])
    rows = [[x["name"][:30], x["cls"], x["sector"], x["region"], fpl(x["w"], 1), fp(x["p1m"])]
            for x in L[:10]]
    rows.append(["Total 10 premieres", "", "", "", fpl(top10, 1), ""])
    y = _trows(c, ML, y, rows, wcol, aligns=["left"] * 4 + ["right", "right"],
               rh=4.1 * mm, bold_last=True)
    y -= GAP

    # contributions
    y = _sec(c, ML, y, "CONTRIBUTION A LA PERFORMANCE", right=f"Periode : {D['period_1m']}")
    valid = [x for x in L if not _nan(x["contrib"])]
    best = sorted(valid, key=lambda x: -x["contrib"])[:5]
    worst = sorted(valid, key=lambda x: x["contrib"])[:5]
    wc3 = [HW - 36 * mm, 15 * mm, 21 * mm]
    _text(c, ML, y - 7, "5 meilleures contributions", "Helvetica-Bold", 6.8, P_NAVY)
    _text(c, X2, y - 7, "5 contributions les plus negatives", "Helvetica-Bold", 6.8, P_NAVY)
    yy = y - 3.2 * mm
    ya = _thead(c, ML, yy, ["Ligne", "Poids", "Contrib."], wc3, ["left", "right", "right"])
    yb = _thead(c, X2, yy, ["Ligne", "Poids", "Contrib."], wc3, ["left", "right", "right"])
    fmt = lambda r: [[x["name"][:26], fpl(x["w"], 1), fp(x["contrib"])] for x in r]
    ya = _trows(c, ML, ya, fmt(best), wc3, aligns=["left", "right", "right"], rh=4.1 * mm)
    yb = _trows(c, X2, yb, fmt(worst), wc3, aligns=["left", "right", "right"], rh=4.1 * mm)
    y = min(ya, yb) - 1.5 * mm
    _para(c, ML, y, "Contribution = poids cible x performance de la ligne sur 30 jours (en EUR). "
          "Approximation : ne tient pas compte de la derive des poids sur la periode.",
          CW, 5.2, 6.4, P_MUTED, "Helvetica-Oblique")


# ═════════════════════════════════════════════════════════
# PAGE 3
# ═════════════════════════════════════════════════════════
def _page3(c, D):
    y = _header(c, D, 3)
    _footer(c, D)

    # composition complete
    y = _sec(c, ML, y, "COMPOSITION DU PORTEFEUILLE", right=f"Au {D['as_of']}")
    wcol = [46 * mm, 26 * mm, 27 * mm, 27 * mm, 15 * mm, 13 * mm, 13 * mm, 13 * mm]
    y = _thead(c, ML, y, ["Ligne", "Ticker", "Classe", "Zone", "Devise", "Poids", "1 mois", "YTD"],
               wcol, ["left", "left", "left", "left", "left", "right", "right", "right"])
    rows = [[x["name"][:28], x["t"], x["cls"][:16], x["region"][:16], x["ccy"].replace(" (couvert)", " h."),
             fpl(x["w"], 1), fp(x["p1m"], 1), fp(x["pytd"], 1)] for x in D["lines"]]
    if D["cash"] > 0:
        rows.append(["Liquidites", "-", "Liquidites", "-", "EUR", fpl(D["cash"] * 100, 1), "", ""])
    rows.append(["TOTAL", "", "", "", "", "100,0%", "", ""])
    y = _trows(c, ML, y, rows, wcol, aligns=["left"] * 5 + ["right"] * 3, rh=3.85 * mm,
               bold_last=True, size=6.3)
    y -= GAP

    # historique mensuel
    y = _sec(c, ML, y, "HISTORIQUE DE PERFORMANCE MENSUELLE (%)")
    mo = D["monthly"]
    years = sorted({k[0] for k in mo.index})[-6:]
    names = ["Jan", "Fev", "Mar", "Avr", "Mai", "Jun", "Jul", "Aou", "Sep", "Oct", "Nov", "Dec"]
    wy, wa = 15 * mm, 21 * mm
    wm = (CW - wy - wa) / 12
    y = _thead(c, ML, y, ["Annee"] + names + ["Annee"], [wy] + [wm] * 12 + [wa])
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
            _text(c, x0 + wm / 2, y - rh + 1.3 * mm, _fr(f"{v:.1f}"), "Helvetica", 6.2,
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
    y1 = _sec(c, ML, y, "DRAWDOWN", HW)
    _sec(c, X2, y, "VOLATILITE GLISSANTE 3 MOIS (ANN.)", HW)
    ch = 38 * mm
    _img(c, g_drawdown(D, HW, ch), ML, y1 - ch, HW, ch)
    _img(c, g_rollvol(D, HW, ch), X2, y1 - ch, HW, ch)
    y = y1 - ch - GAP

    # risques & mentions
    y = _sec(c, ML, y, "PRINCIPAUX RISQUES ET MENTIONS LEGALES")
    risks = (
        "Risque actions : la valeur des actions varie selon des facteurs propres aux societes et des facteurs "
        "economiques ou politiques ; des pertes importantes sont possibles. Risque de change : les titres "
        "libelles en USD ou CHF exposent le portefeuille aux variations de change ; la couverture USD presentee "
        "est simulee et sans cout. Risque de taux et de credit : la poche fonds de credit est sensible aux "
        "variations de taux et a la qualite des emetteurs. Risque de liquidite : certains fonds (immobilier, "
        "marches frontieres) peuvent etre moins liquides. Risque de concentration : le portefeuille compte un "
        "nombre limite de lignes."
    )
    y = _para(c, ML, y, risks, CW, 5.8, 7.2)
    y -= 1.5 * mm
    legal = (
        "Methodologie : performances calculees a partir des cours ajustes (dividendes reinvestis) convertis en EUR, "
        "avec des poids cibles constants (rebalancement quotidien implicite). Benchmark compose d'ETF cotes en USD "
        "(IEV, SPY, TLT, VNQ, EEM), non convertis en EUR. Sharpe et alpha calcules avec un taux sans risque de "
        f"{RF * 100:.1f}%. Ce document est communique a titre informatif, ne constitue ni un conseil en "
        "investissement ni une offre d'achat ou de vente. Les performances passees ne sont pas un indicateur fiable "
        "des performances futures. Tout investissement comporte un risque de perte en capital."
    )
    _para(c, ML, y, legal, CW, 5.4, 6.8, P_MUTED, "Helvetica-Oblique")


# ═════════════════════════════════════════════════════════
# ENTREE PRINCIPALE
# ═════════════════════════════════════════════════════════
def generate_factsheet_pdf(port_idx, bench_idx, port_ret, allocation, ticker_names,
                           weights, prices_eur=None, bench_weights=None) -> bytes:
    D = build_context(port_idx, bench_idx, prices_eur, allocation, ticker_names,
                      weights, bench_weights, port_ret)
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    c.setTitle(f"Factsheet - {D['as_of']}")
    c.setAuthor(FUND_TITLE)
    _page1(c, D)
    c.showPage()
    _page2(c, D)
    c.showPage()
    _page3(c, D)
    c.showPage()
    c.save()
    return buf.getvalue()


@st.cache_data(ttl=900, show_spinner="Generation de la factsheet...")
def _build_cached(cache_key: str, _payload: dict) -> bytes:
    return generate_factsheet_pdf(**_payload)


def render_factsheet_section(port_idx, bench_idx, port_ret, allocation, ticker_names,
                             weights, prices_eur=None, bench_weights=None):
    """Affiche le bouton de telechargement : PDF genere en 1 clic, mis en cache 15 min."""
    st.divider()
    st.subheader("Factsheet mensuelle PDF")
    payload = dict(port_idx=port_idx, bench_idx=bench_idx, port_ret=port_ret,
                   allocation=allocation, ticker_names=ticker_names, weights=weights,
                   prices_eur=prices_eur, bench_weights=bench_weights)
    key = f"{port_idx.index[-1]:%Y%m%d}-{float(port_idx.iloc[-1]):.6f}-{len(port_idx)}"
    try:
        pdf = _build_cached(key, payload)
        st.download_button(
            "Telecharger la factsheet (PDF, 3 pages)",
            data=pdf,
            file_name=f"factsheet_{datetime.date.today():%Y%m%d}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
        st.caption("Donnees au " + port_idx.index[-1].strftime("%d/%m/%Y")
                   + " - regeneree automatiquement a chaque actualisation des cours.")
    except Exception as e:
        st.error(f"Erreur lors de la generation de la factsheet : {e}")
        raise
