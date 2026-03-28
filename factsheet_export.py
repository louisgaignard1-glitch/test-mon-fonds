"""
factsheet_export.py — v5
--------------------------
Corrections :
- Top 5 mensuel : vrais prix de prices_eur (comme app.py), zero donnee inventee
- Section "contributions mensuelles" remplacee par "Top 5 lignes du mois" avec
  barres horizontales + courbes normalisees (base 100)
- Marges suffisantes : rien ne deborde sur le footer
- Hauteurs de sections recalculees pour tenir sur la page
- Style asset manager : espacement, couleurs, typographie soignee
"""

import io
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# ─────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────
P_NAVY   = colors.HexColor("#0D2B55")
P_BLUE2  = colors.HexColor("#2E6DA4")
P_BLUE3  = colors.HexColor("#5B9BD5")
P_BLUE_L = colors.HexColor("#D6E4F0")
P_BLUE_XL= colors.HexColor("#EBF3FB")
P_STRIPE = colors.HexColor("#F4F8FC")
P_BORDER = colors.HexColor("#B8CDE4")
P_TEXT   = colors.HexColor("#0D2B55")
P_MUTED  = colors.HexColor("#5A7490")
P_POS    = colors.HexColor("#1A6B3C")
P_NEG    = colors.HexColor("#B03030")
P_WHITE  = colors.white
P_GOLD   = colors.HexColor("#C9A84C")

CLS_HEX = {
    "Actions EU": "#0D2B55",
    "Actions US": "#2E6DA4",
    "Fonds":      "#5B9BD5",
    "Cash":       "#A8C8E8",
}
CLS_ORDER = ["Actions EU", "Actions US", "Fonds", "Cash"]

# Couleurs distinctes pour top 5 lignes
TOP5_COLORS = ["#0D2B55", "#2E6DA4", "#5B9BD5", "#C9A84C", "#1A6B3C"]

# ─────────────────────────────────────────────────────────
# GÉOMÉTRIE A4 — marges généreuses pour eviter overflow
# ─────────────────────────────────────────────────────────
W, H    = A4
ML      = 15 * mm
MR      = 15 * mm
CW      = W - ML - MR
HDR_H   = 26 * mm
FTR_H   = 14 * mm          # footer plus haut = zone protegee plus grande
SAFE_Y  = FTR_H + 4 * mm   # on ne descend jamais en dessous de ca
ST_H    = 7.0 * mm
TH_H    = 5.0 * mm
TR_H    = 4.2 * mm
GAP     = 4 * mm
GAP_S   = 2.5 * mm

USD_TICKERS = ["GOOGL", "META", "HWM", "AMZN"]

COUNTRY_MAP = {
    "TTE.PA":"France",    "MC.PA":"France",    "BOI.PA":"France",
    "SAP.DE":"Allemagne", "EOAN.DE":"Allemagne",
    "INGA.AS":"Pays-Bas", "THEON.AS":"Pays-Bas",
    "ACLN.SW":"Suisse",
    "GOOGL":"Etats-Unis", "META":"Etats-Unis",
    "HWM":"Etats-Unis",   "AMZN":"Etats-Unis",
}

def asset_class(t: str) -> str:
    if t.startswith("0P"):      return "Fonds"
    if t in USD_TICKERS:        return "Actions US"
    return "Actions EU"

# ─────────────────────────────────────────────────────────
# MÉTRIQUES — identique a app.py
# ─────────────────────────────────────────────────────────
def compute_metrics(port_idx: pd.Series,
                    bench_idx: pd.Series,
                    port_ret: pd.Series) -> dict:
    r   = port_ret.dropna()
    ann = 252
    rf  = 0.035 / ann

    def _roll(s, days):
        cut = s.index[-1] - pd.Timedelta(days=days)
        sub = s[s.index >= cut]
        return (sub.iloc[-1] / sub.iloc[0] - 1) * 100 if len(sub) >= 2 else np.nan

    def _ytd(s):
        ytd_start = pd.Timestamp(datetime.date.today().year, 1, 1)
        pre = s[s.index < ytd_start]
        sub = s[s.index >= ytd_start]
        if sub.empty:
            return (s.iloc[-1] / s.iloc[0] - 1) * 100
        base = pre.iloc[-1] if not pre.empty else sub.iloc[0]
        return (sub.iloc[-1] / base - 1) * 100

    vol     = r.std() * np.sqrt(ann) * 100
    sharpe  = ((r.mean() - rf) / r.std() * np.sqrt(ann)) if r.std() > 0 else np.nan
    down_s  = r[r < 0].std() * np.sqrt(ann)
    sortino = (r.mean() * ann / down_s) if down_s > 0 else np.nan
    cum     = (1 + r).cumprod()
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    ann_ret = r.mean() * ann * 100

    common = port_ret.index.intersection(bench_idx.index)
    br     = bench_idx.loc[common].pct_change().dropna()
    pr     = port_ret.loc[br.index]
    beta   = (np.cov(pr, br)[0, 1] / np.var(br)) if len(br) > 5 else np.nan
    corr   = np.corrcoef(pr, br)[0, 1]            if len(br) > 5 else np.nan
    diff   = pr - br
    te     = diff.std() * np.sqrt(ann) * 100
    ir     = (diff.mean() * ann / (diff.std() * np.sqrt(ann))) if diff.std() > 0 else np.nan

    return dict(
        perf_ytd=_ytd(port_idx),    perf_1m=_roll(port_idx, 30),
        perf_3m=_roll(port_idx, 91), perf_1y=_roll(port_idx, 365),
        perf_3y=_roll(port_idx, 3*365),
        perf_total=(port_idx.iloc[-1] / port_idx.iloc[0] - 1) * 100,
        ann_ret=ann_ret, vol=vol, sharpe=sharpe, sortino=sortino,
        max_dd=max_dd, beta=beta, corr=corr, te=te, ir=ir,
        bench_ytd=_ytd(bench_idx),    bench_1m=_roll(bench_idx, 30),
        bench_3m=_roll(bench_idx, 91), bench_1y=_roll(bench_idx, 365),
        bench_3y=_roll(bench_idx, 3*365),
    )

def _nan(v):      return v is None or (isinstance(v, float) and np.isnan(v))
def fp(v, d=2):   return "n/a" if _nan(v) else ("+" if v > 0 else "") + f"{v:.{d}f}%"
def fx(v, d=2):   return "n/a" if _nan(v) else f"{v:.{d}f}x"
def fraw(v, d=2): return "n/a" if _nan(v) else f"{v:.{d}f}"

# ─────────────────────────────────────────────────────────
# TOP 5 MENSUEL — vrais prix de prices_eur (comme app.py)
# ─────────────────────────────────────────────────────────
def compute_top5_monthly(prices_eur: pd.DataFrame,
                         allocation: dict,
                         ticker_names: dict,
                         weights: pd.Series,
                         n: int = 5) -> pd.DataFrame:
    """
    Reproduit exactement la logique du Top 5 de app.py :
        month_start = prices_eur.index[-1] - timedelta(days=30)
        prices_month = prices_eur[prices_eur.index >= month_start].ffill()
        monthly_perf = (prices_month.iloc[-1] / prices_month.iloc[0] - 1) * 100
    Retourne un DataFrame avec colonnes : name, ticker, perf, weight
    """
    if prices_eur is None or prices_eur.empty:
        return pd.DataFrame()

    month_start  = prices_eur.index[-1] - timedelta(days=30)
    prices_month = prices_eur[prices_eur.index >= month_start].ffill()

    if len(prices_month) < 2:
        return pd.DataFrame()

    monthly_perf = (prices_month.iloc[-1] / prices_month.iloc[0] - 1) * 100
    monthly_perf = monthly_perf.dropna()

    # Filtrer uniquement les tickers dans l'allocation
    monthly_perf = monthly_perf[monthly_perf.index.isin(allocation.keys())]
    monthly_perf = monthly_perf.sort_values(ascending=False)
    top5 = monthly_perf.head(n)

    records = []
    for ticker, perf in top5.items():
        records.append({
            "name":   ticker_names.get(ticker, ticker),
            "ticker": ticker,
            "perf":   perf,
            "weight": allocation.get(ticker, 0) * 100,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────
# STYLE MATPLOTLIB
# ─────────────────────────────────────────────────────────
_RC = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False,    "axes.spines.right": False,
    "axes.edgecolor": "#B8CDE4", "axes.linewidth": 0.8,
    "xtick.color": "#5A7490",    "ytick.color": "#5A7490",
    "xtick.labelsize": 8,        "ytick.labelsize": 8,
    "grid.color": "#EBF3FB",     "grid.linewidth": 0.6,
    "axes.titlesize": 10,        "axes.titlecolor": "#0D2B55",
    "axes.titleweight": "bold",  "axes.titlepad": 8,
    "legend.fontsize": 8,        "legend.framealpha": 0.95,
    "legend.edgecolor": "#B8CDE4",
}

def _savepng(fig, dpi=160) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ═══════════════════════════════════════════════════════════
# GRAPHIQUES
# ═══════════════════════════════════════════════════════════

def g_perf(port_idx: pd.Series, bench_idx: pd.Series) -> bytes:
    matplotlib.rcParams.update(_RC)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    common = port_idx.index.intersection(bench_idx.index)
    if len(common) < 2:
        ax.text(0.5, 0.5, "Donnees insuffisantes",
                ha="center", va="center", transform=ax.transAxes)
        return _savepng(fig)
    p = (port_idx.loc[common] / port_idx.loc[common].iloc[0]) * 100
    b = (bench_idx.loc[common] / bench_idx.loc[common].iloc[0]) * 100
    ax.fill_between(p.index, p, 100, where=(p >= 100),
                    alpha=0.10, color="#0D2B55", zorder=1)
    ax.fill_between(p.index, p, 100, where=(p < 100),
                    alpha=0.07, color="#B03030", zorder=1)
    ax.plot(p.index, p, color="#0D2B55", lw=2.2, zorder=3,
            label="Portefeuille hedge USD")
    ax.plot(b.index, b, color="#5B9BD5", lw=1.6, ls="--",
            dashes=(6, 3), zorder=2, label="Benchmark composite")
    ax.axhline(100, color="#B8CDE4", lw=0.8, zorder=0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.grid(True, axis="y", alpha=0.5, linestyle="--")
    ax.legend(loc="upper left", handlelength=2.2, fontsize=8,
              frameon=True, facecolor="white")
    ax.set_ylabel("Base 100", fontsize=8, color="#5A7490")
    fig.tight_layout(pad=0.8)
    return _savepng(fig)


def g_drawdown(port_ret: pd.Series) -> bytes:
    matplotlib.rcParams.update(_RC)
    fig, ax = plt.subplots(figsize=(8.5, 2.4))
    cum = (1 + port_ret.dropna()).cumprod()
    dd  = ((cum - cum.cummax()) / cum.cummax()) * 100
    ax.fill_between(dd.index, dd, 0, color="#2E6DA4", alpha=0.15)
    ax.plot(dd.index, dd, color="#2E6DA4", lw=1.4)
    idx_min, val_min = dd.idxmin(), dd.min()
    ax.annotate(f"Max DD : {val_min:.1f}%",
                xy=(idx_min, val_min), xytext=(32, 20),
                textcoords="offset points", fontsize=8, color="#2E6DA4",
                arrowprops=dict(arrowstyle="->", color="#2E6DA4", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          alpha=0.92, ec="#2E6DA4", lw=0.6))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.grid(True, axis="y", alpha=0.4, linestyle="--")
    ax.set_ylabel("Drawdown (%)", fontsize=8, color="#5A7490")
    fig.tight_layout(pad=0.8)
    return _savepng(fig)


def g_donut(cls_pcts: dict) -> bytes:
    matplotlib.rcParams.update(_RC)
    labels = [c for c in CLS_ORDER if cls_pcts.get(c, 0) > 0]
    values = [cls_pcts[c] for c in labels]
    clrs   = [CLS_HEX[c]  for c in labels]
    total  = sum(values)
    fig = plt.figure(figsize=(5.6, 2.8), facecolor="white")
    ax  = fig.add_axes([0.02, 0.04, 0.52, 0.92])
    ax.set_aspect("equal")
    ax.pie(values, colors=clrs,
           wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.6),
           startangle=90)
    ax.add_patch(plt.Circle((0, 0), 0.36, fc="white", zorder=10))
    ax.text(0,  0.10, "Allocation", ha="center", va="center",
            fontsize=8, color="#0D2B55")
    ax.text(0, -0.14, "100%", ha="center", va="center",
            fontsize=11, color="#0D2B55", fontweight="bold")
    ax_leg = fig.add_axes([0.56, 0.04, 0.42, 0.92])
    ax_leg.axis("off")
    handles    = [Patch(facecolor=CLS_HEX[l], edgecolor="white",
                        linewidth=0.6) for l in labels]
    leg_labels = [f"{l}  {cls_pcts[l]/total*100:.1f}%" for l in labels]
    ax_leg.legend(handles, leg_labels, loc="center left", fontsize=8.5,
                  handlelength=1.0, handleheight=0.9,
                  frameon=False, labelspacing=0.7)
    return _savepng(fig, dpi=150)


def g_top5_bars(df_top5: pd.DataFrame) -> bytes:
    """
    Barres horizontales : Top 5 lignes du mois.
    Performance reelle depuis prices_eur (comme app.py).
    """
    matplotlib.rcParams.update(_RC)
    if df_top5.empty:
        fig, ax = plt.subplots(figsize=(8.5, 2.8))
        ax.text(0.5, 0.5, "Donnees insuffisantes pour le Top 5",
                ha="center", va="center", fontsize=10, color="#5A7490")
        ax.axis("off")
        return _savepng(fig)

    names = df_top5["name"].tolist()
    perfs = df_top5["perf"].tolist()
    weights_pct = df_top5["weight"].tolist()
    bar_colors = [TOP5_COLORS[i % len(TOP5_COLORS)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ys = list(range(len(names)))
    bars = ax.barh(ys, perfs, color=bar_colors, height=0.52,
                   edgecolor="white", linewidth=0.5)

    # Labels
    for i, (bar, perf, w) in enumerate(zip(bars, perfs, weights_pct)):
        x_pos = perf + 0.05 if perf >= 0 else perf - 0.05
        ha    = "left" if perf >= 0 else "right"
        col   = "#1A6B3C" if perf >= 0 else "#B03030"
        ax.text(x_pos, i, f"{perf:+.2f}%",
                va="center", ha=ha, fontsize=8.5,
                color=col, fontweight="bold")
        ax.text(-max(abs(p) for p in perfs) * 1.45, i,
                f"Poids : {w:.1f}%",
                va="center", ha="left", fontsize=7.5, color="#5A7490")

    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=9, color="#0D2B55", fontweight="500")
    ax.invert_yaxis()
    ax.axvline(0, color="#B8CDE4", lw=0.9)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    ax.grid(True, axis="x", alpha=0.4, linestyle="--")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    xmax = max(abs(p) for p in perfs) if perfs else 1
    ax.set_xlim(-xmax * 1.7, xmax * 1.55)
    fig.tight_layout(pad=0.8)
    return _savepng(fig)


def g_top5_lines(prices_eur: pd.DataFrame,
                 df_top5: pd.DataFrame,
                 ticker_names: dict) -> bytes:
    """
    Courbes normalisees base 100 sur 30 jours pour le Top 5.
    Donnees reelles de prices_eur.
    """
    matplotlib.rcParams.update(_RC)
    fig, ax = plt.subplots(figsize=(8.5, 2.8))

    if prices_eur is None or prices_eur.empty or df_top5.empty:
        ax.text(0.5, 0.5, "Donnees insuffisantes",
                ha="center", va="center", fontsize=10, color="#5A7490")
        ax.axis("off")
        return _savepng(fig)

    month_start  = prices_eur.index[-1] - timedelta(days=30)
    prices_month = prices_eur[prices_eur.index >= month_start].ffill()

    plotted = 0
    for i, row in df_top5.iterrows():
        ticker = row["ticker"]
        if ticker not in prices_month.columns:
            continue
        series = prices_month[ticker].dropna()
        if len(series) < 2:
            continue
        normalized = (series / series.iloc[0]) * 100
        color = TOP5_COLORS[plotted % len(TOP5_COLORS)]
        ax.plot(normalized.index, normalized,
                color=color, lw=1.8, label=row["name"])
        # Annotation valeur finale
        ax.annotate(f"{row['perf']:+.1f}%",
                    xy=(normalized.index[-1], normalized.iloc[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=color, va="center", fontweight="bold")
        plotted += 1

    ax.axhline(100, color="#B8CDE4", lw=0.8, ls="--")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.grid(True, axis="y", alpha=0.4, linestyle="--")
    ax.legend(loc="upper left", fontsize=7.5, frameon=True,
              facecolor="white", handlelength=1.5, ncol=2)
    ax.set_ylabel("Base 100", fontsize=8, color="#5A7490")
    fig.tight_layout(pad=0.8)
    return _savepng(fig)


def g_country(weights: pd.Series) -> bytes:
    matplotlib.rcParams.update(_RC)
    by_country: dict = {}
    for t, w in weights.items():
        cty = COUNTRY_MAP.get(t)
        if cty is None:
            continue
        by_country[cty] = by_country.get(cty, 0.0) + float(w) * 100

    if not by_country:
        fig, ax = plt.subplots(figsize=(4.2, 2.0))
        ax.axis("off")
        return _savepng(fig)

    total    = sum(by_country.values())
    sorted_c = sorted(by_country.items(), key=lambda x: -x[1])
    names    = [c for c, _ in sorted_c]
    vals     = [v / total * 100 for _, v in sorted_c]
    blues    = ["#0D2B55","#1B3A6B","#2E6DA4","#4A8DC0",
                "#5B9BD5","#7AAFD8","#96C2E0","#B2D4EA"]
    bar_colors = [blues[min(i, len(blues)-1)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(4.2, max(2.0, len(names) * 0.40)))
    bars = ax.barh(range(len(names)), vals, color=bar_colors,
                   alpha=0.92, height=0.58)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% actions individuelles", fontsize=7.5, color="#5A7490")
    for bar, val in zip(bars, vals):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=7.5, color="#0D2B55")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout(pad=0.8)
    return _savepng(fig)

# ═══════════════════════════════════════════════════════════
# PRIMITIVES CANVAS REPORTLAB
# ═══════════════════════════════════════════════════════════

def _rect(c, x, y, w, h, fill=None, stroke=None, sw=0.3):
    c.saveState()
    if fill:   c.setFillColor(fill)
    if stroke: c.setStrokeColor(stroke); c.setLineWidth(sw)
    c.rect(x, y, w, h, fill=1 if fill else 0, stroke=1 if stroke else 0)
    c.restoreState()

def _text(c, x, y, txt, font="Helvetica", size=8,
          color=P_TEXT, align="left"):
    c.saveState()
    c.setFont(font, size); c.setFillColor(color); s = str(txt)
    if align == "right":    c.drawRightString(x, y, s)
    elif align == "center": c.drawCentredString(x, y, s)
    else:                   c.drawString(x, y, s)
    c.restoreState()

def _hline(c, x1, x2, y, color=P_BLUE2, lw=0.7):
    c.saveState(); c.setStrokeColor(color); c.setLineWidth(lw)
    c.line(x1, y, x2, y); c.restoreState()

def _img(c, png_bytes, x, y, w, h, preserve_aspect=False):
    c.drawImage(ImageReader(io.BytesIO(png_bytes)), x, y,
                width=w, height=h,
                preserveAspectRatio=preserve_aspect,
                anchor="c", mask="auto")

def _sec(c, x, y, title, bw=None) -> float:
    bw = bw if bw is not None else CW
    _rect(c, x, y - ST_H, bw, ST_H, fill=P_NAVY)
    _rect(c, x, y - ST_H, 3.0 * mm, ST_H, fill=P_BLUE2)
    _text(c, x + 5*mm, y - ST_H + (ST_H - 8)/2, title,
          font="Helvetica-Bold", size=8.5, color=P_WHITE)
    return y - ST_H - GAP_S

def _thead(c, x, y, cols, widths) -> float:
    tw = sum(widths)
    _rect(c, x, y - TH_H, tw, TH_H, fill=P_NAVY)
    cx = x
    for i, (col, cw) in enumerate(zip(cols, widths)):
        if i > 0:
            c.saveState(); c.setStrokeColor(P_BLUE2); c.setLineWidth(0.3)
            c.line(cx, y - TH_H + 0.8*mm, cx, y - 0.8*mm)
            c.restoreState()
        _text(c, cx + cw/2, y - TH_H + (TH_H - 6.5)/2, col,
              font="Helvetica-Bold", size=6.8, color=P_WHITE, align="center")
        cx += cw
    return y - TH_H

def _trows(c, x, y, rows, widths, aligns=None, rh=TR_H,
           bold_last=False, stripe=True) -> float:
    aligns = aligns or ["left"] * len(widths)
    tw     = sum(widths)
    for i, row in enumerate(rows):
        if y - rh < SAFE_Y:   # securite anti-overflow
            break
        bg = P_STRIPE if (stripe and i % 2 == 0) else P_WHITE
        _rect(c, x, y - rh, tw, rh, fill=bg, stroke=P_BORDER, sw=0.15)
        cx = x; ty = y - rh + (rh - 6.5) / 2
        for j, (cell, cw) in enumerate(zip(row, widths)):
            s   = str(cell)
            fnt = "Helvetica-Bold" if (bold_last and i == len(rows)-1) else "Helvetica"
            col = P_TEXT
            if s.startswith("+"):                             col = P_POS
            elif s.startswith("-") and s not in ("-", "—"):  col = P_NEG
            pad = 2.0 * mm
            if aligns[j] == "left":    _text(c, cx + pad,      ty, s, fnt, 6.8, col, "left")
            elif aligns[j] == "right": _text(c, cx + cw - pad, ty, s, fnt, 6.8, col, "right")
            else:                      _text(c, cx + cw/2,      ty, s, fnt, 6.8, col, "center")
            cx += cw
        y -= rh
    return y

# ═══════════════════════════════════════════════════════════
# EN-TÊTE / PIED DE PAGE
# ═══════════════════════════════════════════════════════════

def _draw_header(c, subtitle, as_of, page_lbl) -> float:
    _rect(c, 0, H - HDR_H, W, HDR_H, fill=P_NAVY)
    _rect(c, 0, H - HDR_H, 5*mm, HDR_H, fill=P_BLUE2)
    _hline(c, 0, W, H - HDR_H, color=P_BLUE3, lw=1.2)
    _text(c, ML + 2*mm, H - 11*mm,
          "PORTEFEUILLE MULTI-ACTIFS - HEDGE USD",
          font="Helvetica-Bold", size=13, color=P_WHITE)
    _text(c, ML + 2*mm, H - 18*mm, subtitle,
          font="Helvetica", size=8,
          color=colors.HexColor("#8AAFD0"))
    _text(c, ML + 2*mm, H - 23.5*mm,
          "Document a usage informatif — Performances passees non garanties",
          font="Helvetica-Oblique", size=6,
          color=colors.HexColor("#6A90B0"))
    _text(c, W - MR, H - 11*mm, f"Au {as_of}",
          font="Helvetica-Bold", size=10,
          color=colors.HexColor("#A0C4E0"), align="right")
    _text(c, W - MR, H - 18*mm, page_lbl,
          font="Helvetica", size=7.5,
          color=colors.HexColor("#8AAFD0"), align="right")
    return H - HDR_H - GAP

def _draw_footer(c, today):
    _rect(c, 0, 0, W, FTR_H, fill=P_NAVY)
    _hline(c, 0, W, FTR_H, color=P_BLUE3, lw=0.6)
    _text(c, ML, 4*mm,
          f"Donnees : Yahoo Finance / FMP via OpenBB  —  "
          f"Couverture FX simulee sans cout  —  Genere le {today}",
          font="Helvetica-Oblique", size=5.5,
          color=colors.HexColor("#6A90B0"))

# ═══════════════════════════════════════════════════════════
# PAGE 1
# ═══════════════════════════════════════════════════════════

def _page1(c, port_idx, bench_idx, port_ret, weights, m, as_of, today):
    y = _draw_header(
        c, "Gestion diversifiee  —  Europe & Etats-Unis  —  Couverture USD/EUR",
        as_of, "Page 1 / 2")
    _draw_footer(c, today)

    # ── KPIs ──────────────────────────────────────────────
    KPI_H = 17 * mm
    kpis  = [
        ("Perf. YTD",    fp(m["perf_ytd"])),
        ("Perf. 1 mois", fp(m["perf_1m"])),
        ("Perf. 1 an",   fp(m["perf_1y"])),
        ("Volatilite",   fp(m["vol"])),
        ("Sharpe",       fx(m["sharpe"])),
        ("Max Drawdown", fp(m["max_dd"])),
    ]
    kw = CW / len(kpis)
    for i, (lbl, val) in enumerate(kpis):
        bx = ML + i * kw
        bg = P_BLUE_XL if i % 2 == 0 else P_BLUE_L
        _rect(c, bx, y - KPI_H, kw, KPI_H, fill=bg, stroke=P_BORDER, sw=0.2)
        neg = isinstance(val, str) and val.startswith("-")
        vc  = P_NEG if neg else P_NAVY
        _text(c, bx + kw/2, y - KPI_H + 9*mm, val,
              font="Helvetica-Bold", size=11, color=vc, align="center")
        _text(c, bx + kw/2, y - KPI_H + 3.5*mm, lbl,
              font="Helvetica", size=6.8, color=P_MUTED, align="center")
        if i > 0:
            c.saveState(); c.setStrokeColor(P_BORDER); c.setLineWidth(0.3)
            c.line(bx, y - KPI_H + 1.5*mm, bx, y - 1.5*mm)
            c.restoreState()
    _hline(c, ML, ML + CW, y - KPI_H, color=P_BLUE2, lw=0.8)
    y -= KPI_H + GAP

    # ── Performance base 100 + tableau ────────────────────
    y = _sec(c, ML, y, "PERFORMANCE CUMULEE — BASE 100")
    chart_w = CW * 0.60
    table_x = ML + chart_w + 6*mm
    table_w = CW - chart_w - 6*mm
    ch      = 54 * mm
    _img(c, g_perf(port_idx, bench_idx), ML, y - ch, chart_w, ch)

    ty  = y - 2*mm
    _text(c, table_x, ty, "Performances cumulees",
          font="Helvetica-Bold", size=7.5, color=P_NAVY)
    ty -= 5*mm
    pw  = [19*mm, (table_w - 19*mm)/2, (table_w - 19*mm)/2]
    ty  = _thead(c, table_x, ty, ["Periode", "Portef.", "Bench."], pw)
    _trows(c, table_x, ty, [
        ["YTD",     fp(m["perf_ytd"]),   fp(m["bench_ytd"])],
        ["1 mois",  fp(m["perf_1m"]),    fp(m["bench_1m"])],
        ["3 mois",  fp(m["perf_3m"]),    fp(m["bench_3m"])],
        ["1 an",    fp(m["perf_1y"]),    fp(m["bench_1y"])],
        ["3 ans",   fp(m["perf_3y"]),    fp(m["bench_3y"])],
        ["Origine", fp(m["perf_total"]), "—"],
    ], pw, aligns=["left", "right", "right"])
    y -= ch + GAP

    # ── Drawdown ──────────────────────────────────────────
    y  = _sec(c, ML, y, "DRAWDOWN — PORTEFEUILLE HEDGE USD")
    dh = 34 * mm
    _img(c, g_drawdown(port_ret), ML, y - dh, CW, dh)
    y -= dh + GAP

    # ── Indicateurs de risque ─────────────────────────────
    y  = _sec(c, ML, y, "INDICATEURS DE RISQUE ET DE PERFORMANCE")
    rw = [72*mm, 52*mm, 41*mm]
    y  = _thead(c, ML, y,
                ["Indicateur", "Portefeuille hedge USD", "Benchmark"], rw)
    _trows(c, ML, y, [
        ["Rendement annualise",   fp(m["ann_ret"]),  "—"],
        ["Volatilite annualisee", fp(m["vol"]),      "—"],
        ["Ratio de Sharpe",       fx(m["sharpe"]),   "—"],
        ["Ratio de Sortino",      fx(m["sortino"]),  "—"],
        ["Max Drawdown",          fp(m["max_dd"]),   "—"],
        ["Beta vs benchmark",     fraw(m["beta"]),   "1,00"],
        ["Correlation",           fraw(m["corr"]),   "1,00"],
        ["Tracking Error",        fp(m["te"]),       "—"],
        ["Information Ratio",     fx(m["ir"]),       "—"],
    ], rw, aligns=["left", "right", "right"])

# ═══════════════════════════════════════════════════════════
# PAGE 2
# ═══════════════════════════════════════════════════════════

def _page2(c, port_ret, allocation, ticker_names, weights,
           prices_eur, as_of, today):
    y = _draw_header(
        c, "Composition  —  Allocation  —  Top 5 mensuel  —  Historique",
        as_of, "Page 2 / 2")
    _draw_footer(c, today)

    # ── Composition ───────────────────────────────────────
    y = _sec(c, ML, y, "COMPOSITION DU PORTEFEUILLE")
    cw_comp = [60*mm, 30*mm, 30*mm, 22*mm, 23*mm]
    y = _thead(c, ML, y,
               ["Actif", "Ticker", "Classe d'actifs",
                "Poids cible", "Poids effectif"], cw_comp)
    crows = [
        [ticker_names.get(t, t), t, asset_class(t),
         f"{allocation.get(t,0)*100:.1f}%", f"{float(w)*100:.1f}%"]
        for t, w in weights.items()
    ]
    crows.append(["Cash", "—", "Liquidites", "5,0%", "5,0%"])
    y = _trows(c, ML, y, crows, cw_comp,
               aligns=["left","left","left","right","right"],
               bold_last=True, rh=3.9*mm)

    # Ligne total
    if y - 5*mm > SAFE_Y:
        _rect(c, ML, y - 5*mm, sum(cw_comp), 5*mm, fill=P_NAVY)
        mid_t = y - 5*mm + (5*mm - 7) / 2
        _text(c, ML + 3*mm, mid_t, "TOTAL",
              font="Helvetica-Bold", size=8, color=P_WHITE)
        _text(c, ML + sum(cw_comp) - 3*mm, mid_t,
              f"{(weights.sum() + 0.05)*100:.1f}%",
              font="Helvetica-Bold", size=8,
              color=colors.HexColor("#A0C4E0"), align="right")
        y -= 5*mm
    y -= GAP

    # ── 3 colonnes : classe | donut | géo ─────────────────
    G3   = 4 * mm
    col3 = (CW - 2*G3) / 3
    H3   = 52 * mm
    ytop = y

    cls_pcts: dict = {}
    for t, w in weights.items():
        k = asset_class(t)
        cls_pcts[k] = cls_pcts.get(k, 0.0) + float(w) * 100
    cls_pcts["Cash"] = 5.0

    # Gauche — tableau classe
    yl  = _sec(c, ML, ytop, "PAR CLASSE D'ACTIFS", bw=col3)
    clw = [col3 * 0.60, col3 * 0.40]
    yl  = _thead(c, ML, yl, ["Classe", "Alloc."], clw)
    cls_rows = [(k, cls_pcts.get(k, 0)) for k in CLS_ORDER if k in cls_pcts]
    cls_rows.append(("Total", sum(cls_pcts.values())))
    yl = _trows(c, ML, yl, [[k, f"{v:.1f}%"] for k, v in cls_rows],
                clw, aligns=["left","right"], bold_last=True, rh=4.2*mm)

    # Centre — donut
    xp = ML + col3 + G3
    yp = _sec(c, xp, ytop, "ALLOCATION", bw=col3)
    ph = H3 - ST_H - 3*mm
    _img(c, g_donut(cls_pcts), xp, yp - ph, col3, ph, preserve_aspect=True)

    # Droite — géographie
    xt = ML + 2*col3 + 2*G3
    yt = _sec(c, xt, ytop, "GEOGRAPHIE", bw=col3)
    gh = H3 - ST_H - 3*mm
    _img(c, g_country(weights), xt, yt - gh, col3, gh, preserve_aspect=True)

    y = min(yl, yp - ph, yt - gh) - GAP

    # ── TOP 5 MENSUEL — vrais prix prices_eur ─────────────
    df_top5 = compute_top5_monthly(prices_eur, allocation,
                                   ticker_names, weights, n=5)

    # Date de la periode
    if prices_eur is not None and not prices_eur.empty:
        date_fin   = prices_eur.index[-1]
        date_debut = date_fin - timedelta(days=30)
        periode_lbl = (f"Periode : {date_debut.strftime('%d/%m/%Y')} "
                       f"au {date_fin.strftime('%d/%m/%Y')}")
    else:
        periode_lbl = ""

    y = _sec(c, ML, y,
             f"TOP 5 LIGNES DU MOIS — PERFORMANCES REELLES  |  {periode_lbl}")

    # Layout 2 colonnes : barres gauche | courbes droite
    half = (CW - 4*mm) / 2
    bh   = 46 * mm

    _img(c, g_top5_bars(df_top5),  ML,          y - bh, half, bh)
    _img(c, g_top5_lines(prices_eur, df_top5, ticker_names),
         ML + half + 4*mm, y - bh, half, bh)

    # Petit tableau résumé sous les graphiques
    y -= bh + 2*mm
    if not df_top5.empty and y - TH_H - len(df_top5) * TR_H > SAFE_Y:
        tw5 = [CW * 0.40, CW * 0.18, CW * 0.22, CW * 0.20]
        y   = _thead(c, ML, y,
                     ["Actif", "Poids", "Perf. 30j", "Classe"], tw5)
        trows5 = [
            [row["name"],
             f"{row['weight']:.1f}%",
             fp(row["perf"]),
             asset_class(row["ticker"])]
            for _, row in df_top5.iterrows()
        ]
        y = _trows(c, ML, y, trows5, tw5,
                   aligns=["left","right","right","left"])
    y -= GAP

    # ── Historique mensuel ────────────────────────────────
    if y - ST_H - TH_H - 4 * TR_H > SAFE_Y:
        y = _sec(c, ML, y,
                 "HISTORIQUE DE PERFORMANCE MENSUELLE — PORTEFEUILLE HEDGE USD")
        m_all = (port_ret.resample("ME")
                 .apply(lambda x: (1 + x).prod() - 1) * 100)
        years = sorted(m_all.index.year.unique())[-4:]
        mois  = ["Jan","Fev","Mar","Avr","Mai","Jun",
                  "Jul","Aou","Sep","Oct","Nov","Dec","Annuel"]
        hw    = [13*mm] + [10.8*mm] * 13
        y     = _thead(c, ML, y, ["Annee"] + mois, hw)
        hrows = []
        for yr in years:
            row = [str(yr)]; acc = 1.0
            for mo in range(1, 13):
                mask = (m_all.index.year == yr) & (m_all.index.month == mo)
                if mask.any():
                    v = m_all[mask].iloc[0]; acc *= (1 + v/100)
                    row.append(fp(v, 1))
                else:
                    row.append("—")
            row.append(fp((acc - 1)*100, 1))
            hrows.append(row)
        y = _trows(c, ML, y, hrows, hw,
                   aligns=["left"] + ["right"] * 13, rh=4.6*mm)
        y -= GAP

    # ── Informations générales ────────────────────────────
    if y - ST_H - 6 * TR_H > SAFE_Y:
        y  = _sec(c, ML, y, "INFORMATIONS GENERALES")
        iw = [52*mm, CW - 52*mm]
        _trows(c, ML, y, [
            ["Strategie",           "Multi-actifs couvert USD"],
            ["Benchmark", "38% IEV | 11% ACWI | 24% AGGG | 5% EPRA | 8% EEM | 14% SHV"],
            ["Couverture de change", "USD/EUR — simulation forward (sans cout inclus)"],
            ["Nombre de lignes",     f"{len(weights) + 1} (dont cash)"],
            ["Univers",              "Europe, Etats-Unis, Marches emergents"],
            ["Sources de donnees",   "Yahoo Finance / FMP via OpenBB"],
        ], iw, aligns=["left","left"], rh=4.0*mm)

# ═══════════════════════════════════════════════════════════
# ENTRÉE PRINCIPALE
# ═══════════════════════════════════════════════════════════

def generate_factsheet_pdf(port_idx, bench_idx, port_ret,
                           allocation, ticker_names, weights,
                           prices_eur=None) -> bytes:
    """
    Parametres identiques a l'appel dans app.py, avec prices_eur en plus
    pour le Top 5 reel.

    Dans app.py, remplacer l'appel par :
        pdf = generate_factsheet_pdf(
            portfolio_index_hedged, bench_index, portfolio_returns_hedged,
            allocation, ticker_names, weights,
            prices_eur=prices_eur          # <— ajouter ce parametre
        )
    """
    buf   = io.BytesIO()
    c     = rl_canvas.Canvas(buf, pagesize=A4)
    m     = compute_metrics(port_idx, bench_idx, port_ret)
    as_of = port_idx.index[-1].strftime("%d/%m/%Y")
    today = datetime.datetime.now().strftime("%d/%m/%Y")

    _page1(c, port_idx, bench_idx, port_ret, weights, m, as_of, today)
    c.showPage()
    _page2(c, port_ret, allocation, ticker_names, weights,
           prices_eur, as_of, today)
    c.save()
    buf.seek(0)
    return buf.read()

# ═══════════════════════════════════════════════════════════
# WIDGET STREAMLIT
# ═══════════════════════════════════════════════════════════

def render_factsheet_section(port_idx, bench_idx, port_ret,
                             allocation, ticker_names, weights,
                             prices_eur=None):
    st.divider()
    st.subheader("Factsheet mensuelle PDF")
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown(
            "**Page 1** : KPIs, performance base 100, drawdown, indicateurs de risque\n\n"
            "**Page 2** : Composition, allocation, **Top 5 lignes reelles du mois**, "
            "historique mensuel"
        )
    with col_btn:
        if st.button("Generer la factsheet", type="primary",
                     use_container_width=True):
            with st.spinner("Generation en cours..."):
                try:
                    pdf = generate_factsheet_pdf(
                        port_idx, bench_idx, port_ret,
                        allocation, ticker_names, weights,
                        prices_eur=prices_eur)
                    fname = (f"factsheet_"
                             f"{datetime.date.today().strftime('%Y%m%d')}.pdf")
                    st.download_button(
                        "Telecharger le PDF",
                        data=pdf, file_name=fname,
                        mime="application/pdf",
                        use_container_width=True)
                    st.success("Factsheet generee avec succes !")
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    raise
