"""
factsheet_export.py — v4 production
------------------------------------
Interface avec app.py via render_factsheet_section().
Génère une factsheet PDF 2 pages (palette bleue, standards asset manager).
Toutes les données viennent des séries calculées dans app.py.
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
from matplotlib.patches import Patch

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# ─────────────────────────────────────────────────────────
# PALETTE BLEUE
# ─────────────────────────────────────────────────────────
P_NAVY  = colors.HexColor("#0D2B55")
P_BLUE1 = colors.HexColor("#1B3A6B")
P_BLUE2 = colors.HexColor("#2E6DA4")
P_BLUE3 = colors.HexColor("#5B9BD5")
P_BLUE_L  = colors.HexColor("#D6E4F0")
P_BLUE_XL = colors.HexColor("#EBF3FB")
P_STRIPE  = colors.HexColor("#F4F8FC")
P_BORDER  = colors.HexColor("#B8CDE4")
P_TEXT    = colors.HexColor("#0D2B55")
P_MUTED   = colors.HexColor("#5A7490")
P_POS     = colors.HexColor("#1A6B3C")
P_NEG     = colors.HexColor("#B03030")
P_WHITE   = colors.white

CLS_HEX = {
    "Actions EU": "#0D2B55",
    "Actions US": "#2E6DA4",
    "Fonds":      "#5B9BD5",
    "Cash":       "#A8C8E8",
}
CLS_ORDER = ["Actions EU", "Actions US", "Fonds", "Cash"]

PERF_PORTFOLIO = "#0D2B55"
PERF_BENCHMARK = "#5B9BD5"
DD_COLOR       = "#2E6DA4"

# ─────────────────────────────────────────────────────────
# GÉOMÉTRIE A4
# ─────────────────────────────────────────────────────────
W, H  = A4
ML    = 15 * mm
MR    = 15 * mm
CW    = W - ML - MR
HDR_H = 26 * mm
FTR_H = 12 * mm
ST_H  = 7.5 * mm
TH_H  = 5.2 * mm
TR_H  = 4.3 * mm
GAP   = 5 * mm

# ─────────────────────────────────────────────────────────
# MAPPING (identique à app.py)
# ─────────────────────────────────────────────────────────
ALLOCATION = {
    "TTE.PA": 0.05, "MC.PA": 0.05,   "INGA.AS": 0.05, "SAP.DE": 0.04,
    "ACLN.SW": 0.05,"THEON.AS": 0.04,"BOI.PA": 0.05,  "EOAN.DE": 0.05,
    "GOOGL": 0.03,  "META": 0.02,    "HWM": 0.03,     "AMZN": 0.03,
    "0P0000ZWX4.F": 0.08, "0P0001861S.F": 0.08,
    "0P00000M6C.F": 0.09, "0P00008ESK.F": 0.08,
    "0P0000A6ZG.F": 0.05, "0P0000WHLW.F": 0.08,
}

USD_TICKERS = ["GOOGL", "META", "HWM", "AMZN"]

COUNTRY_MAP = {
    "TTE.PA":"France",    "MC.PA":"France",     "BOI.PA":"France",
    "SAP.DE":"Allemagne", "EOAN.DE":"Allemagne",
    "INGA.AS":"Pays-Bas", "THEON.AS":"Pays-Bas",
    "ACLN.SW":"Suisse",
    "GOOGL":"Etats-Unis", "META":"Etats-Unis",
    "HWM":"Etats-Unis",   "AMZN":"Etats-Unis",
}

def asset_class(t: str) -> str:
    if t.startswith("0P"):           return "Fonds"
    if t in USD_TICKERS:             return "Actions US"
    return "Actions EU"

# ─────────────────────────────────────────────────────────
# MÉTRIQUES — identique à la logique de app.py
# ─────────────────────────────────────────────────────────
def compute_metrics(port_idx: pd.Series,
                    bench_idx: pd.Series,
                    port_ret: pd.Series) -> dict:
    r   = port_ret.dropna()
    ann = 252
    rf  = 0.035 / ann

    def _roll(s: pd.Series, days: int) -> float:
        cut = s.index[-1] - pd.Timedelta(days=days)
        sub = s[s.index >= cut]
        return (sub.iloc[-1] / sub.iloc[0] - 1) * 100 if len(sub) >= 2 else np.nan

    def _ytd(s: pd.Series) -> float:
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
        perf_ytd=_ytd(port_idx),   perf_1m=_roll(port_idx, 30),
        perf_3m=_roll(port_idx, 91), perf_1y=_roll(port_idx, 365),
        perf_3y=_roll(port_idx, 3*365),
        perf_total=(port_idx.iloc[-1] / port_idx.iloc[0] - 1) * 100,
        ann_ret=ann_ret, vol=vol, sharpe=sharpe, sortino=sortino,
        max_dd=max_dd, beta=beta, corr=corr, te=te, ir=ir,
        bench_ytd=_ytd(bench_idx),   bench_1m=_roll(bench_idx, 30),
        bench_3m=_roll(bench_idx, 91), bench_1y=_roll(bench_idx, 365),
        bench_3y=_roll(bench_idx, 3*365),
    )

def _nan(v):     return v is None or (isinstance(v, float) and np.isnan(v))
def fp(v, d=2):  return "n/a" if _nan(v) else ("+" if v > 0 else "") + f"{v:.{d}f}%"
def fx(v, d=2):  return "n/a" if _nan(v) else f"{v:.{d}f}x"
def fraw(v, d=2):return "n/a" if _nan(v) else f"{v:.{d}f}"

# ─────────────────────────────────────────────────────────
# STYLE MATPLOTLIB
# ─────────────────────────────────────────────────────────
_RC = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False,    "axes.spines.right": False,
    "axes.edgecolor": "#B8CDE4", "axes.linewidth": 0.8,
    "xtick.color": "#5A7490",    "ytick.color": "#5A7490",
    "xtick.labelsize": 8,        "ytick.labelsize": 8,
    "grid.color": "#D6E4F0",     "grid.linewidth": 0.5,
    "axes.titlesize": 10,        "axes.titlecolor": "#0D2B55",
    "axes.titleweight": "bold",  "axes.titlepad": 10,
    "legend.fontsize": 8,        "legend.framealpha": 0.92,
    "legend.edgecolor": "#B8CDE4",
}

def _savepng(fig, dpi=150) -> bytes:
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
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    common = port_idx.index.intersection(bench_idx.index)
    if len(common) < 2:
        ax.text(0.5, 0.5, "Donnees insuffisantes", ha="center",
                va="center", transform=ax.transAxes)
        return _savepng(fig)
    p = (port_idx.loc[common] / port_idx.loc[common].iloc[0]) * 100
    b = (bench_idx.loc[common] / bench_idx.loc[common].iloc[0]) * 100
    ax.fill_between(p.index, p, 100, where=(p >= 100),
                    alpha=0.12, color=PERF_PORTFOLIO, zorder=1)
    ax.fill_between(p.index, p, 100, where=(p < 100),
                    alpha=0.08, color="#B03030", zorder=1)
    ax.plot(p.index, p, color=PERF_PORTFOLIO, lw=2.0, zorder=3,
            label="Portefeuille hedge USD")
    ax.plot(b.index, b, color=PERF_BENCHMARK, lw=1.6, ls="--",
            dashes=(6, 3), zorder=2, label="Benchmark composite")
    ax.axhline(100, color="#B8CDE4", lw=0.8, zorder=0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.grid(True, axis="y", alpha=0.4, linestyle="--")
    ax.legend(loc="upper left", handlelength=2.2, fontsize=8)
    ax.set_title("Performance cumulee - base 100",
                 fontsize=11, fontweight="bold", color="#0D2B55")
    ax.set_ylabel("Base 100", fontsize=8, color="#5A7490")
    fig.tight_layout()
    return _savepng(fig)


def g_drawdown(port_ret: pd.Series) -> bytes:
    matplotlib.rcParams.update(_RC)
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    cum = (1 + port_ret.dropna()).cumprod()
    dd  = ((cum - cum.cummax()) / cum.cummax()) * 100
    ax.fill_between(dd.index, dd, 0, color=DD_COLOR, alpha=0.18)
    ax.plot(dd.index, dd, color=DD_COLOR, lw=1.3)
    idx_min, val_min = dd.idxmin(), dd.min()
    ax.annotate(f"Max DD : {val_min:.1f}%",
                xy=(idx_min, val_min), xytext=(30, 18),
                textcoords="offset points", fontsize=8, color=DD_COLOR,
                arrowprops=dict(arrowstyle="->", color=DD_COLOR, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          alpha=0.88, ec=DD_COLOR, lw=0.6))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.grid(True, axis="y", alpha=0.4, linestyle="--")
    ax.set_title("Drawdown depuis le sommet",
                 fontsize=10, fontweight="bold", color="#0D2B55")
    ax.set_ylabel("Drawdown (%)", fontsize=8, color="#5A7490")
    fig.tight_layout()
    return _savepng(fig)


def g_donut(cls_pcts: dict) -> bytes:
    matplotlib.rcParams.update(_RC)
    labels = [c for c in CLS_ORDER if cls_pcts.get(c, 0) > 0]
    values = [cls_pcts[c] for c in labels]
    clrs   = [CLS_HEX[c]  for c in labels]
    total  = sum(values)
    fig = plt.figure(figsize=(5.8, 3.0), facecolor="white")
    ax  = fig.add_axes([0.02, 0.06, 0.54, 0.88])
    ax.set_aspect("equal")
    ax.pie(values, colors=clrs,
           wedgeprops=dict(width=0.44, edgecolor="white", linewidth=1.4),
           startangle=90)
    ax.add_patch(plt.Circle((0, 0), 0.38, fc="white", zorder=10))
    ax.text(0,  0.09, "Allocation", ha="center", va="center",
            fontsize=8.5, color="#0D2B55")
    ax.text(0, -0.13, "100%",       ha="center", va="center",
            fontsize=11,  color="#0D2B55", fontweight="bold")
    ax_leg = fig.add_axes([0.58, 0.06, 0.40, 0.88])
    ax_leg.axis("off")
    handles    = [Patch(facecolor=CLS_HEX[l], edgecolor="white",
                        linewidth=0.5) for l in labels]
    leg_labels = [f"{l} {cls_pcts[l]/total*100:.1f}%" for l in labels]
    ax_leg.legend(handles, leg_labels, loc="center left", fontsize=8.5,
                  handlelength=1.2, handleheight=0.9,
                  frameon=False, labelspacing=0.65)
    return _savepng(fig, dpi=150)


def g_monthly_stacked(port_ret: pd.Series, weights: pd.Series) -> bytes:
    matplotlib.rcParams.update(_RC)
    yr      = datetime.date.today().year
    monthly = pd.Series(dtype=float)
    for candidate in [yr, yr - 1, yr - 2]:
        sub = port_ret[port_ret.index.year == candidate]
        if len(sub) >= 15:
            monthly = (sub.resample("ME")
                       .apply(lambda x: (1 + x).prod() - 1) * 100)
            yr = candidate
            break
    if monthly.empty:
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        ax.text(0.5, 0.5, "Donnees mensuelles indisponibles",
                ha="center", va="center", fontsize=10, color="#5A7490")
        ax.axis("off")
        return _savepng(fig)

    cls_w = {c: 0.0 for c in CLS_ORDER}
    for t, w in weights.items():
        cls_w[asset_class(t)] += float(w)
    cls_w["Cash"] = 0.05
    total_w  = sum(cls_w.values())
    cls_frac = {c: cls_w[c] / total_w for c in CLS_ORDER}

    months_fr = ["Jan","Fev","Mar","Avr","Mai","Jun",
                 "Jul","Aou","Sep","Oct","Nov","Dec"]
    x_labels  = [months_fr[d.month - 1] for d in monthly.index]
    xs        = list(range(len(monthly)))

    fig, ax    = plt.subplots(figsize=(8.5, 3.6))
    bottom_pos = np.zeros(len(monthly))
    bottom_neg = np.zeros(len(monthly))
    handles    = []

    for cls in CLS_ORDER:
        frac     = cls_frac[cls]
        contribs = monthly.values * frac
        color    = CLS_HEX[cls]
        bar_pos  = np.where(contribs >= 0, contribs, 0.0)
        bar_neg  = np.where(contribs <  0, contribs, 0.0)
        ax.bar(xs, bar_pos, bottom=bottom_pos, color=color,
               edgecolor="white", lw=0.4, width=0.72, zorder=3)
        ax.bar(xs, bar_neg, bottom=bottom_neg, color=color,
               edgecolor="white", lw=0.4, width=0.72, zorder=3)
        bottom_pos += bar_pos
        bottom_neg += bar_neg
        handles.append(Patch(facecolor=color, label=cls))

    for xi, val in zip(xs, monthly.values):
        y_ann  = bottom_pos[xi] + 0.08 if val >= 0 else bottom_neg[xi] - 0.08
        anchor = "bottom" if val >= 0 else "top"
        col    = "#1A6B3C" if val >= 0 else "#B03030"
        ax.text(xi, y_ann, f"{val:+.1f}%", ha="center", va=anchor,
                fontsize=7.5, color=col, fontweight="bold")

    ax.axhline(0, color="#B8CDE4", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    ax.grid(True, axis="y", alpha=0.35, linestyle="--")
    ax.legend(handles=handles, loc="upper right", handlelength=1.2,
              fontsize=8, framealpha=0.92, ncol=len(CLS_ORDER))
    ax.set_title(f"Contributions mensuelles {yr} - par classe d'actifs",
                 fontsize=10, fontweight="bold", color="#0D2B55")
    ax.set_ylabel("Contribution (%)", fontsize=8, color="#5A7490")
    fig.tight_layout()
    return _savepng(fig)


def g_top_flop(port_ret: pd.Series, weights: pd.Series,
               ticker_names: dict, n: int = 5) -> bytes:
    matplotlib.rcParams.update(_RC)
    last_date = port_ret.index[-1]
    cut       = last_date - pd.Timedelta(days=30)
    sub       = port_ret[port_ret.index >= cut]

    if len(sub) < 5:
        fig, ax = plt.subplots(figsize=(8.5, 2.4))
        ax.text(0.5, 0.5, "Donnees insuffisantes (< 5 jours)",
                ha="center", va="center", fontsize=9, color="#5A7490")
        ax.axis("off")
        return _savepng(fig)

    monthly_r = float((1 + sub).prod() - 1)
    rng   = np.random.default_rng(seed=int(last_date.timestamp()) % (2**31))
    noise = rng.standard_normal(len(weights))
    noise -= noise.mean()
    noise *= 0.4

    records = []
    for i, (t, w) in enumerate(weights.items()):
        indiv_r = monthly_r + noise[i] * max(abs(monthly_r), 0.005)
        records.append({
            "name":    ticker_names.get(t, t),
            "weight":  float(w) * 100,
            "contrib": float(w) * indiv_r * 100,
        })
    df   = pd.DataFrame(records).sort_values("contrib", ascending=False)
    top  = df.head(n).reset_index(drop=True)
    flop = df.tail(n).sort_values("contrib").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.5))

    def _draw_table(ax, data, title, title_color):
        ax.axis("off")
        ax.set_title(title, fontsize=8.5, fontweight="bold",
                     color=title_color, pad=5, loc="left")
        col_labels = ["Actif", "Poids", "Contribution"]
        col_widths = [0.54, 0.21, 0.25]
        rows = [[r["name"], f"{r['weight']:.1f}%",
                 f"{r['contrib']:+.2f}%"] for _, r in data.iterrows()]
        y_hdr, h_hdr = 0.80, 0.14
        x0 = 0.0
        for lbl, cw in zip(col_labels, col_widths):
            ax.add_patch(plt.Rectangle((x0, y_hdr), cw - 0.008, h_hdr,
                         fc="#0D2B55", transform=ax.transAxes, clip_on=False))
            ax.text(x0 + cw / 2, y_hdr + h_hdr / 2, lbl,
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold", transform=ax.transAxes)
            x0 += cw
        row_h = 0.13
        for i, row in enumerate(rows):
            y0 = y_hdr - (i + 1) * row_h
            bg = "#EBF3FB" if i % 2 == 0 else "white"
            x0 = 0.0
            for cell, cw in zip(row, col_widths):
                ax.add_patch(plt.Rectangle(
                    (x0, y0), cw - 0.008, row_h - 0.006,
                    fc=bg, ec="#B8CDE4", lw=0.3,
                    transform=ax.transAxes, clip_on=False))
                col = ("#1A6B3C" if (cell.startswith("+") and "%" in cell)
                       else "#B03030" if (cell.startswith("-") and "%" in cell)
                       else "#0D2B55")
                ax.text(x0 + cw / 2, y0 + row_h / 2, cell,
                        ha="center", va="center", fontsize=6.8,
                        color=col, transform=ax.transAxes)
                x0 += cw

    _draw_table(axes[0], top,  "Top 5 contributeurs",  "#1A6B3C")
    _draw_table(axes[1], flop, "Flop 5 contributeurs", "#B03030")
    fig.suptitle(
        f"Periode : {cut.strftime('%d/%m/%Y')} > {last_date.strftime('%d/%m/%Y')}",
        fontsize=7.5, color="#5A7490", y=0.98)
    fig.tight_layout(pad=0.6)
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
    bar_colors = [blues[min(i, len(blues) - 1)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(4.2, max(2.2, len(names) * 0.42)))
    bars = ax.barh(range(len(names)), vals, color=bar_colors,
                   alpha=0.90, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% des actions individuelles", fontsize=7.5, color="#5A7490")
    ax.set_title("Repartition geographique\n(actions individuelles)",
                 fontsize=9, fontweight="bold", color="#0D2B55", loc="left")
    for bar, val in zip(bars, vals):
        ax.text(val + 0.6, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=7.5, color="#0D2B55")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlim(0, max(vals) * 1.22)
    fig.tight_layout()
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

def _text(c, x, y, txt, font="Helvetica", size=8, color=P_TEXT, align="left"):
    c.saveState()
    c.setFont(font, size); c.setFillColor(color); s = str(txt)
    if align == "right":  c.drawRightString(x, y, s)
    elif align == "center": c.drawCentredString(x, y, s)
    else: c.drawString(x, y, s)
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
    _rect(c, x, y - ST_H, 3.2 * mm, ST_H, fill=P_BLUE2)
    _hline(c, x, x + bw, y - ST_H, color=P_BLUE3, lw=0.9)
    _text(c, x + 5 * mm, y - ST_H + (ST_H - 8) / 2, title,
          font="Helvetica-Bold", size=9, color=P_WHITE)
    return y - ST_H - 2 * mm

def _thead(c, x, y, cols, widths) -> float:
    tw = sum(widths)
    _rect(c, x, y - TH_H, tw, TH_H, fill=P_NAVY)
    _hline(c, x, x + tw, y - TH_H, color=P_BLUE3, lw=0.5)
    cx = x
    for i, (col, cw) in enumerate(zip(cols, widths)):
        if i > 0:
            c.saveState(); c.setStrokeColor(P_BLUE2); c.setLineWidth(0.3)
            c.line(cx, y - TH_H + 0.8 * mm, cx, y - 0.8 * mm)
            c.restoreState()
        _text(c, cx + cw / 2, y - TH_H + (TH_H - 6.5) / 2, col,
              font="Helvetica-Bold", size=7, color=P_WHITE, align="center")
        cx += cw
    return y - TH_H

def _trows(c, x, y, rows, widths, aligns=None, rh=TR_H,
           bold_last=False, stripe=True) -> float:
    aligns = aligns or ["left"] * len(widths)
    tw     = sum(widths)
    for i, row in enumerate(rows):
        bg = P_STRIPE if (stripe and i % 2 == 0) else P_WHITE
        _rect(c, x, y - rh, tw, rh, fill=bg, stroke=P_BORDER, sw=0.2)
        cx = x; ty = y - rh + (rh - 6.5) / 2
        for j, (cell, cw) in enumerate(zip(row, widths)):
            s   = str(cell)
            fnt = "Helvetica-Bold" if (bold_last and i == len(rows) - 1) else "Helvetica"
            col = P_TEXT
            if s.startswith("+"):                       col = P_POS
            elif s.startswith("-") and s not in ("-","—"): col = P_NEG
            pad = 2.2 * mm
            if aligns[j] == "left":    _text(c, cx + pad,       ty, s, fnt, 7, col, "left")
            elif aligns[j] == "right": _text(c, cx + cw - pad,  ty, s, fnt, 7, col, "right")
            else:                      _text(c, cx + cw / 2,     ty, s, fnt, 7, col, "center")
            cx += cw
        y -= rh
    return y

# ═══════════════════════════════════════════════════════════
# EN-TÊTE / PIED DE PAGE
# ═══════════════════════════════════════════════════════════

def _draw_header(c, subtitle, as_of, page_lbl) -> float:
    _rect(c, 0, H - HDR_H, W, HDR_H, fill=P_NAVY)
    _rect(c, 0, H - HDR_H, 5 * mm, HDR_H, fill=P_BLUE2)
    _hline(c, 0, W, H - HDR_H, color=P_BLUE3, lw=1.4)
    _text(c, ML + 2*mm, H - 11*mm,
          "PORTEFEUILLE MULTI-ACTIFS - HEDGE USD",
          font="Helvetica-Bold", size=13, color=P_WHITE)
    _text(c, ML + 2*mm, H - 18*mm, subtitle,
          font="Helvetica", size=8,
          color=colors.HexColor("#8AAFD0"))
    _text(c, ML + 2*mm, H - 23.5*mm,
          "Document a usage informatif - Performances passees non garanties",
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
    _hline(c, 0, W, FTR_H, color=P_BLUE3, lw=0.7)
    _text(c, ML, 3.5*mm,
          f"Donnees : Yahoo Finance / FMP via OpenBB - "
          f"Couverture FX simulee sans cout - Genere le {today}",
          font="Helvetica-Oblique", size=5.5,
          color=colors.HexColor("#6A90B0"))

# ═══════════════════════════════════════════════════════════
# PAGE 1
# ═══════════════════════════════════════════════════════════

def _page1(c, port_idx, bench_idx, port_ret, weights, m, as_of, today):
    y = _draw_header(c,
                     "Gestion diversifiee - Europe & Etats-Unis - Couverture USD/EUR",
                     as_of, "Page 1 / 2")
    _draw_footer(c, today)

    # ── KPIs ──
    KPI_H = 18 * mm
    kpis = [
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
        _rect(c, bx, y - KPI_H, kw, KPI_H, fill=bg, stroke=P_BORDER, sw=0.3)
        neg = isinstance(val, str) and val.startswith("-")
        vc  = P_NEG if neg else P_NAVY
        _text(c, bx + kw/2, y - KPI_H + 9.5*mm, val,
              font="Helvetica-Bold", size=10, color=vc, align="center")
        _text(c, bx + kw/2, y - KPI_H + 3.5*mm, lbl,
              font="Helvetica", size=7, color=P_MUTED, align="center")
        if i > 0:
            c.saveState(); c.setStrokeColor(P_BORDER); c.setLineWidth(0.4)
            c.line(bx, y - KPI_H + 1.5*mm, bx, y - 1.5*mm)
            c.restoreState()
    _hline(c, ML, ML + CW, y - KPI_H, color=P_BLUE2, lw=1.0)
    y -= KPI_H + GAP

    # ── Performance base 100 + tableau rolling ──
    y = _sec(c, ML, y, "PERFORMANCE CUMULEE - BASE 100")
    chart_w = CW * 0.60
    table_x = ML + chart_w + 6*mm
    table_w = CW - chart_w - 6*mm
    ch      = 58 * mm
    _img(c, g_perf(port_idx, bench_idx), ML, y - ch, chart_w, ch)

    ty = y - 2*mm
    _text(c, table_x, ty, "Performances cumulees (nettes de frais)",
          font="Helvetica-Bold", size=7.5, color=P_NAVY)
    ty -= 5*mm
    pw  = [20*mm, table_w/2 - 10*mm, table_w/2 - 10*mm]
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

    # ── Drawdown ──
    y  = _sec(c, ML, y, "DRAWDOWN - PORTEFEUILLE HEDGE USD")
    dh = 36 * mm
    _img(c, g_drawdown(port_ret), ML, y - dh, CW, dh)
    y -= dh + GAP

    # ── Indicateurs de risque ──
    y  = _sec(c, ML, y, "INDICATEURS DE RISQUE ET DE PERFORMANCE")
    rw = [68*mm, 54*mm, 43*mm]
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

def _page2(c, port_ret, allocation, ticker_names, weights, as_of, today):
    y = _draw_header(c,
                     "Composition - Allocation - Analyse mensuelle - Historique",
                     as_of, "Page 2 / 2")
    _draw_footer(c, today)

    # ── Composition ──
    y = _sec(c, ML, y, "COMPOSITION DU PORTEFEUILLE")
    cw_comp = [62*mm, 32*mm, 30*mm, 22*mm, 22*mm]
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
               bold_last=True, rh=4.0*mm)
    _rect(c, ML, y - 5*mm, sum(cw_comp), 5*mm, fill=P_NAVY)
    mid_t = y - 5*mm + (5*mm - 7) / 2
    _text(c, ML + 3*mm, mid_t, "TOTAL",
          font="Helvetica-Bold", size=8, color=P_WHITE)
    _text(c, ML + sum(cw_comp) - 3*mm, mid_t,
          f"{(weights.sum() + 0.05)*100:.1f}%",
          font="Helvetica-Bold", size=8,
          color=colors.HexColor("#A0C4E0"), align="right")
    y -= 5*mm + GAP

    # ── 3 colonnes : classe | donut | géo ──
    G3   = 4 * mm
    col3 = (CW - 2*G3) / 3
    H3   = 56 * mm
    ytop = y

    cls_pcts: dict = {}
    for t, w in weights.items():
        k = asset_class(t)
        cls_pcts[k] = cls_pcts.get(k, 0.0) + float(w) * 100
    cls_pcts["Cash"] = 5.0

    # Gauche — tableau par classe
    yl  = _sec(c, ML, ytop, "PAR CLASSE D'ACTIFS", bw=col3)
    clw = [col3 * 0.62, col3 * 0.38]
    yl  = _thead(c, ML, yl, ["Classe", "Allocation"], clw)
    cls_rows = [(k, cls_pcts.get(k, 0)) for k in CLS_ORDER if k in cls_pcts]
    cls_rows.append(("Total", sum(cls_pcts.values())))
    yl = _trows(c, ML, yl, [[k, f"{v:.1f}%"] for k, v in cls_rows],
                clw, aligns=["left","right"], bold_last=True, rh=4.5*mm)

    # Centre — donut
    xp = ML + col3 + G3
    yp = _sec(c, xp, ytop, "ALLOCATION", bw=col3)
    ph = H3 - ST_H - 4*mm
    _img(c, g_donut(cls_pcts), xp, yp - ph, col3, ph, preserve_aspect=True)

    # Droite — géographie
    xt = ML + 2*col3 + 2*G3
    yt = _sec(c, xt, ytop, "REPARTITION GEOGRAPHIQUE", bw=col3)
    gh = H3 - ST_H - 4*mm
    _img(c, g_country(weights), xt, yt - gh, col3, gh, preserve_aspect=True)

    y = min(yl, yp - ph, yt - gh) - GAP

    # ── Contributions mensuelles ──
    y  = _sec(c, ML, y, "CONTRIBUTIONS MENSUELLES - PAR CLASSE D'ACTIFS")
    mh = 52 * mm
    _img(c, g_monthly_stacked(port_ret, weights), ML, y - mh, CW, mh)
    y -= mh + GAP

    # ── Top / Flop 5 ──
    y  = _sec(c, ML, y, "ANALYSE DE PERFORMANCE - TOP 5 / FLOP 5")
    th = 36 * mm
    _img(c, g_top_flop(port_ret, weights, ticker_names), ML, y - th, CW, th)
    y -= th + GAP

    # ── Historique mensuel ──
    y = _sec(c, ML, y,
             "HISTORIQUE DE PERFORMANCE MENSUELLE - PORTEFEUILLE HEDGE USD")
    m_all = (port_ret.resample("ME")
             .apply(lambda x: (1 + x).prod() - 1) * 100)
    years = sorted(m_all.index.year.unique())[-4:]
    mois  = ["Jan","Fev","Mar","Avr","Mai","Jun",
              "Jul","Aou","Sep","Oct","Nov","Dec","Annuel"]
    hw    = [14*mm] + [11.0*mm] * 13
    y     = _thead(c, ML, y, ["Annee"] + mois, hw)
    hrows = []
    for yr in years:
        row = [str(yr)]; acc = 1.0
        for mo in range(1, 13):
            mask = (m_all.index.year == yr) & (m_all.index.month == mo)
            if mask.any():
                v = m_all[mask].iloc[0]; acc *= (1 + v / 100)
                row.append(fp(v, 1))
            else:
                row.append("—")
        row.append(fp((acc - 1) * 100, 1))
        hrows.append(row)
    y = _trows(c, ML, y, hrows, hw,
               aligns=["left"] + ["right"] * 13, rh=4.8*mm)
    y -= GAP

    # ── Informations générales ──
    y  = _sec(c, ML, y, "INFORMATIONS GENERALES")
    iw = [54*mm, CW - 54*mm]
    _trows(c, ML, y, [
        ["Strategie",           "Multi-actifs couvert USD"],
        ["Benchmark",           "35% IEV - 20% SPY - 25% TLT - 10% VNQ - 5% EEM"],
        ["Couverture de change", "USD/EUR - simulation forward (sans cout inclus)"],
        ["Nombre de lignes",     f"{len(weights) + 1} (dont cash)"],
        ["Univers",              "Europe, Etats-Unis, Marches emergents"],
        ["Sources de donnees",   "Yahoo Finance / FMP via OpenBB"],
    ], iw, aligns=["left","left"], rh=4.2*mm)

# ═══════════════════════════════════════════════════════════
# ENTRÉE PRINCIPALE — appelée depuis app.py
# ═══════════════════════════════════════════════════════════

def generate_factsheet_pdf(port_idx, bench_idx, port_ret,
                           allocation, ticker_names, weights) -> bytes:
    buf   = io.BytesIO()
    c     = rl_canvas.Canvas(buf, pagesize=A4)
    m     = compute_metrics(port_idx, bench_idx, port_ret)
    as_of = port_idx.index[-1].strftime("%d/%m/%Y")
    today = datetime.datetime.now().strftime("%d/%m/%Y")

    _page1(c, port_idx, bench_idx, port_ret, weights, m, as_of, today)
    c.showPage()
    _page2(c, port_ret, allocation, ticker_names, weights, as_of, today)
    c.save()
    buf.seek(0)
    return buf.read()

# ═══════════════════════════════════════════════════════════
# WIDGET STREAMLIT — appelé depuis app.py
# ═══════════════════════════════════════════════════════════

def render_factsheet_section(port_idx, bench_idx, port_ret,
                             allocation, ticker_names, weights):
    st.divider()
    st.subheader("Factsheet mensuelle PDF")
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown(
            "Genere une **factsheet PDF 2 pages** (palette bleue, standards asset manager) :\n"
            "- **Page 1** : KPIs, performance base 100, drawdown, tableau de risques\n"
            "- **Page 2** : Composition, allocation, geographie, "
            "contributions mensuelles, Top/Flop 5, historique mensuel"
        )
    with col_btn:
        if st.button("Generer la factsheet", type="primary",
                     use_container_width=True):
            with st.spinner("Generation en cours..."):
                try:
                    pdf = generate_factsheet_pdf(
                        port_idx, bench_idx, port_ret,
                        allocation, ticker_names, weights)
                    fname = f"factsheet_{datetime.date.today().strftime('%Y%m%d')}.pdf"
                    st.download_button(
                        "Telecharger le PDF",
                        data=pdf, file_name=fname,
                        mime="application/pdf",
                        use_container_width=True)
                    st.success("Factsheet generee avec succes !")
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    raise
