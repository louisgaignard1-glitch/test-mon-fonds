"""
factsheet_export.py — Version Asset Manager (BlackRock/Amundi Style)
---------------------------------------------------------------------
Standards appliqués :
  • Couleurs : Bleu marine (#1B3A6B), Or (#B8972A), Vert/rouge pour les performances
  • Polices : Helvetica, hiérarchie claire (titres 12pt, texte 8pt)
  • Graphiques : Axes datés, légendes externes, grilles discrètes
  • Tableaux : En-têtes bleu marine, alternance de couleurs, totaux en gras
  • Mise en page : Marges 15mm, espacements généreux, séparateurs dorés
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

# =============================================
# PALETTE DE COULEURS (Standards Asset Manager)
# =============================================
P_NAVY    = colors.HexColor("#1B3A6B")  # Bleu marine (titres, en-têtes)
P_GOLD    = colors.HexColor("#B8972A")  # Or (accents, lignes)
P_GOLD_L  = colors.HexColor("#FBF3DF")  # Or clair (fond KPI)
P_BLUE_L  = colors.HexColor("#EDF2FB")  # Bleu très clair (alternance tableaux)
P_STRIPE  = colors.HexColor("#F5F8FC")  # Fond lignes paires
P_BORDER  = colors.HexColor("#C8D4E8")  # Bordures
P_TEXT    = colors.HexColor("#1A2E4A")  # Texte principal
P_MUTED   = colors.HexColor("#607080")  # Texte secondaire
P_POS     = colors.HexColor("#1A6B3C")  # Vert (performances positives)
P_NEG     = colors.HexColor("#B03030")  # Rouge (performances négatives)
P_WHITE   = colors.white

# Couleurs par classe d'actif (cohérentes partout)
CLS_COLORS = {
    "Actions EU": "#2E5FA3",  # Bleu
    "Actions US": "#E07B39",  # Orange
    "Fonds":      "#4A9B7F",  # Vert
    "Cash":       "#A0A8B8",  # Gris
}
CLS_ORDER = ["Actions EU", "Actions US", "Fonds", "Cash"]

# =============================================
# GÉOMÉTRIE (Marges, hauteurs, espacements)
# =============================================
W, H = A4  # 595 x 842 pts
ML = 15 * mm  # Marge gauche
MR = 15 * mm  # Marge droite
CW = W - ML - MR  # Largeur utile
HDR_H = 26 * mm  # Hauteur en-tête
FTR_H = 12 * mm  # Hauteur pied de page
ST_H = 7.5 * mm  # Hauteur titre de section
TH_H = 5.2 * mm  # Hauteur en-tête tableau
TR_H = 4.3 * mm  # Hauteur ligne tableau
GAP = 5 * mm    # Espacement entre sections

# =============================================
# CLASSIFIEUR D'ACTIFS
# =============================================
def asset_class(ticker):
    if ticker.startswith("0P"):
        return "Fonds"
    if ticker in ["GOOGL", "META", "HWM", "AMZN"]:
        return "Actions US"
    return "Actions EU"

# =============================================
# MÉTRIQUES (Calcul des indicateurs)
# =============================================
def compute_metrics(port_idx, bench_idx, port_ret):
    r = port_ret.dropna()
    ann = 252
    rf = 0.035 / ann

    def _p(s, days):
        cut = s.index[-1] - pd.Timedelta(days=days)
        sub = s[s.index >= cut]
        return (sub.iloc[-1] / sub.iloc[0] - 1) * 100 if len(sub) > 1 else np.nan

    ytd_start = pd.Timestamp(datetime.date.today().year, 1, 1)
    def _ytd(s):
        sub = s[s.index >= ytd_start]
        return (sub.iloc[-1] / sub.iloc[0] - 1) * 100 if len(sub) > 1 else np.nan

    vol = r.std() * np.sqrt(ann) * 100
    sharpe = (r.mean() - rf) / r.std() * np.sqrt(ann) if r.std() > 0 else np.nan
    down_std = r[r < 0].std() * np.sqrt(ann)
    sortino = r.mean() * ann / down_std if down_std > 0 else np.nan
    cum = (1 + r).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    ann_ret = r.mean() * ann * 100

    common = port_ret.index.intersection(bench_idx.index)
    br = bench_idx.loc[common].pct_change().dropna()
    pr = port_ret.loc[br.index]
    beta = np.cov(pr, br)[0, 1] / np.var(br) if len(br) > 5 else np.nan
    corr = np.corrcoef(pr, br)[0, 1] if len(br) > 5 else np.nan
    te = (pr - br).std() * np.sqrt(ann) * 100
    ir = (pr - br).mean() * ann / ((pr - br).std() * np.sqrt(ann)) if (pr - br).std() > 0 else np.nan

    return {
        "perf_ytd": _ytd(port_idx), "perf_1m": _p(port_idx, 30),
        "perf_3m": _p(port_idx, 91), "perf_1y": _p(port_idx, 365),
        "perf_3y": _p(port_idx, 3*365), "perf_total": (port_idx.iloc[-1]/port_idx.iloc[0]-1)*100,
        "ann_ret": ann_ret, "vol": vol, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "beta": beta, "corr": corr, "te": te, "ir": ir,
        "bench_ytd": _ytd(bench_idx), "bench_1m": _p(bench_idx, 30),
        "bench_3m": _p(bench_idx, 91), "bench_1y": _p(bench_idx, 365),
        "bench_3y": _p(bench_idx, 3*365),
    }

# Formatters pour les valeurs
def _f(v, dec=2, suf=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return ("+" if v > 0 else "") + f"{v:.{dec}f}{suf}"

def fp(v, d=2):
    return _f(v, d, "%")

def fx(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{d}f}x"

def fraw(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{d}f}"

# =============================================
# STYLE MATPLOTLIB (Commun à tous les graphiques)
# =============================================
_RC = {
    "figure.facecolor": "white",  "axes.facecolor": "white",
    "axes.spines.top": False,     "axes.spines.right": False,
    "axes.spines.left": True,     "axes.spines.bottom": True,
    "axes.edgecolor": "#C8D4E8",  "axes.linewidth": 0.8,
    "xtick.color": "#607080",     "ytick.color": "#607080",
    "xtick.labelsize": 7,         "ytick.labelsize": 7,
    "grid.color": "#E8EEF6",      "grid.linewidth": 0.5,
    "axes.titlesize": 9,          "axes.titlecolor": "#1B3A6B",
    "axes.titleweight": "bold",   "axes.titlepad": 8,
    "legend.fontsize": 7,         "legend.framealpha": 0.9,
    "legend.edgecolor": "#C8D4E8",
}

def _png(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# =============================================
# GRAPHIQUE 1 : PERFORMANCE CUMULÉE (avec axes datés)
# =============================================
def g_perf(port_idx, bench_idx, w_pt, h_pt):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(max(w_pt/72, 4.0), max(h_pt/72, 2.5)))
        common = port_idx.index.intersection(bench_idx.index)
        p = (port_idx.loc[common] - 1) * 100
        b = (bench_idx.loc[common] - 1) * 100

        # Remplissage bleu clair sous la courbe du portefeuille
        ax.fill_between(p.index, p, 0, alpha=0.12, color="#2E5FA3", zorder=1)
        ax.plot(p.index, p, color="#2E5FA3", lw=2.0, zorder=3, label="Portefeuille hedgé USD")
        ax.plot(b.index, b, color="#B8972A", lw=1.5, ls="--", dashes=(5, 3), zorder=2, label="Benchmark composite")
        ax.axhline(0, color="#C8D4E8", lw=0.7, ls=":", zorder=0)

        # Axes et grille
        ax.set_xlabel("Date", fontsize=7, color="#607080", labelpad=5)
        ax.set_ylabel("Performance (%)", fontsize=7, color="#607080", labelpad=5)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
        ax.grid(True, axis="both", zorder=0, alpha=0.4, linestyle="--")
        ax.legend(loc="upper left", handlelength=2.0, fontsize=7)
        ax.set_title("Performance cumulée nette de frais", fontsize=9, fontweight="bold", pad=10)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.15)
        return _png(fig)

# =============================================
# GRAPHIQUE 2 : DRAWDOWN (avec annotation)
# =============================================
def g_dd(port_ret, w_pt, h_pt):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(max(w_pt/72, 3.0), max(h_pt/72, 2.0)))
        cum = (1 + port_ret.dropna()).cumprod()
        dd = ((cum - cum.cummax()) / cum.cummax()) * 100

        ax.fill_between(dd.index, dd, 0, color="#B03030", alpha=0.15, label="Drawdown")
        ax.plot(dd.index, dd, color="#B03030", lw=1.2)

        # Annotation du max drawdown
        idx_min = dd.idxmin()
        val_min = dd.min()
        ax.annotate(
            f"Max DD\n{val_min:.1f}%",
            xy=(idx_min, val_min), xytext=(25, 15), textcoords="offset points",
            fontsize=6, color="#B03030",
            arrowprops=dict(arrowstyle="->", color="#B03030", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="#B03030", lw=0.5)
        )

        ax.set_xlabel("Date", fontsize=7, color="#607080", labelpad=5)
        ax.set_ylabel("Drawdown (%)", fontsize=7, color="#607080", labelpad=5)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.grid(True, axis="both", zorder=0, alpha=0.4, linestyle="--")
        ax.legend(loc="lower left", handlelength=1.5, fontsize=7)
        ax.set_title("Drawdown depuis le sommet", fontsize=9, fontweight="bold", pad=10)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.18)
        return _png(fig)

# =============================================
# GRAPHIQUE 3 : BARRES MENSUELLES EMPILÉES
# =============================================
def g_monthly(port_ret, weights, w_pt, h_pt):
    today_yr = datetime.date.today().year
    monthly_total = pd.Series(dtype=float)
    for cand in [today_yr, today_yr - 1]:
        sub = port_ret[port_ret.index.year == cand]
        if not sub.empty:
            monthly_total = sub.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
            break

    # Poids par classe
    cls_weights = {c: 0.0 for c in CLS_ORDER}
    for t, w in weights.items():
        ac = asset_class(t)
        cls_weights[ac] = cls_weights.get(ac, 0.0) + float(w)
    cls_weights["Cash"] = 0.05
    total_w = sum(cls_weights.values())
    cls_frac = {c: cls_weights[c] / total_w for c in CLS_ORDER}

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(max(w_pt/72, 3.5), max(h_pt/72, 2.5)))
        if monthly_total.empty:
            ax.text(0.5, 0.5, "Données mensuelles\nindisponibles", ha="center", va="center", fontsize=8, color="#607080")
        else:
            xs = list(range(len(monthly_total)))
            mois_labels = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
            x_labels = [mois_labels[d.month - 1] for d in monthly_total.index]

            bottom_pos = np.zeros(len(monthly_total))
            bottom_neg = np.zeros(len(monthly_total))
            legend_handles = []

            for cls in CLS_ORDER:
                frac = cls_frac[cls]
                contribs = monthly_total.values * frac
                color = CLS_COLORS[cls]
                bar_pos = np.where(contribs >= 0, contribs, 0.0)
                bar_neg = np.where(contribs < 0, contribs, 0.0)

                ax.bar(xs, bar_pos, bottom=bottom_pos, color=color, edgecolor="white", lw=0.5, width=0.7, zorder=3)
                ax.bar(xs, bar_neg, bottom=bottom_neg, color=color, edgecolor="white", lw=0.5, width=0.7, zorder=3)
                bottom_pos += bar_pos
                bottom_neg += bar_neg
                legend_handles.append(Patch(facecolor=color, label=cls))

            # Valeurs totales au-dessus des barres
            for xi, val in zip(xs, monthly_total.values):
                ax.text(xi, val + 0.5, f"{val:+.1f}%", ha="center", va="bottom", fontsize=6, color="#1A2E4A", fontweight="bold")

            ax.set_xticks(xs)
            ax.set_xticklabels(x_labels, fontsize=6.5)
            ax.set_xlabel("Mois", fontsize=7, color="#607080", labelpad=5)
            ax.set_ylabel("Contribution (%)", fontsize=7, color="#607080", labelpad=5)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
            ax.grid(True, axis="y", zorder=0, alpha=0.4, linestyle="--")
            ax.legend(handles=legend_handles, loc="upper right", handlelength=1.2, fontsize=6.5, framealpha=0.9)

        ax.set_title(f"Contributions mensuelles {datetime.date.today().year} · par classe d'actifs", fontsize=9, fontweight="bold", pad=10)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.15)
        return _png(fig)

# =============================================
# GRAPHIQUE 4 : DONUT D'ALLOCATION (légende externe)
# =============================================
def g_pie(cls_data, side_pt):
    labels = [c for c in CLS_ORDER if c in cls_data and cls_data[c] > 0]
    values = [cls_data[c] for c in labels]
    clrs = [CLS_COLORS[c] for c in labels]
    total = sum(values)

    side_in = max(side_pt / 72, 2.0)
    fig = plt.figure(figsize=(side_in * 1.8, side_in), facecolor="white")

    # Donut (45% de la largeur)
    ax = fig.add_axes([0.05, 0.10, 0.45, 0.80])
    ax.set_aspect("equal")
    wedges, _ = ax.pie(
        values, colors=clrs,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5),
        startangle=90,
    )
    centre_circle = plt.Circle((0, 0), 0.40, fc="white", zorder=10)
    ax.add_patch(centre_circle)
    ax.text(0, 0, "Allocation\n100%", ha="center", va="center", fontsize=8, color="#1B3A6B", fontweight="bold")

    # Légende externe (50% de la largeur)
    ax_leg = fig.add_axes([0.55, 0.10, 0.40, 0.80])
    ax_leg.axis("off")
    handles = [Patch(facecolor=CLS_COLORS[l], edgecolor="white", linewidth=0.5) for l in labels]
    leg_labels = [f"{l}\n{cls_data[l]/total*100:.1f}%" for l in labels]
    ax_leg.legend(
        handles, leg_labels, loc="center left", fontsize=7,
        handlelength=1.2, handleheight=0.8, frameon=False, labelspacing=0.8
    )

    fig.text(0.05, 0.95, "Répartition par classe d'actifs", fontsize=8, fontweight="bold", color="#1B3A6B", va="top")
    return _png(fig, dpi=120)

# =============================================
# PRIMITIVES CANVAS (ReportLab)
# =============================================
def _rect(c, x, y, w, h, fill=None, stroke=None, sw=0.3):
    c.saveState()
    if fill: c.setFillColor(fill)
    if stroke: c.setStrokeColor(stroke); c.setLineWidth(sw)
    c.rect(x, y, w, h, fill=1 if fill else 0, stroke=1 if stroke else 0)
    c.restoreState()

def _text(c, x, y, text, font="Helvetica", size=8, color=P_TEXT, align="left"):
    c.saveState()
    c.setFont(font, size); c.setFillColor(color)
    if align == "right": c.drawRightString(x, y, str(text))
    elif align == "center": c.drawCentredString(x, y, str(text))
    else: c.drawString(x, y, str(text))
    c.restoreState()

def _hline(c, x1, x2, y, color=P_GOLD, lw=0.7):
    c.saveState(); c.setStrokeColor(color); c.setLineWidth(lw)
    c.line(x1, y, x2, y); c.restoreState()

def _img_rect(c, png_bytes, x, y, w, h):
    c.drawImage(ImageReader(io.BytesIO(png_bytes)), x, y, width=w, height=h, preserveAspectRatio=False)

def _img_fit(c, png_bytes, x, y, w, h):
    c.drawImage(ImageReader(io.BytesIO(png_bytes)), x, y, width=w, height=h, preserveAspectRatio=True, anchor="c")

def sec(c, x, y, title, w=None):
    bw = w if w is not None else CW
    _rect(c, x, y - ST_H, bw, ST_H, fill=P_NAVY)
    _rect(c, x, y - ST_H, 3.2*mm, ST_H, fill=P_GOLD)  # Bande dorée à gauche
    _hline(c, x, x + bw, y - ST_H, color=P_GOLD, lw=0.9)
    mid_y = y - ST_H + (ST_H - 8) / 2
    _text(c, x + 5*mm, mid_y, title, font="Helvetica-Bold", size=8, color=P_WHITE)
    return y - ST_H - 2*mm

def thead(c, x, y, cols, widths):
    _rect(c, x, y - TH_H, sum(widths), TH_H, fill=P_NAVY)  # Fond bleu marine
    _hline(c, x, x + sum(widths), y - TH_H, color=P_GOLD, lw=0.5)
    cx = x
    for i, (col, w) in enumerate(zip(cols, widths)):
        if i > 0:
            c.saveState(); c.setStrokeColor(colors.HexColor("#4A7AC8"))
            c.setLineWidth(0.3)
            c.line(cx, y - TH_H + 0.8*mm, cx, y - 0.8*mm)  # Séparateurs verticaux
            c.restoreState()
        mid = y - TH_H + (TH_H - 6.5) / 2
        _text(c, cx + w/2, mid, col, font="Helvetica-Bold", size=6.5, color=P_WHITE, align="center")
        cx += w
    return y - TH_H

def trows(c, x, y, rows, widths, aligns=None, rh=TR_H, bold_last=False, stripe=True):
    aligns = aligns or ["left"] * len(widths)
    for i, row in enumerate(rows):
        bg = P_STRIPE if (stripe and i % 2 == 0) else P_WHITE
        _rect(c, x, y - rh, sum(widths), rh, fill=bg, stroke=P_BORDER, sw=0.2)
        cx = x
        ty = y - rh + (rh - 6.5) / 2
        for j, (cell, w) in enumerate(zip(row, widths)):
            s = str(cell)
            fnt = "Helvetica-Bold" if (bold_last and i == len(rows)-1) else "Helvetica"
            col = P_TEXT
            if s.startswith("+"): col = P_POS
            elif s.startswith("-") and s not in ("-", "—"): col = P_NEG
            pad = 2.2 * mm
            if aligns[j] == "left": _text(c, cx + pad, ty, s, fnt, 6.5, col, "left")
            elif aligns[j] == "right": _text(c, cx + w - pad, ty, s, fnt, 6.5, col, "right")
            else: _text(c, cx + w/2, ty, s, fnt, 6.5, col, "center")
            cx += w
        y -= rh
    return y

# =============================================
# EN-TÊTE ET PIED DE PAGE (Style Asset Manager)
# =============================================
def draw_header(c, subtitle, as_of, page_lbl):
    _rect(c, 0, H - HDR_H, W, HDR_H, fill=P_NAVY)  # Bandeau bleu marine
    _rect(c, 0, H - HDR_H, 4*mm, HDR_H, fill=P_GOLD)  # Bande dorée à gauche
    _hline(c, 0, W, H - HDR_H, color=P_GOLD, lw=1.4)  # Ligne dorée en haut
    _rect(c, 0, H - 1.3*mm, W, 1.3*mm, fill=P_GOLD)  # Ligne dorée en bas

    # Titre principal
    _text(c, ML+2*mm, H-11*mm, "PORTEFEUILLE MULTI-ACTIFS  ·  HEDGÉ USD",
          font="Helvetica-Bold", size=13, color=P_WHITE)
    # Sous-titre
    _text(c, ML+2*mm, H-18*mm, subtitle,
          font="Helvetica", size=8, color=colors.HexColor("#AABFD8"))
    # Disclaimer léger
    _text(c, ML+2*mm, H-23.5*mm,
          "Document à usage informatif — Performances passées non garanties",
          font="Helvetica-Oblique", size=6, color=colors.HexColor("#7A90A8"))
    # Date et numéro de page
    _text(c, W-MR, H-11*mm, f"Au {as_of}",
          font="Helvetica-Bold", size=10, color=P_GOLD, align="right")
    _text(c, W-MR, H-18*mm, page_lbl,
          font="Helvetica", size=7.5, color=colors.HexColor("#AABFD8"), align="right")
    return H - HDR_H - GAP

def draw_footer(c, today):
    _rect(c, 0, 0, W, FTR_H, fill=P_NAVY)  # Fond bleu marine
    _hline(c, 0, W, FTR_H, color=P_GOLD, lw=0.7)  # Ligne dorée
    _text(c, ML, 3*mm,
          f"Données : Yahoo Finance / FMP via OpenBB  —  Couverture FX simulée sans coût  —  Généré le {today}",
          font="Helvetica-Oblique", size=5.5, color=colors.HexColor("#7A90A8"))

# =============================================
# PAGE 1 (KPIs, Performance, Drawdown, Risques)
# =============================================
def page1(c, port_idx, bench_idx, port_ret, weights, m, as_of, today):
    y = draw_header(
        c,
        "Gestion diversifiée  ·  Europe & États-Unis  ·  Couverture de change USD / EUR",
        as_of, "Page 1 / 2",
    )
    draw_footer(c, today)

    # --- KPIs (6 indicateurs clés) ---
    KPI_H = 18 * mm
    kpis = [
        ("Perf. YTD", fp(m["perf_ytd"])),
        ("Perf. 1 mois", fp(m["perf_1m"])),
        ("Perf. 1 an", fp(m["perf_1y"])),
        ("Volatilité", fp(m["vol"])),
        ("Sharpe", fx(m["sharpe"])),
        ("Max Drawdown", fp(m["max_dd"])),
    ]
    kw = CW / len(kpis)
    for i, (lbl, val) in enumerate(kpis):
        bx = ML + i * kw
        bg = P_GOLD_L if i % 2 == 0 else P_BLUE_L  # Alternance or/bleu clair
        _rect(c, bx, y - KPI_H, kw, KPI_H, fill=bg, stroke=P_BORDER, sw=0.3)
        neg = isinstance(val, str) and val.startswith("-") and val != "n/a"
        vc = P_NEG if neg else P_NAVY  # Rouge si négatif, bleu marine sinon
        _text(c, bx+kw/2, y-KPI_H+9.5*mm, val,
              font="Helvetica-Bold", size=11, color=vc, align="center")
        _text(c, bx+kw/2, y-KPI_H+3.5*mm, lbl,
              font="Helvetica", size=6.2, color=P_MUTED, align="center")
        if i > 0:
            c.saveState(); c.setStrokeColor(P_BORDER); c.setLineWidth(0.4)
            c.line(bx, y-KPI_H+1.5*mm, bx, y-1.5*mm)  # Séparateurs verticaux
            c.restoreState()
    _hline(c, ML, ML+CW, y-KPI_H, color=P_GOLD, lw=1.0)  # Ligne dorée sous les KPIs
    y -= KPI_H + GAP

    # --- Performance cumulée (graphique) ---
    y = sec(c, ML, y, "PERFORMANCE CUMULÉE")
    ch = 63 * mm
    _img_rect(c, g_perf(port_idx, bench_idx, CW*(72/mm), ch*(72/mm)), ML, y-ch, CW, ch)
    y -= ch + GAP

    # --- Performances rolling (tableau) ---
    y = sec(c, ML, y, "PERFORMANCES CUMULÉES  —  nettes de frais")
    pw = [46*mm, 19*mm, 19*mm, 19*mm, 19*mm, 19*mm, 24*mm-0.1]
    y = thead(c, ML, y,
        ["", "1 mois", "3 mois", "YTD", "1 an", "3 ans", "Depuis origine"], pw)
    y = trows(c, ML, y, [
        ["Portefeuille hedgé USD",
         fp(m["perf_1m"]), fp(m["perf_3m"]), fp(m["perf_ytd"]),
         fp(m["perf_1y"]), fp(m["perf_3y"]), fp(m["perf_total"])],
        ["Benchmark composite",
         fp(m["bench_1m"]), fp(m["bench_3m"]), fp(m["bench_ytd"]),
         fp(m["bench_1y"]), fp(m["bench_3y"]), "—"],
    ], pw, aligns=["left"]+["right"]*6)
    y -= GAP

    # --- Drawdown + Barres mensuelles (2 colonnes) ---
    col2 = (CW - 5*mm) / 2
    H2 = 48 * mm
    ytop = y

    # Drawdown (gauche)
    yl = sec(c, ML, ytop, "DRAWDOWN  ·  portefeuille hedgé USD", w=col2)
    _img_rect(c, g_dd(port_ret, col2*(72/mm), H2*(72/mm)), ML, yl-H2, col2, H2)

    # Barres mensuelles (droite)
    x2 = ML + col2 + 5*mm
    yr = sec(c, x2, ytop, f"CONTRIBUTIONS MENSUELLES {datetime.date.today().year}  ·  par classe d'actifs", w=col2)
    _img_rect(c, g_monthly(port_ret, weights, col2*(72/mm), H2*(72/mm)), x2, yr-H2, col2, H2)

    y = min(yl-H2, yr-H2) - GAP

    # --- Indicateurs de risque (tableau) ---
    y = sec(c, ML, y, "INDICATEURS DE RISQUE ET DE PERFORMANCE")
    rw = [66*mm, 52*mm, 47*mm-0.1]
    y = thead(c, ML, y, ["Indicateur", "Portefeuille hedgé USD", "Benchmark"], rw)
    y = trows(c, ML, y, [
        ["Rendement annualisé",   fp(m["ann_ret"]),  "—"],
        ["Volatilité annualisée", fp(m["vol"]),      "—"],
        ["Ratio de Sharpe",       fx(m["sharpe"]),   "—"],
        ["Ratio de Sortino",      fx(m["sortino"]),  "—"],
        ["Max Drawdown",          fp(m["max_dd"]),   "—"],
        ["Bêta vs benchmark",     fraw(m["beta"]),   "1,00"],
        ["Corrélation",           fraw(m["corr"]),   "1,00"],
        ["Tracking Error",        fp(m["te"]),       "—"],
        ["Information Ratio",     fx(m["ir"]),       "—"],
    ], rw, aligns=["left", "right", "right"])

# =============================================
# PAGE 2 (Composition, Donut, Top 5, Historique)
# =============================================
def page2(c, port_ret, allocation, ticker_names, weights, as_of, today):
    y = draw_header(
        c,
        "Composition du portefeuille  ·  Analyse  ·  Historique mensuel",
        as_of, "Page 2 / 2",
    )
    draw_footer(c, today)

    # --- Composition (tableau) ---
    y = sec(c, ML, y, "COMPOSITION DU PORTEFEUILLE")
    cw = [63*mm, 33*mm, 30*mm, 22*mm, 22*mm-0.1]
    y = thead(c, ML, y,
        ["Actif", "Ticker", "Classe d'actifs", "Poids cible", "Poids effectif"], cw)
    crows = [[ticker_names.get(t, t), t, asset_class(t),
              f"{allocation.get(t,0)*100:.1f}%", f"{w*100:.1f}%"]
             for t, w in weights.items()]
    crows.append(["Cash", "—", "Liquidités", "5,0%", "5,0%"])
    y = trows(c, ML, y, crows, cw,
              aligns=["left","left","left","right","right"],
              bold_last=True, rh=4.0*mm)

    # --- Ligne total (fond bleu marine) ---
    _rect(c, ML, y-5*mm, sum(cw), 5*mm, fill=P_NAVY)
    mid_tot = y - 5*mm + (5*mm - 7) / 2
    _text(c, ML+3*mm, mid_tot, "TOTAL", font="Helvetica-Bold", size=7, color=P_WHITE)
    _text(c, ML+sum(cw)-3*mm, mid_tot,
          f"{(weights.sum()+0.05)*100:.1f}%",
          font="Helvetica-Bold", size=7, color=P_GOLD, align="right")
    y -= 5*mm + GAP

    # --- 3 colonnes : Classe | Donut | Top 5 ---
    G3 = 4 * mm
    col3 = (CW - 2*G3) / 3
    H3 = 52 * mm
    ytop = y

    # Répartition par classe (gauche)
    cls = {}
    for t, w in weights.items():
        k = asset_class(t); cls[k] = cls.get(k, 0) + float(w) * 100
    cls["Cash"] = 5.0

    yl = sec(c, ML, ytop, "PAR CLASSE D'ACTIFS", w=col3)
    clw = [col3 * 0.63, col3 * 0.37]
    yl = thead(c, ML, yl, ["Classe", "Allocation"], clw)
    cls_sorted = [(k, cls.get(k, 0)) for k in CLS_ORDER if k in cls]
    cls_sorted.append(("Total", sum(cls.values())))
    yl = trows(c, ML, yl, [[k, f"{v:.1f}%"] for k, v in cls_sorted],
               clw, aligns=["left","right"], bold_last=True, rh=4.2*mm)

    # Pastilles couleur sous le tableau
    ly = yl - 2*mm
    for k in CLS_ORDER:
        if k not in cls: continue
        c.saveState()
        c.setFillColor(colors.HexColor(CLS_COLORS[k]))
        c.rect(ML + 1*mm, ly - 2.5*mm, 3*mm, 2.5*mm, fill=1, stroke=0)
        c.restoreState()
        _text(c, ML + 5.5*mm, ly - 2.2*mm, k, font="Helvetica", size=6, color=P_TEXT)
        ly -= 4*mm

    # Donut (centre)
    xp = ML + col3 + G3
    yp = sec(c, xp, ytop, "ALLOCATION", w=col3)
    ph = H3 - ST_H - 2*mm
    pie_png = g_pie({k: v for k, v in cls.items()}, ph * (72/mm))
    _img_fit(c, pie_png, xp, yp - ph, col3, ph)

    # Top 5 (droite)
    xt = ML + 2*col3 + 2*G3
    yt = sec(c, xt, ytop, "TOP 5 POSITIONS", w=col3)
    t5w = [col3 * 0.68, col3 * 0.32]
    yt = thead(c, xt, yt, ["Actif", "Poids"], t5w)
    top5 = sorted(weights.items(), key=lambda x: -x[1])[:5]
    yt = trows(c, xt, yt,
               [[ticker_names.get(t, t), f"{w*100:.1f}%"] for t, w in top5],
               t5w, aligns=["left","right"], rh=4.2*mm)

    y = min(yl, yp - ph, yt) - GAP

    # --- Historique mensuel (tableau) ---
    y = sec(c, ML, y, "HISTORIQUE DE PERFORMANCE MENSUELLE  —  PORTEFEUILLE HEDGÉ USD")
    m_all = (port_ret.resample("ME")
             .apply(lambda x: (1+x).prod()-1) * 100)
    years = sorted(m_all.index.year.unique())[-4:]
    mois = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc","Annuel"]
    hw = [14*mm] + [11.0*mm] * 13
    y = thead(c, ML, y, ["Année"] + mois, hw)
    hrows = []
    for yr in years:
        row = [str(yr)]; acc = 1.0
        for mo in range(1, 13):
            mask = (m_all.index.year == yr) & (m_all.index.month == mo)
            if mask.any():
                v = m_all[mask].iloc[0]; acc *= (1 + v/100); row.append(fp(v, 1))
            else: row.append("—")
        row.append(fp((acc - 1) * 100, 1)); hrows.append(row)
    y = trows(c, ML, y, hrows, hw, aligns=["left"] + ["right"] * 13, rh=4.8*mm)
    y -= GAP

    # --- Informations générales (tableau) ---
    y = sec(c, ML, y, "INFORMATIONS GÉNÉRALES")
    iw = [54*mm, CW - 54*mm]
    y = trows(c, ML, y, [
        ["Stratégie",                "Multi-actifs couvert USD"],
        ["Benchmark",                "35% IEV  ·  20% SPY  ·  25% TLT  ·  10% VNQ  ·  5% EEM"],
        ["Couverture de change",     "USD/EUR — simulation forward (sans coût inclus)"],
        ["Nombre de lignes",         f"{len(weights)+1} (dont cash)"],
        ["Univers d'investissement", "Europe, États-Unis, Marchés émergents"],
        ["Sources de données",       "Yahoo Finance  /  FMP via OpenBB"],
    ], iw, aligns=["left","left"], rh=4.2*mm)

    # --- Disclaimer (pied de page étendu) ---
    dz = 18 * mm
    _rect(c, 0, FTR_H, W, dz - FTR_H, fill=colors.HexColor("#F4F7FC"))
    _hline(c, ML, ML+CW, dz - 0.5, color=P_BORDER, lw=0.5)
    disc = (
        "Ce document est établi à titre purement informatif et ne constitue pas un conseil "
        "en investissement. Les performances passées ne préjugent pas des résultats futurs. "
        "La couverture de change USD/EUR est simulée sans coût de transaction. "
        "Les données proviennent de sources publiques (Yahoo Finance, FMP) et peuvent contenir des imprécisions. "
        "Document non soumis à l'approbation de l'AMF."
    )
    chunk = len(disc) // 3
    c1 = disc.rfind(" ", 0, chunk); c2 = disc.rfind(" ", 0, chunk * 2)
    for i, ln in enumerate([disc[:c1], disc[c1+1:c2], disc[c2+1:]]):
        _text(c, ML, dz - 5*mm - i*3.5*mm, ln,
              font="Helvetica-Oblique", size=5.5, color=P_MUTED)

# =============================================
# GÉNÉRATEUR PDF PRINCIPAL
# =============================================
def generate_factsheet_pdf(port_idx, bench_idx, port_ret, allocation, ticker_names, weights):
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    m = compute_metrics(port_idx, bench_idx, port_ret)
    as_of = port_idx.index[-1].strftime("%d/%m/%Y")
    today = datetime.datetime.now().strftime("%d/%m/%Y")

    page1(c, port_idx, bench_idx, port_ret, weights, m, as_of, today)
    c.showPage()
    page2(c, port_ret, allocation, ticker_names, weights, as_of, today)
    c.save()
    buf.seek(0)
    return buf.read()

# =============================================
# WIDGET STREAMLIT (pour l'export)
# =============================================
def render_factsheet_section(port_idx, bench_idx, port_ret, allocation, ticker_names, weights):
    st.divider()
    st.subheader("📄 Export Factsheet mensuelle")
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown(
            "Génère une **factsheet PDF 2 pages** aux standards des meilleurs asset managers :\n"
            "- **Graphiques professionnels** : axes datés, légendes claires, couleurs cohérentes\n"
            "- **Tableaux lisibles** : alternance de couleurs, totaux en gras, alignement parfait\n"
            "- **Mise en page institutionnelle** : marges généreuses, en-tête/pied de page sobres\n"
            "- **Donut et barres empilées** : répartition par classe d'actif visible en un coup d'œil"
        )
    with col_btn:
        if st.button("⬇️ Générer la factsheet", type="primary", use_container_width=True):
            with st.spinner("Génération en cours…"):
                try:
                    pdf = generate_factsheet_pdf(
                        port_idx, bench_idx, port_ret, allocation, ticker_names, weights)
                    fname = f"factsheet_{datetime.date.today().strftime('%Y%m%d')}.pdf"
                    st.download_button(
                        "📥 Télécharger le PDF",
                        data=pdf, file_name=fname, mime="application/pdf",
                        use_container_width=True,
                    )
                    st.success("✅ Factsheet générée avec succès !")
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")
                    raise
