"""
factsheet_export.py
-------------------
Factsheet PDF style "maison de gestion" inspirée de la factsheet EdR.
Utilise reportlab canvas pour un contrôle pixel-perfect.

Usage dans app.py (coller à la fin) :
    from factsheet_export import render_factsheet_section
    render_factsheet_section(
        portfolio_index_hedged, bench_index,
        portfolio_returns_hedged, allocation,
        ticker_names, weights
    )

requirements.txt :
    reportlab
    matplotlib
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

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib import colors


# ─────────────────────────────────────────────
# Palette — sobre, institutionnelle
# ─────────────────────────────────────────────
C_DARK   = colors.HexColor("#1A2B4A")   # bleu nuit header
C_MID    = colors.HexColor("#2C4A7C")   # bleu moyen section titles
C_ACCENT = colors.HexColor("#C8A951")   # or bande décorative (style EdR)
C_LIGHT  = colors.HexColor("#EEF2F7")   # fond lignes paires tableaux
C_BORDER = colors.HexColor("#C8D0DC")   # bordures tableaux
C_TEXT   = colors.HexColor("#1A2B4A")   # texte principal
C_MUTED  = colors.HexColor("#6B7A8D")   # texte secondaire
C_GREEN  = colors.HexColor("#1B6B3A")
C_RED    = colors.HexColor("#A0282A")
C_WHITE  = colors.white

W, H = A4          # 595 x 842 pts
ML = 14 * mm       # marge gauche
MR = 14 * mm       # marge droite
CW = W - ML - MR   # largeur utile


# ─────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────
def compute_metrics(port_idx, bench_idx, port_ret):
    r = port_ret.dropna()
    ann = 252

    def perf_over(days):
        cutoff = port_idx.index[-1] - pd.Timedelta(days=days)
        s = port_idx[port_idx.index >= cutoff]
        return (s.iloc[-1] / s.iloc[0] - 1) * 100 if len(s) > 1 else np.nan

    ytd_start = pd.Timestamp(datetime.date.today().year, 1, 1)
    s_ytd = port_idx[port_idx.index >= ytd_start]
    perf_ytd = (s_ytd.iloc[-1] / s_ytd.iloc[0] - 1) * 100 if len(s_ytd) > 1 else np.nan

    vol = r.std() * np.sqrt(ann) * 100
    rf  = 0.035 / ann
    sharpe   = (r.mean() - rf) / r.std() * np.sqrt(ann) if r.std() > 0 else np.nan
    down_std = r[r < 0].std() * np.sqrt(ann)
    sortino  = r.mean() * ann / down_std if down_std > 0 else np.nan
    cum = (1 + r).cumprod()
    dd  = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    ann_ret = r.mean() * ann * 100
    calmar  = ann_ret / abs(dd) if dd != 0 else np.nan

    common  = port_ret.index.intersection(bench_idx.index)
    br      = bench_idx.loc[common].pct_change().dropna()
    pr      = port_ret.loc[br.index]
    beta    = np.cov(pr, br)[0, 1] / np.var(br) if len(br) > 5 else np.nan
    corr    = np.corrcoef(pr, br)[0, 1] if len(br) > 5 else np.nan
    te      = (pr - br).std() * np.sqrt(ann) * 100
    ir      = (pr - br).mean() * ann / ((pr - br).std() * np.sqrt(ann)) if (pr - br).std() > 0 else np.nan

    # Perf benchmark
    def bench_perf(days):
        cutoff = bench_idx.index[-1] - pd.Timedelta(days=days)
        s = bench_idx[bench_idx.index >= cutoff]
        return (s.iloc[-1] / s.iloc[0] - 1) * 100 if len(s) > 1 else np.nan

    bench_ytd_s = bench_idx[bench_idx.index >= ytd_start]
    bench_ytd   = (bench_ytd_s.iloc[-1] / bench_ytd_s.iloc[0] - 1) * 100 if len(bench_ytd_s) > 1 else np.nan

    return {
        "perf_ytd": perf_ytd, "perf_1m": perf_over(30),
        "perf_3m": perf_over(91), "perf_1y": perf_over(365),
        "perf_3y": perf_over(3*365), "perf_total": (port_idx.iloc[-1] / port_idx.iloc[0] - 1) * 100,
        "ann_ret": ann_ret, "vol": vol, "sharpe": sharpe,
        "sortino": sortino, "max_dd": dd, "calmar": calmar,
        "beta": beta, "corr": corr, "te": te, "ir": ir,
        "bench_ytd": bench_ytd, "bench_1m": bench_perf(30),
        "bench_3m": bench_perf(91), "bench_1y": bench_perf(365),
        "bench_3y": bench_perf(3*365),
    }


def f(v, decimals=2, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{decimals}f}{suffix}"

def fp(v, decimals=2):   # pct avec signe
    return f(v, decimals, "%")

def fx(v, decimals=2):   # ratio sans signe
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{decimals}f}x"


# ─────────────────────────────────────────────
# Graphiques matplotlib → bytes PNG
# ─────────────────────────────────────────────
PLOT_STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.edgecolor": "#C8D0DC", "axes.linewidth": 0.6,
    "xtick.color": "#6B7A8D", "ytick.color": "#6B7A8D",
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "grid.color": "#E8ECF2", "grid.linewidth": 0.4,
}

def _fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_perf_chart(port_idx, bench_idx, w_pts, h_pts):
    """Graphique performance cumulée — courbe portfolio + benchmark."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(w_pts / 72, h_pts / 72))
        common = port_idx.index.intersection(bench_idx.index)
        p = (port_idx.loc[common] - 1) * 100
        b = (bench_idx.loc[common] - 1) * 100
        ax.plot(p.index, p, color="#2C4A7C", linewidth=1.3,
                label="Portefeuille hedgé USD", zorder=3)
        ax.fill_between(p.index, p, alpha=0.07, color="#2C4A7C")
        ax.plot(b.index, b, color="#C8A951", linewidth=1.0,
                linestyle="--", label="Benchmark composite", alpha=0.85, zorder=2)
        ax.axhline(0, color="#C8D0DC", linewidth=0.5, linestyle=":")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.grid(True, axis="y")
        ax.legend(fontsize=6.5, framealpha=0, loc="upper left",
                  handlelength=1.5, handletextpad=0.5)
        plt.tight_layout(pad=0.3)
        return _fig_to_bytes(fig)


def build_drawdown_chart(port_ret, w_pts, h_pts):
    """Graphique drawdown."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(w_pts / 72, h_pts / 72))
        cum = (1 + port_ret.dropna()).cumprod()
        dd  = ((cum - cum.cummax()) / cum.cummax()) * 100
        ax.fill_between(dd.index, dd, 0, color="#A0282A", alpha=0.55)
        ax.plot(dd.index, dd, color="#A0282A", linewidth=0.7)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.grid(True, axis="y")
        plt.tight_layout(pad=0.3)
        return _fig_to_bytes(fig)


def build_monthly_bar_chart(port_ret, w_pts, h_pts):
    """Barres des performances mensuelles de l'année en cours."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(w_pts / 72, h_pts / 72))
        year = datetime.date.today().year
        monthly = port_ret[port_ret.index.year == year].resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        if monthly.empty:
            ax.text(0.5, 0.5, "Pas de données", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="#6B7A8D")
        else:
            colors_bar = ["#1B6B3A" if v >= 0 else "#A0282A" for v in monthly.values]
            ax.bar(monthly.index, monthly.values, color=colors_bar,
                   width=20, edgecolor="white", linewidth=0.3)
            ax.axhline(0, color="#C8D0DC", linewidth=0.5)
            ax.set_xticks(monthly.index)
            ax.set_xticklabels([d.strftime("%b") for d in monthly.index], fontsize=6)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
            ax.grid(True, axis="y")
        plt.tight_layout(pad=0.3)
        return _fig_to_bytes(fig)


# ─────────────────────────────────────────────
# Primitives canvas
# ─────────────────────────────────────────────
def draw_rect(c, x, y, w, h, fill=None, stroke=None, stroke_w=0.4):
    c.saveState()
    if fill:
        c.setFillColor(fill)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(stroke_w)
    if fill and stroke:
        c.rect(x, y, w, h, fill=1, stroke=1)
    elif fill:
        c.rect(x, y, w, h, fill=1, stroke=0)
    elif stroke:
        c.rect(x, y, w, h, fill=0, stroke=1)
    c.restoreState()


def draw_text(c, x, y, text, font="Helvetica", size=8,
              color=C_TEXT, align="left"):
    c.saveState()
    c.setFont(font, size)
    c.setFillColor(color)
    if align == "right":
        c.drawRightString(x, y, str(text))
    elif align == "center":
        c.drawCentredString(x, y, str(text))
    else:
        c.drawString(x, y, str(text))
    c.restoreState()


def draw_section_title(c, x, y, title, w=None):
    """Bande bleue + titre blanc style EdR."""
    bw = w or CW
    bh = 5.5 * mm
    draw_rect(c, x, y, bw, bh, fill=C_MID)
    # Barre or à gauche
    draw_rect(c, x, y, 1.8 * mm, bh, fill=C_ACCENT)
    draw_text(c, x + 3 * mm, y + 1.8 * mm, title,
              font="Helvetica-Bold", size=7.5, color=C_WHITE)
    return y - 2 * mm  # retourne y sous la section title


def draw_table_header(c, x, y, cols, widths, row_h=5 * mm):
    """Header de tableau bleu foncé."""
    draw_rect(c, x, y, sum(widths), row_h, fill=C_DARK)
    cx = x
    for col, w in zip(cols, widths):
        draw_text(c, cx + w / 2, y + 1.5 * mm, col,
                  font="Helvetica-Bold", size=6.5, color=C_WHITE, align="center")
        cx += w
    return y - row_h


def draw_table_rows(c, x, y, rows, widths, aligns=None,
                    row_h=4.5 * mm, bold_last=False):
    """Lignes de tableau avec alternance de fond."""
    aligns = aligns or ["left"] * len(widths)
    for i, row in enumerate(rows):
        bg = C_LIGHT if i % 2 == 0 else C_WHITE
        draw_rect(c, x, y, sum(widths), row_h, fill=bg, stroke=C_BORDER, stroke_w=0.3)
        cx = x
        for j, (cell, w) in enumerate(zip(row, widths)):
            font = "Helvetica-Bold" if (bold_last and i == len(rows) - 1) else "Helvetica"
            # Couleur pour les valeurs perf
            col = C_TEXT
            if isinstance(cell, str) and cell.startswith("+"):
                col = C_GREEN
            elif isinstance(cell, str) and cell.startswith("-") and cell != "—":
                col = C_RED
            pad = 1.5 * mm if aligns[j] == "left" else -1.5 * mm
            ax_ = cx + pad if aligns[j] == "left" else cx + w + pad
            draw_text(c, ax_, y + 1.4 * mm, cell, size=6.5,
                      color=col, align=aligns[j])
            cx += w
        y -= row_h
    return y


# ─────────────────────────────────────────────
# Générateur PDF principal — canvas pur
# ─────────────────────────────────────────────
def generate_factsheet_pdf(port_idx, bench_idx, port_ret,
                            allocation, ticker_names, weights):
    buf    = io.BytesIO()
    c      = rl_canvas.Canvas(buf, pagesize=A4)
    metrics = compute_metrics(port_idx, bench_idx, port_ret)
    as_of   = port_idx.index[-1].strftime("%d/%m/%Y")
    today   = datetime.datetime.now().strftime("%d/%m/%Y")

    # ── PAGE 1 ─────────────────────────────────────────────────────
    y = H  # curseur vertical, on descend

    # === HEADER BANNER ===
    banner_h = 22 * mm
    draw_rect(c, 0, H - banner_h, W, banner_h, fill=C_DARK)
    # Barre or décorative
    draw_rect(c, 0, H - banner_h, W, 1.2 * mm, fill=C_ACCENT)
    draw_rect(c, 0, H - 1.2 * mm, W, 1.2 * mm, fill=C_ACCENT)

    draw_text(c, ML, H - 8 * mm,
              "PORTEFEUILLE HEDGÉ USD",
              font="Helvetica-Bold", size=13, color=C_WHITE)
    draw_text(c, ML, H - 13.5 * mm,
              "Gestion multi-actifs — Couverture de change USD/EUR",
              font="Helvetica", size=8, color=colors.HexColor("#A8B8CC"))
    draw_text(c, W - MR, H - 8 * mm,
              f"Au {as_of}",
              font="Helvetica-Bold", size=8.5, color=C_ACCENT, align="right")
    draw_text(c, W - MR, H - 13.5 * mm,
              "Document à usage informatif uniquement",
              font="Helvetica", size=7, color=colors.HexColor("#A8B8CC"), align="right")

    y = H - banner_h - 4 * mm

    # === LIGNE KPI (style EdR : 6 cases horizontales) ===
    kpi_h   = 14 * mm
    kpis    = [
        ("Perf. YTD",     fp(metrics["perf_ytd"])),
        ("Perf. 1 mois",  fp(metrics["perf_1m"])),
        ("Perf. 1 an",    fp(metrics["perf_1y"])),
        ("Volatilité ann.", fp(metrics["vol"])),
        ("Sharpe",        fx(metrics["sharpe"])),
        ("Max Drawdown",  fp(metrics["max_dd"])),
    ]
    kw = CW / len(kpis)
    for i, (label, val) in enumerate(kpis):
        bx = ML + i * kw
        bg = C_DARK if i % 2 == 0 else C_MID
        draw_rect(c, bx, y - kpi_h, kw, kpi_h, fill=bg)
        # Valeur grande
        is_neg = isinstance(val, str) and val.startswith("-") and val != "n/a"
        vc = colors.HexColor("#F4B942") if not is_neg else colors.HexColor("#F4A0A0")
        draw_text(c, bx + kw / 2, y - kpi_h + 6 * mm, val,
                  font="Helvetica-Bold", size=9.5, color=vc, align="center")
        draw_text(c, bx + kw / 2, y - kpi_h + 2 * mm, label,
                  font="Helvetica", size=6, color=colors.HexColor("#A8B8CC"), align="center")
    y -= kpi_h + 4 * mm

    # === GRAPHIQUE PERFORMANCE ===
    y = draw_section_title(c, ML, y, "PERFORMANCE CUMULÉE")
    chart_h = 48 * mm
    perf_png = build_perf_chart(port_idx, bench_idx, CW * (72 / mm), chart_h * (72 / mm))
    from reportlab.lib.utils import ImageReader
    c.drawImage(ImageReader(io.BytesIO(perf_png)),
                ML, y - chart_h, width=CW, height=chart_h)
    y -= chart_h + 4 * mm

    # === TABLEAU PERFORMANCES ROLLING ===
    y = draw_section_title(c, ML, y, "PERFORMANCES CUMULÉES (nettes de frais)")
    col_labels = ["", "1 mois", "3 mois", "YTD", "1 an", "3 ans", "Depuis origine"]
    col_w = [38 * mm, 20 * mm, 20 * mm, 20 * mm, 20 * mm, 20 * mm, 25 * mm]
    y = draw_table_header(c, ML, y, col_labels, col_w)
    perf_rows = [
        ["Portefeuille hedgé",
         fp(metrics["perf_1m"]), fp(metrics["perf_3m"]),
         fp(metrics["perf_ytd"]), fp(metrics["perf_1y"]),
         fp(metrics["perf_3y"]), fp(metrics["perf_total"])],
        ["Benchmark composite",
         fp(metrics["bench_1m"]), fp(metrics["bench_3m"]),
         fp(metrics["bench_ytd"]), fp(metrics["bench_1y"]),
         fp(metrics["bench_3y"]), "—"],
    ]
    aligns_perf = ["left"] + ["right"] * 6
    y = draw_table_rows(c, ML, y, perf_rows, col_w, aligns=aligns_perf)
    y -= 4 * mm

    # === DEUX COLONNES : DRAWDOWN + BARRES MENSUELLES ===
    col2 = (CW - 4 * mm) / 2
    # Drawdown
    y_dd = draw_section_title(c, ML, y, "DRAWDOWN", w=col2)
    dd_h = 30 * mm
    dd_png = build_drawdown_chart(port_ret, col2 * (72 / mm), dd_h * (72 / mm))
    c.drawImage(ImageReader(io.BytesIO(dd_png)),
                ML, y_dd - dd_h, width=col2, height=dd_h)

    # Barres mensuelles
    x2 = ML + col2 + 4 * mm
    y_bar = draw_section_title(c, x2, y,
                               f"PERFORMANCES MENSUELLES {datetime.date.today().year}",
                               w=col2)
    bar_png = build_monthly_bar_chart(port_ret, col2 * (72 / mm), dd_h * (72 / mm))
    c.drawImage(ImageReader(io.BytesIO(bar_png)),
                x2, y_bar - dd_h, width=col2, height=dd_h)

    y = min(y_dd, y_bar) - dd_h - 4 * mm

    # === INDICATEURS DE RISQUE ===
    y = draw_section_title(c, ML, y, "INDICATEURS DE RISQUE")
    risk_cols    = ["Indicateur", "Portefeuille hedgé", "Benchmark"]
    risk_widths  = [55 * mm, 45 * mm, 45 * mm]
    y = draw_table_header(c, ML, y, risk_cols, risk_widths)
    risk_rows = [
        ["Rendement annualisé",      fp(metrics["ann_ret"]),  "—"],
        ["Volatilité annualisée",    fp(metrics["vol"]),      "—"],
        ["Ratio de Sharpe",          fx(metrics["sharpe"]),   "—"],
        ["Ratio de Sortino",         fx(metrics["sortino"]),  "—"],
        ["Max Drawdown",             fp(metrics["max_dd"]),   "—"],
        ["Beta vs benchmark",        f"{metrics['beta']:.2f}" if not np.isnan(metrics['beta']) else "n/a", "1.00"],
        ["Corrélation benchmark",    f"{metrics['corr']:.2f}" if not np.isnan(metrics['corr']) else "n/a", "1.00"],
        ["Tracking Error",           fp(metrics["te"]),       "—"],
        ["Information Ratio",        fx(metrics["ir"]),       "—"],
    ]
    aligns_risk = ["left", "right", "right"]
    y = draw_table_rows(c, ML, y, risk_rows, risk_widths, aligns=aligns_risk)
    y -= 3 * mm

    # === DISCLAIMER PAGE 1 ===
    draw_rect(c, 0, 0, W, 8 * mm, fill=C_DARK)
    draw_text(c, ML, 3 * mm,
              f"Document informatif — Performances passées non garanties — "
              f"Couverture FX simulée sans coût — Données Yahoo Finance / FMP — Généré le {today}",
              font="Helvetica-Oblique", size=5.5,
              color=colors.HexColor("#7A8FA8"))
    draw_text(c, W - MR, 3 * mm, "1 / 2",
              font="Helvetica", size=5.5,
              color=colors.HexColor("#7A8FA8"), align="right")

    c.showPage()

    # ── PAGE 2 ─────────────────────────────────────────────────────
    y = H

    # Header simplifié page 2
    draw_rect(c, 0, H - 10 * mm, W, 10 * mm, fill=C_DARK)
    draw_rect(c, 0, H - 10 * mm, W, 0.8 * mm, fill=C_ACCENT)
    draw_text(c, ML, H - 6.5 * mm, "PORTEFEUILLE HEDGÉ USD — COMPOSITION ET ANALYSE",
              font="Helvetica-Bold", size=8, color=C_WHITE)
    draw_text(c, W - MR, H - 6.5 * mm, f"Au {as_of}",
              font="Helvetica-Bold", size=7, color=C_ACCENT, align="right")
    y = H - 10 * mm - 4 * mm

    # === COMPOSITION — tableau principal ===
    y = draw_section_title(c, ML, y, "COMPOSITION DU PORTEFEUILLE")

    comp_cols   = ["Actif", "Ticker", "Classe", "Poids cible", "Poids effectif"]
    comp_widths = [60 * mm, 34 * mm, 30 * mm, 25 * mm, 25 * mm]

    def asset_class(ticker):
        if ticker.startswith("0P"):
            return "Fonds"
        if ticker in ["GOOGL", "META", "HWM", "AMZN"]:
            return "Actions US"
        return "Actions EU"

    y = draw_table_header(c, ML, y, comp_cols, comp_widths)
    comp_rows = []
    for t, w in weights.items():
        comp_rows.append([
            ticker_names.get(t, t),
            t,
            asset_class(t),
            f"{allocation.get(t, 0) * 100:.1f}%",
            f"{w * 100:.1f}%",
        ])
    comp_rows.append(["Cash", "—", "Liquidités", "5.0%", "5.0%"])
    aligns_comp = ["left", "left", "left", "right", "right"]
    y = draw_table_rows(c, ML, y, comp_rows, comp_widths,
                        aligns=aligns_comp, bold_last=True)
    # Total
    draw_rect(c, ML, y, sum(comp_widths), 4.5 * mm, fill=C_DARK)
    total_eff = (weights.sum() + 0.05) * 100
    draw_text(c, ML + 1.5 * mm, y + 1.2 * mm, "TOTAL",
              font="Helvetica-Bold", size=6.5, color=C_WHITE)
    draw_text(c, ML + sum(comp_widths) - 1.5 * mm, y + 1.2 * mm,
              f"{total_eff:.1f}%",
              font="Helvetica-Bold", size=6.5, color=C_ACCENT, align="right")
    y -= 4.5 * mm + 5 * mm

    # === DEUX COLONNES : RÉPARTITION PAR CLASSE + TOP POSITIONS ===
    col2 = (CW - 4 * mm) / 2

    # Répartition par classe
    y_left = draw_section_title(c, ML, y, "RÉPARTITION PAR CLASSE D'ACTIFS", w=col2)
    class_data = {}
    for t, w in weights.items():
        cls = asset_class(t)
        class_data[cls] = class_data.get(cls, 0) + w * 100
    class_data["Cash"] = 5.0
    cls_cols   = ["Classe d'actifs", "Allocation"]
    cls_widths = [col2 * 0.65, col2 * 0.35]
    y_left = draw_table_header(c, ML, y_left, cls_cols, cls_widths, row_h=4.5 * mm)
    cls_rows = [[k, f"{v:.1f}%"] for k, v in sorted(class_data.items(), key=lambda x: -x[1])]
    cls_rows.append(["Total", f"{sum(class_data.values()):.1f}%"])
    y_left = draw_table_rows(c, ML, y_left, cls_rows, cls_widths,
                             aligns=["left", "right"], bold_last=True)

    # Top 5 positions (dernier mois)
    x2 = ML + col2 + 4 * mm
    y_right = draw_section_title(c, x2, y, "TOP 5 POSITIONS (POIDS EFFECTIF)", w=col2)
    top5_data  = sorted(weights.items(), key=lambda x: -x[1])[:5]
    top5_cols  = ["Actif", "Poids"]
    top5_widths= [col2 * 0.70, col2 * 0.30]
    y_right = draw_table_header(c, x2, y_right, top5_cols, top5_widths, row_h=4.5 * mm)
    top5_rows = [[ticker_names.get(t, t), f"{w * 100:.1f}%"] for t, w in top5_data]
    y_right = draw_table_rows(c, x2, y_right, top5_rows, top5_widths,
                              aligns=["left", "right"])

    y = min(y_left, y_right) - 5 * mm

    # === TABLEAU PERFORMANCE MENSUELLE HISTORIQUE ===
    y = draw_section_title(c, ML, y, "HISTORIQUE DE PERFORMANCE MENSUELLE (PORTEFEUILLE HEDGÉ)")

    # Construire la grille mois × années
    monthly_all = port_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    years_avail = sorted(monthly_all.index.year.unique())[-4:]  # 4 dernières années
    month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                    "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc", "Annuel"]
    hist_cols   = ["Année"] + month_labels
    hist_widths = [14 * mm] + [11.5 * mm] * 13
    y = draw_table_header(c, ML, y, hist_cols, hist_widths, row_h=4.5 * mm)

    hist_rows = []
    for yr in years_avail:
        row = [str(yr)]
        ann_yr = 1.0
        for m in range(1, 13):
            mask = (monthly_all.index.year == yr) & (monthly_all.index.month == m)
            if mask.any():
                v = monthly_all[mask].iloc[0]
                ann_yr *= (1 + v / 100)
                row.append(fp(v, 1))
            else:
                row.append("—")
        row.append(fp((ann_yr - 1) * 100, 1))
        hist_rows.append(row)

    aligns_hist = ["left"] + ["right"] * 13
    y = draw_table_rows(c, ML, y, hist_rows, hist_widths,
                        aligns=aligns_hist, row_h=4.5 * mm)
    y -= 5 * mm

    # === INFORMATIONS FONDS ===
    y = draw_section_title(c, ML, y, "INFORMATIONS GÉNÉRALES")
    info_rows = [
        ["Stratégie", "Multi-actifs couvert USD"],
        ["Benchmark", "35% IEV + 20% SPY + 25% TLT + 10% VNQ + 5% EEM"],
        ["Couverture de change", "USD/EUR — simulation forward (sans coût)"],
        ["Nombre de lignes", str(len(weights) + 1)],  # +1 cash
        ["Univers géographique", "Europe, États-Unis, Marchés émergents"],
        ["Source données", "Yahoo Finance / FMP via OpenBB"],
    ]
    info_widths = [50 * mm, CW - 50 * mm]
    y = draw_table_rows(c, ML, y, info_rows, info_widths, aligns=["left", "left"])

    # === DISCLAIMER PAGE 2 ===
    draw_rect(c, 0, 0, W, 14 * mm, fill=colors.HexColor("#F4F6F7"))
    draw_rect(c, 0, 13.3 * mm, W, 0.5, fill=C_BORDER)
    disclaimer = (
        "Ce document est établi à titre purement informatif et ne constitue pas un conseil en investissement, ni une offre ou sollicitation "
        "d'achat ou de vente de valeurs mobilières. Les performances passées ne constituent pas un indicateur fiable des performances futures. "
        "La simulation de couverture de change USD/EUR est effectuée sans coût de couverture. "
        "Toutes les données sont issues de sources publiques (Yahoo Finance, FMP) et peuvent contenir des imprécisions."
    )
    # Découpe manuelle en deux lignes
    mid = len(disclaimer) // 2
    cut = disclaimer.rfind(" ", 0, mid)
    l1, l2 = disclaimer[:cut], disclaimer[cut + 1:]
    draw_text(c, ML, 9 * mm, l1, font="Helvetica-Oblique", size=5.5, color=C_MUTED)
    draw_text(c, ML, 5.5 * mm, l2, font="Helvetica-Oblique", size=5.5, color=C_MUTED)
    draw_text(c, W - MR, 3 * mm, "2 / 2",
              font="Helvetica", size=5.5, color=C_MUTED, align="right")
    draw_rect(c, 0, 0, W, 2.5 * mm, fill=C_DARK)

    c.save()
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# Widget Streamlit
# ─────────────────────────────────────────────
def render_factsheet_section(port_idx, bench_idx, port_ret,
                              allocation, ticker_names, weights):
    st.divider()
    st.subheader("📄 Export Factsheet mensuelle")

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown(
            "Génère une **factsheet PDF 2 pages** style institutionnel : "
            "performances rolling, drawdown, barres mensuelles, "
            "composition détaillée, historique mensuel et métriques de risque."
        )
    with col_btn:
        if st.button("⬇️ Générer la factsheet", type="primary", use_container_width=True):
            with st.spinner("Génération en cours…"):
                try:
                    pdf = generate_factsheet_pdf(
                        port_idx, bench_idx, port_ret,
                        allocation, ticker_names, weights
                    )
                    fname = f"factsheet_{datetime.date.today().strftime('%Y%m')}.pdf"
                    st.download_button(
                        "📥 Télécharger le PDF",
                        data=pdf,
                        file_name=fname,
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("Factsheet générée ✓")
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    raise
