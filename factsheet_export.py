"""
factsheet_export.py
-------------------
Module autonome à importer dans ton app Streamlit principale.

Usage dans portfolio_app.py :
    from factsheet_export import render_factsheet_section
    render_factsheet_section(
        portfolio_index_hedged,
        bench_index,
        portfolio_returns_hedged,
        allocation,
        ticker_names,
        weights
    )

Dépendance à ajouter dans requirements.txt :
    reportlab
    matplotlib
"""

import io
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non-interactif obligatoire dans Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


# ─────────────────────────────────────────────
# Palette couleurs
# ─────────────────────────────────────────────
DARK_BLUE   = colors.HexColor("#0D1B2A")
MID_BLUE    = colors.HexColor("#1B4F72")
ACCENT_BLUE = colors.HexColor("#2E86C1")
LIGHT_GRAY  = colors.HexColor("#F4F6F7")
MID_GRAY    = colors.HexColor("#BDC3C7")
TEXT_GRAY   = colors.HexColor("#566573")
GREEN       = colors.HexColor("#1E8449")
RED         = colors.HexColor("#C0392B")
WHITE       = colors.white
BLACK       = colors.black


# ─────────────────────────────────────────────
# Styles typographiques
# ─────────────────────────────────────────────
def get_styles():
    return {
        "title": ParagraphStyle(
            "title", fontName="Helvetica-Bold", fontSize=22,
            textColor=WHITE, alignment=TA_LEFT, leading=26
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontName="Helvetica", fontSize=10,
            textColor=colors.HexColor("#A9CCE3"), alignment=TA_LEFT, leading=14
        ),
        "section": ParagraphStyle(
            "section", fontName="Helvetica-Bold", fontSize=10,
            textColor=DARK_BLUE, spaceBefore=6, spaceAfter=4,
            borderPad=2, leading=14,
            backColor=LIGHT_GRAY, leftIndent=4
        ),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=8,
            textColor=TEXT_GRAY, leading=12
        ),
        "metric_label": ParagraphStyle(
            "metric_label", fontName="Helvetica", fontSize=7,
            textColor=TEXT_GRAY, alignment=TA_CENTER, leading=10
        ),
        "metric_value": ParagraphStyle(
            "metric_value", fontName="Helvetica-Bold", fontSize=13,
            textColor=DARK_BLUE, alignment=TA_CENTER, leading=16
        ),
        "metric_value_green": ParagraphStyle(
            "metric_value_green", fontName="Helvetica-Bold", fontSize=13,
            textColor=GREEN, alignment=TA_CENTER, leading=16
        ),
        "metric_value_red": ParagraphStyle(
            "metric_value_red", fontName="Helvetica-Bold", fontSize=13,
            textColor=RED, alignment=TA_CENTER, leading=16
        ),
        "table_header": ParagraphStyle(
            "table_header", fontName="Helvetica-Bold", fontSize=7,
            textColor=WHITE, alignment=TA_CENTER
        ),
        "table_cell": ParagraphStyle(
            "table_cell", fontName="Helvetica", fontSize=7,
            textColor=DARK_BLUE, alignment=TA_LEFT
        ),
        "table_cell_right": ParagraphStyle(
            "table_cell_right", fontName="Helvetica", fontSize=7,
            textColor=DARK_BLUE, alignment=TA_RIGHT
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer", fontName="Helvetica-Oblique", fontSize=6,
            textColor=MID_GRAY, alignment=TA_LEFT, leading=8
        ),
    }


# ─────────────────────────────────────────────
# Calcul des métriques de risque/perf
# ─────────────────────────────────────────────
def compute_metrics(portfolio_index: pd.Series,
                    bench_index: pd.Series,
                    portfolio_returns: pd.Series) -> dict:
    """
    Calcule les métriques clés depuis les séries de prix/rendements.
    portfolio_index et bench_index : series indexées (1.0 à la date de début)
    portfolio_returns : rendements journaliers du portefeuille hedgé
    """
    r = portfolio_returns.dropna()
    ann_factor = 252

    # Perf cumulée
    total_return = (portfolio_index.iloc[-1] / portfolio_index.iloc[0] - 1) * 100

    # Perf 1 an
    idx_1y = portfolio_index[portfolio_index.index >= portfolio_index.index[-1] - pd.Timedelta(days=365)]
    perf_1y = (idx_1y.iloc[-1] / idx_1y.iloc[0] - 1) * 100 if len(idx_1y) > 1 else np.nan

    # Perf YTD
    year_start = pd.Timestamp(datetime.date.today().year, 1, 1)
    idx_ytd = portfolio_index[portfolio_index.index >= year_start]
    perf_ytd = (idx_ytd.iloc[-1] / idx_ytd.iloc[0] - 1) * 100 if len(idx_ytd) > 1 else np.nan

    # Volatilité annualisée
    vol = r.std() * np.sqrt(ann_factor) * 100

    # Sharpe (risk-free = 3.5% approximation BCE 2024)
    rf_daily = 0.035 / ann_factor
    excess = r - rf_daily
    sharpe = (excess.mean() / r.std() * np.sqrt(ann_factor)) if r.std() > 0 else np.nan

    # Sortino
    downside = r[r < 0].std() * np.sqrt(ann_factor)
    sortino = (r.mean() * ann_factor / downside) if downside > 0 else np.nan

    # Max Drawdown
    cumulative = (1 + r).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    # Calmar
    ann_return = r.mean() * ann_factor * 100
    calmar = (ann_return / abs(max_dd)) if max_dd != 0 else np.nan

    # Beta vs benchmark
    common = portfolio_returns.index.intersection(bench_index.index)
    bench_r = bench_index.loc[common].pct_change().dropna()
    port_r  = portfolio_returns.loc[bench_r.index]
    if len(bench_r) > 10:
        cov = np.cov(port_r, bench_r)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
        corr = np.corrcoef(port_r, bench_r)[0, 1]
    else:
        beta, corr = np.nan, np.nan

    # Tracking Error
    diff = port_r - bench_r
    tracking_error = diff.std() * np.sqrt(ann_factor) * 100

    # Information Ratio
    ir = (diff.mean() * ann_factor / (diff.std() * np.sqrt(ann_factor))) if diff.std() > 0 else np.nan

    return {
        "total_return": total_return,
        "perf_1y": perf_1y,
        "perf_ytd": perf_ytd,
        "ann_return": ann_return,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "beta": beta,
        "correlation": corr,
        "tracking_error": tracking_error,
        "information_ratio": ir,
    }


def fmt(val, suffix="", decimals=2, is_pct=False):
    """Formate une valeur numérique, gère les NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if is_pct:
        suffix = "%"
    return f"{val:.{decimals}f}{suffix}"


# ─────────────────────────────────────────────
# Génération des graphiques matplotlib → bytes
# ─────────────────────────────────────────────
def build_perf_chart(portfolio_index: pd.Series, bench_index: pd.Series) -> bytes:
    """Graphique performance cumulée portefeuille hedgé vs benchmark."""
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    common = portfolio_index.index.intersection(bench_index.index)
    p = portfolio_index.loc[common]
    b = bench_index.loc[common]

    ax.plot(p.index, (p - 1) * 100, color="#2E86C1", linewidth=1.5, label="Portefeuille hedgé USD")
    ax.fill_between(p.index, (p - 1) * 100, alpha=0.08, color="#2E86C1")
    ax.plot(b.index, (b - 1) * 100, color="#E74C3C", linewidth=1.2,
            linestyle="--", label="Benchmark composite", alpha=0.8)

    ax.axhline(0, color=MID_GRAY.hexval() if hasattr(MID_GRAY, 'hexval') else "#BDC3C7",
               linewidth=0.5, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E0E0E0")
    ax.spines["bottom"].set_color("#E0E0E0")
    ax.tick_params(labelsize=7, colors="#566573")
    ax.legend(fontsize=7, framealpha=0, loc="upper left")
    ax.set_title("Performance cumulée (base 0%)", fontsize=8, color="#0D1B2A", pad=6)

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_drawdown_chart(portfolio_returns: pd.Series) -> bytes:
    """Graphique drawdown."""
    r = portfolio_returns.dropna()
    cumulative = (1 + r).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = ((cumulative - rolling_max) / rolling_max) * 100

    fig, ax = plt.subplots(figsize=(7.2, 1.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.fill_between(drawdown.index, drawdown, 0, color="#C0392B", alpha=0.6)
    ax.plot(drawdown.index, drawdown, color="#C0392B", linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E0E0E0")
    ax.spines["bottom"].set_color("#E0E0E0")
    ax.tick_params(labelsize=7, colors="#566573")
    ax.set_title("Drawdown", fontsize=8, color="#0D1B2A", pad=4)

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_allocation_chart(allocation: dict, ticker_names: dict, weights: pd.Series) -> bytes:
    """Donut chart allocation effective."""
    labels = [ticker_names.get(t, t) for t in weights.index]
    sizes  = [w * 100 for w in weights.values]

    # Grouper les petites positions (<3%) dans "Autres"
    threshold = 3.0
    main_labels, main_sizes, other_size = [], [], 0.0
    for l, s in zip(labels, sizes):
        if s >= threshold:
            main_labels.append(l)
            main_sizes.append(s)
        else:
            other_size += s
    if other_size > 0:
        main_labels.append(f"Autres (<{threshold:.0f}%)")
        main_sizes.append(other_size)

    palette = [
        "#2E86C1", "#1B4F72", "#148F77", "#1E8449", "#B7950B",
        "#884EA0", "#CB4335", "#D35400", "#717D7E", "#2C3E50",
        "#2471A3", "#0E6655"
    ]

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    fig.patch.set_facecolor("white")
    wedges, texts, autotexts = ax.pie(
        main_sizes,
        labels=None,
        autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
        colors=palette[:len(main_sizes)],
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.2),
        startangle=90,
        pctdistance=0.78
    )
    for at in autotexts:
        at.set_fontsize(6)
        at.set_color("white")

    ax.legend(
        wedges, main_labels,
        loc="center left", bbox_to_anchor=(1.0, 0.5),
        fontsize=6, framealpha=0,
        handlelength=1.0, handleheight=1.0
    )
    ax.set_title("Allocation effective", fontsize=8, color="#0D1B2A", pad=4)

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# Assemblage du PDF
# ─────────────────────────────────────────────
def generate_factsheet_pdf(
    portfolio_index_hedged: pd.Series,
    bench_index: pd.Series,
    portfolio_returns_hedged: pd.Series,
    allocation: dict,
    ticker_names: dict,
    weights: pd.Series
) -> bytes:
    """
    Génère la factsheet complète et retourne les bytes du PDF.
    """
    buf = io.BytesIO()
    styles = get_styles()
    metrics = compute_metrics(portfolio_index_hedged, bench_index, portfolio_returns_hedged)

    W, H = A4
    margin = 15 * mm

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=10 * mm, bottomMargin=12 * mm,
        title="Factsheet Portefeuille Hedgé USD"
    )

    content = []

    # ── Header Banner ──────────────────────────────────────────────
    as_of = portfolio_index_hedged.index[-1].strftime("%d %B %Y")
    header_data = [[
        Paragraph(f"<b>Portfolio Hedgé USD</b>", styles["title"]),
        Paragraph(f"Factsheet mensuelle<br/>Au {as_of}", styles["subtitle"])
    ]]
    header_table = Table(header_data, colWidths=[120 * mm, 60 * mm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 8),
        ("RIGHTPADDING", (-1, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    content.append(header_table)
    content.append(Spacer(1, 5 * mm))

    # ── Métriques clés (6 cartes) ──────────────────────────────────
    content.append(Paragraph("Indicateurs clés", styles["section"]))
    content.append(Spacer(1, 2 * mm))

    def metric_style(val, is_pct=True):
        """Choisit le style selon le signe de la valeur."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return styles["metric_value"]
        return styles["metric_value_green"] if val >= 0 else styles["metric_value_red"]

    kpis = [
        ("Perf. YTD",      metrics["perf_ytd"],       True,  "%"),
        ("Perf. 1 an",     metrics["perf_1y"],         True,  "%"),
        ("Perf. totale",   metrics["total_return"],    True,  "%"),
        ("Volatilité ann.", metrics["volatility"],     False, "%"),
        ("Sharpe",         metrics["sharpe"],          False, "x"),
        ("Max Drawdown",   metrics["max_drawdown"],    True,  "%"),
    ]

    metric_cells = []
    for label, val, signed, suffix in kpis:
        s = metric_style(val) if signed else styles["metric_value"]
        metric_cells.append([
            Paragraph(fmt(val, suffix, decimals=2), s),
            Paragraph(label, styles["metric_label"])
        ])

    # 3 colonnes x 2 lignes
    row1 = [metric_cells[i][0] for i in range(3)]
    row1b = [metric_cells[i][1] for i in range(3)]
    row2 = [metric_cells[i][0] for i in range(3, 6)]
    row2b = [metric_cells[i][1] for i in range(3, 6)]

    kpi_table_data = [row1, row1b, row2, row2b]
    col_w = (W - 2 * margin) / 3
    kpi_table = Table(kpi_table_data, colWidths=[col_w] * 3)
    kpi_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
        ("ROWBACKGROUND", (0, 0), (-1, -1), [LIGHT_GRAY, colors.white, LIGHT_GRAY, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    content.append(kpi_table)
    content.append(Spacer(1, 5 * mm))

    # ── Graphique performance cumulée ───────────────────────────────
    content.append(Paragraph("Performance cumulée vs Benchmark", styles["section"]))
    content.append(Spacer(1, 2 * mm))

    perf_img_bytes = build_perf_chart(portfolio_index_hedged, bench_index)
    perf_img = RLImage(io.BytesIO(perf_img_bytes), width=W - 2 * margin, height=68 * mm)
    content.append(perf_img)
    content.append(Spacer(1, 4 * mm))

    # ── Drawdown + Allocation côte à côte ───────────────────────────
    dd_img_bytes    = build_drawdown_chart(portfolio_returns_hedged)
    alloc_img_bytes = build_allocation_chart(allocation, ticker_names, weights)

    col_half = (W - 2 * margin - 6 * mm) / 2

    dd_img    = RLImage(io.BytesIO(dd_img_bytes),    width=col_half, height=44 * mm)
    alloc_img = RLImage(io.BytesIO(alloc_img_bytes), width=col_half, height=44 * mm)

    two_col = Table([[dd_img, alloc_img]], colWidths=[col_half, col_half])
    two_col.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("INNERGRID",     (0, 0), (-1, -1), 0, colors.white),
    ]))
    content.append(two_col)
    content.append(Spacer(1, 4 * mm))

    # ── Tableau métriques avancées ──────────────────────────────────
    content.append(Paragraph("Statistiques de risque avancées", styles["section"]))
    content.append(Spacer(1, 2 * mm))

    advanced = [
        ["Métrique", "Portefeuille hedgé", "Benchmark"],
        ["Rendement annualisé",    fmt(metrics["ann_return"], "%"),     "—"],
        ["Volatilité annualisée",  fmt(metrics["volatility"], "%"),     "—"],
        ["Ratio de Sharpe",        fmt(metrics["sharpe"], "x"),         "—"],
        ["Ratio de Sortino",       fmt(metrics["sortino"], "x"),        "—"],
        ["Max Drawdown",           fmt(metrics["max_drawdown"], "%"),   "—"],
        ["Ratio de Calmar",        fmt(metrics["calmar"], "x"),         "—"],
        ["Beta vs benchmark",      fmt(metrics["beta"]),                "1.00"],
        ["Corrélation benchmark",  fmt(metrics["correlation"]),         "1.00"],
        ["Tracking Error",         fmt(metrics["tracking_error"], "%"), "—"],
        ["Information Ratio",      fmt(metrics["information_ratio"]),   "—"],
    ]

    col_w3 = [(W - 2 * margin) * r for r in [0.45, 0.275, 0.275]]
    adv_table = Table(advanced, colWidths=col_w3)
    adv_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), MID_BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 7.5),
        ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",        (0, 0), (0, -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("GRID",         (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING",  (0, 0), (0, -1), 6),
    ]))
    content.append(adv_table)
    content.append(Spacer(1, 4 * mm))

    # ── Tableau composition ─────────────────────────────────────────
    content.append(Paragraph("Composition du portefeuille hedgé", styles["section"]))
    content.append(Spacer(1, 2 * mm))

    comp_data = [["Actif", "Ticker", "Poids cible", "Poids effectif"]]
    for t, w in weights.items():
        comp_data.append([
            ticker_names.get(t, t),
            t,
            f"{allocation.get(t, 0) * 100:.1f}%",
            f"{w * 100:.1f}%"
        ])
    comp_data.append(["Cash", "—", "5.0%", "5.0%"])

    col_w4 = [(W - 2 * margin) * r for r in [0.42, 0.22, 0.18, 0.18]]
    comp_table = Table(comp_data, colWidths=col_w4)
    comp_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), ACCENT_BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("ALIGN",         (2, 0), (-1, -1), "CENTER"),
        ("ALIGN",         (0, 0), (1, -1), "LEFT"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.3, MID_GRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING",   (0, 0), (1, -1), 5),
        ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
    ]))
    content.append(comp_table)
    content.append(Spacer(1, 5 * mm))

    # ── Disclaimer ──────────────────────────────────────────────────
    content.append(HRFlowable(width="100%", thickness=0.3, color=MID_GRAY))
    content.append(Spacer(1, 2 * mm))
    disclaimer_text = (
        "Document à caractère informatif uniquement. Les performances passées ne préjugent pas des performances futures. "
        "Ce document ne constitue pas un conseil en investissement. La couverture de change USD est simulée "
        "sans coût de couverture. Données issues de Yahoo Finance / FMP via OpenBB. "
        f"Généré le {datetime.datetime.now().strftime('%d/%m/%Y à %H:%M')}."
    )
    content.append(Paragraph(disclaimer_text, styles["disclaimer"]))

    doc.build(content)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# Section Streamlit à appeler dans l'app principale
# ─────────────────────────────────────────────
def render_factsheet_section(
    portfolio_index_hedged: pd.Series,
    bench_index: pd.Series,
    portfolio_returns_hedged: pd.Series,
    allocation: dict,
    ticker_names: dict,
    weights: pd.Series
):
    """
    Ajoute le bouton d'export factsheet à l'interface Streamlit.
    Appelle cette fonction à la fin de ton portfolio_app.py.
    """
    st.divider()
    st.subheader("📄 Export Factsheet")

    col_info, col_btn = st.columns([3, 1])

    with col_info:
        st.markdown(
            "Génère une factsheet PDF du portefeuille **hedgé USD** incluant : "
            "indicateurs clés, performance vs benchmark, drawdown, "
            "allocation et statistiques de risque avancées."
        )

    with col_btn:
        if st.button("⬇️ Générer la factsheet PDF", type="primary"):
            with st.spinner("Génération du PDF en cours..."):
                try:
                    pdf_bytes = generate_factsheet_pdf(
                        portfolio_index_hedged=portfolio_index_hedged,
                        bench_index=bench_index,
                        portfolio_returns_hedged=portfolio_returns_hedged,
                        allocation=allocation,
                        ticker_names=ticker_names,
                        weights=weights
                    )
                    filename = f"factsheet_{datetime.date.today().strftime('%Y%m%d')}.pdf"
                    st.download_button(
                        label="📥 Télécharger le PDF",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("Factsheet générée avec succès !")
                except Exception as e:
                    st.error(f"Erreur lors de la génération : {e}")
                    raise

# =====================
# Export Factsheet
# =====================
from factsheet_export import render_factsheet_section

render_factsheet_section(
    portfolio_index_hedged=portfolio_index_hedged,
    bench_index=bench_index,
    portfolio_returns_hedged=portfolio_returns_hedged,
    allocation=allocation,
    ticker_names=ticker_names,
    weights=weights
)
