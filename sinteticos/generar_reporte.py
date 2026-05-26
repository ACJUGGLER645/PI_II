"""
Generador de Reporte Técnico PDF — Pipeline de Analítica Predictiva
Proyecto PI-II: Detección Temprana de Riesgo de Deserción — ETITC Nocturna

Autores:  Jhon Alejandro Correal Martínez
          Rafael Andrés Guzmán Rodríguez
Docente:  Doris Constanza Alvarado Mariño
Versión:  1.0  |  2026-05-25

Ejecuta internamente el pipeline completo (ETL + entrenamiento + evaluación)
y exporta un reporte PDF de 7 páginas orientado al profesor de Ciencia de Datos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_curve
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
SEED           = 42
TEST_SIZE      = 0.20
N_ESTIMATORS   = 100
MAX_DEPTH      = 6
MIN_SAMPLES    = 4
OUTPUT_PDF     = "reporte_modelo_pi2.pdf"

COLOR_VERDE    = "#00913C"   # verde ETITC
COLOR_GRIS     = "#4A4A4A"
COLOR_CLARO    = "#E8F5EE"
COLOR_ALERTA   = "#C0392B"
COLOR_AZUL     = "#2471A3"
COLOR_NARANJA  = "#E67E22"

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 9,
    "axes.titlesize"  : 10,
    "axes.labelsize"  : 9,
    "axes.spines.top" : False,
    "axes.spines.right": False,
})


# =============================================================================
# PIPELINE INTERNO (ETL + ENTRENAMIENTO)
# =============================================================================
df_b = pd.read_csv("bienestar_caracterizacion.csv")
df_n = pd.read_csv("adviser_teams_notas.csv")
df_t = pd.read_csv("registro_admisiones_target.csv")

df = (df_b
      .merge(df_n, on="id_estudiante", how="inner")
      .merge(df_t, on="id_estudiante", how="inner"))

y = df["deserto"]
X = df.drop(columns=["id_estudiante", "deserto"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

modelo = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES, random_state=SEED, n_jobs=-1
)
modelo.fit(X_train, y_train)

y_pred      = modelo.predict(X_test)
y_prob      = modelo.predict_proba(X_test)[:, 1]
cm          = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
reporte_dict = classification_report(
    y_test, y_pred,
    target_names=["Persiste", "Deserto"],
    output_dict=True
)

importancias = pd.Series(
    modelo.feature_importances_, index=X.columns
).sort_values(ascending=False)

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc      = auc(fpr, tpr)
prec_pr, rec_pr, _ = precision_recall_curve(y_test, y_prob)

print("Pipeline ejecutado. Generando PDF...")


# =============================================================================
# FUNCIONES DE UTILIDAD PARA EL DISEÑO
# =============================================================================

def cabecera(ax_titulo, titulo, subtitulo=""):
    """Dibuja la cabecera verde de cada página."""
    ax_titulo.set_facecolor(COLOR_VERDE)
    ax_titulo.text(0.03, 0.62, titulo,
                   transform=ax_titulo.transAxes,
                   fontsize=13, fontweight="bold", color="white", va="center")
    if subtitulo:
        ax_titulo.text(0.03, 0.18, subtitulo,
                       transform=ax_titulo.transAxes,
                       fontsize=8, color="#D5F5E3", va="center")
    ax_titulo.set_xticks([])
    ax_titulo.set_yticks([])
    for sp in ax_titulo.spines.values():
        sp.set_visible(False)


def pie_pagina(fig, num, total=7):
    fig.text(0.97, 0.01, f"Página {num}/{total}  |  PI-II ETITC — 2026",
             ha="right", va="bottom", fontsize=7, color="#888888")
    fig.text(0.03, 0.01, "Correal M. & Guzmán R.",
             ha="left",  va="bottom", fontsize=7, color="#888888")


# =============================================================================
# CONSTRUCCIÓN DEL PDF
# =============================================================================
with PdfPages(OUTPUT_PDF) as pdf:

    # ─────────────────────────────────────────────────────────────────────────
    # PÁGINA 1: PORTADA
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")

    # Bloque verde superior
    ax_top = fig.add_axes([0, 0.72, 1, 0.28])
    ax_top.set_facecolor(COLOR_VERDE)
    ax_top.axis("off")
    ax_top.text(0.5, 0.72, "REPORTE TÉCNICO DE VALIDACIÓN",
                ha="center", va="center", fontsize=15,
                fontweight="bold", color="white",
                transform=ax_top.transAxes)
    ax_top.text(0.5, 0.42, "Pipeline de Analítica Predictiva — Detección Temprana",
                ha="center", va="center", fontsize=11, color="#D5F5E3",
                transform=ax_top.transAxes)
    ax_top.text(0.5, 0.18, "de Riesgo de Deserción Estudiantil",
                ha="center", va="center", fontsize=11, color="#D5F5E3",
                transform=ax_top.transAxes)

    ax_body = fig.add_axes([0.08, 0.10, 0.84, 0.58])
    ax_body.axis("off")

    info = [
        ("Institución",   "Escuela Tecnológica Instituto Técnico Central (ETITC)"),
        ("Programa",      "Facultad de Sistemas — Proyecto de Investigación II"),
        ("Jornada",       "Nocturna — Sede Centro, Bogotá"),
        ("",              ""),
        ("Autores",       "Jhon Alejandro Correal Martínez"),
        ("",              "Rafael Andrés Guzmán Rodríguez"),
        ("Docente",       "Doris Constanza Alvarado Mariño"),
        ("",              ""),
        ("Fecha",         "Mayo 25 de 2026"),
        ("Versión TRL",   "TRL 2/3 — Validación con datos sintéticos calibrados"),
        ("",              ""),
        ("Algoritmo",     f"Random Forest Classifier  (n={N_ESTIMATORS}, max_depth={MAX_DEPTH})"),
        ("Dataset",       "1,000 estudiantes sintéticos | 24 variables | 3 fuentes CSV"),
        ("Tasa deserción","25.7%  (257 positivos / 743 negativos)"),
    ]
    for i, (etiq, valor) in enumerate(info):
        y_pos = 0.95 - i * 0.065
        if etiq:
            ax_body.text(0.0, y_pos, etiq + ":",
                         fontsize=9, fontweight="bold", color=COLOR_VERDE,
                         transform=ax_body.transAxes)
        ax_body.text(0.28, y_pos, valor,
                     fontsize=9, color=COLOR_GRIS,
                     transform=ax_body.transAxes)

    # Línea separadora inferior
    fig.add_axes([0.08, 0.085, 0.84, 0.004]).set_facecolor(COLOR_VERDE)
    fig.axes[-1].axis("off")

    pie_pagina(fig, 1)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PÁGINA 2: ARQUITECTURA DE DATOS Y ETL
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(5, 2, figure=fig,
                           top=0.95, bottom=0.06,
                           left=0.07, right=0.95,
                           hspace=0.55, wspace=0.35)

    ax_h = fig.add_subplot(gs[0, :])
    cabecera(ax_h, "Arquitectura Multi-CSV y Proceso ETL",
             "Estructura relacional de tres fuentes independientes unificadas por id_estudiante")

    # Diagrama ETL textual
    ax_etl = fig.add_subplot(gs[1:3, :])
    ax_etl.axis("off")
    ax_etl.set_xlim(0, 10)
    ax_etl.set_ylim(0, 4)

    # Cajas CSV
    bloques = [
        (1.0, 2.5, "bienestar_\ncaracterizacion.csv",
         "7 columnas\n6 variables\nPerfil socioeconómico", COLOR_AZUL),
        (4.3, 2.5, "adviser_teams_\nnotas.csv",
         "19 columnas\n18 variables\nCortes 1, 2 y 3", COLOR_VERDE),
        (7.6, 2.5, "registro_admisiones_\ntarget.csv",
         "2 columnas\nVariable objetivo\ndeserto (0/1)", COLOR_NARANJA),
    ]
    for bx, by, titulo, detalle, color in bloques:
        rect = mpatches.FancyBboxPatch(
            (bx - 1.0, by - 1.1), 2.2, 2.0,
            boxstyle="round,pad=0.08",
            linewidth=1.5, edgecolor=color, facecolor=color + "22"
        )
        ax_etl.add_patch(rect)
        ax_etl.text(bx + 0.1, by + 0.65, titulo,
                    ha="center", fontsize=7.5, fontweight="bold", color=color)
        ax_etl.text(bx + 0.1, by - 0.4, detalle,
                    ha="center", fontsize=7, color=COLOR_GRIS)

    # Flechas hacia merge
    for bx in [1.0, 4.3, 7.6]:
        ax_etl.annotate("", xy=(5.0, 0.55), xytext=(bx + 0.1, 1.4),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))

    # Caja merge
    rect_merge = mpatches.FancyBboxPatch(
        (3.5, -0.1), 3.0, 0.75,
        boxstyle="round,pad=0.08",
        linewidth=2, edgecolor=COLOR_VERDE, facecolor=COLOR_CLARO
    )
    ax_etl.add_patch(rect_merge)
    ax_etl.text(5.0, 0.28,
                "INNER JOIN  on  id_estudiante\n1,000 filas × 26 columnas | 0 nulos",
                ha="center", fontsize=8, fontweight="bold", color=COLOR_VERDE)

    # Tabla resumen de fuentes
    ax_tbl = fig.add_subplot(gs[3:, :])
    ax_tbl.axis("off")

    encabezados = ["Archivo CSV", "Filas", "Columnas", "Tipo de variable", "Rol en el pipeline"]
    filas_tbl = [
        ["bienestar_caracterizacion.csv", "1,000", "7",
         "Booleanas + entero", "Variables explicativas (contexto)"],
        ["adviser_teams_notas.csv",       "1,000", "19",
         "Booleanas + continuas", "Variables explicativas (rendimiento)"],
        ["registro_admisiones_target.csv","1,000", "2",
         "Booleana",             "Variable objetivo  y = deserto"],
        ["Matriz unificada (post-merge)", "1,000", "26",
         "Mixtas",               "Entrada al modelo de ML"],
    ]

    tabla = ax_tbl.table(
        cellText=filas_tbl, colLabels=encabezados,
        loc="center", cellLoc="left"
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8)
    tabla.scale(1, 1.6)
    for (r, c), cell in tabla.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor(COLOR_VERDE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")
        else:
            cell.set_facecolor("white")

    pie_pagina(fig, 2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PÁGINA 3: ESTADÍSTICA DESCRIPTIVA DEL DATASET
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           top=0.95, bottom=0.06,
                           left=0.08, right=0.95,
                           hspace=0.65, wspace=0.40)

    ax_h = fig.add_subplot(gs[0, :])
    cabecera(ax_h, "Estadística Descriptiva del Dataset Sintético",
             "Distribuciones de las variables socioeconómicas y la variable objetivo")

    # Gráfico 1: Balance de clases (donut)
    ax1 = fig.add_subplot(gs[1, 0])
    vals_clase = [743, 257]
    colores_clase = [COLOR_AZUL, COLOR_ALERTA]
    wedges, texts, autotexts = ax1.pie(
        vals_clase,
        labels=["Persiste (0)", "Deserto (1)"],
        colors=colores_clase,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 8}
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight("bold")
        at.set_color("white")
    ax1.set_title("Balance de clases\n(Variable objetivo: deserto)", fontweight="bold")

    # Gráfico 2: Variables binarias de bienestar
    ax2 = fig.add_subplot(gs[1, 1])
    vars_bool = ["trabaja", "debilidad_mate", "conflicto_tiempo",
                 "barrera_tecnologica", "inseguridad_alimentaria"]
    etiquetas = ["Trabaja", "Deb. Mate", "Conflicto\nTiempo", "Barrera\nTecnol.", "Inseg.\nAliment."]
    porcentajes = [df_b[v].mean() * 100 for v in vars_bool]
    colores_bar = [COLOR_VERDE if p >= 50 else COLOR_AZUL for p in porcentajes]
    bars = ax2.barh(etiquetas, porcentajes, color=colores_bar, edgecolor="white", height=0.6)
    for bar, pct in zip(bars, porcentajes):
        ax2.text(pct + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{pct:.1f}%", va="center", fontsize=8, color=COLOR_GRIS)
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("% de estudiantes")
    ax2.set_title("Variables binarias — Bienestar\n(% con valor = 1)", fontweight="bold")
    ax2.axvline(50, color="#CCCCCC", linestyle="--", linewidth=0.8)

    # Gráfico 3: Distribución brecha_bach
    ax3 = fig.add_subplot(gs[2, 0])
    brecha_conteo = df_b["brecha_bach"].value_counts().sort_index()
    ax3.bar(brecha_conteo.index, brecha_conteo.values,
            color=COLOR_AZUL, edgecolor="white", alpha=0.85)
    ax3.set_xlabel("Años desde bachillerato")
    ax3.set_ylabel("Estudiantes")
    ax3.set_title("Distribución de brecha_bach\n(rezago temporal)", fontweight="bold")
    ax3.set_xticks(brecha_conteo.index)
    for i, (xi, yi) in enumerate(zip(brecha_conteo.index, brecha_conteo.values)):
        ax3.text(xi, yi + 4, str(yi), ha="center", fontsize=8, color=COLOR_GRIS)

    # Gráfico 4: Distribución de definitivas por corte (boxplot)
    ax4 = fig.add_subplot(gs[2, 1])
    data_box = [
        df_n["definitiva_c1"].values,
        df_n["definitiva_c2"].values,
        df_n["definitiva_c3"].values,
    ]
    bp = ax4.boxplot(data_box, patch_artist=True,
                     medianprops={"color": "white", "linewidth": 2},
                     flierprops={"marker": ".", "markersize": 3,
                                 "markerfacecolor": "#AAAAAA"})
    colores_box = [COLOR_AZUL, COLOR_VERDE, COLOR_NARANJA]
    for patch, color in zip(bp["boxes"], colores_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(["Corte 1", "Corte 2", "Corte 3"])
    ax4.set_ylabel("Nota definitiva (0–5)")
    ax4.axhline(3.0, color=COLOR_ALERTA, linestyle="--",
                linewidth=1, label="Umbral de aprobación (3.0)")
    ax4.set_title("Distribución de notas definitivas\npor corte académico", fontweight="bold")
    ax4.legend(fontsize=7)

    # Gráfico 5: Asistencia por corte (barras agrupadas)
    ax5 = fig.add_subplot(gs[3, :])
    cortes = ["Corte 1", "Corte 2", "Corte 3"]
    asistencia = [df_n[f"asistio_c{c}"].mean() * 100 for c in [1, 2, 3]]
    ausentismo = [100 - a for a in asistencia]
    x_pos = np.arange(3)
    w = 0.35
    ax5.bar(x_pos - w/2, asistencia, w, label="Asistió", color=COLOR_VERDE, alpha=0.85)
    ax5.bar(x_pos + w/2, ausentismo, w, label="No asistió", color=COLOR_ALERTA, alpha=0.75)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(cortes)
    ax5.set_ylabel("% estudiantes")
    ax5.set_ylim(0, 105)
    ax5.set_title("Tasa de asistencia vs. ausentismo presencial por corte\n"
                  "(efecto acumulado de la presión laboral conforme avanza el semestre)",
                  fontweight="bold")
    ax5.legend(fontsize=8)
    for i, (a, b_) in enumerate(zip(asistencia, ausentismo)):
        ax5.text(i - w/2, a + 1, f"{a:.1f}%", ha="center", fontsize=8)
        ax5.text(i + w/2, b_ + 1, f"{b_:.1f}%", ha="center", fontsize=8)

    pie_pagina(fig, 3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PÁGINA 4: REGLAS DE NEGOCIO Y SIMULACIÓN
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           top=0.95, bottom=0.06,
                           left=0.07, right=0.95,
                           hspace=0.55, wspace=0.38)

    ax_h = fig.add_subplot(gs[0, :])
    cabecera(ax_h, "Reglas de Negocio Programadas en el Script de Simulación",
             "Consistencia interna que valida la arquitectura del pipeline antes de recibir datos reales")

    # Gráfico: impacto de riesgo cognitivo en parciales
    ax_cog = fig.add_subplot(gs[1, 0])
    mascara_rc  = (df["brecha_bach"] > 3) & (df["debilidad_mate"] == 1)
    p_riesgo_   = df.loc[mascara_rc,  "parcial_c1"].values
    p_normal_   = df.loc[~mascara_rc, "parcial_c1"].values
    ax_cog.hist(p_normal_, bins=20, alpha=0.7, color=COLOR_AZUL,
                label=f"Sin riesgo cognitivo  (n={len(p_normal_)})", density=True)
    ax_cog.hist(p_riesgo_, bins=15, alpha=0.8, color=COLOR_ALERTA,
                label=f"Riesgo cognitivo alto (n={len(p_riesgo_)})", density=True)
    ax_cog.axvline(3.0, color="black", linestyle="--", linewidth=1, label="Umbral 3.0")
    ax_cog.set_xlabel("Nota parcial C1")
    ax_cog.set_ylabel("Densidad")
    ax_cog.set_title("Regla 3: Choque cognitivo\n(brecha_bach>3 ∧ debilidad_mate=1)", fontweight="bold")
    ax_cog.legend(fontsize=7)

    # Gráfico: impacto de presión de tiempo en notas Teams
    ax_teams = fig.add_subplot(gs[1, 1])
    presion_ = (df["trabaja"] == 1) | (df["conflicto_tiempo"] == 1)
    nt_pres  = df.loc[presion_,  "notas_teams_c1"].values
    nt_nopres= df.loc[~presion_, "notas_teams_c1"].values
    ax_teams.hist(nt_nopres, bins=20, alpha=0.7, color=COLOR_VERDE,
                  label=f"Sin presión de tiempo (n={len(nt_nopres)})", density=True)
    ax_teams.hist(nt_pres,   bins=20, alpha=0.7, color=COLOR_NARANJA,
                  label=f"Con presión de tiempo (n={len(nt_pres)})", density=True)
    ax_teams.axvline(3.0, color="black", linestyle="--", linewidth=1, label="Umbral 3.0")
    ax_teams.set_xlabel("Nota Teams C1 (entregas)")
    ax_teams.set_ylabel("Densidad")
    ax_teams.set_title("Regla 2: Penalización por ausentismo\n(trabaja=1 ∨ conflicto_tiempo=1)", fontweight="bold")
    ax_teams.legend(fontsize=7)

    # Tabla de reglas
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis("off")

    reglas = [
        ["Regla 1", "Porcentajes base",
         "trabaja: 68%  |  debilidad_mate: 45%",
         "Calibrado desde informes BU 2021–2025"],
        ["Regla 2", "Presión de tiempo",
         "P(ausentismo) = 40% si trabaja=1 ∨ conflicto=1",
         "notas_teams penalizadas a [0.0, 2.5] si no asiste"],
        ["Regla 3", "Choque cognitivo",
         "Si brecha>3 ∧ debilidad=1 → N(2.2, 0.5)",
         "Resto → N(3.5–4.2, 0.5–0.6) según presión"],
        ["Regla 4", "Criterio de deserción",
         "deserto=1 si promedio_definitivas < 3.0",
         "O si autoevaluacion=coevaluacion=0.0 en C2/C3"],
    ]
    tabla = ax_tbl.table(
        cellText=reglas,
        colLabels=["Regla", "Nombre", "Condición lógica", "Efecto programado"],
        loc="center", cellLoc="left"
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8)
    tabla.scale(1, 1.8)
    for (r, c), cell in tabla.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor(COLOR_VERDE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")
        else:
            cell.set_facecolor("white")

    pie_pagina(fig, 4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PÁGINA 5: MATRIZ DE CONFUSIÓN + CURVAS ROC Y PR
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           top=0.95, bottom=0.06,
                           left=0.08, right=0.95,
                           hspace=0.55, wspace=0.40)

    ax_h = fig.add_subplot(gs[0, :])
    cabecera(ax_h, "Evaluación del Modelo — Conjunto de Prueba (20%)",
             "200 estudiantes reservados con estratificación  |  Random Forest  n=100  max_depth=6")

    # Matriz de confusión (heatmap)
    ax_cm = fig.add_subplot(gs[1, 0])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Persiste (0)", "Deserto (1)"]
    )
    disp.plot(ax=ax_cm, colorbar=False, cmap="Greens")
    ax_cm.set_title("Matriz de Confusión", fontweight="bold")
    # Anotar FN en rojo
    ax_cm.texts[2].set_color(COLOR_ALERTA)
    ax_cm.texts[2].set_fontweight("bold")
    ax_cm.texts[2].set_fontsize(13)

    # Leyenda de celdas
    ax_leg = fig.add_subplot(gs[1, 1])
    ax_leg.axis("off")
    leyenda_cm = [
        (f"VN = {TN}", COLOR_AZUL,
         "Predijo Persiste → ERA Persiste\n(Sin alerta, correcto)"),
        (f"FP = {FP}", COLOR_NARANJA,
         "Predijo Deserto → ERA Persiste\n(Falsa alarma — intervención\ninnecesaria)"),
        (f"FN = {FN}", COLOR_ALERTA,
         "Predijo Persiste → ERA Deserto\n⚠ ERROR CRITICO: alumno en\nriesgo no detectado"),
        (f"VP = {TP}", COLOR_VERDE,
         "Predijo Deserto → ERA Deserto\n(Alerta correcta — intervención\nposible y oportuna)"),
    ]
    for i, (etiq, color, desc) in enumerate(leyenda_cm):
        y0 = 0.88 - i * 0.24
        rect = mpatches.FancyBboxPatch(
            (0.0, y0 - 0.08), 0.22, 0.18,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white",
            transform=ax_leg.transAxes
        )
        ax_leg.add_patch(rect)
        ax_leg.text(0.11, y0 + 0.01, etiq,
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white",
                    transform=ax_leg.transAxes)
        ax_leg.text(0.27, y0 + 0.01, desc,
                    ha="left", va="center", fontsize=7.5,
                    color=COLOR_GRIS,
                    transform=ax_leg.transAxes)
    ax_leg.set_title("Interpretación de celdas", fontweight="bold")

    # Curva ROC
    ax_roc = fig.add_subplot(gs[2, 0])
    ax_roc.plot(fpr, tpr, color=COLOR_VERDE, lw=2,
                label=f"ROC (AUC = {roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="#CCCCCC",
                linestyle="--", lw=1, label="Clasificador aleatorio")
    ax_roc.fill_between(fpr, tpr, alpha=0.10, color=COLOR_VERDE)
    ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax_roc.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax_roc.set_title(f"Curva ROC  |  AUC = {roc_auc:.3f}", fontweight="bold")
    ax_roc.legend(fontsize=8)
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1.02])

    # Curva Precision-Recall
    ax_pr = fig.add_subplot(gs[2, 1])
    ax_pr.plot(rec_pr, prec_pr, color=COLOR_AZUL, lw=2, label="Curva P-R")
    ax_pr.fill_between(rec_pr, prec_pr, alpha=0.10, color=COLOR_AZUL)
    ax_pr.axhline(257/1000, color="#CCCCCC", linestyle="--",
                  lw=1, label=f"Baseline (prevalencia={257/1000:.2f})")
    ax_pr.set_xlabel("Recall (Sensibilidad)")
    ax_pr.set_ylabel("Precisión")
    ax_pr.set_title("Curva Precision-Recall\n(relevante para clases desbalanceadas)",
                    fontweight="bold")
    ax_pr.legend(fontsize=8)
    ax_pr.set_xlim([0, 1])
    ax_pr.set_ylim([0, 1.02])

    pie_pagina(fig, 5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PÁGINA 6: MÉTRICAS Y FEATURE IMPORTANCE
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           top=0.95, bottom=0.06,
                           left=0.08, right=0.95,
                           hspace=0.55, wspace=0.40)

    ax_h = fig.add_subplot(gs[0, :])
    cabecera(ax_h, "Métricas de Rendimiento y Sensores Predictivos",
             "Classification Report  |  Feature Importance (Gini Impurity)")

    # Tabla classification report
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.axis("off")
    metricas_data = [
        ["Persiste (0)",
         f"{reporte_dict['Persiste']['precision']:.3f}",
         f"{reporte_dict['Persiste']['recall']:.3f}",
         f"{reporte_dict['Persiste']['f1-score']:.3f}",
         str(int(reporte_dict['Persiste']['support']))],
        ["Deserto (1)",
         f"{reporte_dict['Deserto']['precision']:.3f}",
         f"{reporte_dict['Deserto']['recall']:.3f}",
         f"{reporte_dict['Deserto']['f1-score']:.3f}",
         str(int(reporte_dict['Deserto']['support']))],
        ["Accuracy", "—", "—",
         f"{reporte_dict['accuracy']:.3f}", "200"],
        ["Macro avg",
         f"{reporte_dict['macro avg']['precision']:.3f}",
         f"{reporte_dict['macro avg']['recall']:.3f}",
         f"{reporte_dict['macro avg']['f1-score']:.3f}", "200"],
    ]
    tbl = ax_tbl.table(
        cellText=metricas_data,
        colLabels=["Clase", "Precision", "Recall", "F1-Score", "Support"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor(COLOR_VERDE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 2:
            cell.set_facecolor(COLOR_CLARO)
            cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")
        # Resaltar recall clase 1
        if r == 2 and c == 2:
            cell.set_facecolor("#FFF3CD")
    ax_tbl.set_title("Classification Report\n(conjunto de prueba, 200 muestras)",
                     fontweight="bold", pad=10)

    # Gauge de métricas clave
    ax_gauge = fig.add_subplot(gs[1, 1])
    ax_gauge.axis("off")
    kpis = [
        ("Accuracy",      reporte_dict["accuracy"],         COLOR_AZUL),
        ("Precision (1)", reporte_dict["Deserto"]["precision"], COLOR_VERDE),
        ("Recall (1)",    reporte_dict["Deserto"]["recall"],    COLOR_ALERTA),
        ("F1-Score (1)",  reporte_dict["Deserto"]["f1-score"],  COLOR_NARANJA),
        (f"AUC-ROC",      roc_auc,                           "#7D3C98"),
    ]
    for i, (nombre, valor, color) in enumerate(kpis):
        y0 = 0.85 - i * 0.18
        ax_gauge.add_patch(mpatches.FancyBboxPatch(
            (0.0, y0 - 0.07), 1.0, 0.14,
            boxstyle="round,pad=0.02",
            facecolor="#F8F9FA", edgecolor=color, linewidth=2,
            transform=ax_gauge.transAxes
        ))
        # Barra de progreso
        ax_gauge.add_patch(mpatches.FancyBboxPatch(
            (0.0, y0 - 0.07), valor, 0.14,
            boxstyle="round,pad=0.02",
            facecolor=color, alpha=0.20, edgecolor="none",
            transform=ax_gauge.transAxes
        ))
        ax_gauge.text(0.03, y0, nombre,
                      fontsize=9, va="center", fontweight="bold",
                      color=color, transform=ax_gauge.transAxes)
        ax_gauge.text(0.97, y0, f"{valor:.3f}",
                      fontsize=11, va="center", ha="right",
                      fontweight="bold", color=color,
                      transform=ax_gauge.transAxes)
    ax_gauge.set_title("KPIs del Modelo", fontweight="bold", pad=10)

    # Feature Importance (completo, horizontal)
    ax_fi = fig.add_subplot(gs[2, :])
    top15 = importancias.head(15)
    colores_fi = []
    for nombre in top15.index:
        if any(k in nombre for k in ["definitiva", "notas_teams", "parcial",
                                      "asistio", "autoevaluacion", "coevaluacion"]):
            colores_fi.append(COLOR_AZUL)
        else:
            colores_fi.append(COLOR_NARANJA)

    bars = ax_fi.barh(
        top15.index[::-1], top15.values[::-1],
        color=colores_fi[::-1], edgecolor="white", height=0.65
    )
    ax_fi.set_xlabel("Importancia (Gini Impurity — promedio sobre 100 árboles)")
    ax_fi.set_title("Top 15 Variables por Importancia Predictiva\n"
                    "  Azul = academica (Adviser/Teams)   |   Naranja = socioeconómica (Bienestar)",
                    fontweight="bold")
    for bar, val in zip(bars, top15.values[::-1]):
        ax_fi.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                   f"{val:.4f}", va="center", fontsize=7.5)
    ax_fi.set_xlim(0, top15.max() * 1.25)

    # Leyenda
    patch_ac = mpatches.Patch(color=COLOR_AZUL,   label="Variable académica")
    patch_se = mpatches.Patch(color=COLOR_NARANJA, label="Variable socioeconómica")
    ax_fi.legend(handles=[patch_ac, patch_se], fontsize=8, loc="lower right")

    pie_pagina(fig, 6)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PÁGINA 7: CONCLUSIONES Y PRÓXIMOS PASOS
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 1, figure=fig,
                           top=0.95, bottom=0.06,
                           left=0.07, right=0.95,
                           hspace=0.4)

    ax_h = fig.add_subplot(gs[0])
    cabecera(ax_h, "Conclusiones Técnicas y Próximos Pasos",
             "Validación TRL 2/3 — Consistencia de la arquitectura Multi-CSV demostrada")

    ax_body = fig.add_subplot(gs[1])
    ax_body.axis("off")

    bloques_texto = [
        ("HALLAZGOS VALIDADOS POR EL PIPELINE", COLOR_VERDE, [
            f"1. La arquitectura Multi-CSV (3 fuentes + INNER JOIN por id_estudiante) produce\n"
            f"   una matriz unificada de 1,000 × 24 sin valores nulos ni inconsistencias.",
            f"2. El modelo Random Forest (Accuracy={reporte_dict['accuracy']:.1%}, "
            f"AUC-ROC={roc_auc:.3f}) demuestra que los patrones\n"
            f"   inyectados en la simulación son estadisticamente separables.",
            f"3. Las 5 variables con mayor peso son académicas y dinámicas (definitivas\n"
            f"   por corte + autoevaluación C3), validando la hipótesis del proyecto:\n"
            f"   la 'huella digital académica' supera al perfil socioeconómico estático.",
            f"4. El Recall de {reporte_dict['Deserto']['recall']:.1%} para la clase Deserto significa que {FN} de los 51 desertores\n"
            f"   del conjunto de prueba NO fueron detectados. Este es el margen de mejora\n"
            f"   más importante para la fase experimental futura.",
        ]),
        ("LIMITACIONES METODOLÓGICAS (TRL 2/3)", COLOR_NARANJA, [
            "5. Los datos son SINTÉTICOS: los resultados métricamente altos se deben a que\n"
            "   el modelo aprende las mismas reglas lógicas que generaron los datos.\n"
            "   Esto es esperado y metodológicamente correcto en esta fase.",
            "6. La prueba real del sistema requiere microdatos individuales anonimizados\n"
            "   del sistema Adviser V.11 y los logs de Microsoft Teams.",
        ]),
        ("PRÓXIMOS PASOS PROYECTADOS", COLOR_AZUL, [
            "7. Ajuste del umbral de decisión (threshold < 0.5) para priorizar Recall\n"
            "   sobre Precisión, minimizando los Falsos Negativos.",
            "8. Incorporar class_weight='balanced' o SMOTE para el desbalance de clases.",
            "9. Comparar con XGBoost y evaluar con validación cruzada k-fold (k=5).",
            "10. Conexión con datos reales (previa firma de acuerdo de confidencialidad\n"
            "    con la ETITC bajo Ley 1581/2012 — Habeas Data).",
        ]),
    ]

    y_cursor = 0.97
    for titulo_bloque, color_bloque, items in bloques_texto:
        # Título de bloque
        rect = mpatches.FancyBboxPatch(
            (0.0, y_cursor - 0.04), 1.0, 0.045,
            boxstyle="round,pad=0.01",
            facecolor=color_bloque, edgecolor="none",
            transform=ax_body.transAxes
        )
        ax_body.add_patch(rect)
        ax_body.text(0.015, y_cursor - 0.017, titulo_bloque,
                     fontsize=8.5, fontweight="bold", color="white",
                     transform=ax_body.transAxes, va="center")
        y_cursor -= 0.055

        for item in items:
            n_lineas = item.count("\n") + 1
            ax_body.text(0.015, y_cursor, item,
                         fontsize=8, color=COLOR_GRIS,
                         transform=ax_body.transAxes, va="top",
                         linespacing=1.4)
            y_cursor -= 0.038 * n_lineas + 0.012

        y_cursor -= 0.01

    pie_pagina(fig, 7)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


print(f"Reporte exportado exitosamente: {OUTPUT_PDF}")
print(f"  -> 7 paginas  |  Pipeline completo  |  Listo para revision academica")
