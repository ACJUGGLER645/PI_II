"""
Generador de Presentación Técnica PDF — Pipeline de Analítica Predictiva
Proyecto PI-II: Detección Temprana de Riesgo de Deserción — ETITC Nocturna

Autores:  Jhon Alejandro Correal Martínez
          Rafael Andrés Guzmán Rodríguez
Docente:  Doris Constanza Alvarado Mariño
Versión:  2.0  |  2026-05-25

Formato: 16:9 landscape — una idea por diapositiva — fuente grande
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_curve
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
SEED         = 42
TEST_SIZE    = 0.20
N_EST        = 100
MAX_D        = 6
MIN_LEAF     = 4
OUTPUT_PDF   = "presentacion_modelo_pi2.pdf"
SLIDE_W      = 13.33
SLIDE_H      = 7.50

COLOR_VERDE   = "#00913C"
COLOR_OSCURO  = "#006B2D"
COLOR_CLARO   = "#E8F5EE"
COLOR_GRIS    = "#3D3D3D"
COLOR_GRIS2   = "#888888"
COLOR_ALERTA  = "#C0392B"
COLOR_AZUL    = "#1A5276"
COLOR_AZUL2   = "#2471A3"
COLOR_NARANJA = "#D35400"
COLOR_FONDO   = "#FAFAFA"

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.color"       : "#E5E5E5",
    "grid.linewidth"   : 0.7,
})


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE INTERNO
# ─────────────────────────────────────────────────────────────────────────────
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
    n_estimators=N_EST, max_depth=MAX_D,
    min_samples_leaf=MIN_LEAF, random_state=SEED, n_jobs=-1
)
modelo.fit(X_train, y_train)
y_pred   = modelo.predict(X_test)
y_prob   = modelo.predict_proba(X_test)[:, 1]
cm       = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()
rep      = classification_report(
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

print("Pipeline listo. Generando presentación...")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def nueva_diapositiva(num_slide, total=11):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H))
    fig.patch.set_facecolor(COLOR_FONDO)
    # Franja verde izquierda — fig.add_patch es confiable en el backend PDF
    fig.add_artist(mpatches.Rectangle(
        (0, 0), 0.008, 1.0,
        transform=fig.transFigure,
        facecolor=COLOR_VERDE, edgecolor="none", zorder=10
    ))
    fig.text(0.985, 0.018, f"{num_slide} / {total}",
             ha="right", va="bottom", fontsize=9, color=COLOR_GRIS2, zorder=11)
    fig.text(0.015, 0.018, "PI-II ETITC  |  Correal & Guzmán  |  2026",
             ha="left", va="bottom", fontsize=9, color=COLOR_GRIS2, zorder=11)
    return fig


def barra_titulo(fig, titulo, subtitulo="", color=COLOR_VERDE):
    # fig.add_patch garantiza fondo sólido en PDF (axes.set_facecolor puede
    # quedar transparente en algunos backends vectoriales)
    fig.add_artist(mpatches.Rectangle(
        (0.008, 0.868), 0.992, 0.122,
        transform=fig.transFigure,
        facecolor=color, edgecolor="none", zorder=10
    ))
    fig.text(0.025, 0.921, titulo,
             fontsize=16, fontweight="bold", color="white",
             va="center", zorder=11)
    if subtitulo:
        fig.text(0.025, 0.882, subtitulo,
                 fontsize=9.5, color="#D5F5E3",
                 va="center", zorder=11)


def area_contenido(fig, left=0.04, bottom=0.10, width=0.92, height=0.74):
    """Devuelve la caja de contenido principal debajo del título."""
    return [left, bottom, width, height]


# =============================================================================
# DIAPOSITIVA 1 — PORTADA
# =============================================================================
fig = nueva_diapositiva(1)

# Dos columnas de color sólido usando parches de figura
fig.add_artist(mpatches.Rectangle(
    (0, 0), 0.60, 1.0,
    transform=fig.transFigure,
    facecolor=COLOR_VERDE, edgecolor="none", zorder=0
))
fig.add_artist(mpatches.Rectangle(
    (0.60, 0), 0.40, 1.0,
    transform=fig.transFigure,
    facecolor=COLOR_OSCURO, edgecolor="none", zorder=0
))
# Línea divisoria sutil
fig.add_artist(mpatches.Rectangle(
    (0.598, 0), 0.004, 1.0,
    transform=fig.transFigure,
    facecolor="#004D20", edgecolor="none", zorder=1
))

fig.text(0.04, 0.93, "Reporte Técnico de Validación — PI-II ETITC",
         fontsize=11, color="#A9DFBF", fontweight="normal", zorder=5)
fig.text(0.04, 0.72,
         "Pipeline de Analítica Predictiva\npara Detección Temprana\nde Riesgo de Deserción",
         fontsize=24, color="white", fontweight="bold", linespacing=1.4, zorder=5)
fig.text(0.04, 0.48, "Facultad de Sistemas  |  Jornada Nocturna",
         fontsize=11, color="#A9DFBF", zorder=5)

lineas_meta = [
    ("Autores",    "Correal Martínez & Guzmán Rodríguez"),
    ("Docente",    "Doris Constanza Alvarado Mariño"),
    ("",           ""),
    ("Algoritmo",  f"Random Forest  |  n={N_EST}  |  max_depth={MAX_D}"),
    ("Dataset",    "1,000 estudiantes sintéticos — 24 variables"),
    ("",           ""),
    ("Fecha",      "Mayo 25 de 2026"),
    ("Nivel TRL",  "TRL 2/3 — Validación con datos sintéticos"),
]
y_meta = 0.82
for etiq, val in lineas_meta:
    if not etiq:
        y_meta -= 0.028
        continue
    fig.text(0.635, y_meta, etiq + ":", fontsize=9.5,
             color="#A9DFBF", fontweight="bold", zorder=5)
    fig.text(0.635, y_meta - 0.045, val, fontsize=10,
             color="white", zorder=5)
    y_meta -= 0.100

with PdfPages(OUTPUT_PDF) as pdf:
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # DIAPOSITIVA 2 — PROBLEMA Y DATASET
    # ─────────────────────────────────────────────────────────────────────
    fig = nueva_diapositiva(2)
    barra_titulo(fig,
                 "¿Qué problema resuelve este pipeline?",
                 "Contexto institucional y estructura del dataset sintético")

    # Columna izquierda: 5 tarjetas de contexto con altura fija y espacio claro
    CARD_H   = 0.128   # altura de cada tarjeta en coordenadas de figura
    CARD_GAP = 0.016   # separación entre tarjetas
    CARD_X   = 0.04
    CARD_W   = 0.40
    CARD_Y0  = 0.83    # top de la primera tarjeta (debajo de barra de título)

    puntos = [
        ("Problema",      "El 12.05% de estudiantes nocturnos de la ETITC\nabandonan cada semestre."),
        ("Restricción",   "Sin acceso a microdatos individuales reales\n(Ley 1581/2012 — Habeas Data)."),
        ("Solución",      "Dataset sintético de 1,000 estudiantes calibrado\ncon 10 semestres de informes de Bienestar."),
        ("Arquitectura",  "3 CSV relacionales: Bienestar + Notas + Target\nunificados por id_estudiante (INNER JOIN)."),
        ("Meta",          "Validar consistencia técnica del pipeline\nantes de recibir datos reales anonimizados."),
    ]
    for i, (titulo_p, cuerpo_p) in enumerate(puntos):
        y_card = CARD_Y0 - i * (CARD_H + CARD_GAP)
        # Fondo de tarjeta — parche de figura para renderizado confiable
        fig.add_artist(mpatches.Rectangle(
            (CARD_X, y_card - CARD_H), CARD_W, CARD_H,
            transform=fig.transFigure,
            facecolor=COLOR_CLARO, edgecolor=COLOR_VERDE,
            linewidth=1.2, zorder=5
        ))
        fig.text(CARD_X + 0.013, y_card - 0.028,
                 titulo_p,
                 fontsize=10, fontweight="bold", color=COLOR_VERDE,
                 va="top", zorder=6)
        fig.text(CARD_X + 0.013, y_card - 0.062,
                 cuerpo_p,
                 fontsize=9, color=COLOR_GRIS,
                 va="top", linespacing=1.4, zorder=6)

    # Columna derecha: donut + tabla de fuentes
    ax_donut = fig.add_axes([0.50, 0.38, 0.22, 0.46])
    vals = [743, 257]
    wedges, _, autotexts = ax_donut.pie(
        vals,
        labels=["Persiste\n743", "Deserto\n257"],
        colors=[COLOR_AZUL2, COLOR_ALERTA],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2.5},
        textprops={"fontsize": 10}
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color("white")
    ax_donut.set_title("Balance de clases\n(variable objetivo)", fontsize=11,
                       fontweight="bold", pad=8)

    ax_tbl = fig.add_axes([0.73, 0.12, 0.24, 0.72])
    ax_tbl.axis("off")
    filas = [
        ["bienestar_\ncaracterizacion", "7 col.", "Socio-\neconómica"],
        ["adviser_teams_\nnotas",        "19 col.", "Rendimiento\nacadémico"],
        ["registro_\nadmisiones_target", "2 col.",  "Variable\nobjetivo"],
        ["Unificada\n(post-merge)",       "26 col.", "Entrada\nal modelo"],
    ]
    tbl = ax_tbl.table(
        cellText=filas,
        colLabels=["Archivo CSV", "Tamaño", "Tipo"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor(COLOR_VERDE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 4:
            cell.set_facecolor(COLOR_CLARO)
            cell.set_text_props(fontweight="bold", color=COLOR_VERDE)
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # DIAPOSITIVA 3 — DISTRIBUCIONES DE BIENESTAR
    # ─────────────────────────────────────────────────────────────────────
    fig = nueva_diapositiva(3)
    barra_titulo(fig,
                 "Perfil socioeconómico simulado — Bienestar Universitario",
                 "Calibrado con los informes de los 10 semestres consecutivos (2021-I a 2025-II)")

    # Gráfico 1: barras horizontales variables binarias
    # x=0.16 da margen suficiente para las etiquetas largas del eje Y
    ax1 = fig.add_axes([0.16, 0.10, 0.30, 0.62])
    vars_b = ["trabaja", "debilidad_mate", "conflicto_tiempo",
              "barrera_tecnologica", "inseguridad_alimentaria"]
    etiq   = ["Trabaja\nactivamente", "Debilidad en\nmatemáticas",
              "Conflicto\ntrabajo-estudio", "Solo celular para\ningeniería",
              "Inseguridad\nalimentaria"]
    pcts   = [df_b[v].mean() * 100 for v in vars_b]
    colores_h = [COLOR_VERDE if p >= 40 else COLOR_AZUL2 for p in pcts]

    bars = ax1.barh(etiq[::-1], pcts[::-1], color=colores_h[::-1],
                    edgecolor="white", height=0.55)
    for bar, pct in zip(bars, pcts[::-1]):
        ax1.text(pct + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{pct:.1f}%", va="center", fontsize=11, fontweight="bold",
                 color=COLOR_GRIS)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("% de la población estudiantil", fontsize=11)
    ax1.set_title("Variables de caracterización\n(Bienestar — % con valor = 1)",
                  fontweight="bold", fontsize=13)
    ax1.axvline(50, color="#CCCCCC", linestyle="--", linewidth=1)
    ax1.set_yticks(range(len(etiq)))
    ax1.set_yticklabels(etiq[::-1], fontsize=10)
    ax1.grid(axis="x")
    ax1.set_axisbelow(True)

    # Gráfico 2: brecha_bach
    ax2 = fig.add_axes([0.52, 0.10, 0.44, 0.62])
    brecha = df_b["brecha_bach"].value_counts().sort_index()
    colores_br = [COLOR_VERDE if i <= 2 else COLOR_ALERTA
                  for i in brecha.index]
    bars2 = ax2.bar(brecha.index, brecha.values,
                    color=colores_br, edgecolor="white", width=0.65)
    for bar, val in zip(bars2, brecha.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 4,
                 str(val), ha="center", fontsize=11,
                 fontweight="bold", color=COLOR_GRIS)
    ax2.set_xticks(brecha.index)
    ax2.set_xticklabels([f"{i} año{'s' if i != 1 else ''}" for i in brecha.index],
                        fontsize=10)
    ax2.set_ylabel("Número de estudiantes", fontsize=11)
    ax2.set_title("Años desde el bachillerato (brecha_bach)\n"
                  "  Verde = 0-2 años  |  Rojo = más de 3 años (rezago crítico)",
                  fontweight="bold", fontsize=13)
    ax2.set_ylim(0, brecha.max() * 1.20)
    ax2.grid(axis="y")
    ax2.set_axisbelow(True)

    pct_rezago = (df_b["brecha_bach"] > 3).mean() * 100
    ax2.text(0.97, 0.94,
             f"Rezago > 3 años: {pct_rezago:.1f}%\nde la población",
             transform=ax2.transAxes, ha="right", fontsize=10,
             color=COLOR_ALERTA, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FDEDEC",
                       edgecolor=COLOR_ALERTA, linewidth=1.2))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # DIAPOSITIVA 4 — RENDIMIENTO ACADÉMICO POR CORTE
    # ─────────────────────────────────────────────────────────────────────
    fig = nueva_diapositiva(4)
    barra_titulo(fig,
                 "Rendimiento académico dinámico — Los tres cortes institucionales",
                 "Adviser V.11  ×  Microsoft Teams  |  Tendencia de degradación semestral")

    ax1 = fig.add_axes([0.04, 0.10, 0.44, 0.62])
    data_box = [df_n[f"definitiva_c{c}"].values for c in [1, 2, 3]]
    bp = ax1.boxplot(data_box, patch_artist=True, widths=0.45,
                     medianprops={"color": "white", "linewidth": 2.5},
                     flierprops={"marker": ".", "markersize": 4,
                                 "markerfacecolor": "#AAAAAA", "alpha": 0.5})
    colores_box = [COLOR_AZUL2, COLOR_VERDE, COLOR_NARANJA]
    for patch, color in zip(bp["boxes"], colores_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(["Corte 1", "Corte 2", "Corte 3"], fontsize=12)
    ax1.set_ylabel("Nota definitiva (escala 0 – 5)", fontsize=11)
    ax1.axhline(3.0, color=COLOR_ALERTA, linestyle="--",
                linewidth=1.5, label="Umbral de aprobación (3.0)")
    ax1.set_title("Distribución de notas definitivas por corte\n"
                  "(tendencia a la baja por fatiga laboral acumulada)",
                  fontweight="bold", fontsize=13)
    ax1.legend(fontsize=10)
    medias = [df_n[f"definitiva_c{c}"].mean() for c in [1, 2, 3]]
    for xi, m in enumerate(medias, 1):
        ax1.text(xi, m + 0.08, f"x̄ = {m:.2f}",
                 ha="center", fontsize=10, color=COLOR_GRIS, fontweight="bold")

    ax2 = fig.add_axes([0.54, 0.10, 0.42, 0.62])
    cortes_lbl = ["Corte 1", "Corte 2", "Corte 3"]
    asist = [df_n[f"asistio_c{c}"].mean() * 100 for c in [1, 2, 3]]
    ausen = [100 - a for a in asist]
    x_pos = np.arange(3)
    w = 0.35
    bars_a = ax2.bar(x_pos - w/2, asist, w, label="Asistió",
                     color=COLOR_VERDE, edgecolor="white", alpha=0.88)
    bars_b = ax2.bar(x_pos + w/2, ausen, w, label="No asistió",
                     color=COLOR_ALERTA, edgecolor="white", alpha=0.80)
    for bar, val in zip(list(bars_a) + list(bars_b),
                        asist + ausen):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                 f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cortes_lbl, fontsize=12)
    ax2.set_ylabel("% de estudiantes", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.set_title("Asistencia vs. ausentismo por corte\n"
                  "(presión laboral → ausentismo crece en C3)",
                  fontweight="bold", fontsize=13)
    ax2.legend(fontsize=10)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # DIAPOSITIVA 5 — REGLAS DE NEGOCIO VALIDADAS
    # ─────────────────────────────────────────────────────────────────────
    fig = nueva_diapositiva(5)
    barra_titulo(fig,
                 "Validación de las Reglas de Negocio Programadas",
                 "Regla 2: penalización por ausentismo  |  Regla 3: choque cognitivo en ciencias básicas")

    ax1 = fig.add_axes([0.04, 0.10, 0.42, 0.62])
    presion_ = (df["trabaja"] == 1) | (df["conflicto_tiempo"] == 1)
    nt_np = df.loc[~presion_, "notas_teams_c1"]
    nt_p  = df.loc[presion_,  "notas_teams_c1"]
    ax1.hist(nt_np, bins=22, alpha=0.75, color=COLOR_VERDE,
             label=f"Sin presión de tiempo  (n={len(nt_np)})",
             density=True, edgecolor="white")
    ax1.hist(nt_p,  bins=22, alpha=0.75, color=COLOR_NARANJA,
             label=f"Con presión de tiempo  (n={len(nt_p)})",
             density=True, edgecolor="white")
    ax1.axvline(3.0, color="black", linestyle="--",
                linewidth=1.5, label="Umbral 3.0")
    ax1.set_xlabel("Nota de entregas en Teams — Corte 1", fontsize=11)
    ax1.set_ylabel("Densidad", fontsize=11)
    ax1.set_title("Regla 2: Penalización de entregas\n"
                  "si trabaja=1  ∨  conflicto_tiempo=1",
                  fontweight="bold", fontsize=13)
    ax1.legend(fontsize=10)
    medias_r2 = [nt_np.mean(), nt_p.mean()]
    for m, color, y_pos in zip(medias_r2, [COLOR_VERDE, COLOR_NARANJA], [0.87, 0.78]):
        ax1.text(0.97, y_pos, f"x̄ = {m:.2f}",
                 transform=ax1.transAxes, ha="right",
                 fontsize=10, color=color, fontweight="bold")

    ax2 = fig.add_axes([0.54, 0.10, 0.42, 0.62])
    mascara_rc = (df["brecha_bach"] > 3) & (df["debilidad_mate"] == 1)
    p_rc  = df.loc[mascara_rc,  "parcial_c1"]
    p_nrc = df.loc[~mascara_rc, "parcial_c1"]
    ax2.hist(p_nrc, bins=22, alpha=0.75, color=COLOR_AZUL2,
             label=f"Sin riesgo cognitivo  (n={len(p_nrc)})",
             density=True, edgecolor="white")
    ax2.hist(p_rc,  bins=16, alpha=0.85, color=COLOR_ALERTA,
             label=f"Riesgo cognitivo alto  (n={len(p_rc)})",
             density=True, edgecolor="white")
    ax2.axvline(3.0, color="black", linestyle="--",
                linewidth=1.5, label="Umbral 3.0")
    ax2.set_xlabel("Nota de examen presencial — Corte 1", fontsize=11)
    ax2.set_ylabel("Densidad", fontsize=11)
    ax2.set_title("Regla 3: Choque cognitivo\n"
                  "si brecha_bach > 3  ∧  debilidad_mate = 1",
                  fontweight="bold", fontsize=13)
    ax2.legend(fontsize=10)
    medias_r3 = [p_nrc.mean(), p_rc.mean()]
    for m, color, y_pos in zip(medias_r3, [COLOR_AZUL2, COLOR_ALERTA], [0.87, 0.78]):
        ax2.text(0.97, y_pos, f"x̄ = {m:.2f}",
                 transform=ax2.transAxes, ha="right",
                 fontsize=10, color=color, fontweight="bold")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # DIAPOSITIVA 6 — MATRIZ DE CONFUSIÓN
    # ─────────────────────────────────────────────────────────────────────
    fig = nueva_diapositiva(6)
    barra_titulo(fig,
                 "Evaluación del Modelo — Matriz de Confusión",
                 f"Conjunto de prueba: 200 estudiantes  |  Accuracy = {rep['accuracy']:.1%}")

    # Heatmap centrado grande
    ax_cm = fig.add_axes([0.08, 0.11, 0.36, 0.72])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Persiste (0)", "Deserto (1)"])
    disp.plot(ax=ax_cm, colorbar=False, cmap="Greens",
              text_kw={"fontsize": 22, "fontweight": "bold"})
    ax_cm.set_title("Matriz de Confusión", fontsize=14, fontweight="bold", pad=12)
    ax_cm.set_xlabel("Predicción del modelo", fontsize=12)
    ax_cm.set_ylabel("Realidad observada", fontsize=12)
    ax_cm.tick_params(labelsize=11)
    # Marcar FN en rojo
    ax_cm.texts[2].set_color(COLOR_ALERTA)

    # Tarjetas de interpretación
    tarjetas = [
        (TN, "Verdaderos\nNegativos",
         "Persiste → predijo Persiste\nSin alerta. Correcto.", COLOR_AZUL2),
        (FP, "Falsos\nPositivos",
         "Persiste → predijo Deserto\nFalsa alarma.\nBienestar interviene sin necesidad.", COLOR_NARANJA),
        (FN, "Falsos\nNegativos  ⚠",
         "Deserto → predijo Persiste\nERROR CRÍTICO: alumno en\nriesgo no detectado.", COLOR_ALERTA),
        (TP, "Verdaderos\nPositivos",
         "Deserto → predijo Deserto\nAlerta correcta.\nIntervención posible.", COLOR_VERDE),
    ]
    posiciones = [
        (0.50, 0.535),
        (0.745, 0.535),
        (0.50, 0.11),
        (0.745, 0.11),
    ]
    for (x0, y0), (valor, nombre, desc, color) in zip(posiciones, tarjetas):
        ax_t = fig.add_axes([x0, y0, 0.225, 0.36])
        ax_t.set_facecolor(color + "18")
        for sp in ax_t.spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(2)
        ax_t.axis("off")
        ax_t.text(0.5, 0.80, str(valor),
                  ha="center", va="center", fontsize=32,
                  fontweight="bold", color=color,
                  transform=ax_t.transAxes)
        ax_t.text(0.5, 0.56, nombre,
                  ha="center", va="center", fontsize=11,
                  fontweight="bold", color=color,
                  transform=ax_t.transAxes)
        ax_t.text(0.5, 0.22, desc,
                  ha="center", va="center", fontsize=8.5,
                  color=COLOR_GRIS, transform=ax_t.transAxes,
                  linespacing=1.4)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # DIAPOSITIVA 7 — CURVAS ROC Y PRECISION-RECALL
    # ─────────────────────────────────────────────────────────────────────
    fig = nueva_diapositiva(7)
    barra_titulo(fig,
                 "Curvas de Rendimiento del Clasificador",
                 f"ROC-AUC = {roc_auc:.3f}  |  Un modelo aleatorio obtendría AUC = 0.500")

    ax1 = fig.add_axes([0.06, 0.10, 0.40, 0.62])
    ax1.plot(fpr, tpr, color=COLOR_VERDE, lw=2.5,
             label=f"Random Forest  (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="#CCCCCC",
             linestyle="--", lw=1.5, label="Clasificador aleatorio")
    ax1.fill_between(fpr, tpr, alpha=0.12, color=COLOR_VERDE)
    ax1.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11)
    ax1.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=11)
    ax1.set_title(f"Curva ROC  —  AUC = {roc_auc:.3f}",
                  fontweight="bold", fontsize=13)
    ax1.legend(fontsize=10, loc="lower right")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])
    ax1.text(0.55, 0.30,
             f"AUC = {roc_auc:.3f}\n\nUn AUC > 0.90 indica que el\nmodelo separa muy bien\nambas clases.",
             fontsize=10, color=COLOR_GRIS,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=COLOR_CLARO,
                       edgecolor=COLOR_VERDE, linewidth=1.2))

    ax2 = fig.add_axes([0.54, 0.10, 0.40, 0.62])
    ax2.plot(rec_pr, prec_pr, color=COLOR_AZUL2, lw=2.5,
             label="Curva Precision-Recall")
    ax2.fill_between(rec_pr, prec_pr, alpha=0.10, color=COLOR_AZUL2)
    baseline = 257 / 1000
    ax2.axhline(baseline, color="#CCCCCC", linestyle="--",
                lw=1.5, label=f"Baseline (prevalencia = {baseline:.2f})")
    ax2.set_xlabel("Recall (% desertores detectados)", fontsize=11)
    ax2.set_ylabel("Precisión (% alertas correctas)", fontsize=11)
    ax2.set_title("Curva Precision-Recall\n(clave para clases desbalanceadas)",
                  fontweight="bold", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])
    ax2.text(0.05, 0.15,
             "Cuanto más alto y a la derecha,\nmejor equilibrio entre detectar\ntodos los desertores y no\ngenerar falsas alarmas.",
             fontsize=10, color=COLOR_GRIS,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#EBF5FB",
                       edgecolor=COLOR_AZUL2, linewidth=1.2))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # DIAPOSITIVA 8 — MÉTRICAS CLAVE
    # ─────────────────────────────────────────────────────────────────────
    fig = nueva_diapositiva(8)
    barra_titulo(fig,
                 "Métricas de Rendimiento — Classification Report",
                 "Evaluado sobre el 20% de datos reservados (200 muestras estratificadas)")

    # KPI cards grandes
    kpis = [
        ("Accuracy", rep["accuracy"],       COLOR_AZUL2,
         "Predicciones correctas\nsobre el total"),
        ("Precision\n(Deserto)",  rep["Deserto"]["precision"], COLOR_VERDE,
         "De 100 alertas emitidas,\ncuántas son reales"),
        ("Recall\n(Deserto)",     rep["Deserto"]["recall"],    COLOR_ALERTA,
         "De 100 desertores reales,\ncuántos detecta el modelo"),
        ("F1-Score\n(Deserto)",   rep["Deserto"]["f1-score"],  COLOR_NARANJA,
         "Media armónica Precisión\ny Recall (métrica balanceada)"),
        ("AUC-ROC",               roc_auc,                    "#7D3C98",
         "Capacidad discriminante\nglobal del clasificador"),
    ]
    posiciones_kpi = [0.02, 0.21, 0.40, 0.59, 0.78]
    for x0, (nombre, valor, color, desc) in zip(posiciones_kpi, kpis):
        ax_k = fig.add_axes([x0, 0.32, 0.18, 0.52])
        ax_k.set_facecolor("white")
        for sp in ax_k.spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(2.5)
        ax_k.axis("off")
        # Barra de progreso de fondo
        ax_k.add_patch(mpatches.Rectangle(
            (0, 0), 1, valor,
            facecolor=color, alpha=0.12,
            transform=ax_k.transAxes
        ))
        ax_k.text(0.5, 0.68, f"{valor:.1%}",
                  ha="center", va="center", fontsize=28,
                  fontweight="bold", color=color,
                  transform=ax_k.transAxes)
        ax_k.text(0.5, 0.46, nombre,
                  ha="center", va="center", fontsize=11,
                  fontweight="bold", color=color,
                  transform=ax_k.transAxes, linespacing=1.3)
        ax_k.text(0.5, 0.16, desc,
                  ha="center", va="center", fontsize=8.5,
                  color=COLOR_GRIS2, transform=ax_k.transAxes,
                  linespacing=1.35)

    # Tabla debajo
    ax_tbl = fig.add_axes([0.04, 0.05, 0.92, 0.24])
    ax_tbl.axis("off")
    metricas_rows = [
        ["Persiste (0)",
         f"{rep['Persiste']['precision']:.3f}",
         f"{rep['Persiste']['recall']:.3f}",
         f"{rep['Persiste']['f1-score']:.3f}",
         f"{int(rep['Persiste']['support'])}"],
        ["Deserto (1)",
         f"{rep['Deserto']['precision']:.3f}",
         f"{rep['Deserto']['recall']:.3f}",
         f"{rep['Deserto']['f1-score']:.3f}",
         f"{int(rep['Deserto']['support'])}"],
    ]
    tbl = ax_tbl.table(
        cellText=metricas_rows,
        colLabels=["Clase", "Precision", "Recall", "F1-Score", "Support"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor(COLOR_VERDE)
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 1:
            cell.set_facecolor("#EBF5FB")
        elif r == 2:
            cell.set_facecolor("#FDEDEC")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # DIAPOSITIVA 9 — FEATURE IMPORTANCE
    # ─────────────────────────────────────────────────────────────────────
    fig = nueva_diapositiva(9)
    barra_titulo(fig,
                 "Top 12 Sensores Predictivos — Importancia de Variables",
                 "Gini Impurity promediada sobre 100 árboles  |  Azul = académica  |  Naranja = socioeconómica")

    ax = fig.add_axes([0.04, 0.10, 0.60, 0.76])
    top12 = importancias.head(12)
    colores_fi = []
    for nombre in top12.index:
        if any(k in nombre for k in ["definitiva", "notas_teams", "parcial",
                                      "asistio", "autoevaluacion", "coevaluacion"]):
            colores_fi.append(COLOR_AZUL2)
        else:
            colores_fi.append(COLOR_NARANJA)

    bars_fi = ax.barh(
        range(len(top12)), top12.values,
        color=colores_fi, edgecolor="white", height=0.65
    )
    ax.set_yticks(range(len(top12)))
    ax.set_yticklabels(list(top12.index), fontsize=11)
    ax.set_xlabel("Importancia (contribución al poder predictivo)", fontsize=11)
    ax.set_title("Importancia de variables — Random Forest",
                 fontweight="bold", fontsize=13)
    ax.set_xlim(0, top12.max() * 1.30)
    for bar, val in zip(bars_fi, top12.values):
        ax.text(val + top12.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", fontsize=10, color=COLOR_GRIS, fontweight="bold")

    patch_ac = mpatches.Patch(color=COLOR_AZUL2,   label="Variable académica (Adviser/Teams)")
    patch_se = mpatches.Patch(color=COLOR_NARANJA, label="Variable socioeconómica (Bienestar)")
    ax.legend(handles=[patch_ac, patch_se], fontsize=10, loc="lower right")
    ax.invert_yaxis()

    # Insight lateral
    ax_ins = fig.add_axes([0.67, 0.10, 0.30, 0.76])
    ax_ins.set_facecolor(COLOR_CLARO)
    for sp in ax_ins.spines.values():
        sp.set_edgecolor(COLOR_VERDE)
        sp.set_linewidth(1.5)
    ax_ins.axis("off")

    top3 = importancias.head(3)
    peso_top5 = importancias.head(5).sum()

    insights = [
        ("HALLAZGO CLAVE", COLOR_VERDE,
         f"Las 5 variables más\nimportantes explican\nel {peso_top5:.1%} del poder\npredictivo total."),
        (f"#{1}  {top3.index[0]}", COLOR_AZUL2,
         f"Peso: {top3.iloc[0]:.4f}\n\nLa nota final del\núltimo corte es el\nindicador más crítico."),
        (f"#{2}  {top3.index[1]}", COLOR_AZUL2,
         f"Peso: {top3.iloc[1]:.4f}\n\nEl corte 2 anticipa\nel resultado final\ncon alta correlación."),
        (f"#{3}  {top3.index[2]}", COLOR_VERDE,
         f"Peso: {top3.iloc[2]:.4f}\n\nValor = 0.0 activa\nla Regla 4 de\ndeserción tardía."),
    ]
    y_c = 0.97
    for titulo_i, color_i, texto_i in insights:
        n_lineas = texto_i.count("\n") + 1
        ax_ins.text(0.5, y_c, titulo_i,
                    ha="center", va="top", fontsize=10,
                    fontweight="bold", color=color_i,
                    transform=ax_ins.transAxes)
        y_c -= 0.055
        ax_ins.text(0.5, y_c, texto_i,
                    ha="center", va="top", fontsize=9,
                    color=COLOR_GRIS, transform=ax_ins.transAxes,
                    linespacing=1.4)
        y_c -= 0.045 * n_lineas + 0.045
        ax_ins.plot([0.05, 0.95], [y_c + 0.01, y_c + 0.01],
                    color="#CCCCCC", linewidth=0.8,
                    transform=ax_ins.transAxes)
        y_c -= 0.015

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # DIAPOSITIVA 10 — INTERPRETACIÓN: ¿QUÉ SABE EL MODELO?
    # ─────────────────────────────────────────────────────────────────────
    fig = nueva_diapositiva(10)
    barra_titulo(fig,
                 "¿Qué aprendió el modelo? — Interpretación para Bienestar Universitario",
                 "Traducción de las métricas a acciones institucionales concretas")

    ax_body = fig.add_axes([0.03, 0.09, 0.94, 0.76])
    ax_body.axis("off")

    hallazgos = [
        (COLOR_AZUL2, "DOMINIO ACADÉMICO",
         "Las 5 variables top son cortes académicos, no factores sociales.",
         "La 'huella digital' (definitivas + autoevaluación) supera al perfil inicial de Bienestar\n"
         "como predictor. Esto valida la arquitectura Multi-CSV del proyecto."),
        (COLOR_ALERTA, "EL ERROR CRÍTICO: 11 Falsos Negativos",
         f"El modelo falló en detectar {FN} de los {TP + FN} desertores reales del conjunto de prueba.",
         "Son alumnos que el sistema dejaría pasar sin intervención. Reducirlos requiere\n"
         "bajar el umbral de decisión (< 0.5) o usar class_weight='balanced'."),
        (COLOR_VERDE, "RECALL del 78.4% — ¿Es suficiente?",
         "El modelo detecta 8 de cada 10 estudiantes en riesgo real.",
         "En producción, si hay 100 desertores potenciales en un semestre, el modelo alertaría\n"
         "sobre ~78. Los 22 restantes requerirían un segundo modelo de alerta tardía."),
        (COLOR_NARANJA, "PRECISION del 97.6% — Bajo ruido de falsas alarmas",
         "Solo 1 estudiante recibió alerta incorrecta en las 200 muestras de prueba.",
         "Bienestar podría intervenir sobre las alertas del modelo sin saturar su capacidad,\n"
         "ya que casi todas las alertas corresponden a riesgo real confirmado."),
    ]

    xs = [0.0, 0.505, 0.0, 0.505]
    ys = [0.985, 0.985, 0.48, 0.48]

    for (x0, y0), (color, titulo_h, subtit, desc) in zip(
            zip(xs, ys), hallazgos):
        ax_body.add_patch(mpatches.FancyBboxPatch(
            (x0, y0 - 0.44), 0.47, 0.42,
            boxstyle="round,pad=0.02",
            facecolor="white", edgecolor=color, linewidth=2,
            transform=ax_body.transAxes
        ))
        ax_body.add_patch(mpatches.FancyBboxPatch(
            (x0, y0 - 0.07), 0.47, 0.065,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="none",
            transform=ax_body.transAxes
        ))
        ax_body.text(x0 + 0.015, y0 - 0.037, titulo_h,
                     fontsize=10, fontweight="bold", color="white",
                     transform=ax_body.transAxes, va="center")
        ax_body.text(x0 + 0.015, y0 - 0.13, subtit,
                     fontsize=9.5, fontweight="bold", color=color,
                     transform=ax_body.transAxes)
        ax_body.text(x0 + 0.015, y0 - 0.24, desc,
                     fontsize=9, color=COLOR_GRIS,
                     transform=ax_body.transAxes, linespacing=1.45)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # DIAPOSITIVA 11 — CONCLUSIONES Y PRÓXIMOS PASOS
    # ─────────────────────────────────────────────────────────────────────
    fig = nueva_diapositiva(11)
    barra_titulo(fig,
                 "Conclusiones y Próximos Pasos",
                 "TRL 2/3 validado — Arquitectura consistente — Lista para datos reales")

    ax_body = fig.add_axes([0.03, 0.09, 0.94, 0.76])
    ax_body.axis("off")

    col_izq = [
        (COLOR_VERDE, "Arquitectura Multi-CSV validada",
         f"El INNER JOIN de 3 fuentes CSV produce una matriz de\n"
         f"1,000 × 24 sin nulos, demostrado para n=1,000 sintéticos."),
        (COLOR_VERDE, "Patrones estadísticos separables",
         f"Accuracy={rep['accuracy']:.1%} | AUC={roc_auc:.3f} confirman que las\n"
         f"reglas de negocio generan señales que el algoritmo puede aprender."),
        (COLOR_VERDE, "Hipótesis central validada",
         f"Las variables académicas dinámicas (cortes) tienen\n"
         f"mayor peso predictivo que el perfil socioeconómico estático."),
        (COLOR_NARANJA, "Limitación metodológica (TRL 2/3)",
         f"Las métricas altas se deben a que el modelo aprende\n"
         f"las mismas reglas que generaron los datos sintéticos.\n"
         f"Requiere validación con microdatos reales anonimizados."),
    ]

    col_der = [
        (COLOR_AZUL2, "Paso 1: Ajuste de umbral",
         "Bajar el threshold de 0.5 a ~0.35 para priorizar\nRecall sobre Precisión y reducir los 11 FN."),
        (COLOR_AZUL2, "Paso 2: Comparar con XGBoost",
         "Entrenar XGBoost con los mismos datos y evaluar\nsi mejora el F1-Score para la clase minoritaria."),
        (COLOR_AZUL2, "Paso 3: Validación cruzada k-fold",
         "Reemplazar el split 80/20 por k=5 folds para\nestimar varianza del modelo con más robustez."),
        (COLOR_AZUL2, "Paso 4: Datos reales",
         "Acuerdo de confidencialidad con ETITC (Ley 1581/2012)\npara acceder a sábanas anonimizadas de Adviser V.11."),
    ]

    for col_data, x0 in [(col_izq, 0.0), (col_der, 0.505)]:
        y0 = 0.96
        for color, titulo_c, desc_c in col_data:
            n = desc_c.count("\n") + 1
            alto = 0.18 + (n - 2) * 0.04 if n > 2 else 0.18
            ax_body.add_patch(mpatches.FancyBboxPatch(
                (x0, y0 - alto), 0.47, alto - 0.01,
                boxstyle="round,pad=0.02",
                facecolor=color + "15",
                edgecolor=color, linewidth=1.5,
                transform=ax_body.transAxes
            ))
            ax_body.text(x0 + 0.015, y0 - 0.045, titulo_c,
                         fontsize=10, fontweight="bold", color=color,
                         transform=ax_body.transAxes)
            ax_body.text(x0 + 0.015, y0 - 0.11, desc_c,
                         fontsize=9, color=COLOR_GRIS,
                         transform=ax_body.transAxes, linespacing=1.4)
            y0 -= alto + 0.025

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"Presentacion exportada exitosamente: {OUTPUT_PDF}")
print(f"  -> 11 diapositivas  |  Formato 16:9  |  Lista para revision academica")
