"""
Script de Generación de Datos Sintéticos Relacionales
Proyecto PI-II: Sistema de Analítica Predictiva para Detección Temprana
de Riesgo de Deserción — Jornada Nocturna ETITC Sede Centro

Autores:  Jhon Alejandro Correal Martínez
          Rafael Andrés Guzmán Rodríguez
Docente:  Doris Constanza Alvarado Mariño
Versión:  2.1  |  2026-05-26

Arquitectura Multi-CSV (tres archivos independientes vinculados por id_estudiante):
  1. bienestar_caracterizacion.csv   — perfil socioeconómico inicial
  2. adviser_teams_notas.csv         — historial académico por 3 cortes
  3. registro_admisiones_target.csv  — variable objetivo (deserto: 0/1)

Calibración: porcentajes extraídos de los informes de Bienestar Universitario
ETITC semestres 2021-I a 2025-II (10 semestres consecutivos).
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
N    = 1_000
RNG  = np.random.default_rng(SEED)

OUTPUT_BIENESTAR = "bienestar_caracterizacion.csv"
OUTPUT_NOTAS     = "adviser_teams_notas.csv"
OUTPUT_TARGET    = "registro_admisiones_target.csv"

# Pesos del promedio ponderado por corte (estructura de planilla Adviser V.11)
W_PARCIAL  = 0.40
W_TEAMS    = 0.30
W_AUTOEVAL = 0.15
W_COEVAL   = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def truncated_normal(mean: float, std: float, size: int,
                     lo: float = 0.0, hi: float = 5.0) -> np.ndarray:
    """
    Genera 'size' muestras de una distribución Normal(mean, std) truncada al
    intervalo [lo, hi] mediante rechazo iterativo.  Garantiza que ningún valor
    esté fuera del rango de calificaciones válidas (0.0 – 5.0).
    """
    collected = []
    while len(collected) < size:
        batch = RNG.normal(mean, std, size * 5)
        collected.extend(batch[(batch >= lo) & (batch <= hi)].tolist())
    return np.round(np.array(collected[:size]), 2)


def clamp(arr: np.ndarray, lo: float = 0.0, hi: float = 5.0) -> np.ndarray:
    """Limita todos los valores de un array al intervalo [lo, hi]."""
    return np.clip(arr, lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 1 — IDENTIFICADORES ÚNICOS
# ─────────────────────────────────────────────────────────────────────────────
ids = [f"EST-{i:04d}" for i in range(1, N + 1)]


# =============================================================================
# ARCHIVO 1: bienestar_caracterizacion.csv
# =============================================================================

# ── Regla 1: Porcentajes base calibrados con informes 2021-I → 2025-II ───────

trabaja = RNG.choice([1, 0], size=N, p=[0.68, 0.32])
# 68 % de la jornada nocturna ETITC está vinculada laboralmente (fuente BU)

debilidad_mate = RNG.choice([1, 0], size=N, p=[0.45, 0.55])
# 45 % reporta autopercepción de vacíos en matemáticas (fuente BU)

# brecha_bach: distribución asimétrica positiva con cola derecha
#   0 años 25 % | 1 año 25 % | 2 años 20 % | 3 años 15 % | 4 años 10 % | 5 años 5 %
brecha_bach = RNG.choice(
    [0, 1, 2, 3, 4, 5],
    size=N,
    p=[0.25, 0.25, 0.20, 0.15, 0.10, 0.05]
)

# conflicto_tiempo: correlacionado con trabaja (no independiente)
#   → Si trabaja == 1 : 70 % de probabilidad de conflicto trabajo-estudio
#   → Si trabaja == 0 : 20 % de probabilidad de conflicto (carga familiar, etc.)
conflicto_tiempo = np.where(
    trabaja == 1,
    RNG.binomial(1, 0.70, N),
    RNG.binomial(1, 0.20, N)
).astype(int)

barrera_tecnologica     = RNG.choice([1, 0], size=N, p=[0.35, 0.65])
# ~35 % depende exclusivamente de celular para actividades de ingeniería

inseguridad_alimentaria = RNG.choice([1, 0], size=N, p=[0.28, 0.72])
# ~28 % ha omitido comidas por restricciones económicas (fuente BU 2023-2025)

df_bienestar = pd.DataFrame({
    "id_estudiante"         : ids,
    "trabaja"               : trabaja,
    "debilidad_mate"        : debilidad_mate,
    "brecha_bach"           : brecha_bach,
    "conflicto_tiempo"      : conflicto_tiempo,
    "barrera_tecnologica"   : barrera_tecnologica,
    "inseguridad_alimentaria": inseguridad_alimentaria,
})


# =============================================================================
# ARCHIVO 2: adviser_teams_notas.csv
# =============================================================================

# ── Máscaras de perfil de riesgo (se derivan de bienestar) ───────────────────

# Regla 2: presión de tiempo = trabaja O conflicto de tiempo
presion_tiempo = (trabaja == 1) | (conflicto_tiempo == 1)

# Regla 3: riesgo cognitivo = rezago temporal + debilidad en matemáticas
riesgo_cognitivo = (brecha_bach > 3) & (debilidad_mate == 1)

# Número de sesiones presenciales por corte
# (punto medio entre materias regulares ~5 y materias intensivas ~10)
TOTAL_CLASES = 8

# Medias de inasistencias por perfil de presión laboral (R1)
# C3 suma +1 sesión por fatiga laboral acumulada
MU_INASIST_PRESION = {1: 3.0, 2: 3.0, 3: 4.0}
MU_INASIST_NORMAL  = {1: 0.8, 2: 0.8, 3: 1.0}

# Probabilidad de abandono tácito dado que el estudiante perdió >= 6 sesiones (R4)
# Aumenta progresivamente a medida que avanza el semestre
P_ABANDONO_DADO_FALTA_CRITICA = {1: 0.50, 2: 0.70, 3: 0.90}

df_notas = pd.DataFrame({"id_estudiante": ids})

for c in [1, 2, 3]:

    # ── R1: Presión laboral → inasistencias físicas y virtuales ──────────────
    # Con presión:  N(mu_presion, 1.2) truncada en [1, TOTAL_CLASES]
    # Sin presión:  N(mu_normal,  0.6) truncada en [0, 2]
    raw_presion  = truncated_normal(MU_INASIST_PRESION[c], 1.2, N,
                                    lo=1.0, hi=float(TOTAL_CLASES))
    raw_normal   = truncated_normal(MU_INASIST_NORMAL[c],  0.6, N,
                                    lo=0.0, hi=2.0)
    inasist_raw  = np.where(presion_tiempo, raw_presion, raw_normal)
    inasistencias = np.clip(np.round(inasist_raw).astype(int), 0, TOTAL_CLASES)

    # ── R1 cont. + R3: Notas Teams = base por perfil − penalización inasist. − barrera ──
    # Cada sesión faltada descuenta 0.30 en la entrega de Teams del corte.
    # La barrera tecnológica (solo celular) añade −0.50 adicional (R3).
    nt_base = np.where(
        presion_tiempo,
        truncated_normal(3.8, 0.6, N),
        truncated_normal(4.4, 0.5, N)
    )
    pen_inasist  = inasistencias * 0.30
    pen_barrera  = barrera_tecnologica * 0.50
    notas_teams  = clamp(nt_base - pen_inasist - pen_barrera)

    # ── R2: Riesgo cognitivo → parciales bajos en ciencias básicas ───────────
    # brecha_bach > 3 AND debilidad_mate → N(2.2, 0.5)
    parcial_riesgo_alto   = truncated_normal(2.2, 0.5, N)
    parcial_presion_media = truncated_normal(3.5, 0.6, N)
    parcial_perfil_normal = truncated_normal(4.2, 0.5, N)

    parcial = np.where(
        riesgo_cognitivo,
        parcial_riesgo_alto,
        np.where(presion_tiempo, parcial_presion_media, parcial_perfil_normal)
    )
    parcial = clamp(parcial)

    # ── R4: Abandono tácito → autoevaluación = coevaluación = 0.0 ────────────
    # Aplica cuando el estudiante perdió >= 6 de 8 sesiones (≥ 75 % de falta),
    # con probabilidad creciente por corte para simular la deserción tardía.
    falta_critica  = inasistencias >= 6
    abandono_total = falta_critica & (RNG.random(N) < P_ABANDONO_DADO_FALTA_CRITICA[c])

    autoeval_base  = clamp(truncated_normal(3.9, 0.5, N))
    coeval_base    = clamp(truncated_normal(3.8, 0.5, N))
    autoevaluacion = np.where(abandono_total, 0.0, autoeval_base)
    coevaluacion   = np.where(abandono_total, 0.0, coeval_base)

    # ── Definitiva ponderada del corte + ruido de medición ───────────────────
    # El ±0.20 de ruido Gaussiano simula discrecionalidad docente, correcciones
    # tardías y redondeo institucional. Evita que la definitiva sea la síntesis
    # perfecta de sus componentes, introduciendo ambigüedad real para el modelo.
    definitiva = (
        W_PARCIAL  * parcial        +
        W_TEAMS    * notas_teams    +
        W_AUTOEVAL * autoevaluacion +
        W_COEVAL   * coevaluacion
    )
    definitiva += RNG.normal(0, 0.20, N)
    definitiva  = np.round(clamp(definitiva), 2)

    # ── Persistir columnas en el DataFrame ───────────────────────────────────
    df_notas[f"inasistencias_c{c}"]  = inasistencias
    df_notas[f"notas_teams_c{c}"]    = notas_teams
    df_notas[f"parcial_c{c}"]        = parcial
    df_notas[f"autoevaluacion_c{c}"] = autoevaluacion
    df_notas[f"coevaluacion_c{c}"]   = coevaluacion
    df_notas[f"definitiva_c{c}"]     = definitiva


# =============================================================================
# ARCHIVO 3: registro_admisiones_target.csv
# =============================================================================

# ── Condición de deserción (target) ──────────────────────────────────────────

promedio_definitivas = (
    df_notas["definitiva_c1"] +
    df_notas["definitiva_c2"] +
    df_notas["definitiva_c3"]
) / 3.0

# ── Asignación probabilística del target ─────────────────────────────────────
# Se usa una función logística centrada en el umbral institucional (3.0):
#   p_persistir(prom) = sigmoid(2 × (prom − 3.0) / 0.5)
# Esto crea ambigüedad real en la zona limítrofe (2.5–3.5) que el modelo
# no puede aprender perfectamente, produciendo FP y FN realistas.
# Sin este paso la separabilidad es perfecta → Precision = 100 % (artefacto).
condicion_reprobacion = promedio_definitivas < 3.0

abandono_c2 = (
    (df_notas["autoevaluacion_c2"] == 0.0) &
    (df_notas["coevaluacion_c2"]   == 0.0)
)
abandono_c3 = (
    (df_notas["autoevaluacion_c3"] == 0.0) &
    (df_notas["coevaluacion_c3"]   == 0.0)
)
abandono_efectivo = (
    (abandono_c2 | abandono_c3) &
    (RNG.random(N) < 0.80)
)

deserto_base = condicion_reprobacion | abandono_efectivo

# ── Zona limítrofe: casos que el modelo no puede determinar con certeza ───────
# Estos dos grupos generan FP y FN realistas sin destruir el rendimiento global.

# FP: 20 % de los que reprueban por notas (2.75–3.0) se recuperan
# (habilitación, retiro no registrado en Adviser, repetición)
recuperacion = (
    condicion_reprobacion &
    (promedio_definitivas >= 2.75) &
    ~abandono_efectivo &
    (RNG.random(N) < 0.20)
)

# FN: 8 % de los que aprueban con nota justa (3.0–3.3) desertan
# por causas externas (pérdida de empleo, crisis familiar, salud)
crisis_externa = (
    ~condicion_reprobacion &
    (promedio_definitivas < 3.3) &
    ~abandono_efectivo &
    (RNG.random(N) < 0.08)
)

deserto = np.where(
    recuperacion, 0,
    np.where(crisis_externa, 1, deserto_base.astype(int))
).astype(int)

df_target = pd.DataFrame({
    "id_estudiante": ids,
    "deserto"      : deserto,
})


# =============================================================================
# EXPORTAR ARCHIVOS CSV
# =============================================================================
df_bienestar.to_csv(OUTPUT_BIENESTAR, index=False, encoding="utf-8")
df_notas.to_csv(    OUTPUT_NOTAS,     index=False, encoding="utf-8")
df_target.to_csv(   OUTPUT_TARGET,    index=False, encoding="utf-8")


# =============================================================================
# INFORME DE VALIDACION AUTOMATICA
# =============================================================================
SEP = "=" * 62

print(f"\n{SEP}")
print("  INFORME DE VALIDACION — DATOS SINTETICOS ETITC PI-II")
print(f"{SEP}\n")

# ── Archivo 1 ────────────────────────────────────────────────────────────────
b = df_bienestar["brecha_bach"]
print(f"[1] {OUTPUT_BIENESTAR}")
print(f"    Registros generados    : {len(df_bienestar)}")
print(f"    Trabaja          (%)   : {df_bienestar['trabaja'].mean()*100:.1f}%  (esperado ~68%)")
print(f"    Debilidad mate   (%)   : {df_bienestar['debilidad_mate'].mean()*100:.1f}%  (esperado ~45%)")
print(f"    Conflicto tiempo (%)   : {df_bienestar['conflicto_tiempo'].mean()*100:.1f}%")
print(f"    Brecha bach 0-2  (%)   : {(b <= 2).mean()*100:.1f}%  (mayoria esperada)")
print(f"    Brecha bach > 3  (%)   : {(b  > 3).mean()*100:.1f}%  (cola larga)")
print(f"    Riesgo cognitivo (n)   : {int(riesgo_cognitivo.sum())}  (brecha>3 AND debilidad_mate)")

# ── Archivo 2 ────────────────────────────────────────────────────────────────
print(f"\n[2] {OUTPUT_NOTAS}")
print(f"    Registros generados    : {len(df_notas)}")
for c in [1, 2, 3]:
    inas = df_notas[f"inasistencias_c{c}"].mean()
    inas0 = (df_notas[f"inasistencias_c{c}"] == 0).mean() * 100
    nt   = df_notas[f"notas_teams_c{c}"].mean()
    par  = df_notas[f"parcial_c{c}"].mean()
    def_ = df_notas[f"definitiva_c{c}"].mean()
    print(f"    Corte {c}: inasistencias_media={inas:.1f}/{TOTAL_CLASES}  "
          f"sin_inasist={inas0:.1f}%  teams={nt:.2f}  "
          f"parcial={par:.2f}  definitiva={def_:.2f}")

# ── Archivo 3 ────────────────────────────────────────────────────────────────
print(f"\n[3] {OUTPUT_TARGET}")
print(f"    Registros generados    : {len(df_target)}")
print(f"    Tasa de desercion (%)  : {df_target['deserto'].mean()*100:.1f}%")
print(f"    Desertores (n)         : {int(df_target['deserto'].sum())}")
print(f"      -> Por reprobacion   : {int(condicion_reprobacion.sum())}")
print(f"      -> Por abandono C2   : {int(abandono_c2.sum())}")
print(f"      -> Por abandono C3   : {int(abandono_c3.sum())}")

print(f"\n{'─' * 62}")
print(f"  Tres archivos CSV generados exitosamente.")
print(f"  -> {OUTPUT_BIENESTAR}")
print(f"  -> {OUTPUT_NOTAS}")
print(f"  -> {OUTPUT_TARGET}")
print(f"{'─' * 62}\n")
