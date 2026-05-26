"""
Pipeline de Entrenamiento Predictivo — Detección Temprana de Riesgo de Deserción
Proyecto PI-II: Sistema de Analítica Predictiva — Jornada Nocturna ETITC Sede Centro

Autores:  Jhon Alejandro Correal Martínez
          Rafael Andrés Guzmán Rodríguez
Docente:  Doris Constanza Alvarado Mariño
Versión:  1.0  |  2026-05-25

Arquitectura de entrada (Multi-CSV relacional):
  bienestar_caracterizacion.csv   → perfil socioeconómico inicial
  adviser_teams_notas.csv         → historial académico por 3 cortes
  registro_admisiones_target.csv  → variable objetivo (deserto: 0/1)

Algoritmo: Random Forest Classifier (scikit-learn)
Nivel TRL: 2/3 — validación de consistencia con datos sintéticos calibrados
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
SEED           = 42
TEST_SIZE      = 0.20
N_ESTIMATORS   = 100
MAX_DEPTH      = 6       # Limita la profundidad para reducir el sobreajuste
MIN_SAMPLES_LEAF = 4     # Mínimo de muestras en una hoja para evitar ruido
TOP_N_FEATURES = 5       # Número de variables más importantes a reportar

FILE_BIENESTAR = "bienestar_caracterizacion.csv"
FILE_NOTAS     = "adviser_teams_notas.csv"
FILE_TARGET    = "registro_admisiones_target.csv"

SEP_DOBLE  = "=" * 65
SEP_SIMPLE = "-" * 65


# =============================================================================
# FASE 1: ETL Y UNIFICACIÓN DE FUENTES RELACIONALES
# =============================================================================
print(f"\n{SEP_DOBLE}")
print("  FASE 1: ETL — CARGA Y UNION DE FUENTES RELACIONALES")
print(f"{SEP_DOBLE}\n")

df_bienestar = pd.read_csv(FILE_BIENESTAR)
df_notas     = pd.read_csv(FILE_NOTAS)
df_target    = pd.read_csv(FILE_TARGET)

print(f"  Fuente 1 - {FILE_BIENESTAR:<38}: {df_bienestar.shape[0]} filas x {df_bienestar.shape[1]} columnas")
print(f"  Fuente 2 - {FILE_NOTAS:<38}: {df_notas.shape[0]} filas x {df_notas.shape[1]} columnas")
print(f"  Fuente 3 - {FILE_TARGET:<38}: {df_target.shape[0]} filas x {df_target.shape[1]} columnas")

# Merge secuencial usando 'id_estudiante' como llave relacional (equivale a un
# JOIN SQL). El uso de how='inner' garantiza que solo entren al entrenamiento
# los estudiantes que existan en las tres fuentes simultáneamente, preservando
# la integridad referencial del dataset.
df = (df_bienestar
      .merge(df_notas,   on="id_estudiante", how="inner")
      .merge(df_target,  on="id_estudiante", how="inner"))

print(f"\n  Matriz unificada (post-merge)              : {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"  Valores nulos en la matriz combinada       : {df.isnull().sum().sum()}")
print(f"  Distribucion de la variable objetivo:")
vc = df["deserto"].value_counts()
print(f"    Clase 0 (Persiste)  : {vc.get(0, 0)} registros  ({vc.get(0, 0)/len(df)*100:.1f}%)")
print(f"    Clase 1 (Deserto)   : {vc.get(1, 0)} registros  ({vc.get(1, 0)/len(df)*100:.1f}%)")


# =============================================================================
# FASE 2: PREPARACIÓN DE CARACTERÍSTICAS (FEATURE ENGINEERING)
# =============================================================================
print(f"\n{SEP_DOBLE}")
print("  FASE 2: PREPARACION DE CARACTERISTICAS")
print(f"{SEP_DOBLE}\n")

# Se elimina 'id_estudiante' del conjunto de entrenamiento porque es un
# identificador arbitrario sin contenido predictivo. Si el modelo aprendiera
# de él, memorizaría individuos del conjunto de entrenamiento en lugar de
# generalizar patrones: esto produce sobreajuste perfecto pero rendimiento
# nulo sobre datos reales de nuevos estudiantes.
y = df["deserto"]
X = df.drop(columns=["id_estudiante", "deserto"])

print(f"  Variable objetivo (y)          : 'deserto'  (0 = persiste, 1 = deserto)")
print(f"  Numero de caracteristicas (X)  : {X.shape[1]} variables explicativas")
print(f"  Listado de variables:\n")
for i, col in enumerate(X.columns, 1):
    print(f"    {i:>2}. {col}")


# =============================================================================
# FASE 3: DIVISIÓN ESTRATIFICADA TRAIN / TEST
# =============================================================================
print(f"\n{SEP_DOBLE}")
print("  FASE 3: DIVISION TRAIN / TEST  (80% / 20%)")
print(f"{SEP_DOBLE}\n")

# stratify=y garantiza que la proporción de desertores (clase 1) sea idéntica
# en ambos bloques. Sin estratificación, el split aleatorio podría concentrar
# la mayoría de los casos positivos en train y privar a test de ejemplos
# suficientes para una evaluación representativa.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y
)

print(f"  Conjunto de entrenamiento (80%): {X_train.shape[0]} filas")
print(f"    Clase 0 (Persiste) : {(y_train == 0).sum()}  ({(y_train == 0).mean()*100:.1f}%)")
print(f"    Clase 1 (Deserto)  : {(y_train == 1).sum()}  ({(y_train == 1).mean()*100:.1f}%)")
print(f"\n  Conjunto de prueba    (20%): {X_test.shape[0]} filas")
print(f"    Clase 0 (Persiste) : {(y_test == 0).sum()}  ({(y_test == 0).mean()*100:.1f}%)")
print(f"    Clase 1 (Deserto)  : {(y_test == 1).sum()}  ({(y_test == 1).mean()*100:.1f}%)")


# =============================================================================
# FASE 4: ENTRENAMIENTO DEL MODELO
# =============================================================================
print(f"\n{SEP_DOBLE}")
print("  FASE 4: ENTRENAMIENTO — RANDOM FOREST CLASSIFIER")
print(f"{SEP_DOBLE}\n")

# Random Forest construye un ensamble de árboles de decisión entrenados sobre
# subconjuntos aleatorios de los datos y las características. La diversidad
# entre árboles reduce la varianza y controla el sobreajuste, lo que lo hace
# especialmente robusto con tablas de características mixtas (booleanas +
# continuas) como las que genera la arquitectura Multi-CSV de este proyecto.
#
# Hiperparámetros elegidos para controlar el sobreajuste:
#   max_depth=6       → cada árbol no puede especializarse más de 6 niveles,
#                       forzando reglas generales y no memorización de datos.
#   min_samples_leaf=4 → una hoja necesita al menos 4 muestras para existir,
#                        eliminando divisiones ruidosas sobre casos aislados.
#   n_estimators=100  → suficientes árboles para estabilizar las predicciones
#                       sin costo computacional excesivo.
modelo = RandomForestClassifier(
    n_estimators     = N_ESTIMATORS,
    max_depth        = MAX_DEPTH,
    min_samples_leaf = MIN_SAMPLES_LEAF,
    random_state     = SEED,
    n_jobs           = -1       # usa todos los núcleos disponibles
)

print(f"  Algoritmo          : RandomForestClassifier (scikit-learn)")
print(f"  n_estimators       : {N_ESTIMATORS}")
print(f"  max_depth          : {MAX_DEPTH}")
print(f"  min_samples_leaf   : {MIN_SAMPLES_LEAF}")
print(f"  random_state       : {SEED}\n")
print("  Entrenando...", end=" ", flush=True)

modelo.fit(X_train, y_train)

print("completado.\n")

y_pred = modelo.predict(X_test)


# =============================================================================
# FASE 5A: MATRIZ DE CONFUSIÓN
# =============================================================================
print(f"\n{SEP_DOBLE}")
print("  FASE 5A: MATRIZ DE CONFUSION")
print(f"{SEP_DOBLE}\n")

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print("  Formato de la matriz:  [ Clase real 0 ]  [ Clase real 1 ]")
print()
print(f"  {'':30} Predicho: 0    Predicho: 1")
print(f"  {'':30} {'-'*13}  {'-'*13}")
print(f"  Real: 0 (Persiste)          {'':4} {TN:^13}  {FP:^13}")
print(f"  Real: 1 (Deserto)           {'':4} {FN:^13}  {TP:^13}")
print()
print(f"  {SEP_SIMPLE}")
print(f"  Verdaderos Negativos  (VN = {TN:>3})")
print(f"    Estudiantes que el modelo predijo correctamente como 'Persiste'.")
print(f"    Son los casos donde el sistema NO emite una alerta innecesaria.")
print()
print(f"  Falsos Positivos      (FP = {FP:>3})")
print(f"    Estudiantes que PERSISTIERON pero el modelo los clasifico como")
print(f"    'Desertor'. Son falsas alarmas: Bienestar intervendria sin")
print(f"    necesidad, generando carga de trabajo innecesaria.")
print()
print(f"  Falsos Negativos      (FN = {FN:>3})  <-- CRITICO para el proyecto")
print(f"    Estudiantes que DESERTARON pero el modelo NO los detecto.")
print(f"    Este es el error de mayor costo institucional: son los alumnos")
print(f"    en riesgo real que caen sin que Bienestar reciba ninguna alerta.")
print(f"    Reducir este numero es el objetivo central del sistema predictivo.")
print()
print(f"  Verdaderos Positivos  (VP = {TP:>3})")
print(f"    Estudiantes en riesgo que el modelo detecto correctamente.")
print(f"    Cada uno de estos representa una intervencion oportuna posible.")
print(f"  {SEP_SIMPLE}")


# =============================================================================
# FASE 5B: REPORTE DE CLASIFICACIÓN
# =============================================================================
print(f"\n{SEP_DOBLE}")
print("  FASE 5B: REPORTE DE CLASIFICACION")
print(f"{SEP_DOBLE}\n")

reporte = classification_report(
    y_test, y_pred,
    target_names=["0 - Persiste", "1 - Deserto"],
    digits=3
)
print(reporte)

# Calcular métricas individuales para el reporte narrativo
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

precision_1 = precision_score(y_test, y_pred, pos_label=1)
recall_1    = recall_score(y_test,    y_pred, pos_label=1)
f1_1        = f1_score(y_test,        y_pred, pos_label=1)
accuracy    = accuracy_score(y_test, y_pred)

print(f"  {SEP_SIMPLE}")
print(f"  LECTURA DE METRICAS (para la clase 1 = Deserto):")
print()
print(f"  Precision  = {precision_1:.3f}")
print(f"    De cada 100 alertas emitidas por el modelo, {precision_1*100:.1f} son reales.")
print(f"    Un valor bajo genera 'fatiga de alerta' en el equipo de Bienestar.")
print()
print(f"  Recall (Sensibilidad) = {recall_1:.3f}")
print(f"    El modelo detecta el {recall_1*100:.1f}% de los estudiantes en riesgo real.")
print(f"    Es la metrica mas importante para este sistema: maximizarla")
print(f"    significa minimizar los Falsos Negativos (estudiantes perdidos).")
print()
print(f"  F1-Score = {f1_1:.3f}")
print(f"    Media armonica de Precision y Recall. Equilibra ambas metas.")
print(f"    Es la metrica recomendada para datasets con clases desbalanceadas.")
print()
print(f"  Exactitud global (Accuracy) = {accuracy:.3f}")
print(f"    {accuracy*100:.1f}% de predicciones correctas en total (ambas clases).")
print(f"  {SEP_SIMPLE}")


# =============================================================================
# FASE 5C: IMPORTANCIA DE VARIABLES (FEATURE IMPORTANCE)
# =============================================================================
print(f"\n{SEP_DOBLE}")
print(f"  FASE 5C: IMPORTANCIA DE VARIABLES — TOP {TOP_N_FEATURES} SENSORES PREDICTIVOS")
print(f"{SEP_DOBLE}\n")

importancias = pd.Series(
    modelo.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

top_features = importancias.head(TOP_N_FEATURES)
total_peso   = top_features.sum()

print(f"  Las siguientes variables explican el {total_peso*100:.1f}% del peso")
print(f"  predictivo total del modelo:\n")
print(f"  {'Rango':<6} {'Variable':<30} {'Importancia':>12}  {'Barra visual'}")
print(f"  {'-'*6} {'-'*30} {'-'*12}  {'-'*25}")

for rango, (nombre, valor) in enumerate(top_features.items(), 1):
    barra = "#" * int(valor * 200)
    print(f"  {rango:<6} {nombre:<30} {valor:>11.4f}  {barra}")

print(f"\n  {SEP_SIMPLE}")
print(f"  INTERPRETACION PARA BIENESTAR UNIVERSITARIO:")
print()

# Narrativa dinámica de las top 3 variables
for rango, (nombre, valor) in enumerate(top_features.head(3).items(), 1):
    fuente = "Adviser/Teams (academica)" if any(k in nombre for k in ["notas", "parcial", "definitiva", "inasistencias", "autoeval", "coeval"]) else "Bienestar (socioecon.)"
    print(f"  #{rango} '{nombre}' (peso={valor:.4f})")
    print(f"     Fuente: {fuente}")

    if "definitiva" in nombre:
        c = nombre[-1]
        print(f"     La nota definitiva del Corte {c} es el indicador academico de")
        print(f"     mayor poder discriminante. Una caida aqui es la señal mas clara")
        print(f"     de riesgo antes del cierre del semestre.")
    elif "parcial" in nombre:
        c = nombre[-1]
        print(f"     El resultado del examen presencial del Corte {c} captura el")
        print(f"     impacto del choque cognitivo y la fatiga laboral acumulada.")
    elif "notas_teams" in nombre:
        c = nombre[-1]
        print(f"     La regularidad de entregas en Teams del Corte {c} actua como")
        print(f"     el 'sensor de huella digital academica': mide desconexion")
        print(f"     antes de que la inasistencia se consolide en el sistema Adviser.")
    elif "inasistencias" in nombre:
        c = nombre[-1]
        print(f"     El numero de inasistencias al Corte {c} captura la presion laboral")
        print(f"     en forma continua: cada sesion perdida penaliza las entregas de Teams")
        print(f"     y reduce la probabilidad de aprobar el parcial presencial.")
    elif "autoeval" in nombre or "coeval" in nombre:
        c = nombre[-1]
        print(f"     Las metricas de evaluacion formativa del Corte {c} son el sensor")
        print(f"     de abandono: un valor 0.0 indica que el alumno dejo de participar")
        print(f"     activamente en el proceso de aprendizaje colaborativo.")
    elif "brecha_bach" in nombre:
        print(f"     Los años transcurridos desde el bachillerato son el predictor")
        print(f"     socieconomico de mayor peso: captura el rezago cognitivo y la")
        print(f"     dificultad de reintegracion al ritmo academico universitario.")
    elif "trabaja" in nombre:
        print(f"     La condicion laboral activa es el principal factor externo del")
        print(f"     entorno del estudiante nocturno que restringe el tiempo de estudio.")
    elif "conflicto_tiempo" in nombre:
        print(f"     El conflicto trabajo-estudio captura la percepcion subjetiva del")
        print(f"     alumno sobre su incapacidad de cubrir ambas demandas, lo que es")
        print(f"     un predictor mas fino que la simple condicion laboral.")
    elif "debilidad_mate" in nombre:
        print(f"     La autopercepcion de debilidades en matematicas activa el patron")
        print(f"     de evitacion academica, especialmente en ciencias basicas.")
    elif "barrera_tecnologica" in nombre:
        print(f"     Depender exclusivamente de un celular para actividades de")
        print(f"     ingenieria limita la calidad y regularidad de las entregas en Teams.")
    elif "inseguridad_alimentaria" in nombre:
        print(f"     La inseguridad alimentaria refleja una vulnerabilidad material")
        print(f"     critica que el sistema Adviser V.11 no puede capturar por si solo.")
    print()

print(f"  {SEP_SIMPLE}")
print(f"\n  CONCLUSION DEL PIPELINE:")
print(f"  El modelo Random Forest con {N_ESTIMATORS} estimadores y max_depth={MAX_DEPTH}")
print(f"  ha sido entrenado y evaluado exitosamente sobre el 20% de datos reservados.")
print(f"  El pipeline Multi-CSV demuestra consistencia estructural y viabilidad")
print(f"  tecnica para la arquitectura TRL 2/3 del proyecto PI-II.")
print(f"\n{SEP_DOBLE}\n")
