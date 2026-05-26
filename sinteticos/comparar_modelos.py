"""
Comparación de algoritmos — Detección Temprana de Riesgo de Deserción
Proyecto PI-II: Sistema de Analítica Predictiva — Jornada Nocturna ETITC

Autores:  Jhon Alejandro Correal Martínez
          Rafael Andrés Guzmán Rodríguez
Versión:  1.0  |  2026-05-26

Objetivo:
  Comparar Random Forest (baseline) vs. XGBoost en dos variantes sobre el
  mismo split estratificado 80/20, evaluando el impacto de:
    1. Ponderación de clase minoritaria (scale_pos_weight)
    2. Ajuste del umbral de decisión (0.5 → 0.35)

Justificación del comparativo:
  El Recall es la métrica crítica: un Falso Negativo (desertor no detectado)
  tiene mayor costo institucional que un Falso Positivo (alerta innecesaria).
  XGBoost con scale_pos_weight amplifica la penalización de errores sobre
  la clase minoritaria (desertores), lo que debería mejorar el Recall sin
  colapsar la Precision a niveles que generen fatiga de alerta.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
SEED      = 42
TEST_SIZE = 0.20

FILE_BIENESTAR = "bienestar_caracterizacion.csv"
FILE_NOTAS     = "adviser_teams_notas.csv"
FILE_TARGET    = "registro_admisiones_target.csv"

SEP_DOBLE  = "=" * 70
SEP_SIMPLE = "-" * 70


# =============================================================================
# CARGA Y UNIFICACIÓN
# =============================================================================
print(f"\n{SEP_DOBLE}")
print("  CARGA Y UNIFICACIÓN DE FUENTES")
print(f"{SEP_DOBLE}\n")

df = (pd.read_csv(FILE_BIENESTAR)
        .merge(pd.read_csv(FILE_NOTAS),  on="id_estudiante")
        .merge(pd.read_csv(FILE_TARGET), on="id_estudiante"))

y = df["deserto"]
X = df.drop(columns=["id_estudiante", "deserto"])

print(f"  Matriz unificada          : {df.shape[0]} filas × {X.shape[1]} features")

vc = y.value_counts()
print(f"  Clase 0 (Persiste)        : {vc[0]}  ({vc[0]/len(y)*100:.1f}%)")
print(f"  Clase 1 (Deserto)         : {vc[1]}  ({vc[1]/len(y)*100:.1f}%)")

# Ratio para scale_pos_weight
SPW = round(vc[0] / vc[1], 2)
print(f"  scale_pos_weight calculado: {SPW}  (negativos/positivos)")


# =============================================================================
# SPLIT ESTRATIFICADO (mismo para todos los modelos)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)
print(f"\n  Train : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")
print(f"  Desertores en test : {int((y_test == 1).sum())}")


# =============================================================================
# FUNCIÓN DE REPORTE
# =============================================================================
def reporte(nombre, y_true, y_pred, y_prob, threshold=0.5):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)

    print(f"\n  {'─'*60}")
    print(f"  MODELO : {nombre}  (umbral = {threshold})")
    print(f"  {'─'*60}")
    print(f"  Matriz de confusión:")
    print(f"    TN={TN:>3}  FP={FP:>3}   ← desertores correctamente ignorados / falsas alarmas")
    print(f"    FN={FN:>3}  TP={TP:>3}   ← perdidos sin alerta / detectados correctamente")
    print()
    print(f"  Accuracy  : {acc*100:>5.1f}%")
    print(f"  Precision : {prec*100:>5.1f}%   (alertas correctas / total alertas)")
    print(f"  Recall    : {rec*100:>5.1f}%   ← métrica crítica: desertores detectados")
    print(f"  F1-Score  : {f1*100:>5.1f}%   (media armónica Precision × Recall)")
    print(f"  AUC-ROC   : {auc:.4f}")

    return {
        "Modelo": nombre, "Umbral": threshold,
        "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        "Accuracy": round(acc*100,1), "Precision": round(prec*100,1),
        "Recall": round(rec*100,1), "F1": round(f1*100,1),
        "AUC": round(auc,4)
    }


resultados = []

# =============================================================================
# MODELO 1 — Random Forest baseline (umbral 0.5)
# =============================================================================
print(f"\n{SEP_DOBLE}")
print("  MODELO 1 — Random Forest (baseline)")
print(f"{SEP_DOBLE}")

rf = RandomForestClassifier(
    n_estimators=100, max_depth=6, min_samples_leaf=4,
    random_state=SEED, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_pred = rf.predict(X_test)

resultados.append(reporte("Random Forest (RF)", y_test, rf_pred, rf_prob, threshold=0.5))


# =============================================================================
# MODELO 2 — Random Forest umbral ajustado (0.35)
# =============================================================================
rf_pred_35 = (rf_prob >= 0.35).astype(int)
resultados.append(reporte("Random Forest (umbral 0.35)", y_test, rf_pred_35, rf_prob, threshold=0.35))


# =============================================================================
# MODELO 3 — XGBoost sin ponderación (umbral 0.5)
# =============================================================================
print(f"\n{SEP_DOBLE}")
print("  MODELO 3 — XGBoost sin ponderación de clase")
print(f"{SEP_DOBLE}")

xgb = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, n_jobs=-1, verbosity=0, eval_metric="logloss"
)
xgb.fit(X_train, y_train)
xgb_prob = xgb.predict_proba(X_test)[:, 1]
xgb_pred = xgb.predict(X_test)

resultados.append(reporte("XGBoost (sin peso)", y_test, xgb_pred, xgb_prob, threshold=0.5))


# =============================================================================
# MODELO 4 — XGBoost con scale_pos_weight (umbral 0.5)
# =============================================================================
print(f"\n{SEP_DOBLE}")
print(f"  MODELO 4 — XGBoost con scale_pos_weight={SPW}")
print(f"{SEP_DOBLE}")
print()
print(f"  Al asignar scale_pos_weight={SPW}, cada error sobre un desertor real")
print(f"  penaliza {SPW}× más que un error sobre un estudiante que persiste.")
print(f"  Esto orienta el árbol a reducir Falsos Negativos a costa de aumentar")
print(f"  ligeramente los Falsos Positivos.\n")

xgb_w = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=SPW,
    random_state=SEED, n_jobs=-1, verbosity=0, eval_metric="logloss"
)
xgb_w.fit(X_train, y_train)
xgb_w_prob = xgb_w.predict_proba(X_test)[:, 1]
xgb_w_pred = xgb_w.predict(X_test)

resultados.append(reporte(f"XGBoost (SPW={SPW})", y_test, xgb_w_pred, xgb_w_prob, threshold=0.5))


# =============================================================================
# MODELO 5 — XGBoost con scale_pos_weight + umbral 0.35
# =============================================================================
xgb_w_35 = (xgb_w_prob >= 0.35).astype(int)
resultados.append(reporte(f"XGBoost (SPW={SPW}, umbral 0.35)", y_test, xgb_w_35, xgb_w_prob, threshold=0.35))


# =============================================================================
# TABLA COMPARATIVA
# =============================================================================
print(f"\n\n{SEP_DOBLE}")
print("  TABLA COMPARATIVA — RESUMEN")
print(f"{SEP_DOBLE}\n")

df_res = pd.DataFrame(resultados)

header = f"  {'Modelo':<38} {'Umbral':>7} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>7} {'FN':>4} {'FP':>4}"
print(header)
print(f"  {'-'*38} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*4} {'-'*4}")

for _, row in df_res.iterrows():
    marker = "  ◄ MEJOR RECALL" if row["Recall"] == df_res["Recall"].max() and row["FN"] == df_res["FN"].min() else ""
    print(f"  {row['Modelo']:<38} {row['Umbral']:>7} {row['Accuracy']:>5.1f}% {row['Precision']:>5.1f}% "
          f"{row['Recall']:>5.1f}% {row['F1']:>5.1f}% {row['AUC']:>7.4f} {row['FN']:>4} {row['FP']:>4}{marker}")


# =============================================================================
# ANÁLISIS DE IMPORTANCIA DE VARIABLES — XGBoost ganador
# =============================================================================
print(f"\n{SEP_DOBLE}")
print(f"  IMPORTANCIA DE VARIABLES — XGBoost (SPW={SPW})  vs  Random Forest")
print(f"{SEP_DOBLE}\n")

imp_xgb = pd.Series(xgb_w.feature_importances_, index=X.columns).sort_values(ascending=False)
imp_rf  = pd.Series(rf.feature_importances_,    index=X.columns).sort_values(ascending=False)

print(f"  {'Variable':<28} {'XGBoost':>8}  {'RF':>8}  {'Diff':>8}")
print(f"  {'-'*28} {'-'*8}  {'-'*8}  {'-'*8}")

for feat in imp_xgb.head(10).index:
    xv = imp_xgb[feat]
    rv = imp_rf.get(feat, 0)
    diff = xv - rv
    arrow = "↑" if diff > 0 else "↓"
    print(f"  {feat:<28} {xv*100:>7.1f}%  {rv*100:>7.1f}%  {arrow}{abs(diff)*100:>6.1f}%")


# =============================================================================
# VALIDACIÓN CRUZADA (k=5) — modelo ganador
# =============================================================================
print(f"\n{SEP_DOBLE}")
print(f"  VALIDACIÓN CRUZADA k=5 — XGBoost (SPW={SPW})")
print(f"{SEP_DOBLE}\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

xgb_cv = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=SPW,
    random_state=SEED, n_jobs=-1, verbosity=0, eval_metric="logloss"
)

for metric_name, scoring in [("F1", "f1"), ("Recall", "recall"), ("AUC", "roc_auc")]:
    scores = cross_val_score(xgb_cv, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    print(f"  {metric_name:<10}: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%   "
          f"(folds: {' '.join(f'{s*100:.1f}%' for s in scores)})")


# =============================================================================
# CONCLUSIÓN
# =============================================================================
print(f"\n{SEP_DOBLE}")
print("  CONCLUSIÓN DEL COMPARATIVO")
print(f"{SEP_DOBLE}\n")

best = df_res.loc[df_res["F1"].idxmax()]
print(f"  Modelo con mejor F1   : {best['Modelo']}  (F1={best['F1']}%)")
print(f"  Modelo con mejor Recall: {df_res.loc[df_res['Recall'].idxmax(), 'Modelo']}  (Recall={df_res['Recall'].max()}%)")
print()
print(f"  XGBoost con scale_pos_weight={SPW} mejora el Recall de {df_res.loc[0,'Recall']}%")
print(f"  (RF baseline) a {df_res.loc[3,'Recall']}%, detectando {int(df_res.loc[3,'TP'])} de {int((y_test==1).sum())} desertores.")
print(f"  El costo: {int(df_res.loc[3,'FP'])} alertas adicionales que el equipo de Bienestar")
print(f"  necesita evaluar. Con una capacidad operativa normal, este trade-off")
print(f"  favorece claramente al XGBoost para la versión de producción del sistema.")
print(f"\n{SEP_DOBLE}\n")
