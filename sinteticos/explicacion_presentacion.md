# Explicación técnica de la presentación — PI-II ETITC
**Pipeline de Analítica Predictiva para Detección Temprana de Riesgo de Deserción**  
Autores: Correal Martínez & Guzmán Rodríguez | Docente: Doris Constanza Alvarado Mariño

---

## Diapositiva 1 — Portada

**Contexto del proyecto:**  
No se tienen datos reales de estudiantes porque la Ley 1581 de 2012 (Habeas Data) impide que la ETITC entregue microdatos individuales. Para avanzar, se construyeron 1,000 estudiantes sintéticos usando los porcentajes de los informes de Bienestar Universitario de los últimos 10 semestres (2021-I a 2025-II) como fuente de calibración.

**Lo que se validó en esta fase (TRL 2/3):**  
- Que la arquitectura de tres CSV relacionales funciona sin errores
- Que las reglas de negocio producen patrones estadísticamente separables
- Que el pipeline ETL → Entrenamiento → Evaluación es reproducible

**Lo que NO se puede afirmar todavía:** ninguna métrica de rendimiento (Accuracy, Recall, AUC) es transferible a producción — son resultado de datos sintéticos, no de comportamiento humano real.

---

## Diapositiva 2 — ¿Qué problema resuelve este pipeline?

### Donut: Balance de clases

| Clase | Estudiantes | Porcentaje |
|---|---|---|
| Persiste (0) | 743 | 74.3% |
| Deserto (1) | 257 | 25.7% |

**Qué significa este desbalance:**  
El dataset tiene casi 3 veces más estudiantes que persisten que desertores. Esto refleja la distribución real reportada por la ETITC (~12% deserción anual real, aunque en los datos sintéticos se elevó a 25.7% para tener suficientes casos positivos que el modelo pueda aprender).

**Por qué importa para el modelo:**  
Un algoritmo que ignorase todo y dijera siempre "este estudiante va a persistir" tendría 74.3% de accuracy sin aprender absolutamente nada. Por eso accuracy sola es una métrica engañosa aquí — las métricas que realmente importan son **Recall** y **F1-Score sobre la clase 1 (Deserto)**.

**Sobre el desbalance:** no se forzó un balance artificial 50/50 porque eso distorsionaría las probabilidades del modelo. Se mantuvo la distribución real y se usó `stratify=y` en el split para garantizar que la proporción de desertores sea idéntica en el bloque de entrenamiento y en el de prueba.

### Tabla: Tres fuentes CSV

| Archivo | Columnas | Rol |
|---|---|---|
| bienestar_caracterizacion.csv | 7 | Perfil socioeconómico inicial del estudiante |
| adviser_teams_notas.csv | 19 | Rendimiento académico por los 3 cortes |
| registro_admisiones_target.csv | 2 | Variable objetivo `deserto` (0/1) |
| **Unificada post-merge** | **26** | **Entrada al modelo de ML** |

Los tres archivos se unen con un INNER JOIN usando `id_estudiante` como llave — el equivalente en Python a un `pd.merge(..., how='inner')`. Solo entran al entrenamiento los estudiantes que existen en los tres archivos simultáneamente.

---

## Diapositiva 3 — Perfil socioeconómico simulado

### Gráfica izquierda: Variables de caracterización (barras horizontales)

Cada barra muestra qué porcentaje de los 1,000 estudiantes tiene ese factor de riesgo activado (valor = 1).

| Variable | % simulado | Fuente de calibración |
|---|---|---|
| Trabaja activamente | 67.8% | Informes BU 2021-2025 confirman ~68% |
| Conflicto trabajo-estudio | 54.8% | Derivado condicionalmente: si trabaja → 70% prob. de conflicto |
| Debilidad en matemáticas | 44.7% | Informes BU confirman ~45% |
| Solo celular para ingeniería | 36.7% | Variable proxy de brecha digital |
| Inseguridad alimentaria | 28.3% | Informes BU 2023-2025 |

**Punto clave sobre `conflicto_tiempo` vs. `trabaja`:**  
No son la misma variable. `trabaja` es un hecho objetivo (tiene empleo activo). `conflicto_tiempo` es la percepción subjetiva del estudiante de que ese trabajo le impide cumplir con el estudio. Un estudiante puede trabajar con horario flexible y no sentir conflicto, o puede no trabajar pero tener hijos pequeños y sí sentirlo. En el Feature Importance del modelo, ambas variables aparecen con pesos distintos, confirmando que aportan información independiente.

**Por qué estas variables entran al modelo:**  
La literatura (Bernal, 2024) demuestra que incluir variables del contexto vital del estudiante incrementa el poder predictivo ~14% comparado con modelos que solo usan historial de calificaciones. Estas son las variables que el sistema Adviser V.11 actual no conecta con las notas.

### Gráfica derecha: Distribución de brecha_bach (barras verticales)

Muestra cuántos años transcurrieron desde que cada estudiante se graduó del bachillerato.

- **Verde (0-2 años):** 715 estudiantes — mayoría reciente, menor rezago cognitivo
- **Rojo (3+ años):** 285 estudiantes — rezago crítico, dificultad de readaptación

**Por qué es una variable importante:**  
A más años fuera del sistema educativo, más difícil recuperar hábitos de estudio y competencias en ciencias básicas. En el modelo, esta variable no aparece sola en el top del ranking sino que su efecto se refleja en las notas de los parciales — primero bajan las notas y luego viene la deserción.

**Por qué se truncó en 5 años:**  
Los informes de Bienestar categorizan la brecha en rangos hasta "más de 5 años". Solo el 5% de la distribución cae en ese extremo y extenderlo no cambiaría materialmente los resultados.

---

## Diapositiva 4 — Rendimiento académico dinámico

### Gráfica izquierda: Boxplots de notas definitivas por corte

Cada caja representa la distribución de la nota definitiva de los 1,000 estudiantes en ese corte.

- **La caja:** contiene el 50% central de los datos (entre el percentil 25 y el 75)
- **La línea blanca:** la mediana (la nota que divide exactamente la mitad)
- **Los bigotes:** llegan hasta 1.5 veces el rango intercuartílico
- **Los puntos:** outliers — estudiantes con notas extremadamente bajas o altas

| Corte | Mediana | Interpretación |
|---|---|---|
| Corte 1 | 3.37 | Mayoría por encima del umbral de aprobación (3.0) |
| Corte 2 | 3.33 | Leve degradación — fatiga laboral comienza a sentirse |
| Corte 3 | 3.23 | Degradación mayor — la caja baja del umbral 3.0 |

**Qué valida esta tendencia:**  
La mediana desciende sistemáticamente de C1 a C3. Esto confirma que la **Regla 2** del simulador (presión laboral → ausentismo → notas más bajas) funciona como se programó. En un dataset real, esta trayectoria descendente sería la señal que el modelo aprendería a detectar antes del cierre del semestre.

**Sobre los valores extremos:**  
Los puntos por debajo de 1.5 en C3 corresponden a los 65 estudiantes con riesgo cognitivo alto (`brecha_bach > 3` y `debilidad_mate = 1`). Son los casos que en producción dispararían las alertas más urgentes hacia Bienestar.

### Gráfica derecha: Asistencia vs. ausentismo por corte

| Corte | Asistió | No asistió |
|---|---|---|
| Corte 1 | 68.6% | 31.4% |
| Corte 2 | 66.9% | 33.1% |
| Corte 3 | 61.4% | 38.6% |

**Qué muestra el patrón:**  
La asistencia cae 7.2 puntos porcentuales de C1 a C3. Esto es la **Regla 2** expresada en comportamiento observable: los estudiantes con presión laboral empiezan a fallar en C1, y para C3 el ausentismo ya es casi 4 de cada 10 estudiantes. El modelo captura este patrón temporal como señal predictiva — la variable `asistio_c3` aparece en el Feature Importance precisamente porque ausentismo en el tercer corte es casi determinante de reprobación.

---

## Diapositiva 5 — Validación de reglas de negocio

### Gráfica izquierda: Regla 2 — Penalización de entregas en Teams

Compara las notas de entregas de talleres en Microsoft Teams entre dos grupos:
- **Verde** (sin presión): estudiantes que no trabajan ni reportan conflicto de tiempo (n=263)
- **Naranja** (con presión): estudiantes que trabajan o tienen conflicto de tiempo (n=737)

| Grupo | Media | Forma de la distribución |
|---|---|---|
| Sin presión | 3.64 | Cargada hacia 4-5, pocos valores bajos |
| Con presión | 2.41 | Bimodal — pico en 0-2 (entregas tardías) y otro en 3-4 |

**Qué demuestra esta separación:**  
La distancia entre las dos medias (3.64 vs. 2.41 = 1.23 puntos) es estadísticamente significativa. Esto significa que la variable `notas_teams` no es ruido — lleva información real sobre el comportamiento del estudiante. En términos de ML: la separación entre clases hace que la variable sea **discriminante**, es decir, útil para predecir.

**Limitación importante:**  
En datos sintéticos esta separación es más limpia que en la realidad porque las reglas son deterministas. En datos reales habrá más solapamiento entre los grupos (habrá estudiantes con presión laboral que igualmente entreguen todo a tiempo), lo que es esperable y no invalida el enfoque.

### Gráfica derecha: Regla 3 — Choque cognitivo en ciencias básicas

Compara notas de examen presencial entre:
- **Rojo** (riesgo cognitivo alto): `brecha_bach > 3` Y `debilidad_mate = 1` (n=65, el 6.5%)
- **Azul** (sin riesgo cognitivo): el resto (n=935, el 93.5%)

| Grupo | Media | Forma de la distribución |
|---|---|---|
| Riesgo cognitivo | 2.26 | Pico pronunciado entre 1.5-2.5, casi todo por debajo del umbral |
| Sin riesgo | 3.63 | Distribución más amplia, mayoría por encima del umbral |

**Qué demuestra este gráfico:**  
Aunque el grupo de riesgo cognitivo es solo el 6.5% del dataset, su distribución de notas es completamente distinta — casi ninguno supera el umbral de 3.0 en los parciales. Esto simula el "choque cognitivo" real que ocurre cuando un estudiante adulto con años fuera del sistema educativo enfrenta matemáticas universitarias.

**Valor para el Feature Importance:**  
Esta señal tan clara hace que las variables de `parcial_c1`, `parcial_c2` y `parcial_c3` aparezcan con peso relevante en el modelo, aunque no en el top 5 porque su efecto ya está capturado por las `definitiva_cX` (que las incluyen ponderadas).

---

## Diapositiva 6 — Matriz de confusión

Los 200 estudiantes del conjunto de prueba (el 20% reservado) se clasifican en cuatro celdas:

|  | **Modelo predijo: Persiste** | **Modelo predijo: Deserto** |
|---|---|---|
| **Realidad: Persiste** | **148 Verdaderos Negativos** | **1 Falso Positivo** |
| **Realidad: Deserto** | **11 Falsos Negativos ⚠** | **40 Verdaderos Positivos** |

**Lectura de cada celda:**

**148 Verdaderos Negativos (VN)**  
El modelo dijo "este estudiante va a persistir" y tenía razón. Son casos donde el sistema no emite alerta — correcto, no consume recursos de Bienestar innecesariamente.

**1 Falso Positivo (FP)**  
El modelo dijo "este estudiante va a desertar" pero en realidad persistió. Es una falsa alarma. Bienestar intervendría sin necesidad. Con solo 1 en 200 casos, el costo de falsas alarmas es mínimo.

**40 Verdaderos Positivos (VP)**  
El modelo detectó correctamente a 40 de los 51 desertores reales del conjunto de prueba. Cada uno de estos representa una intervención oportuna posible antes de que el abandono se consolide.

**11 Falsos Negativos (FN) — el número más importante**  
El modelo dijo "este estudiante va a persistir" pero en realidad desertó. Estos 11 son los estudiantes que caerían sin ninguna alerta hacia Bienestar. Son el error de mayor costo institucional y el indicador principal a reducir en la siguiente fase.

**Cómo se interpreta en escala real:**  
Si en un semestre hay 500 estudiantes reales en riesgo, con este Recall del 78.4% el modelo alertaría sobre ~392 y dejaría pasar ~108 sin intervención. La meta de la Fase 4 es subir ese Recall por encima del 85%.

---

## Diapositiva 7 — Curvas ROC y Precision-Recall

### Curva ROC (AUC = 0.996)

El eje X es la tasa de falsas alarmas (FPR) y el eje Y es la tasa de detección correcta (TPR). Cada punto en la curva corresponde a un umbral de decisión diferente.

- **Diagonal punteada:** un modelo que adivina al azar (AUC = 0.5)
- **Esquina superior izquierda:** el punto perfecto (0 falsas alarmas, 100% detección)
- **Nuestra curva:** pegada a la esquina superior izquierda (AUC = 0.996)

**Qué significa AUC = 0.996:**  
Si se toman al azar un estudiante desertor y un estudiante persistente, el modelo les asigna una probabilidad de riesgo correctamente ordenada el 99.6% de las veces. Es una medida de la capacidad discriminante global del clasificador, independiente del umbral elegido.

**Por qué el AUC es tan alto:**  
Porque el modelo está aprendiendo las mismas reglas deterministas con las que se generaron los datos. En datos reales el comportamiento humano tiene mucho más ruido — el AUC esperado sería entre 0.75 y 0.85.

**Para qué sirve esta curva en la práctica:**  
Permite elegir el umbral óptimo. Actualmente se usa threshold = 0.5. Si se baja a 0.35, el modelo emitiría más alertas (sube el Recall, bajan los Falsos Negativos) a costa de generar algunas falsas alarmas adicionales. Ese trade-off es una decisión institucional, no técnica.

### Curva Precision-Recall

El eje X es el Recall (qué porcentaje de desertores reales detectamos) y el eje Y es la Precisión (qué porcentaje de las alertas emitidas son correctas).

**La línea horizontal punteada** es el baseline — si el modelo dijera que todos desertan, tendría Recall = 100% pero Precisión = 25.7% (la prevalencia de la clase). Nuestra curva está muy por encima de ese baseline.

**Por qué esta curva es más honesta que la ROC con clases desbalanceadas:**  
La ROC puede verse optimista porque el denominador del FPR incluye a todos los negativos (743 persistentes). Con muchos negativos, incluso modelos mediocres tienen FPR bajo. La curva PR no tiene ese sesgo — su eje X depende solo de los positivos reales.

---

## Diapositiva 8 — Métricas de rendimiento

| Métrica | Valor | Qué mide |
|---|---|---|
| **Accuracy** | 94.0% | Porcentaje total de predicciones correctas (ambas clases) |
| **Precision (clase 1)** | 97.6% | De cada 100 alertas emitidas, 97.6 son desertores reales |
| **Recall (clase 1)** | 78.4% | De cada 100 desertores reales, el modelo detecta 78.4 |
| **F1-Score (clase 1)** | 87.0% | Media armónica entre Precision y Recall |
| **AUC-ROC** | 99.6% | Capacidad discriminante global del clasificador |

**Por qué Recall es la métrica más importante para este proyecto:**  
Porque el costo de no detectar un desertor (Falso Negativo) es mucho mayor que el costo de emitir una falsa alarma (Falso Positivo). Un FN representa un estudiante que abandona sin intervención. Un FP representa una llamada de Bienestar innecesaria. Institucionalmente, la segunda es mucho más barata que la primera.

**Por qué Accuracy del 94% no es el número a celebrar:**  
Un modelo que siempre diga "persiste" tendría 74.3% de Accuracy sin aprender nada. El 94% es mejor, pero la mejora real está en que el modelo detecta 40 de 51 desertores — eso no lo hace el modelo naive.

**Por qué F1-Score es la métrica de referencia:**  
Porque balancea Precision y Recall. Un F1 de 0.87 sobre la clase 1 en datos sintéticos es la línea base. La meta realista con datos reales sería F1 ≥ 0.70.

---

## Diapositiva 9 — Feature Importance

Muestra el peso predictivo de cada variable, medido como la reducción promedio de impureza Gini a través de los 100 árboles del Random Forest.

| Rank | Variable | Peso | Fuente |
|---|---|---|---|
| 1 | `definitiva_c3` | 17.7% | Adviser — nota final del último corte |
| 2 | `definitiva_c2` | 14.2% | Adviser — nota final del segundo corte |
| 3 | `autoevaluacion_c3` | 12.5% | Planilla docente — evaluación formativa C3 |
| 4 | `coevaluacion_c3` | 9.1% | Planilla docente — evaluación formativa C3 |
| 5 | `definitiva_c1` | 6.9% | Adviser — nota final del primer corte |

**Las 5 variables top explican el 60.4% del peso predictivo total.**

**Lectura del ranking:**

`definitiva_c3` (17.7%): cuando la nota final del tercer corte cae, la deserción ya casi no tiene vuelta atrás. Es el indicador de resultado más tardío pero más definitivo.

`definitiva_c2` (14.2%): el segundo corte anticipa el resultado final con alta correlación. Es el punto óptimo para intervenir — hay tiempo suficiente antes del cierre del semestre.

`autoevaluacion_c3` y `coevaluacion_c3` (12.5% y 9.1%): cuando ambas caen a 0.0 simultáneamente, el modelo interpreta que el estudiante abandonó físicamente el aula. Esta fue la Regla 4 del simulador y el modelo la detectó como señal crítica.

`definitiva_c1` (6.9%): establece la trayectoria inicial. Un C1 bajo predice que la degradación continuará en C2 y C3.

**El hallazgo central del proyecto:**  
Las 5 variables más importantes son todas académicas y dinámicas (por corte), no socioeconómicas estáticas. Los factores de Bienestar como `trabaja`, `brecha_bach` y `conflicto_tiempo` aparecen en el ranking pero con pesos menores. Esto sugiere que el efecto socioeconómico no actúa directamente sobre la deserción — actúa primero sobre las notas, y son las notas las que predicen la deserción.

**Limitación técnica del Feature Importance de Random Forest:**  
El índice de Gini favorece variables continuas sobre variables binarias porque tienen más puntos de corte posibles. Las variables binarias como `trabaja` o `asistio_c1` pueden estar subestimadas en el ranking. Para un análisis más robusto en la siguiente fase se usaría **Permutation Importance** o **SHAP values**, que no tienen este sesgo.

---

## Diapositiva 10 — ¿Qué aprendió el modelo?

Esta diapositiva traduce los números técnicos a acciones institucionales concretas.

**Hallazgo 1 — Las notas por corte son el sensor activo:**  
Las variables académicas dinámicas tienen mayor peso que el perfil socioeconómico estático de Bienestar. Esto valida la hipótesis del proyecto: conectar Adviser con Teams y procesar las notas por corte agrega valor predictivo que el sistema actual no captura.

**Hallazgo 2 — Los 11 Falsos Negativos son el problema a resolver:**  
Son los estudiantes que el sistema dejaría pasar sin intervención. La forma de reducirlos es bajar el umbral de decisión de 0.5 a ~0.35: el modelo emitiría más alertas pero capturaría más desertores reales. Es un trade-off que decide Bienestar, no el algoritmo.

**Hallazgo 3 — Recall del 78.4% como línea base:**  
El modelo detecta 8 de cada 10 estudiantes en riesgo real. En producción con datos reales, esta tasa bajará por el ruido natural del comportamiento humano. La meta es no caer por debajo del 70%.

**Hallazgo 4 — Precision del 97.6% como garantía de calidad de alertas:**  
Solo 1 estudiante recibió alerta incorrecta en 200 pruebas. Esto significa que Bienestar podría actuar sobre cada alerta del modelo con alta confianza de que es un caso real, sin saturar su capacidad operativa.

---

## Diapositiva 11 — Conclusiones y próximos pasos

### Lo que quedó demostrado en esta fase

- La arquitectura Multi-CSV produce una matriz unificada de 1,000 × 24 variables sin errores ni valores nulos
- Los patrones inyectados son estadísticamente separables (AUC = 0.996 en datos sintéticos)
- Las variables académicas por corte tienen mayor peso predictivo que el perfil inicial estático
- El pipeline ETL → RandomForest → Evaluación es reproducible con `random_state=42`

### Lo que NO se puede afirmar todavía

- Ninguna métrica de rendimiento es válida para producción — los datos sintéticos son "demasiado limpios"
- El AUC 0.996 no es una promesa, es un artefacto del proceso de simulación

### Próximos pasos proyectados

| Paso | Acción | Por qué |
|---|---|---|
| 1 | Bajar el threshold de 0.5 a ~0.35 | Subir Recall, reducir los 11 FN |
| 2 | Comparar con XGBoost | Evaluar si mejora el F1 sobre la clase minoritaria |
| 3 | Reemplazar split 80/20 por k-fold (k=5) | Estimación más robusta de la varianza del modelo |
| 4 | Acuerdo de confidencialidad con ETITC | Acceder a sábanas anonimizadas de Adviser V.11 bajo Ley 1581 |

---

## Pregunta más difícil que puede hacer el profesor

> *"¿Qué tan válido es todo esto si los datos son sintéticos?"*

**Respuesta:**  
Completamente válido para el objetivo de esta fase. El objetivo de TRL 2/3 no es predecir desertores reales — es demostrar que la arquitectura es técnicamente coherente y que el instrumento de recolección (las encuestas de Bienestar) captura las variables correctas.

Tres cosas concretas quedan demostradas:
1. El INNER JOIN de 3 CSVs produce una matriz sin errores — la arquitectura relacional funciona
2. Los patrones inyectados son estadísticamente separables — las variables elegidas tienen poder discriminante
3. Las variables académicas por corte tienen mayor peso que el perfil estático — la hipótesis del proyecto tiene soporte empírico

Lo que NO se puede reclamar es ningún número de rendimiento en producción. El AUC 0.996 es un artefacto de los datos sintéticos. Con datos reales, el AUC esperado es entre 0.75 y 0.85, y ese es el número que la Fase 4 del proyecto tiene que medir y reportar.
