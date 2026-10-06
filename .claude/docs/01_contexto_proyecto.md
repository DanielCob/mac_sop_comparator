# 01 · Contexto del proyecto (resumen del Plan de Proyecto y del Documento de Diseño)

## Datos generales

- **Proyecto:** Simulación comparativa de eficiencia teórica entre arquitectura neuromórfica y
  tradicional para clasificación espacial molecular.
- **Producto:** Comparador de eficiencia MAC-SOP.
- **Curso:** CE-1114 Proyecto de Aplicación de la Ingeniería en Computadores, TEC, II Semestre 2026.
- **Estudiante:** Daniel Cob Beirute (2021084824). Proyecto individual.
- **Supervisor:** Ronald García Fernández. **Profesor:** Luis Diego Noguera Mena.
- **Propiedad intelectual:** del estudiante; se usan bibliotecas open source acreditadas.
- **Datos:** públicos (MoleculeNet), sin confidencialidad.

## Problema

No existe un marco reproducible que, implementando una metodología SNN publicada
(Nascimben y Rimondini, 2023) junto con una ANN homóloga, cuantifique **MACs vs SOPs** y el
desempeño de clasificación sobre el mismo benchmark de toxicidad. Se quiere saber **bajo qué
condiciones** (dispersidad, T, etc.) la SNN tiene ventaja teórica de eficiencia, solo con
simulación por software (sin Loihi ni GPU para medir).

**Hipótesis:** la SNN reduce las operaciones por inferencia (SOPs < MACs) con una diferencia
marginal de exactitud.

## Objetivos

- **General:** contrastar eficiencia algorítmica y capacidad predictiva de una SNN frente a una
  ANN homóloga en clasificación de toxicidad molecular sobre huellas de MoleculeNet.
- **OE1:** adaptar huellas binarias (p. ej. MACCS 166 bits) de un benchmark de toxicidad con
  RDKit + limpieza + partición.
- **OE2:** ANN densa en PyTorch como línea base de exactitud y MACs por inferencia.
- **OE3:** replicar la SNN LIF con codificación por tasa de Nascimben y Rimondini (2023) en
  snnTorch; exactitud y SOPs por inferencia.
- **OE4:** contrastar MACs vs SOPs y desempeño para determinar condiciones de ventaja.

## Alcance

**Dentro:** preprocesar MoleculeNet a firmas (MACCS 166, Morgan 1024 r=2); ANN y SNN homólogas
(misma topología, datos y particiones); medir MACs, SOPs, actualizaciones de neurona,
intensidad aritmética y energía estimada; comparar y explorar espacio de diseño
(T, firma, ancho de capas, β, umbral, tasa de disparo).

**Fuera:** hardware real (Loihi 2, GPU como medida), vóxeles 3D, grafos, modelos generativos,
redes profundas VGG/ResNet, optimización exhaustiva de hiperparámetros, costos de
entrenamiento (solo se mide inferencia), gem5, Lava-dl. Reimplementación en lenguaje menos
abstracto = trabajo futuro.

## Entregables (Plan, sección 2.2)

| ID | Entregable | Criterio de aceptación |
|---|---|---|
| ENT01 | Dataset preprocesado (huellas) | Partición train/val/test documentada; sin faltantes ni duplicados; huellas validadas contra al menos una muestra de control generada manualmente con RDKit. |
| ENT02 | ANN de referencia | Convergencia estable en validación; exactitud balanceada (o AUC) y MACs por inferencia con herramienta de profiling reproducible. |
| ENT03 | SNN implementada | Mismas particiones y métricas que la ANN; exactitud balanceada (o AUC) y SOPs por inferencia con método de conteo propio, justificado y reproducible. |
| ENT04 | Módulo de instrumentación y comparación | Extrae MACs y SOPs automáticamente y genera ≥1 tabla y ≥1 gráfica de exactitud vs eficiencia. |
| ENT05 | Informe técnico / artículo | Análisis comparativo, condiciones de eficiencia, limitaciones y trabajo futuro. |
| ENT06 | Repositorio documentado | README con instalación y reproducción; historial de control de versiones activo; licencias acreditadas. |

## Actividades y cronograma (supuesto 120 h, ~8 h/semana, inicio 14/09/2026)

| ID | Actividad | h | Inicio | Fin |
|---|---|---|---|---|
| A1 | Revisión bibliográfica | 6 | 14/09 | 17/09 |
| A2 | Alcance y plan | 4 | 18/09 | 21/09 |
| A3 | Selección benchmark y SMILES | 3 | 22/09 | 23/09 |
| A4 | Huellas MACCS/ECFP con RDKit | 5 | 24/09 | 28/09 |
| A5 | Limpieza, partición, validación | 4 | 29/09 | 30/09 |
| A6 | Diseño ANN | 3 | 01/10 | 02/10 |
| A7 | Implementación y entrenamiento ANN | 10 | 05/10 | 12/10 |
| A8 | Instrumentación MACs | 5 | 13/10 | 15/10 |
| A9 | Ajuste y validación ANN | 5 | 16/10 | 20/10 |
| A10 | Diseño SNN (LIF, tasa) | 4 | 21/10 | 22/10 |
| A11 | Implementación SNN en snnTorch | 12 | 23/10 | 03/11 |
| A12 | Entrenamiento SNN (BPTT, gradiente sustituto) | 8 | 04/11 | 10/11 |
| A13 | Instrumentación SOPs | 5 | 11/11 | 13/11 |
| A14 | Ajuste y validación SNN | 5 | 16/11 | 18/11 |
| A15 | Diseño módulo de comparación | 3 | 19/11 | 20/11 |
| A16 | Experimentos comparativos | 5 | 23/11 | 25/11 |
| A17 | Análisis estadístico y gráficas | 4 | 26/11 | 27/11 |
| A18 | Informe final | 9 | 30/11 | 07/12 |
| A19 | README, licencias (paralelo) | 3 | 14/09 | 07/12 |
| A20 | Informes de avance quincenales (paralelo) | 5 | 14/09 | 07/12 |
| R1 | Reserva de riesgo | 12 | 08/12 | 14/12 |

> Nota: a la fecha (05/10/2026) A3–A5 aún no tienen código; el plan de implementación las
> aborda primero. El total de horas es provisional.

## Riesgos (Plan, sección 2.4)

| ID | Riesgo | Prob. | Impacto (h) | Mitigación en la implementación |
|---|---|---|---|---|
| RSG01 | Conflictos de tiempo del estudiante | 0.35 | 6 | Pasos pequeños y documentados; bitácora para retomar. |
| RSG02 | Incompatibilidades/curva de aprendizaje de snnTorch/RDKit | 0.40 | 8 | Fijar versiones en requisitos desde la fase 0; SnnClassifier aísla la dependencia (SpikingJelly como reemplazo). |
| RSG03 | Conteo de SOPs no estandarizado / no comparable con MACs | 0.50 | 10 | Métrica unificada con r̄ y r̄*, supuestos parametrizables, pruebas con spikes sintéticos. |
| RSG04 | BPTT no converge en CPU en tiempo | 0.30 | 8 | Lotes pequeños, T acotado, límite de puntos del barrido, MPS opcional para entrenar. |

## Literatura clave

- [5] Bosîi et al. (2026): fpCSNN (conv 1D + LIF), β=0.95, gradiente arctan, Tox21 7832 compuestos.
- [6] Nascimben y Rimondini (2023): SNN LIF cuantizada, MACCS/ECFP con codificación por tasa, Tox21/SIDER. **Metodología a replicar (OE3).**
- [7] Küppers (2024): SNN en Loihi 2 vs ANN en GPU; dispersidad MACCS 0.676, Morgan 1024 r2 0.949.
- [8] snnTorch (Eshraghian et al.). [9] SpikingJelly. [4] Loihi. [10] gem5.
