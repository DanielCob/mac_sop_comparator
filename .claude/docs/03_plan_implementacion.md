# 03 · Plan de implementación

Orden guiado por las dependencias del diseño (config → data → models → training → profiling →
comparison → experiments) y por el cronograma del Plan (A3–A19). Cada paso es una unidad
pequeña: se implementa, se prueba, se registra en la bitácora y se consulta al usuario antes
de pasar al siguiente. Los IDs `D-xx` refieren a `04_dudas_y_decisiones.md`.

Convención: `[ ]` pendiente, `[~]` en curso, `[x]` hecho (el estado vivo está en la bitácora).

---

## Fase 0 · Infraestructura del repositorio (antes de A3–A5)

Objetivo: repositorio reproducible, instalable y con pruebas desde el día uno (ENT06).

- **P0.1** Inicializar git, `.gitignore` (results/, snnenv/, `__pycache__`, `.DS_Store`, data_raw/). Remoto GitHub (confirmar con el usuario). (D-01)
- **P0.2** Borrar `snn/` (confirmar con el usuario antes). (D-02)
- **P0.3** Reparar `snnenv/` (rutas tras el movimiento) e instalar lo que falta
  (scikit-learn, pandas, pyyaml, matplotlib, pytest). Usar `snnenv/bin/python -m pip`. (D-03)
- **P0.4** `pyproject.toml` con layout `src/`, instalación editable, `requirements.lock` con versiones exactas.
- **P0.5** Esqueleto de paquetes vacíos según la Estructura del diseño (`__init__.py`, `__main__.py`).
- **P0.6** Utilidades transversales: semillas (`torch`, `numpy`, `random`), número de hilos fijo,
  selección de dispositivo, logging a consola + `.log` con versiones de bibliotecas.
- **P0.7** `pytest` funcionando con una prueba trivial.

Criterio de cierre: `pip install -e .` + `python -m mac_sop_comparator --help` + `pytest` pasan.

## Fase 1 · Configuración (paquete `config`)

- **P1.1** Dataclasses del esquema (en inglés, D-04): `ExperimentConfig`, `DataConfig`, `FingerprintConfig`,
  `SplitConfig`, `ModelConfig`, `SnnConfig`, `TrainingConfig`, `ArchitectureParams`, `SweepConfig`, y `Config` raíz.
- **P1.2** Carga YAML + validación con mensajes que indiquen el campo que falla (proporciones
  suman 1, topología coherente con n_bits, T>0, 0<β≤1, etc.). (entrada derivada de la firma, D-05)
- **P1.3** `configs/tox21_sr_are.yaml` (la configuración acordada en `02_diseno_tecnico.md`).
- **P1.4** Pruebas: config válida carga; config inválida falla con el campo correcto.

## Fase 2 · Datos (paquete `data`) — A3, A4, A5 · ENT01

- **P2.1** `MoleculeNetLoader`: descargar/cachear CSV del conjunto, seleccionar tarea, descartar
  etiquetas faltantes. (D-06 dataset/tarea, D-07 fuente de descarga)
- **P2.2** `FingerprintGenerator`: SMILES → MACCS (166 vs 167) y Morgan (n_bits, radio);
  descarta inválidas y registra cuántas. (D-08 MACCS bit 0, D-09 ECFP 2048)
- **P2.3** Limpieza: duplicados (por SMILES canónico / por firma con etiquetas en conflicto). (D-10)
- **P2.4** `Splitter`: partición estratificada reproducible 80/10/10. (D-11 aleatoria vs scaffold)
- **P2.5** `MolecularDataset` (`__getitem__`, `sparsity()`) + `lotes(particion, tamaño)`.
- **P2.6** Persistencia `.npz` (X, y, índices) + resumen (n, desbalance, dispersidad) en JSON/log.
- **P2.7** `SpikeEncoder.rate_encode(x)` → (T, B, F) replicando la firma (determinista).
- **P2.8** Comando CLI `prepare`.
- **P2.9** Pruebas: muestra de control manual con RDKit (criterio ENT01), dispersidad cercana a
  0.676 (MACCS) y 0.949 (Morgan), sin NaN ni duplicados, particiones disjuntas y estables con
  la misma semilla, codificador replica exactamente.

## Fase 3 · ANN + MACs — A6, A7, A8, A9 · ENT02

- **P3.1** `BaseClassifier` (ABC sobre `nn.Module`): `topology`, `forward`, `layers()`.
- **P3.2** `AnnClassifier`: MLP denso desde la topología (activación configurable; sin dropout/BN
  salvo decisión). Conv1D opcional queda para después. (D-12 Conv1D, D-13 capa de salida)
- **P3.3** `ModelFactory.build(config, kind)` (ANN ahora; SNN en fase 4).
- **P3.4** `Trainer` (Adam, entropía cruzada, lote 16, épocas, selección por validación). (D-14 desbalance, D-15 early stopping)
- **P3.5** `Evaluator`: ROC-AUC y precisión balanceada.
- **P3.6** `OpReport` (dataclass + serialización JSON con el esquema del diseño).
- **P3.7** `OpProfiler` (ABC) + `MacProfiler` (recorre `capas()`; Linear y Conv1D).
  Validación cruzada contra `torch.utils.flop_counter.FlopCounterMode`. (D-16)
- **P3.8** Checkpoints `.pt` y CLI `train --model ann` y `profile` (parcial).
- **P3.9** Pruebas: MACs de capa lineal conocida (p. ej. 1024·128+128·64+64·2 = 139 392),
  entrenamiento de humo converge en subconjunto pequeño.
- **P3.10** Ajuste ligero y validación (A9): curva de pérdida estable en validación.

## Fase 4 · SNN + SOPs — A10, A11, A12, A13, A14 · ENT03

- **P4.1** `SnnClassifier`: misma topología con `nn.Linear` + `snn.Leaky` (β, umbral, gradiente
  arctan, mecanismo de reset), bucle sobre T, registra spikes por capa y paso. (D-17 Nascimben vs Bosîi, D-18 reset)
- **P4.2** Decodificación por conteo de spikes de salida; score para ROC-AUC. (D-19)
- **P4.3** Trainer en modo SNN (BPTT, pérdida sobre conteo de spikes). (D-19)
- **P4.4** `ModelFactory` construye ANN y SNN con la misma topología (prueba de homología: mismas
  formas de pesos).
- **P4.5** `SopProfiler`: forward hooks sobre cada capa LIF + spikes de entrada (S_0);
  N_SOP = Σ_t Σ_l S_l[t]·F_l promediado por muestra; N_UPD = T·Σ_{l≥1} N_l; spikes por capa.
  (D-20 notación/índices)
- **P4.6** CLI `train --model snn` y `profile` completo.
- **P4.7** Pruebas: SOPs con spikes sintéticos de conteo conocido (todo 1 → SOP = T·N_MAC,
  todo 0 → 0); r̄ ∈ [0,1]; implementación manual de la Fig. 8 coincide con el profiler.
- **P4.8** Ajuste y validación SNN (A14).

## Fase 5 · Tráfico de memoria y métricas unificadas — A15

> ⚠️ P5.1–P5.2 bloqueados por D-21/D-23/D-24 (reunión con el supervisor). P5.3 puede avanzar
> con N_MAC, N_SOP, N_UPD, r̄, costos y r̄* usando energías ilustrativas parametrizables.

- **P5.1** `ArchitectureParams` completos (bits por peso y bytes b_w, b_a, b_s, b_e, b_o; energías). (D-21, D-23)
- **P5.2** `TrafficModel`: B_global por arquitectura y escenario (von Neumann / neuromórfico). (D-24)
- **P5.3** `EfficiencyMetrics.compute` → `MetricSet` (N_MAC, N_SOP, N_UPD, r̄, C_ANN, C_SNN,
  razón de ops, AI, energía) y `critical_rate` (r̄*).
- **P5.4** Pruebas numéricas a mano de cada fórmula (incl. r̄* negativo cuando UPD domina).

## Fase 6 · Comparación, Runner y CLI completo — A15, A16 · ENT04

- **P6.1** `ReportBuilder`: CSV de métricas, JSON de OpReports, PNG (barras MAC vs SOP,
  AUC vs costo, r̄ vs r̄*).
- **P6.2** `ExperimentRunner.run()` (Template Method) con la máquina de estados del diseño,
  manejo de Error → registro y liberación.
- **P6.3** Layout de `results/` y localización de artefactos entre comandos separados. (D-25)
- **P6.4** CLI `compare` y `run`.
- **P6.5** Prueba de flujo end-to-end con un subconjunto pequeño y pocas épocas.

## Fase 7 · Barrido y Pareto — A16, A17

- **P7.1** `SweepManager`: producto cartesiano de `barrido` + límite de puntos. (D-26 reutilizar ANN)
- **P7.2** `ParetoAnalyzer` (no dominados en costo vs ROC-AUC) + gráficos de sensibilidad.
- **P7.3** CLI `sweep`, consolidación de resultados.
- **P7.4** Repeticiones con varias semillas para el análisis estadístico (si se decide). (D-27)
- **P7.5** Corrida de los experimentos finales (A16) y gráficas (A17).

## Fase 8 · Cierre — A19 (y apoyo a A18)

- **P8.1** README (instalación, reproducción paso a paso, comandos), licencias de terceros.
- **P8.2** Actualizar el Documento de Diseño a "lo que realmente se construyó" (historia de cambios).
- **P8.3** Exportar tablas/figuras para el informe final (ENT05).

---

## Trazabilidad rápida

| Entregable | Fases |
|---|---|
| ENT01 | 2 |
| ENT02 | 3 |
| ENT03 | 4 |
| ENT04 | 5, 6, 7 |
| ENT05 | 7, 8 (insumos) |
| ENT06 | 0, 8 (continuo) |
