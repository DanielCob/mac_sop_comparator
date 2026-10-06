# 02 · Diseño técnico (extraído del Documento de Diseño v1.0, 02/10/2026)

> **Decisiones que modifican el PDF** (ver `04_dudas_y_decisiones.md`): identificadores y YAML en
> inglés (D-04); entrada derivada de la firma, MACCS = 166 bits (D-05, D-08); N_SOP cuenta S_0 y
> no la capa de salida (D-20); pesos por clase + mejor checkpoint por validación (D-14, D-15);
> `ce_count_loss` con desempate por membrana (D-19); ANN cacheada en el barrido y 5 semillas en
> puntos clave (D-26, D-27). Tráfico/AI y energías pendientes del supervisor (D-21, D-23, D-24).

Estándar: IEEE 1016-2009. Este archivo resume todo lo necesario para implementar; ante
contradicción con el PDF, manda el PDF salvo que `04_dudas_y_decisiones.md` registre una
decisión distinta.

## 1. Casos de uso (perspectiva de Contexto)

Actor principal: Investigador (CLI). Actor secundario: MoleculeNet (provee datos).

| CU | Nombre | Entradas | Salidas | Relación |
|---|---|---|---|---|
| CU-01 | Preparar dataset | conjunto, tarea, tipo/tamaño firma, radio, semilla, proporciones | X (N×F binaria), y, índices de partición, resumen (n moléculas, desbalance, dispersidad) | — |
| CU-02 | Entrenar/evaluar ANN | firmas, particiones, topología, lr, épocas, lote, semilla | modelo (.pt), ROC-AUC, precisión balanceada, OpReport (MACs) | incluye CU-04 |
| CU-03 | Entrenar/evaluar SNN | + T, β, umbral | modelo, ROC-AUC, prec. balanceada, OpReport (SOPs, UPD, spikes/capa) | incluye CU-04 |
| CU-04 | Medir complejidad | modelo, lotes de prueba, params arquitectura | OpReport por inferencia (promedio sobre test), bytes en 2 escenarios de memoria | — |
| CU-05 | Comparar métricas | OpReports, métricas de precisión, params arquitectura | razón de ops, AI, energía, r̄*, Pareto; JSON/CSV/PNG | — |
| CU-06 | Explorar espacio de diseño | listas/rangos de params, límite de puntos | tabla por punto, Pareto, gráficos de sensibilidad | extiende CU-05 |

Detalles: CU-01 descarta moléculas inválidas. CU-02 ANN = MLP denso (variante Conv1D opcional),
Adam + entropía cruzada. CU-03 SNN = misma topología con LIF, codificación por tasa en T pasos,
BPTT + gradiente sustituto, clase decodificada por conteo de spikes de salida.

## 2. Composición (paquetes)

Pipeline orquestado por `experiments`. Dependencias **sin ciclos**:

| Paquete | Contenido | Depende de |
|---|---|---|
| `experiments` | ExperimentRunner, CLI, SweepManager | config, data, training, profiling, comparison (y models) |
| `config` | Esquema YAML + validación, ArchitectureParams | — |
| `data` | MoleculeNetLoader, FingerprintGenerator, Splitter, SpikeEncoder, MolecularDataset | config |
| `models` | BaseClassifier, AnnClassifier, SnnClassifier, ModelFactory | config |
| `training` | Trainer, Evaluator | models, data |
| `profiling` | OpProfiler, MacProfiler, SopProfiler, TrafficModel, OpReport | models, data |
| `comparison` | EfficiencyMetrics (MetricSet), ParetoAnalyzer, ReportBuilder | profiling, config |
| `results/` | directorio de artefactos (no es paquete, no se versiona) | — |

> Nota: la Tabla 4 dice "config y models no dependen de ningún otro paquete", pero también dice
> "models depende de config". Ver duda D-22.

## 3. Lógica (clases, Figuras 3 y 4)

### Datos, modelos, entrenamiento
- **FingerprintGenerator**: `tipo: MACCS|Morgan`, `n_bits: int`, `radio: int`; `generar(smiles) -> ndarray`.
- **MolecularDataset**: `-X: ndarray`, `-y: ndarray`; `__getitem__(i)`, `sparsity() -> float`. (FingerprintGenerator la *produce*.)
- **SpikeEncoder**: `T: int`; `codificar_tasa(x) -> Tensor` (forma (T, F) o (T, B, F)). *Transforma* el dataset.
- **BaseClassifier** (abstracta): `topologia: list[int]`; `forward(x)`, `capas() -> list`.
- **AnnClassifier** (hereda): `activacion: str`; `forward(x)`.
- **SnnClassifier** (hereda): `T`, `beta`, `umbral`; `forward(x)`, `conteo_spikes() -> dict`. *Usa* SpikeEncoder.
- **ModelFactory**: `construir(config) -> BaseClassifier` (crea ambos con la misma topología).
- **Trainer**: `optimizador: str`, `lr: float`, `epocas: int`; `entrenar(modelo, datos)`.
- **Evaluator**: `roc_auc(modelo, datos)`, `acc_balanceada(modelo, datos)`.

### Medición, comparación, ejecución
- **OpReport**: `macs|sops: int`, `actualizaciones: int`, `spikes_por_capa: dict`, `bytes_acceso: int`.
- **OpProfiler** (abstracta): `modelo: BaseClassifier`; `medir(datos) -> OpReport`.
- **MacProfiler**: cuenta MACs desde la topología (independiente del dato).
- **SopProfiler**: SOPs, UPD y spikes por capa vía *forward hooks* sobre las salidas de cada capa LIF.
- **TrafficModel**: bytes transferidos por escenario de memoria (no aparece en el diagrama de clases; sí en paquetes).
- **ArchitectureParams**: `bits_peso: int`, `e_mac, e_ac, e_upd: float`, `escenario_memoria: str`.
- **EfficiencyMetrics**: `calcular(rep, params) -> MetricSet`, `tasa_critica(params) -> float`.
- **ParetoAnalyzer**: frontera (costo, ROC-AUC) (no aparece en el diagrama de clases).
- **ReportBuilder**: `tabla(resultados)`, `grafico_pareto(resultados)`.
- **ExperimentRunner**: `config: dict`; `ejecutar()`, `barrido()`.

## 4. Patrones

| Patrón | Roles |
|---|---|
| Strategy | Contexto: Trainer/Evaluator; estrategias: AnnClassifier/SnnClassifier (y MacProfiler/SopProfiler). |
| Factory Method | ModelFactory crea AnnClassifier/SnnClassifier desde la misma topología (homología). |
| Observer (hooks) | Capas observadas, perfiladores observan con hooks de PyTorch sin tocar el modelo. |
| Template Method | `ExperimentRunner.ejecutar()`: preparar → entrenar → evaluar → medir → comparar. |

## 5. Interfaces (Tabla 8)

| Interfaz | Proveedor → Consumidor | Contrato |
|---|---|---|
| Datos | data → training, profiling | `cargar(config) -> (X, y, particiones)`; `lotes(particion, tamaño) -> iterador` |
| Clasificador | models → training, profiling | `forward(x)`, `capas()`, SNN: `conteo_spikes()` (spikes por capa y paso) |
| Perfilador | profiling → experiments | `medir(modelo, datos) -> OpReport` |
| Métricas | comparison → experiments | `calcular(reportes, parametros) -> MetricSet`; `tasa_critica(parametros) -> float` |
| Config | Investigador → experiments | YAML validado contra esquema |
| Resultados | comparison → Investigador | JSON (OpReport), CSV (métricas), PNG (gráficos) |

### Ejemplo de configuración — versión del PDF (claves en español, solo referencia)

```yaml
experimento: {nombre: tox21_sr_are, semilla: 42}
datos:
  conjunto: tox21
  tarea: SR-ARE
  firma: {tipo: morgan, n_bits: 1024, radio: 2}
  particion: {entrenamiento: 0.8, validacion: 0.1, prueba: 0.1}
modelo:
  topologia: [1024, 128, 64, 2]
  snn: {T: 10, beta: 0.95, umbral: 1.0, gradiente: arctan}
entrenamiento: {optimizador: adam, lr: 1.0e-4, lote: 16, epocas: 100}
arquitectura:
  bits_peso: 32
  e_mac: 1.0        # energías relativas (ilustrativas)
  e_ac: 0.2
  e_upd: 0.3
  escenario_memoria: [von_neumann, neuromorfico]
barrido: {T: [4, 8, 10, 16], firma: [maccs, morgan]}
```

### Configuración ACORDADA para la implementación (D-04 inglés, D-05, D-08, D-14, D-15, D-18, D-28)

```yaml
experiment: {name: tox21_sr_are, seed: 42, device: cpu, n_threads: 4}
data:
  dataset: tox21
  task: SR-ARE
  fingerprint: {type: morgan, n_bits: 1024, radius: 2}   # maccs -> 166 bits (sin bit 0)
  split: {train: 0.8, val: 0.1, test: 0.1, method: stratified}
model:
  hidden_layers: [128, 64]        # la entrada se deriva de la firma
  n_outputs: 2
  ann: {activation: relu}
  snn: {T: 10, beta: 0.95, threshold: 1.0, surrogate: arctan, reset: subtract}
training: {optimizer: adam, lr: 1.0e-4, batch_size: 16, epochs: 100, patience: 15, class_weights: true}
architecture:                     # valores ilustrativos; calibrar con supervisor (D-21, D-23, D-24)
  weight_bits: 32
  e_mac: 1.0
  e_ac: 0.2
  e_upd: 0.3
  memory_scenarios: [von_neumann, neuromorphic]
sweep: {T: [4, 8, 10, 16], fingerprint: [maccs, morgan], max_points: 50}
```

### Nombres en inglés de los métodos del diseño (D-04)

| Diseño (PDF) | Código |
|---|---|
| `generar(smiles)` | `generate(smiles)` |
| `codificar_tasa(x)` | `rate_encode(x)` |
| `capas()` | `layers()` |
| `conteo_spikes()` | `spike_counts()` |
| `construir(config)` | `build(config)` |
| `entrenar(modelo, datos)` | `train(model, data)` |
| `roc_auc`, `acc_balanceada` | `roc_auc`, `balanced_accuracy` |
| `medir(datos)` | `measure(data)` |
| `calcular(rep, params)` | `compute(reports, params)` |
| `tasa_critica(params)` | `critical_rate(params)` |
| `tabla`, `grafico_pareto` | `table`, `pareto_plot` |
| `ejecutar()`, `barrido()` | `run()`, `sweep()` |
| `cargar(config)`, `lotes(...)` | `load(config)`, `batches(split, size)` |
| OpReport: `macs, sops, actualizaciones, spikes_por_capa, bytes_global` | `macs, sops, updates, spikes_per_layer, bytes_global` |
| `--modelo ann\|snn` | `--model ann\|snn` |

### Esquema OpReport (JSON)

```json
{
  "arquitectura": "ANN | SNN",
  "por_inferencia": {
    "macs": int,                 // solo ANN
    "sops": float,               // solo SNN (promedio por muestra)
    "actualizaciones": float,    // solo SNN
    "spikes_por_capa": [float],  // solo SNN
    "bytes_global": {"von_neumann": float, "neuromorfico": float}
  }
}
```

## 6. Interfaz de usuario (CLI)

`python -m mac_sop_comparator <comando> --config <archivo.yaml>`

| Comando | Descripción | CU |
|---|---|---|
| `prepare` | firmas, etiquetas y particiones | CU-01 |
| `train --modelo ann\|snn` | entrena y evalúa | CU-02, CU-03 |
| `profile` | mide complejidad de los modelos entrenados | CU-04 |
| `compare` | métricas unificadas, tablas y gráficos | CU-05 |
| `sweep` | barrido del espacio de diseño | CU-06 |
| `run` | prepare + train + profile + compare | CU-01..05 |

Requisitos: informar progreso por etapa, validar config antes de ejecutar, errores que indiquen
el campo/recurso que falló. Sin GUI.

## 7. Estructura del repositorio

```
mac_sop_comparator/
├── configs/                  # YAML de experimentos
├── src/mac_sop_comparator/
│   ├── experiments/          # ExperimentRunner, CLI, SweepManager
│   ├── config/               # esquema YAML, ArchitectureParams
│   ├── data/                 # carga, firmas, particiones, codificación
│   ├── models/               # AnnClassifier, SnnClassifier, ModelFactory
│   ├── training/             # Trainer, Evaluator
│   ├── profiling/            # MacProfiler, SopProfiler, TrafficModel
│   └── comparison/           # EfficiencyMetrics, ParetoAnalyzer, ReportBuilder
├── results/                  # sin versionar
├── tests/                    # pruebas de conteo y de flujo
└── docs/                     # requerimientos y diseño
```

## 8. Datos persistentes (Figura 5 + Tabla 6)

Entidades: **Experimento** (id, config_yaml, semilla, fecha) —usa N:1→ **ConjuntoDatos**
(id, nombre, tarea, n_moleculas) —contiene 1:N→ **Firma** (id, smiles, tipo, n_bits, radio,
vector, etiqueta). Experimento —genera 1:2→ **CorridaModelo** (id, arquitectura ANN|SNN,
topologia, T/beta/umbral, checkpoint .pt) —1:1→ **ReporteOps** (macs|sops,
actualizaciones_estado, spikes_por_capa, bytes_accedidos) y —1:1→ **Metricas** (roc_auc,
acc_balanceada, intensidad_aritmetica, energia_estimada, tasa_disparo_media).

| Artefacto | Formato | Produce |
|---|---|---|
| Configuración | YAML | Investigador |
| Firmas, etiquetas, particiones | .npy/.npz | data |
| Checkpoints | .pt | training |
| Reporte de operaciones | JSON | profiling |
| Métricas | CSV | comparison |
| Gráficos | PNG | comparison |
| Bitácora | .log (semilla, versiones, eventos) | experiments |

Sin base de datos: todo en sistema de archivos local.

## 9. Interacción y estados

Secuencia `run` con barrido: Investigador → Runner `ejecutar(config.yaml)` → validar → data
(X, y, particiones) → models construir(ANN), construir(SNN) → training (modelos + ROC-AUC) →
profiling (OpReports) → comparison (tablas, gráficos) → reporte; pasos 5–11 se repiten por
punto. Solo el Runner habla con los demás paquetes.

Estados: Inactivo → Configurado → Datos preparados → Entrenando (ANN y luego SNN) → Evaluando
→ Midiendo complejidad → Comparando → (Configurado si quedan puntos | Reportando) → Inactivo.
Cualquier excepción → Error (registrar, liberar recursos) → Inactivo.

## 10. Algoritmos y métricas

Símbolos: L capas densas; N_l neuronas en la capa l (N_0 = F = tamaño de firma);
fan-out F_l = N_(l+1); T pasos; S_l[t] spikes de la capa l en el paso t (l=0: bits activos).

1. **Codificación por tasa:** la firma binaria se replica en T pasos (bit 1 dispara en cada paso). Determinista.
2. **MACs (ANN):** Lineal: `in·out`; Conv1D: `largo_sal·canales_sal·canales_ent·kernel`. Independiente del dato.
3. **SOPs (SNN):** cada spike de la capa l genera F_l SOPs; hooks en salidas LIF.
   Flujo (Fig. 8): U_l=0, SOP=UPD=0 → codificar → para t: para cada capa
   `I_l = W_l·s_(l-1)[t]; U_l = β·U_l + I_l; s_l[t] = (U_l ≥ θ)`; reset donde disparó;
   `SOP += Σ_l n_spikes(s_(l-1)[t])·F_l`; `UPD += Σ_l N_l` → dividir por tamaño de lote → OpReport.
   UPD se reporta como **cota superior** (hardware por eventos podría hacer menos).
4. **Métricas unificadas (Tabla 12):**

| Métrica | Definición |
|---|---|
| N_MAC | Σ_{l=0}^{L-1} N_l·N_{l+1} |
| N_SOP | promedio sobre test de Σ_t Σ_l S_l[t]·F_l |
| N_UPD | T·Σ_{l=1}^{L} N_l |
| r̄ | N_SOP / (T·N_MAC), en [0,1] |
| Costo | C_ANN = N_MAC·e_mac; C_SNN = N_SOP·e_ac + N_UPD·e_upd |
| r̄* | (e_mac·N_MAC − e_upd·N_UPD) / (T·e_ac·N_MAC); SNN más barata si r̄ < r̄* |
| AI | N_ops / B_global. ANN von Neumann: B = b_w·P/B_lote + b_a·A. SNN von Neumann: B = b_w·N_SOP + 2·b_s·N_UPD. SNN neuromórfica: B = b_e·S_0 + b_o·N_L |
| Pareto | no dominados en (costo, ROC-AUC) |

   (P: pesos; A: activaciones accedidas; B_lote: lote que reutiliza pesos; b_w, b_a, b_s, b_e,
   b_o: bytes por peso, activación, estado, evento, salida; S_0: spikes de entrada; N_L:
   neuronas de salida.)

5. **Barrido:** producto cartesiano de `config.barrido`; por punto: construir ANN y SNN,
   entrenar ambos, medir, calcular; misma semilla en todos los puntos; devuelve resultados +
   frontera de Pareto.

Supuestos explícitos y parametrizables: valores de e_mac, e_ac, e_upd y la equivalencia MAC~SOP
(a validar con el supervisor y calibrar con literatura).

## 11. Recursos

CPU multinúcleo, ejecución secuencial con hilos fijos; RAM ≥ 8 GB; disco ≥ 5 GB; GPU opcional
(solo acelerar entrenamiento, nunca para medir); lote 16; T acotado; límite de puntos del
barrido configurable; bitácora con semilla y versiones. **El tiempo de pared no es métrica de
comparación.**

## 12. Bibliotecas

Python + PyTorch, snnTorch, RDKit, scikit-learn, NumPy, pandas, PyYAML, matplotlib.
Versiones exactas se fijan al iniciar la implementación. No se usa Lava-dl.
