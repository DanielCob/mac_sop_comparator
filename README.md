# mac_sop_comparator

[![tests](https://github.com/DanielCob/mac_sop_comparator/actions/workflows/tests.yml/badge.svg)](https://github.com/DanielCob/mac_sop_comparator/actions/workflows/tests.yml)
[![licencia: MIT](https://img.shields.io/badge/licencia-MIT-blue.svg)](LICENSE)

Herramienta en Python que compara la eficiencia teórica de una **red neuronal densa (ANN)** y de una **red neuronal de
impulsos (SNN) homóloga**, ambas entrenadas para clasificar firmas moleculares de MoleculeNet. La ANN se mide en
multiplicaciones-acumulaciones (**MAC**) y la SNN en operaciones sinápticas (**SOP**), que solo ocurren cuando llega un
impulso. La herramienta entrena las dos redes, cuenta sus operaciones, estima el tráfico de memoria y la energía con
parámetros configurables, y compara el costo contra la exactitud (frontera de Pareto, sensibilidad y prueba estadística).

Proyecto CE-1114 (Proyecto de Aplicación de la Ingeniería en Computadores), Instituto Tecnológico de Costa Rica,
II Semestre 2026. Autor: Daniel Cob Beirute. Persona supervisora: Ronald García Fernández.

## Qué se compara

| | ANN | SNN |
|---|---|---|
| Arquitectura | MLP denso con ReLU | Las mismas capas lineales con neuronas LIF (snnTorch) durante `T` pasos de tiempo |
| Entrada | Firma molecular (MACCS 166 bits o Morgan/ECFP) | La misma firma, repetida en los `T` pasos (codificación por tasa) |
| Salida | 2 neuronas, clase por `argmax` | 2 neuronas, clase por conteo de impulsos |
| Operación contada | MAC: una multiplicación por cada peso | SOP: una suma por cada peso alcanzado por un impulso; más actualizaciones de estado de las neuronas |
| Entrenamiento | Adam, entropía cruzada ponderada por clase | Igual, con gradiente sustituto (arctan) y BPTT |

Las dos redes se construyen con la misma semilla, así que **parten de los mismos pesos iniciales**, y se evalúan en la misma
partición. Las métricas principales:

- **r** (tasa efectiva) = SOPs / (T · MACs): qué fracción de las sinapsis de la ANN trabaja la SNN.
- **r\*** (tasa crítica): el valor de r por debajo del cual la SNN cuesta menos que la ANN con las energías configuradas.
- **Costo de cómputo** = MACs · `e_mac` (ANN) frente a SOPs · `e_ac` + actualizaciones · `e_upd` (SNN).
- **Energía total** = costo de cómputo + `e_byte` · bytes movidos. La ANN se evalúa en un escenario von Neumann (los pesos
  vienen de memoria principal) y la SNN en un escenario neuromórfico (los pesos son locales y solo viajan eventos).
- **Intensidad aritmética** = operaciones por byte movido.

## Instalación

Requiere **Python 3.13 o superior**. Se probó en macOS con Apple Silicon (aceleración MPS) y en Linux (solo CPU).

```bash
git clone https://github.com/DanielCob/mac_sop_comparator.git
cd mac_sop_comparator
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

En **Linux** conviene instalar antes la versión de PyTorch para CPU, porque la que trae `pip` por defecto incluye las
bibliotecas CUDA (varios GB):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

`requirements.lock` fija las versiones exactas con las que se obtuvieron los resultados (`pip install -r requirements.lock`).

La primera vez que se ejecuta `prepare` (o cualquier comando que lo necesite) se descarga Tox21 desde el repositorio de
MoleculeNet/DeepChem, se verifica su SHA-256 y se guarda en `data_raw/`. No hace falta nada más.

## Uso

Todos los comandos se ejecutan desde la raíz del repositorio con un archivo de configuración YAML (ver
[`configs/`](configs)):

```bash
python -m mac_sop_comparator <comando> --config configs/tox21_sr_are.yaml [--results-dir results]
```

| Comando | Qué hace |
|---|---|
| `prepare` | Descarga y limpia el conjunto, genera las firmas y la partición entrenamiento/validación/prueba |
| `train --model ann\|snn` | Entrena y evalúa una red; guarda el modelo, las métricas, la curva de entrenamiento y su conteo de operaciones |
| `profile` | Vuelve a medir las operaciones de los modelos ya entrenados |
| `compare` | Métricas unificadas, tabla y gráficos de un punto |
| `run` | `prepare` + `train` (ANN y SNN) + `profile` + `compare` |
| `sweep [--force]` | Barrido del espacio de diseño (T, firma, topología, semillas); calcula la frontera de Pareto y la sensibilidad |
| `analyze` | Tablas, figuras y la prueba estadística ANN–SNN a partir de resultados ya guardados; no entrena |

Un punto completo, de datos a informe (menos de un minuto con MPS):

```bash
python -m mac_sop_comparator run --config configs/tox21_sr_are.yaml
```

`sweep` retoma donde quedó: salta los puntos que ya tienen resultados entrenados con **la misma configuración** y vuelve
a entrenar los que cambiaron o quedaron a medias. `--force` los rehace todos. Los códigos de salida son 0 (correcto),
1 (falló una etapa o un punto) y 2 (configuración inválida).

## Configuración

Un solo YAML describe el experimento (ver [`configs/tox21_sr_are.yaml`](configs/tox21_sr_are.yaml), comentado):

| Sección | Contenido |
|---|---|
| `experiment` | Nombre, semilla, dispositivo (`mps` o `cpu`) e hilos |
| `data` | Conjunto y tarea de MoleculeNet, firma (`maccs` o `morgan` con `n_bits` y `radius`) y proporciones de la partición |
| `model` | Capas ocultas, salidas, y la SNN: `T`, `beta`, umbral, función sustituta y reinicio |
| `training` | Optimizador, tasa de aprendizaje, lote, épocas máximas, paciencia y ponderación de clases |
| `architecture` | Energías relativas (`e_mac`, `e_ac`, `e_upd`, `e_byte_von_neumann`, `e_byte_neuromorphic`), bytes por elemento, lote de inferencia y **la fuente de cada valor** |
| `sweep` | Valores de `T`, firmas, `n_bits`, topologías y semillas, y el tope de puntos |

**Ninguna energía ni parámetro de memoria está fijo en el código**: no existen valores válidos para todas las
arquitecturas (dependen del nodo tecnológico, de la microarquitectura, etc.). Todos son configurables, la sección
`architecture.sources` guarda de dónde sale cada uno, los informes la copian, y los comandos avisan si falta alguna fuente.
Los valores de los archivos de ejemplo se expresan relativos a un MAC (`e_mac = 1`) y salen de Horowitz (ISSCC 2014, 45 nm)
para las operaciones y la DRAM, y de Davies et al. (2018, Loihi) para el costo por byte del escenario neuromórfico. Combinan
dos tecnologías distintas, así que son una aproximación y **se recomienda barrerlos** (el comando `sweep` ya genera las
gráficas de sensibilidad).

## Reproducir los resultados

Los resultados se obtuvieron con tres comandos. Los tiempos son de un MacBook con Apple Silicon y MPS.

```bash
# 1. Barrido de 48 puntos con una semilla: 4 valores de T x 3 firmas x 4 topologías (unos 15 minutos)
python -m mac_sop_comparator sweep --config configs/tox21_sr_are.yaml

# 2. Los dos puntos principales (MACCS 166 y Morgan 1024, una capa oculta de 1000, T = 10) con 30 semillas (unos 20 minutos)
python -m mac_sop_comparator sweep --config configs/tox21_sr_are_principales.yaml

# 3. Tablas, figuras y prueba estadística de cada experimento (segundos; no entrena)
python -m mac_sop_comparator analyze --config configs/tox21_sr_are.yaml
python -m mac_sop_comparator analyze --config configs/tox21_sr_are_principales.yaml
```

En un portátil con batería conviene envolver los comandos largos en `caffeinate -i` (macOS) para que el equipo no entre en
reposo; si algo se interrumpe, basta con volver a lanzar el mismo comando.

Qué esperar al terminar:

- `results/tox21_sr_are/sweep/` y `results/tox21_sr_are_principales/sweep/`: `points.csv` (una fila por punto),
  `pareto.csv` y `pareto_cost.png`/`pareto_energy.png` (frontera de Pareto), `sensitivity_e_upd.png` y
  `sensitivity_e_byte.png`, `statistics.csv`/`statistics.png` (media ± desviación sobre las semillas) y `failed.csv` solo si
  algún punto falla.
- `results/<experimento>/analysis/`: `summary.csv` y `summary.md` (la tabla del informe), `comparison.csv` (diferencia ANN–SNN
  con su intervalo de confianza del 95 % y la prueba de Wilcoxon pareada), `accuracy_gap.png`, `cost_vs_T.png`,
  `rates_vs_T.png` y `energy_breakdown.png`.

**Sobre la reproducibilidad.** La semilla fija la partición de los datos, los pesos iniciales y el orden de los lotes; la
codificación de entrada no tiene azar. Con la misma semilla, dispositivo y versiones de las bibliotecas se obtiene el mismo
resultado, pero **MPS y CPU dan resultados ligeramente distintos** (por ejemplo, el conteo de SOPs de una misma red difirió
en un 1.4 %). Por eso el dispositivo forma parte de la configuración y de la clave que decide si un resultado guardado se
puede reutilizar.

### Resultados de referencia

Tox21 · SR-ARE, una capa oculta de 1000 neuronas, T = 10, 30 semillas (media ± desviación estándar), energías de los
archivos de ejemplo:

| | MACCS 166 | Morgan 1024 |
|---|---|---|
| ROC-AUC de prueba, ANN | 0.805 ± 0.023 | 0.757 ± 0.030 |
| ROC-AUC de prueba, SNN | 0.795 ± 0.023 | 0.736 ± 0.029 |
| Brecha ANN − SNN [IC 95 %], Wilcoxon | 0.010 [0.005, 0.014], p = 0.0002 | 0.021 [0.015, 0.027], p = 1.6e-7 |
| Operaciones: MACs / SOPs | 168 000 / 305 576 | 1 026 000 / 267 245 |
| Tasa efectiva r (crítica r\*) | 0.182 (0.491) | 0.026 (0.499) |
| Costo de cómputo SNN / ANN | 0.38 | 0.055 |
| Energía total SNN / ANN | 0.0019 | 0.00027 |

Con MACCS la SNN hace más operaciones que la ANN y aun así cuesta menos, solo porque una suma cuesta una fracción de una
multiplicación. La brecha de exactitud es pequeña pero estadísticamente distinguible. La energía total la domina el
tráfico de memoria de la ANN, no su cómputo: depende sobre todo de la asignación de escenarios (pesos en DRAM frente a
pesos locales) y de `e_byte`, no de las operaciones.

## Estructura de `results/`

```
results/<experimento>/
├── run.log
├── data/<firma>_s<semilla>/              datos preparados (la semilla fija la partición)
├── <firma>_h<capas>_T<T>_s<semilla>/     un punto, p. ej. morgan1024_r2_h1000_T10_s42
│   ├── {ann,snn}.pt, {ann,snn}_metrics.json, {ann,snn}_curve.png, opreport_{ann,snn}.json
│   └── report/                           metrics.csv, opreports.json y los gráficos del punto
├── sweep/                                resultados consolidados del barrido
└── analysis/                             tablas y figuras de `analyze`
```

`results/` y `data_raw/` no se versionan.

## Estructura del código

```
src/mac_sop_comparator/
├── config/        esquema pydantic del YAML y su validación
├── data/          descarga, limpieza, firmas (RDKit), partición y datos preparados
├── models/        ANN, SNN LIF, codificador de impulsos, fábrica y checkpoints
├── training/      entrenador (parada temprana), evaluador y curvas
├── profiling/     conteo de MACs y SOPs, modelo de tráfico de memoria y OpReport
├── comparison/    métricas, informes, Pareto, sensibilidad, estadística y figuras
├── experiments/   CLI, ejecutor de un punto (máquina de estados), barrido y rutas de resultados
└── utils/         semillas, dispositivo y registro
```

## Pruebas

```bash
python -m pytest -q
```

La suite (unas 450 pruebas) corre en menos de dos minutos. Las pruebas que necesitan MPS, los datos reales de Tox21 o un
modelo ya entrenado se omiten solas cuando no están disponibles. GitHub Actions ejecuta la suite en cada `push` a `main` y
en cada *pull request* (Ubuntu, Python 3.13, PyTorch para CPU).

## Limitaciones

- Es una **simulación teórica**: las energías son relativas y vienen de la literatura, no de mediciones en hardware, y no se
  mide tiempo de pared.
- Se evalúa una sola tarea (Tox21 · SR-ARE) y una sola partición por semilla.
- Se replica la arquitectura de Nascimben y Rimondini (2023), pero **no su protocolo**: aquí se usa pérdida ponderada por
  clase en lugar de sobremuestreo, una partición estratificada por semilla en lugar de validación cruzada anidada, y
  parada temprana (máximo 100 épocas). Sus exactitudes del 98–99 % en Tox21 no se toman como referencia de exactitud.
- El número de pasos de tiempo `T` no está fijado por esa publicación; se usa `T = 10` (Bosîi et al.) y se barre 4, 8, 10 y 16
  (Küppers).

## Licencia

El código de este repositorio se distribuye bajo la licencia [MIT](LICENSE).

### Dependencias

| Paquete | Licencia |
|---|---|
| [PyTorch](https://pytorch.org) | BSD-3-Clause |
| [snnTorch](https://github.com/jeshraghian/snntorch) | MIT |
| [RDKit](https://www.rdkit.org) | BSD-3-Clause |
| NumPy, pandas, scikit-learn, SciPy | BSD-3-Clause |
| pydantic, PyYAML | MIT |
| Matplotlib | Licencia de Matplotlib (basada en PSF) |
| certifi | MPL-2.0 |
| pytest (desarrollo) | MIT |

### Datos

Tox21 es una colaboración del NIH (NCATS y NIEHS/NTP), la EPA y la FDA; la copia usada aquí la distribuyen MoleculeNet y
DeepChem (`deepchemdata.s3-us-west-1.amazonaws.com`). Los datos **no se incluyen en este repositorio**: se descargan al
ejecutar `prepare`. Consulte los términos de uso del conjunto original antes de redistribuirlo.

## Referencias

- Nascimben, M.; Rimondini, L. *Molecular Toxicity Virtual Screening Applying a Quantized Computational SNN-Based Framework.*
  Molecules 2023, 28(3), 1342. Arquitectura y metodología replicadas.
- Bosîi, D.; Torres, L. H. M.; Arrais, J. P.; Ribeiro, B. *Convolutional Spiking Neural Networks with Molecular Fingerprints
  for Drug Discovery.* 2026. Pasos de tiempo, β, función sustituta y tarea SR-ARE.
- Küppers, P. *Exploring the Chemical Universe with Spiking Neural Networks.* Bachelor's thesis, 27 de junio de 2024.
  Valores de T, firmas y energía en Loihi 2.
- Horowitz, M. *1.1 Computing's energy problem (and what we can do about it).* IEEE ISSCC 2014. Energías de la suma, la
  multiplicación y el acceso a DRAM.
- Davies, M. et al. *Loihi: A Neuromorphic Manycore Processor with On-Chip Learning.* IEEE Micro 2018. Costo de una operación
  sináptica y de una actualización de neurona.
- Wu, Z. et al. *MoleculeNet: a benchmark for molecular machine learning.* Chemical Science 2018, 9, 513–530.
