# 04 · Dudas y decisiones

Registro de todo lo que los PDFs no dejan claro o en lo que se contradicen. Cada duda tiene una
**propuesta por defecto** (lo que se haría si no hay respuesta) y una **decisión** (lo que el
usuario resolvió). Estados: `ABIERTA` (se pregunta al llegar al paso), `DECIDIDA`, `CONSULTAR SUPERVISOR`.

> Regla: no implementar un paso que dependa de una duda `ABIERTA` sin preguntar antes.

---

## A. Repositorio y entorno

### D-01 · Control de versiones
- **Duda:** la carpeta no es repo git; ENT06 exige historial activo. ¿Remoto en GitHub? ¿Público o privado?
- **Propuesta:** `git init` en `mac_sop_comparator/`, rama `main`, remoto GitHub privado (público al final).
- **Estado:** DECIDIDA — git en `mac_sop_comparator/` + remoto GitHub (crear el remoto requiere confirmación del usuario al momento).

### D-02 · Prototipos en `snn/`
- **Duda:** `snn/benchmark.py`, `heavy_benchmark.py`, `snn_starter.py`, `test.py` son pruebas de
  capacidad de la Mac. ¿Se conservan?
- **Propuesta:** moverlos a `prototypes/` (versionados, fuera del paquete).
- **Estado:** DECIDIDA — borrar `snn/` (confirmar con el usuario justo antes de borrar).

### D-03 · Entorno virtual y versión de Python
- **Duda:** `snnenv/` fue creado en `/Users/cob/Documents/tec/snnenv` y luego movido; los
  scripts de `bin/` apuntan a la ruta vieja, así que probablemente está roto. Tiene Python
  3.13, torch 2.14, snntorch 1.0, rdkit; le faltan scikit-learn, pandas, PyYAML, matplotlib, pytest.
- **Propuesta:** crear `.venv/` nuevo dentro del repo (Python 3.13), dependencias declaradas en
  `pyproject.toml` y versiones exactas congeladas en `requirements.lock`; borrar `snnenv/`
  cuando el nuevo funcione.
- **Estado:** DECIDIDA — **reparar `snnenv/`** (no crear venv nuevo): corregir rutas tras el movimiento (`pyvenv.cfg`/shebangs, o `python3.13 -m venv --upgrade`), usar siempre `python -m pip`, instalar lo que falta y fijar versiones en `requirements.lock`. `snnenv/` va en `.gitignore`.

### D-04 · Idioma de identificadores
- **Duda:** el diseño mezcla clases en inglés (`SnnClassifier`) con métodos y claves YAML en
  español (`codificar_tasa`, `medir`, `experimento:`, `datos:`).
- **Propuesta:** respetar el diseño tal cual: clases en inglés, métodos públicos y claves YAML en
  español como en el PDF; código interno/variables en español sin tildes.
- **Estado:** DECIDIDA — **todo en inglés**: clases, métodos (`rate_encode`, `measure`, `compute`, `critical_rate`…) y claves YAML (`experiment:`, `data:`, `model:`…). La documentación sigue en español. Actualizar el Documento de Diseño en la próxima versión.

### D-28 · Dispositivo de cómputo
- **Duda:** el diseño dice CPU y GPU opcional; la Mac tiene MPS (Apple Silicon).
- **Propuesta:** `dispositivo: cpu` por defecto (reproducibilidad, hilos fijos); `mps` opcional
  solo para entrenar. Los conteos no dependen del dispositivo.
- **Estado:** DECIDIDA — CPU por defecto; MPS opcional solo para entrenar.

## B. Datos

### D-06 · Conjunto de datos y tarea
- **Duda:** el diseño usa Tox21 tarea SR-ARE (y menciona BBBP); el plan menciona Tox21 o SIDER.
  ¿Una sola tarea binaria? ¿Cuál es la principal?
- **Propuesta:** Tox21 · SR-ARE como experimento principal (tarea única, binaria). BBBP como
  segundo conjunto opcional si sobra tiempo. Multitarea fuera de alcance.
- **Estado:** DECIDIDA — Tox21 · SR-ARE (tarea binaria única). BBBP opcional si sobra tiempo.

### D-07 · Fuente de descarga de MoleculeNet
- **Duda:** el diseño no dice cómo se obtiene el CSV.
- **Propuesta:** descargar el CSV oficial (`tox21.csv.gz` del bucket de DeepChem) una vez,
  cachearlo en `data_raw/` y verificar su hash; no depender de la biblioteca DeepChem.
- **Estado:** ABIERTA

### D-08 · MACCS: 166 o 167 bits
- **Duda:** RDKit entrega 167 posiciones con el bit 0 sin uso. La topología y la dispersidad
  dependen de esto.
- **Propuesta:** descartar el bit 0 → F = 166 (coincide con el glosario "166 bits útiles").
- **Estado:** DECIDIDA — 166 bits (se descarta el bit 0).

### D-09 · Morgan 1024 vs ECFP 2048
- **Duda:** el plan habla de "1024/2048 bits ECFP"; el diseño fija Morgan 1024 r=2.
- **Propuesta:** 1024 r=2 como principal; 2048 solo como punto opcional del barrido (n_bits es parámetro).
- **Estado:** ABIERTA

### D-10 · Política de duplicados (criterio ENT01)
- **Duda:** ¿duplicado = mismo SMILES canónico o misma firma? Con MACCS muchas moléculas
  distintas comparten firma y a veces con etiquetas opuestas.
- **Propuesta:** (1) eliminar duplicados por SMILES canónico (si las etiquetas difieren, eliminar
  ambos); (2) **no** eliminar firmas repetidas, pero reportar cuántas hay y cuántas tienen
  etiqueta en conflicto.
- **Estado:** DECIDIDA — duplicados por SMILES canónico (si las etiquetas difieren se eliminan ambos); firmas repetidas se conservan y se reportan (cuántas y cuántas con conflicto).

### D-11 · Tipo de partición
- **Duda:** el diseño dice "particiones estratificadas" (scikit-learn); MoleculeNet suele
  recomendar *scaffold split*, que da AUC más bajos pero más realistas.
- **Propuesta:** estratificada aleatoria 80/10/10 con semilla (como el diseño). Scaffold como opción futura.
- **Estado:** DECIDIDA — estratificada aleatoria 80/10/10 con semilla.

### D-05 · Primer elemento de la topología
- **Duda:** `topologia: [1024, 128, 64, 2]` fija F=1024; con MACCS sería 166. En el barrido de
  firma, ¿cómo cambia?
- **Propuesta:** la entrada se deriva automáticamente de la firma; en el YAML solo se declaran
  capas ocultas y salida (o se valida que coincida). Las capas ocultas se mantienen iguales
  entre MACCS y Morgan.
- **Estado:** DECIDIDA — la entrada se deriva de la firma; el YAML declara `hidden_layers: [128, 64]` y `n_outputs: 2`.

## C. Modelos y entrenamiento

### D-17 · ¿Qué metodología SNN se replica exactamente?
- **Duda (importante):** el OE3 del plan dice **replicar Nascimben y Rimondini (2023)**, pero los
  parámetros del diseño (β=0.95, gradiente arctan, lote 16, lr 1e-4, 100 épocas) vienen de
  **Bosîi et al. [5]**. Nascimben usa una SNN "cuantizada". ¿Se replica la topología e
  hiperparámetros de Nascimben, o se usa el MLP homólogo del diseño con parámetros de Bosîi?
- **Propuesta:** MLP homólogo del diseño con parámetros de Bosîi (lo que dice el diseño v1.0);
  citar a Nascimben como base metodológica (LIF + tasa + huellas). Si el usuario tiene los
  hiperparámetros de Nascimben, agregarlos como configuración alternativa.
- **Estado:** DECIDIDA — MLP homólogo del diseño con parámetros de Bosîi (β=0.95, arctan, lr 1e-4, lote 16); Nascimben se cita como base metodológica.

### D-12 · Variante Conv1D
- **Duda:** el diseño la marca "opcional" en ANN y en el conteo de MACs.
- **Propuesta:** fuera del MVP; implementar solo si sobra tiempo tras la fase 7. El MacProfiler sí
  soporta Conv1D desde el inicio (es barato).
- **Estado:** DECIDIDA — solo si sobra tiempo tras la fase 7; MacProfiler soporta Conv1D desde el inicio.

### D-13 · Capa de salida: 2 neuronas o 1
- **Duda:** topología termina en 2 (entropía cruzada de 2 clases). En la SNN la clase se decide
  por conteo de spikes entre 2 neuronas.
- **Propuesta:** 2 neuronas en ambos modelos (homología y conteo de spikes directo).
- **Estado:** DECIDIDA — 2 neuronas de salida en ambos modelos.

### D-32 · Activación de la ANN
- **Propuesta:** ReLU en capas ocultas, sin dropout ni batch norm (mantener homología pura).
- **Estado:** DECIDIDA — ReLU, sin dropout ni batch norm.

### D-14 · Desbalance de clases
- **Duda:** Tox21 SR-ARE tiene ~16 % de positivos. ¿Pérdida ponderada?
- **Propuesta:** entropía cruzada con pesos por clase (inverso de la frecuencia en train) en ambos modelos.
- **Estado:** DECIDIDA — entropía cruzada ponderada por clase (inverso de frecuencia en train), en ambos modelos.

### D-15 · Épocas, parada temprana y selección del modelo
- **Duda:** 100 épocas fijas con lr 1e-4; no se dice si se usa validación para elegir el checkpoint.
- **Propuesta:** guardar el checkpoint con mejor ROC-AUC de validación; parada temprana con
  paciencia 15 épocas (configurable); reportar métricas finales en test.
- **Estado:** DECIDIDA — checkpoint con mejor ROC-AUC de validación + parada temprana con paciencia 15 (configurable); métricas finales en test.

### D-18 · Mecanismo de reset LIF
- **Duda:** la Fig. 8 dice "reiniciar U_l donde hubo disparo" sin decir si a cero o restando θ.
- **Propuesta:** `reset_mechanism="subtract"` (default de snnTorch), configurable.
- **Estado:** DECIDIDA — `reset_mechanism="subtract"`, configurable.

### D-19 · Pérdida SNN y score para ROC-AUC
- **Duda:** "entropía cruzada sobre el conteo de spikes de salida". Con T pequeño hay empates en
  el conteo, lo que degrada el ROC-AUC.
- **Propuesta:** pérdida `snntorch.functional.ce_count_loss` (o `ce_rate_loss`); score para AUC =
  softmax(conteo de spikes); en empates, desempatar con el potencial de membrana acumulado de
  salida. Clase predicha = argmax del conteo.
- **Estado:** DECIDIDA — `ce_count_loss`; score = softmax(conteo); desempate con potencial de membrana acumulado; clase = argmax.

### D-29 · Umbral para precisión balanceada
- **Propuesta:** argmax de las 2 salidas (equivale a umbral 0.5). Sin ajuste de umbral.
- **Estado:** DECIDIDA — argmax de las 2 salidas.

## D. Medición y métricas (núcleo del proyecto)

### D-20 · Notación del conteo de SOPs
- **Duda:** la Tabla 12 usa `S_l[t]·F_l` con F_l = N_(l+1) (fan-out de la capa emisora), pero la
  Fig. 8 escribe `n_spikes(s_(l-1)[t])·F_l`, mezclando índices. Además: ¿cuentan los spikes de
  entrada (S_0)? ¿y los de la capa de salida (no tienen fan-out)?
- **Propuesta:** N_SOP = Σ_t Σ_{l=0}^{L-1} S_l[t]·N_(l+1): sí cuentan los spikes de entrada, no los
  de salida. Corregir la notación de la Fig. 8 en la próxima versión del diseño.
- **Estado:** DECIDIDA — N_SOP = Σ_t Σ_{l=0}^{L-1} S_l[t]·N_(l+1): cuentan los spikes de entrada (S_0), no los de salida. Corregir la Fig. 8 en la próxima versión del diseño.

### D-16 · Validación del conteo de MACs
- **Duda:** ENT02 pide "herramienta de profiling reproducible"; el diseño cuenta desde la
  topología. ¿Se cuentan sesgos (bias) y activaciones?
- **Propuesta:** MACs = solo multiplicaciones-acumulaciones de pesos (sin bias ni ReLU), como la
  Tabla 12; validar contra `torch.utils.flop_counter.FlopCounterMode` (FLOPs = 2·MACs) en una prueba.
- **Estado:** ABIERTA

### D-21 · Energías relativas
- **Duda:** e_mac=1.0, e_ac=0.2, e_upd=0.3 son ilustrativos; "se calibrarán con la literatura".
- **Propuesta:** usar Horowitz (ISSCC 2014, 45 nm, FP32): MAC ≈ 4.6 pJ, ADD ≈ 0.9 pJ → e_ac/e_mac ≈
  0.2 (coincide con el valor ilustrativo). e_upd: modelarlo como 1 multiplicación por β + 1 suma
  + 1 comparación (≈ 0.3–1.0 relativo); dejarlo parametrizable y barrerlo.
- **Estado:** CONSULTAR SUPERVISOR — implementar con los valores ilustrativos (1.0/0.2/0.3) como parámetros; calibrar luego (Horowitz 2014 es la propuesta a llevar a la reunión).

### D-23 · Bytes por elemento y lote de inferencia
- **Duda:** la fórmula de AI usa b_w, b_a, b_s, b_e, b_o y B_lote sin valores.
- **Propuesta:** b_w = bits_peso/8 = 4; b_a = 4; b_s = 4 (potencial FP32); b_e = 4 (índice de
  evento de 32 bits); b_o = 4. B_lote de inferencia = 1 por defecto (peor caso von Neumann) y
  parametrizable.
- **Estado:** CONSULTAR SUPERVISOR — junto con D-24. La fase 5 (TrafficModel/AI) queda bloqueada hasta la reunión; las fases 0–4 no dependen de esto.

### D-24 · Escenarios de memoria por arquitectura y definición de N_ops
- **Duda:** el OpReport tiene `bytes_global` von_neumann y neuromórfico para ambas
  arquitecturas, pero la Tabla 12 no define la ANN en escenario neuromórfico. Tampoco define
  A (activaciones accedidas) ni si N_ops de la SNN es N_SOP o N_SOP + N_UPD.
- **Propuesta:** ANN solo en von Neumann (neuromórfico = `null`); A = Σ_l N_l (lectura de entradas
  + escritura de salidas de cada capa); N_ops ANN = N_MAC; N_ops SNN = N_SOP + N_UPD.
- **Estado:** CONSULTAR SUPERVISOR — llevar la propuesta y la alternativa "ANN también en escenario neuromórfico (pesos locales)" para comparación simétrica.

### D-30 · "tasa_disparo_media" en Métricas
- **Duda:** el diagrama ER guarda `tasa_disparo_media`; no se define si es r̄ o spikes por
  neurona por paso.
- **Propuesta:** guardar ambas: `r_efectiva` (r̄) y `tasa_disparo_por_capa` (spikes/(N_l·T)).
- **Estado:** ABIERTA

### D-31 · Conjunto y lote del perfilado
- **Propuesta:** perfilar sobre todo el conjunto de prueba, promediando por muestra; el tamaño de
  lote del perfilado no afecta los conteos (solo B_lote del modelo de tráfico).
- **Estado:** ABIERTA

### D-22 · Dependencia models → config
- **Duda:** Tabla 4 contradice "models no depende de ningún paquete".
- **Propuesta:** models recibe parámetros simples (listas, floats) y no importa `config`; la
  traducción config → argumentos la hace `ModelFactory`... que vive en models. Resolver: la
  factory recibe un objeto `ModeloCfg` (dataclass de config) → models depende de config.
  Corregir el texto del diseño.
- **Estado:** ABIERTA

## E. Experimentos

### D-25 · Organización de `results/` y comandos separados
- **Propuesta:** `results/<nombre_experimento>/<id_punto>/{datos.npz, ann.pt, snn.pt,
  opreport_ann.json, opreport_snn.json, metricas.csv, *.png, run.log}`; `id_punto` = hash corto
  de la config efectiva. Los comandos separados (`train`, `profile`, `compare`) encuentran los
  artefactos por ese hash.
- **Estado:** ABIERTA

### D-26 · Reutilizar la ANN en el barrido
- **Duda:** el algoritmo 5 entrena ANN y SNN en cada punto, pero la ANN no depende de T, β ni umbral.
- **Propuesta:** cachear la ANN por (firma, topología, entrenamiento, semilla) y reentrenar solo
  cuando cambie algo que la afecte. Ahorra mucho tiempo.
- **Estado:** DECIDIDA — cachear la ANN por (firma, topología, entrenamiento, semilla).

### D-27 · Varias semillas para el análisis estadístico
- **Duda:** A17 pide "análisis estadístico", pero el diseño usa una sola semilla por punto.
- **Propuesta:** configuración final con 5 semillas en los puntos principales → media ± desviación;
  el barrido completo con 1 semilla.
- **Estado:** DECIDIDA — 5 semillas en los puntos principales (media ± desviación); barrido completo con 1 semilla.

---

## Decisiones tomadas (resumen cronológico)

| Fecha | ID | Decisión |
|---|---|---|
| 05/10/2026 | D-01 | git en `mac_sop_comparator/` + remoto GitHub (crear el remoto requiere confirmación del usuario al momento). |
| 05/10/2026 | D-02 | borrar `snn/` (confirmar con el usuario justo antes de borrar). |
| 05/10/2026 | D-03 | **reparar `snnenv/`** (no crear venv nuevo): corregir rutas tras el movimiento (`pyvenv.cfg`/shebangs, o `python3.13 -m venv --upgrade`), usar siempre `python -m pip`, instalar lo que falta y fijar versiones en `requirements.lock`. `snnenv/` va en `.gitignore`. |
| 05/10/2026 | D-04 | **todo en inglés**: clases, métodos (`rate_encode`, `measure`, `compute`, `critical_rate`…) y claves YAML (`experiment:`, `data:`, `model:`…). La documentación sigue en español. Actualizar el Documento de Diseño en la próxima versión. |
| 05/10/2026 | D-28 | CPU por defecto; MPS opcional solo para entrenar. |
| 05/10/2026 | D-06 | Tox21 · SR-ARE (tarea binaria única). BBBP opcional si sobra tiempo. |
| 05/10/2026 | D-08 | 166 bits (se descarta el bit 0). |
| 05/10/2026 | D-05 | la entrada se deriva de la firma; el YAML declara `hidden_layers: [128, 64]` y `n_outputs: 2`. |
| 05/10/2026 | D-11 | estratificada aleatoria 80/10/10 con semilla. |
| 05/10/2026 | D-10 | duplicados por SMILES canónico (si las etiquetas difieren se eliminan ambos); firmas repetidas se conservan y se reportan (cuántas y cuántas con conflicto). |
| 05/10/2026 | D-17 | MLP homólogo del diseño con parámetros de Bosîi (β=0.95, arctan, lr 1e-4, lote 16); Nascimben se cita como base metodológica. |
| 05/10/2026 | D-12 | solo si sobra tiempo tras la fase 7; MacProfiler soporta Conv1D desde el inicio. |
| 05/10/2026 | D-13 | 2 neuronas de salida en ambos modelos. |
| 05/10/2026 | D-32 | ReLU, sin dropout ni batch norm. |
| 05/10/2026 | D-14 | entropía cruzada ponderada por clase (inverso de frecuencia en train), en ambos modelos. |
| 05/10/2026 | D-15 | checkpoint con mejor ROC-AUC de validación + parada temprana con paciencia 15 (configurable); métricas finales en test. |
| 05/10/2026 | D-18 | `reset_mechanism="subtract"`, configurable. |
| 05/10/2026 | D-19 | `ce_count_loss`; score = softmax(conteo); desempate con potencial de membrana acumulado; clase = argmax. |
| 05/10/2026 | D-29 | argmax de las 2 salidas. |
| 05/10/2026 | D-20 | N_SOP = Σ_t Σ_{l=0}^{L-1} S_l[t]·N_(l+1): cuentan los spikes de entrada (S_0), no los de salida. Corregir la Fig. 8 en la próxima versión del diseño. |
| 05/10/2026 | D-21 | implementar con los valores ilustrativos (1.0/0.2/0.3) como parámetros; calibrar luego (Horowitz 2014 es la propuesta a llevar a la reunión). |
| 05/10/2026 | D-23 | junto con D-24. La fase 5 (TrafficModel/AI) queda bloqueada hasta la reunión; las fases 0–4 no dependen de esto. |
| 05/10/2026 | D-24 | llevar la propuesta y la alternativa "ANN también en escenario neuromórfico (pesos locales)" para comparación simétrica. |
| 05/10/2026 | D-26 | cachear la ANN por (firma, topología, entrenamiento, semilla). |
| 05/10/2026 | D-27 | 5 semillas en los puntos principales (media ± desviación); barrido completo con 1 semilla. |
