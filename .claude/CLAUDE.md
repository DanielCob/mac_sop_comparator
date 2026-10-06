# Comparador de eficiencia MAC-SOP — Guía para sesiones de Claude

Proyecto CE-1114 (TEC, II Semestre 2026) de Daniel Cob Beirute.
"Simulación comparativa de eficiencia teórica entre arquitectura neuromórfica y tradicional
para clasificación espacial molecular". Producto: herramienta Python que compara una ANN densa
(MACs) contra una SNN homóloga LIF (SOPs) sobre firmas moleculares de MoleculeNet.

## Cómo trabajar en este proyecto (reglas acordadas con el usuario)

1. **Implementación paso a paso.** Se implementa una tarea pequeña por vez siguiendo
   `docs/03_plan_implementacion.md`. No adelantar fases.
2. **Consultar todas las dudas de implementación** antes de escribir código cuando algo no
   esté decidido en `docs/04_dudas_y_decisiones.md`. Si se decide algo nuevo, registrarlo ahí.
3. **Mantener la documentación viva:** al terminar un paso, actualizar
   `docs/05_bitacora_progreso.md` (estado + qué sigue) y, si cambió el diseño,
   `docs/02_diseno_tecnico.md`.
4. Las fuentes de verdad son **solo** dos PDFs en la carpeta padre:
   `../Documento de Diseño.pdf` (v1.0, 02/10/2026) y `../plan de proyecto.pdf` (18/09/2026).
   Ignorar los demás documentos salvo que el usuario diga lo contrario.
5. Idioma: documentación y comunicación en español.

## Índice de documentos

| Archivo | Contenido |
|---|---|
| `docs/01_contexto_proyecto.md` | Resumen del problema, objetivos, entregables, cronograma y riesgos (Plan de Proyecto). |
| `docs/02_diseno_tecnico.md` | Diseño técnico extraído del Documento de Diseño: paquetes, clases, interfaces, config YAML, OpReport, fórmulas, CLI, estructura. |
| `docs/03_plan_implementacion.md` | Plan de implementación por fases y pasos, con criterios de aceptación y pruebas. |
| `docs/04_dudas_y_decisiones.md` | Dudas detectadas, propuesta por defecto y decisión tomada (registro de decisiones). |
| `docs/05_bitacora_progreso.md` | Estado actual, pasos completados y siguiente paso. **Leer primero.** |

## Estado rápido

Ver `docs/05_bitacora_progreso.md`.
