# 05 · Bitácora de progreso

**Leer esto primero en cada sesión nueva.** Actualizar al terminar cada paso.

## Estado actual

- **Fecha última actualización:** 05/10/2026
- **Fase actual:** planificación terminada; la mayoría de las dudas ya están decididas.
- **Siguiente paso:** Fase 0 · P0.1 (git init + `.gitignore`), luego P0.2 (borrar `snn/`,
  confirmar antes) y P0.3 (reparar `snnenv/`).
- **Código existente:** solo prototipos en `snn/` (se van a borrar). No hay código del paquete aún.

## Pendientes con el supervisor (Ronald García)

Llevar a la próxima reunión (bloquean la fase 5 · P5.1–P5.2):
1. **D-21** Energías relativas e_mac, e_ac, e_upd: propuesta Horowitz 2014 (45 nm, FP32:
   MAC ≈ 4.6 pJ, ADD ≈ 0.9 pJ → e_ac/e_mac ≈ 0.2). ¿Cómo modelar e_upd?
2. **D-23** Bytes por elemento (b_w, b_a, b_s, b_e, b_o = 4) y lote de inferencia B_lote = 1.
3. **D-24** ¿ANN solo en escenario von Neumann o también "neuromórfico" (pesos locales) para
   comparación simétrica? Definición de A (activaciones accedidas) y N_ops SNN = N_SOP + N_UPD.

## Dudas abiertas de menor prioridad (preguntar al llegar al paso)

D-07 (fuente de descarga, fase 2), D-09 (Morgan 2048 en el barrido), D-16 (validar MACs con
FlopCounterMode), D-22 (dependencia models→config), D-25 (layout de `results/`), D-30
(definición de tasa de disparo media), D-31 (conjunto/lote de perfilado).

## Historial

| Fecha | Sesión | Qué se hizo |
|---|---|---|
| 05/10/2026 | 1 | Lectura del Documento de Diseño v1.0 y del Plan de Proyecto. Creación de `.claude/` con contexto, diseño técnico, plan de implementación, dudas y bitácora. Resueltas 25 dudas con el usuario (ver tabla al final de `04_dudas_y_decisiones.md`). |

## Checklist por fase

- [ ] Fase 0 · Infraestructura
- [ ] Fase 1 · Configuración
- [ ] Fase 2 · Datos (ENT01)
- [ ] Fase 3 · ANN + MACs (ENT02)
- [ ] Fase 4 · SNN + SOPs (ENT03)
- [ ] Fase 5 · Tráfico y métricas (parcialmente bloqueada por el supervisor)
- [ ] Fase 6 · Comparación, Runner, CLI (ENT04)
- [ ] Fase 7 · Barrido y Pareto
- [ ] Fase 8 · Cierre (ENT06, insumos ENT05; actualizar el Documento de Diseño con los cambios: inglés, Fig. 8, Tabla 4)

## Notas

- `snnenv/` fue creado en `/Users/cob/Documents/tec/snnenv` y movido aquí: los shebangs de
  `bin/` apuntan a la ruta vieja. Usar `snnenv/bin/python -m pip ...` o repararlo (D-03).
- Cronograma: A7 (implementación ANN) empieza oficialmente el 05/10/2026, pero A3–A5 (datos) no
  tienen código → atraso de ~1 semana respecto al plan; la reserva R1 (08–14/12) lo cubre.
- Cambios al Documento de Diseño a registrar en su historia de versiones: identificadores en
  inglés, notación de la Fig. 8 (D-20), contradicción de la Tabla 4 (D-22).
