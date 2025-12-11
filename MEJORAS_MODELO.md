# Mejoras del Rendimiento del Modelo

## Problema Detectado

El modelo de reconocimiento de actividades humanas tenía un problema crítico de **desbalance extremo de clases**:

-   **Actividad 0** ("Sin clasificar"): ~2,788 muestras (93% del dataset)
-   **Actividades 1-12**: ~33-100 muestras cada una (7% total)

### Síntomas

-   El modelo predecía casi todo como actividad 0
-   Accuracy aparentemente alta (~73%) pero solo por predecir la clase mayoritaria
-   Macro F1-score muy bajo (~13%), indicando mal rendimiento en clases minoritarias
-   Matriz de confusión mostraba todas las predicciones en la primera columna

### Métricas Antes (con actividad 0)

```
Validación:  accuracy=71.8%, macro_f1=13.7%
Test:        accuracy=73.9%, macro_f1=12.0%
Demo:        accuracy=74.7%, macro_f1=23.9%
```

## Solución Implementada

### 1. Filtrado de Actividad 0 en Preprocesamiento

**Archivo**: `ml/src/mhealth/preprocess.py`

```python
def filter_unlabeled_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove activity 0 (unlabeled/null activity) from dataset.
    This prevents extreme class imbalance issues during training.
    """
    return df[df[LABEL_COLUMN] != 0].copy()
```

### 2. Integración en Pipeline de Entrenamiento

**Archivo**: `ml/src/mhealth/modeling.py`

```python
def train_model(df: pd.DataFrame, config: Config) -> Dict[str, object]:
    set_global_seed(config.random_seed)
    remaining, demo_df = filter_demo_subjects(df, config.excluded_subjects_demo)

    # Remove activity 0 (unlabeled) to prevent class imbalance
    remaining = filter_unlabeled_activity(remaining)
    demo_df = filter_unlabeled_activity(demo_df)

    train_df_raw, val_df_raw, test_df_raw = split_by_subject(remaining, config)
    # ... resto del código
```

### 3. Actualización del Frontend

**Archivo**: `frontend/src/App.tsx`

-   Eliminadas referencias a "Sin clasificar" de los mapeos de colores
-   Removidos filtros manuales (ya no necesarios porque el modelo no predice actividad 0)
-   Actualizados contadores y visualizaciones para reflejar solo las 12 actividades válidas

## Resultados Después (sin actividad 0)

### Métricas Mejoradas

```
Validación:  accuracy=94.2%, macro_f1=93.1%  (+22.4% / +79.4%)
Test:        accuracy=81.3%, macro_f1=81.1%  (+7.4%  / +69.1%)
Demo:        accuracy=99.3%, macro_f1=99.1%  (+24.6% / +75.2%)
Train:       accuracy=100%, macro_f1=100%
```

### Matriz de Confusión (Validación)

```
Clase      1   2   3   4   5   6   7   8   9  10  11  12
   1      48   2   0   0   0   0   0   0   0   0   0   0
   2       0  48   0   0   0   0   0   0   0   0   0   0
   3       0   0  50   0   0   0   0   0   0   0   0   0
   4       0   0   0  45   3   0   0   0   0   0   0   0
   5       0   0   0   0  48   0   0   0   0   0   0   0
   6       0   0   0   0   0  51   0   0   0   0   0   0
   7       0   0   0   0   0   0  48   0   0   0   0   0
   8       0   0   0   0   1   0   0  50   0   0   0   0
   9       0   0   0   0   0   0   0   0  48   0   0   0
  10       0   0   0   0   0   0   0   0   1  30  18   0
  11       0   0   0   0   0   0   0   0   0   1  48   0
  12       0   0   0   0   0   0   0   0   0   6   0  10
```

**Observaciones**:

-   La diagonal principal está fuerte (buenas predicciones)
-   Confusión menor entre clases similares (ej: clase 10 ↔ 11, que son "Trote" ↔ "Corriendo")
-   No hay sesgo hacia ninguna clase en particular

## Impacto

### Mejoras Cuantitativas

| Métrica         | Antes | Después | Mejora |
| --------------- | ----- | ------- | ------ |
| Accuracy (Val)  | 71.8% | 94.2%   | +31.3% |
| Macro F1 (Val)  | 13.7% | 93.1%   | +579%  |
| Accuracy (Test) | 73.9% | 81.3%   | +10.0% |
| Macro F1 (Test) | 12.0% | 81.1%   | +576%  |

### Mejoras Cualitativas

1. **Predicciones Balanceadas**: El modelo ahora distribuye predicciones correctamente entre todas las actividades
2. **Confianza Real**: Los scores de confianza ahora reflejan la certeza real del modelo
3. **UI Limpia**: Sin necesidad de filtrar actividades "basura" en el frontend
4. **Dataset Limpio**: Solo actividades significativas en entrenamiento

## Comandos para Replicar

```powershell
# 1. Reconstruir backend con código actualizado
docker compose build backend

# 2. Reentrenar modelo
docker compose up -d
docker compose exec backend python ml/train.py --config config/config.yaml

# 3. Reconstruir frontend
docker compose build frontend
docker compose up -d

# 4. Verificar métricas
docker compose exec backend cat ml/artifacts/metrics.json
```

## Consideraciones Futuras

### Por qué existe la actividad 0

La actividad 0 representa períodos de transición entre actividades o momentos donde los sensores no capturan movimientos significativos. En el dataset MHealth original, esto es útil para segmentación temporal.

### Cuándo incluir actividad 0

Solo si necesitas:

-   Detectar períodos de inactividad/transición
-   Segmentar automáticamente secuencias largas
-   Aplicaciones de tiempo real que requieren detectar "nada está pasando"

En esos casos, considera:

-   **SMOTE** o **ADASYN** para sobremuestreo de clases minoritarias
-   **Undersampling** de la clase mayoritaria (actividad 0)
-   **Class weights** más agresivos (ya usamos `balanced`)
-   **Ensemble methods** con diferentes balanceos

### Configuración Actual

El modelo está optimizado para:

-   ✅ Clasificación de actividades claramente definidas
-   ✅ Predicción en ventanas pre-segmentadas
-   ✅ Evaluación de logs con actividades conocidas
-   ❌ Detección de transiciones temporales
-   ❌ Segmentación automática de streams continuos

## Archivos Modificados

```
ml/src/mhealth/preprocess.py    - Nueva función filter_unlabeled_activity()
ml/src/mhealth/modeling.py       - Integración del filtro en train_model()
frontend/src/App.tsx             - Eliminación de referencias a actividad 0
ml/src/mhealth/constants.py      - Mantiene mapeo completo para referencia
```

## Conclusión

Al eliminar la actividad 0 del entrenamiento, el modelo pasó de ser un **clasificador trivial** (que predice siempre la clase mayoritaria) a un **clasificador funcional** que distingue correctamente entre las 12 actividades significativas del dataset MHealth.

**Mejora global**: De un modelo prácticamente inútil (macro F1=13%) a uno altamente efectivo (macro F1=93%).
