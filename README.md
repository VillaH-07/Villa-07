# Predicción de Informalidad Laboral en Colombia

Proyecto de ciencia de datos que predice si un trabajador colombiano es informal o formal, utilizando microdatos de la Gran Encuesta Integrada de Hogares (GEIH) del DANE.

## Datos

- **Fuente:** DANE - GEIH (Gran Encuesta Integrada de Hogares)
- **Periodo:** Enero 2024 - Febrero 2026 (10 meses)
- **Registros:** 287,900 trabajadores ocupados
- **Variable objetivo:** INFORMAL (1 = No cotiza a pensión, 0 = Sí cotiza)

## Estructura del Proyecto

```
Villa-07/
├── data/
│   ├── raw/                           # Microdatos originales del DANE
│   └── processed/                     # Dataset limpio
├── src/
│   ├── 01_cargar_datos_geih.py        # Carga y limpieza de datos
│   ├── 02_eda_informalidad.py         # Análisis Exploratorio (Plotly)
│   ├── 03_modelos_clasificacion.py    # Modelos + Optuna + Experimentos
│   └── 04_interpretabilidad.py        # SHAP + Permutación
├── results/
│   ├── eda/                           # Resultados del EDA
│   ├── modelos/                       # Resultados de modelos
│   └── interpretabilidad/             # Resultados de interpretabilidad
├── .gitignore
├── README.md
└── requirements.txt
```

## Resultados

| Modelo | Accuracy | F1 Score | AUC |
|--------|----------|----------|-----|
| Reg. Logística (Desbalanceado) | 0.8674 | 0.8774 | 0.9311 |
| Random Forest (Desbalanceado) | **0.9145** | **0.9247** | **0.9691** |
| Reg. Logística (SMOTE) | 0.8575 | 0.8661 | 0.9300 |
| Random Forest (SMOTE) | 0.9139 | 0.9238 | 0.9689 |

**Mejor modelo:** Random Forest (Desbalanceado) con AUC de 0.9691.

Para más detalles ver:
- [Resultados EDA](results/eda/README.md)
- [Resultados Modelos](results/modelos/README.md)
- [Resultados Interpretabilidad](results/interpretabilidad/README.md)

## Instalación

```bash
pip install -r requirements.txt
pip install imbalanced-learn optuna shap
```

## Uso

```bash
python src/01_cargar_datos_geih.py
python src/02_eda_informalidad.py
python src/03_modelos_clasificacion.py
python src/04_interpretabilidad.py
```

## Tecnologías

Python 3.11, Pandas, NumPy, Scikit-Learn, Plotly, Optuna, SHAP, Imbalanced-Learn

## Autor

Juan José Villa — Especialización en Ciencia de Datos e IA, Universidad de Medellín
