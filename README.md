# Predicción de Informalidad Laboral en Colombia

## Descripción

Proyecto de ciencia de datos que predice si un trabajador colombiano es informal o formal, utilizando microdatos de la Gran Encuesta Integrada de Hogares (GEIH) del DANE.

## Datos

- **Fuente:** DANE - GEIH (Gran Encuesta Integrada de Hogares)
- **Periodo:** Enero 2024 - Febrero 2026 (10 meses)
- **Registros:** 287,900 trabajadores ocupados
- **Variable objetivo:** Informalidad laboral (no cotiza a pensión)
- **Variables predictoras:** Sexo, Edad, Nivel Educativo, Estado Civil, Afiliación a Salud, Posición Ocupacional, Tenencia de Contrato, Horas Semanales, Ingreso Laboral, Rama de Actividad

## Estructura del Proyecto

```
Villa-07/
├── data/
│   ├── raw/                           # Microdatos originales del DANE (no incluidos en GitHub)
│   └── processed/                     # Dataset limpio para modelado
├── notebooks/                         # Notebooks de EDA y modelado
├── src/
│   ├── 01_cargar_datos_geih.py        # Carga, une y limpia los 10 meses de la GEIH
│   ├── 02_eda_informalidad.py         # Análisis Exploratorio de Datos con Plotly
│   └── 03_modelos_clasificacion.py    # Modelos: Logistic Regression y Random Forest
├── .gitignore
├── README.md
└── requirements.txt
```

## Modelos

| Modelo | Descripción |
|--------|-------------|
| Regresión Logística | Modelo base de clasificación binaria |
| Random Forest | Modelo de ensamble con 200 árboles |

Ambos modelos utilizan **SMOTE** para balancear las clases (57% informal vs 43% formal).

## Instalación

```bash
pip install -r requirements.txt
pip install imbalanced-learn
```

## Uso

Ejecutar los scripts en orden desde la raíz del proyecto:

```bash
python src/01_cargar_datos_geih.py
python src/02_eda_informalidad.py
python src/03_modelos_clasificacion.py
```

## Hallazgos Clave

- La tasa de informalidad promedio es **57%**, consistente con cifras oficiales del DANE
- Los **jornaleros** (98%) y **trabajadores familiares sin pago** (97%) son los más informales
- Los **empleados del gobierno** tienen informalidad prácticamente del 0%
- A **menor nivel educativo**, mayor informalidad
- Los **formales** ganan significativamente más que los **informales**

## Tecnologías

- Python 3.11
- Pandas, NumPy, Scikit-Learn
- Plotly (EDA)
- Imbalanced-Learn (SMOTE)
- Streamlit (App - próximamente)

## Equipo

- Juan José Villa
