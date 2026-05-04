# Predicción de Informalidad Laboral en Colombia

## Descripción
Proyecto de ciencia de datos que predice si un trabajador colombiano es informal o formal, utilizando microdatos de la Gran Encuesta Integrada de Hogares (GEIH) del DANE.

## Datos
- **Fuente:** DANE - GEIH (Gran Encuesta Integrada de Hogares)
- **Periodo:** Enero 2024 - Febrero 2026 (10 meses)
- **Registros:** 287,900 trabajadores ocupados
- **Variable objetivo:** Informalidad laboral (no cotiza a pensión)

## Estructura del Proyecto
Villa-07/
├── data/
│   ├── raw/           # Microdatos originales del DANE (no incluidos en GitHub)
│   └── processed/     # Dataset limpio para modelado
├── notebooks/         # Notebooks de EDA y modelado
├── src/               # Scripts de Python
├── onboarding/        # Ejercicios de onboarding
├── .gitignore
├── README.md
└── requirements.txt

## Instalación
```bash
pip install -r requirements.txt
```

## Equipo
- Juan José Villa

## Tecnologías
- Python 3.11
- Pandas, NumPy, Scikit-Learn
- Plotly (EDA)
- Streamlit (App)