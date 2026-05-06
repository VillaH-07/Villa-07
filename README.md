# Predicción de Informalidad Laboral en Colombia

## Descripción

Proyecto de ciencia de datos que predice si un trabajador colombiano es informal o formal, utilizando microdatos de la Gran Encuesta Integrada de Hogares (GEIH) del DANE.

La **variable objetivo** (INFORMAL) se construyó a partir de la variable P6920 del DANE: un trabajador es **informal** si **no cotiza a pensión**.

---

## Datos

| Característica | Detalle |
|---|---|
| **Fuente** | DANE - GEIH (Gran Encuesta Integrada de Hogares) |
| **Periodo** | Enero 2024 - Febrero 2026 (10 meses) |
| **Registros** | 287,900 trabajadores ocupados |
| **Variable objetivo** | INFORMAL (0 = Formal, 1 = Informal) |
| **Tasa de informalidad** | 56.9% |

---

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

---

## Estadística Descriptiva

### Medidas de tendencia central

| Variable | Media | Mediana | Moda |
|---|---|---|---|
| Edad | 41.4 años | 40 | 30 |
| Ingreso Monetario | $1,049,253 | $480,000 | $0 |
| Ingreso Laboral | $1,710,059 | $1,300,000 | $1,300,000 |
| Horas/Semana | 43.9 | 47 | 48 |

### Medidas de variabilidad

| Variable | Desv. Estándar | Coef. Variación | Rango |
|---|---|---|---|
| Ingreso Monetario | $1,855,140 | 1.77 | $100,000,000 |
| Ingreso Laboral | $2,260,536 | 1.32 | $103,000,000 |
| Edad | 13.88 | 0.34 | 85 |
| Horas/Semana | 12.98 | 0.30 | 129 |

### Medidas de forma

| Variable | Asimetría | Curtosis |
|---|---|---|
| Ingreso Laboral | 12.15 | 320.45 |
| Ingreso Monetario | 8.74 | 218.46 |
| Edad | 0.36 | -0.59 |
| Horas/Semana | -0.08 | 2.55 |

### Cuartiles

| Variable | Q1 (25%) | Q2 (50%) | Q3 (75%) |
|---|---|---|---|
| Edad | 30 | 40 | 52 |
| Ingreso Monetario | $0 | $480,000 | $1,423,500 |
| Ingreso Laboral | $800,000 | $1,300,000 | $1,800,000 |
| Horas/Semana | 40 | 47 | 48 |

---

## Hallazgos del EDA

### Informalidad por sexo

| Sexo | Tasa de informalidad |
|---|---|
| Hombres | 58.5% |
| Mujeres | 55.0% |

### Informalidad por posición ocupacional

| Posición | Tasa de informalidad |
|---|---|
| Jornalero | 98.0% |
| Trab. familiar sin pago | 97.1% |
| Emp. doméstico | 85.6% |
| Cuenta propia | 85.3% |
| Patrón | 63.8% |
| Emp. particular | 29.6% |
| Emp. gobierno | 0.0% |

### Informalidad por nivel educativo

| Nivel educativo | Tasa de informalidad |
|---|---|
| Primaria | 69.8% |
| Preescolar | 53.0% |
| Ninguno | 52.7% |

### Ingreso laboral promedio

| Condición | Ingreso promedio |
|---|---|
| Formal | $2,631,325 |
| Informal | $1,013,461 |

### Horas trabajadas promedio

| Condición | Horas/semana |
|---|---|
| Formal | 46.5 |
| Informal | 42.1 |

### Matriz de correlaciones (variables numéricas vs INFORMAL)

| Variable | Correlación con INFORMAL |
|---|---|
| Ingreso Monetario | -0.459 |
| Ingreso Laboral | -0.354 |
| Horas/Semana | -0.169 |
| Edad | 0.126 |

---

## Modelos de Clasificación

### Variables predictoras (10)

SEXO, EDAD, NIVEL_EDUCATIVO, ESTADO_CIVIL, AFILIADO_SALUD, POSICION_OCUPACIONAL, TIENE_CONTRATO, HORAS_SEMANA, INGRESO_LABORAL, RAMA_ACTIVIDAD.

**Variables excluidas:**
- `COTIZA_PENSION`: define la variable objetivo (sería trampa incluirla)
- `INGRESO_MONETARIO`: alta correlación (0.67) con INGRESO_LABORAL
- `DEPARTAMENTO`: requiere encoding especial
- `MES_PERIODO`: no es variable predictora

### División de datos

| Conjunto | Registros |
|---|---|
| Entrenamiento | 230,320 |
| Prueba | 57,580 |

### Balanceo con SMOTE

| Clase | Antes | Después |
|---|---|---|
| Formal (0) | 99,168 | 131,152 |
| Informal (1) | 131,152 | 131,152 |

---

### Resultados: Regresión Logística

| Métrica | Valor |
|---|---|
| **Accuracy** | 0.8574 |
| **Precision** | 0.9318 |
| **Sensibilidad (Recall)** | 0.8090 |
| **Especificidad** | 0.9215 |
| **F1 Score** | 0.8660 |
| **AUC** | 0.9300 |

**Reporte de clasificación:**

| Clase | Precision | Recall | F1-Score | Soporte |
|---|---|---|---|---|
| Formal | 0.78 | 0.92 | 0.85 | 24,792 |
| Informal | 0.93 | 0.81 | 0.87 | 32,788 |
| **Accuracy** | | | **0.86** | **57,580** |

---

### Resultados: Random Forest

| Métrica | Valor |
|---|---|
| **Accuracy** | 0.9116 |
| **Precision** | 0.9341 |
| **Sensibilidad (Recall)** | 0.9089 |
| **Especificidad** | 0.9152 |
| **F1 Score** | 0.9213 |
| **AUC** | 0.9683 |

**Reporte de clasificación:**

| Clase | Precision | Recall | F1-Score | Soporte |
|---|---|---|---|---|
| Formal | 0.88 | 0.92 | 0.90 | 24,792 |
| Informal | 0.93 | 0.91 | 0.92 | 32,788 |
| **Accuracy** | | | **0.91** | **57,580** |

---

### Comparación de modelos

| Modelo | Accuracy | Precision | Sensibilidad | Especificidad | F1 Score | AUC |
|---|---|---|---|---|---|---|
| Regresión Logística | 0.8574 | 0.9318 | 0.8090 | 0.9215 | 0.8660 | 0.9300 |
| **Random Forest** | **0.9116** | **0.9341** | **0.9089** | **0.9152** | **0.9213** | **0.9683** |

### Importancia de variables (Random Forest)

| Variable | Importancia |
|---|---|
| **INGRESO_LABORAL** | 0.4183 |
| **POSICION_OCUPACIONAL** | 0.1907 |
| **TIENE_CONTRATO** | 0.1718 |
| RAMA_ACTIVIDAD | 0.0910 |
| HORAS_SEMANA | 0.0541 |
| AFILIADO_SALUD | 0.0274 |
| EDAD | 0.0263 |
| ESTADO_CIVIL | 0.0114 |
| NIVEL_EDUCATIVO | 0.0053 |
| SEXO | 0.0036 |

---

### Predicción de ejemplo

**Perfil:** Mujer, 35 años, Secundaria, Soltera, Afiliada a salud, Cuenta propia, Sin contrato, 48 horas/semana, Ingreso $800,000, Comercio.

| Modelo | Predicción | Prob. Formal | Prob. Informal |
|---|---|---|---|
| Regresión Logística | **INFORMAL** | 0.0148 | 0.9852 |
| Random Forest | **INFORMAL** | 0.0209 | 0.9791 |

---

## Conclusiones

1. El **mejor modelo** es **Random Forest** con un AUC de **0.9683** y accuracy de **91.2%**.
2. Las **3 variables más importantes** para predecir informalidad son: **Ingreso Laboral** (41.8%), **Posición Ocupacional** (19.1%) y **Tenencia de Contrato** (17.2%).
3. La tasa de informalidad del dataset (**56.9%**) es consistente con las cifras oficiales del DANE para Colombia.
4. Los trabajadores informales ganan en promedio **$1,013,461**, significativamente menos que los formales (**$2,631,325**).
5. Las posiciones más informales son **jornaleros** (98%) y **trabajadores familiares sin pago** (97.1%).

---

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

## Tecnologías

- Python 3.11
- Pandas, NumPy, Scikit-Learn
- Plotly (EDA)
- Imbalanced-Learn (SMOTE)
- Streamlit (App - próximamente)

## Equipo

- Juan José Villa
