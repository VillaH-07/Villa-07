# Resultados del Análisis Exploratorio de Datos (EDA)

**Script:** `src/02_eda_informalidad.py`

## Dataset

- 287,900 trabajadores ocupados de 10 meses de la GEIH (Ene 2024 - Feb 2026)
- 15 columnas, 0 valores nulos después de limpieza
- Tasa de informalidad: **56.9%**

## Estadísticas Descriptivas

### Medidas de Tendencia Central

| Variable | Media | Mediana | Moda |
|----------|-------|---------|------|
| Edad | 41.4 años | 40.0 | 30.0 |
| Ingreso Monetario | $1,049,253 | $480,000 | $0 |
| Ingreso Laboral | $1,710,059 | $1,300,000 | $1,300,000 |
| Horas/Semana | 44.0 | 47.0 | 48.0 |

### Medidas de Variabilidad

| Variable | Desv. Estándar | Coef. Variación |
|----------|---------------|-----------------|
| Ingreso Monetario | $1,855,140 | 1.77 |
| Ingreso Laboral | $2,260,536 | 1.32 |
| Edad | 13.9 años | 0.34 |
| Horas/Semana | 13.0 | 0.30 |

### Medidas de Forma

| Variable | Asimetría | Curtosis |
|----------|-----------|----------|
| Ingreso Laboral | 12.15 | 320.45 |
| Ingreso Monetario | 8.74 | 218.46 |
| Edad | 0.36 | -0.59 |
| Horas/Semana | -0.08 | 2.55 |

Los ingresos tienen una asimetría positiva muy alta, indicando que la mayoría gana poco y pocos ganan mucho.

## Hallazgos Clave

### Por Sexo
- Hombres: **58.5%** de informalidad
- Mujeres: **55.0%** de informalidad

### Por Nivel Educativo
- Primaria: **69.8%** informal
- Superior: tasa significativamente menor
- A menor nivel educativo, mayor informalidad

### Por Posición Ocupacional
- Jornaleros: **98.0%**
- Trabajadores familiares sin pago: **97.1%**
- Cuenta propia: **85.3%**
- Empleados del gobierno: **0.0%**

### Ingresos
- Formales: **$2,631,325** promedio
- Informales: **$1,013,461** promedio
- Los formales ganan 2.6 veces más que los informales

### Horas Semanales
- Formales: **46.5** horas/semana
- Informales: **42.1** horas/semana

## Matriz de Correlaciones

| Variable | INFORMAL |
|----------|----------|
| INGRESO_MONETARIO | -0.459 |
| INGRESO_LABORAL | -0.354 |
| HORAS_SEMANA | -0.169 |
| EDAD | 0.126 |

Las correlaciones más fuertes con informalidad son el ingreso (negativa) y la edad (positiva débil).
