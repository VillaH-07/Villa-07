import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("Librerías cargadas correctamente.")



RUTA_DATOS = "data/processed/datos_informalidad_10meses.csv"
datos = pd.read_csv(RUTA_DATOS)

print(f"Dataset cargado: {datos.shape[0]:,} filas x {datos.shape[1]} columnas")
print(f"Meses incluidos: {datos['MES_PERIODO'].nunique()}")
print(f"\nColumnas: {list(datos.columns)}")



print("\n" + "=" * 60)
print("INFORMACIÓN DEL DATASET")
print("=" * 60)
datos.info()

print("\n" + "=" * 60)
print("PRIMEROS 10 REGISTROS")
print("=" * 60)
print(datos.head(10))

print("\n" + "=" * 60)
print("VALORES NULOS")
print("=" * 60)
print(datos.isnull().sum())




SEXO_MAP = {1: "Hombre", 2: "Mujer"}
NIVEL_EDUCATIVO_MAP = {
    1: "Ninguno", 2: "Preescolar", 3: "Primaria",
    4: "Secundaria", 5: "Media", 6: "Superior",
}
ESTADO_CIVIL_MAP = {
    1: "Unión libre", 2: "Casado(a)", 3: "Separado(a)",
    4: "Viudo(a)", 5: "Soltero(a)", 6: "No casado(a) 2+ años",
}
POSICION_MAP = {
    1: "Emp. particular", 2: "Emp. gobierno", 3: "Emp. doméstico",
    4: "Cuenta propia", 5: "Patrón", 6: "Trab. familiar s/pago",
    7: "Trab. s/pago otros", 8: "Jornalero",
}
CONTRATO_MAP = {1: "Sí", 2: "No"}
SALUD_MAP = {1: "Sí", 2: "No"}
INFORMAL_MAP = {0: "Formal", 1: "Informal"}

# Crear columnas con etiquetas para gráficas
datos["SEXO_LABEL"] = datos["SEXO"].map(SEXO_MAP)
datos["NIVEL_EDUCATIVO_LABEL"] = datos["NIVEL_EDUCATIVO"].map(NIVEL_EDUCATIVO_MAP)
datos["ESTADO_CIVIL_LABEL"] = datos["ESTADO_CIVIL"].map(ESTADO_CIVIL_MAP)
datos["POSICION_LABEL"] = datos["POSICION_OCUPACIONAL"].map(POSICION_MAP)
datos["CONTRATO_LABEL"] = datos["TIENE_CONTRATO"].map(CONTRATO_MAP)
datos["SALUD_LABEL"] = datos["AFILIADO_SALUD"].map(SALUD_MAP)
datos["INFORMAL_LABEL"] = datos["INFORMAL"].map(INFORMAL_MAP)

# Variables numéricas y categóricas
NUMERICAS = ["EDAD", "INGRESO_MONETARIO", "INGRESO_LABORAL", "HORAS_SEMANA"]
CATEGORICAS = [
    "SEXO_LABEL", "NIVEL_EDUCATIVO_LABEL", "ESTADO_CIVIL_LABEL",
    "POSICION_LABEL", "CONTRATO_LABEL", "INFORMAL_LABEL",
]

print("\nVariables numéricas:", NUMERICAS)
print("Variables categóricas:", CATEGORICAS)



print("\n" + "=" * 60)
print("3. ESTADÍSTICAS DESCRIPTIVAS - VARIABLES NUMÉRICAS")
print("=" * 60)
print(datos[NUMERICAS].describe())



print("\n" + "=" * 60)
print("3.1 MEDIDAS DE TENDENCIA CENTRAL")
print("=" * 60)

stats_central = pd.DataFrame({
    "Media": datos[NUMERICAS].mean(),
    "Mediana": datos[NUMERICAS].median(),
    "Moda": datos[NUMERICAS].agg(lambda x: x.mode().values[0]),
})
print(stats_central.to_string())



print("\n" + "=" * 60)
print("3.2 MEDIDAS DE VARIABILIDAD")
print("=" * 60)

stats_variabilidad = pd.DataFrame({
    "Varianza": datos[NUMERICAS].var(),
    "Desv. Estándar": datos[NUMERICAS].std(),
    "Coef. Variación": datos[NUMERICAS].std() / datos[NUMERICAS].mean(),
    "Rango": datos[NUMERICAS].max() - datos[NUMERICAS].min(),
})
stats_variabilidad = stats_variabilidad.sort_values(
    by="Coef. Variación", ascending=False
)
print(stats_variabilidad.to_string())



print("\n" + "=" * 60)
print("3.3 MEDIDAS DE FORMA")
print("=" * 60)

stats_forma = pd.DataFrame({
    "Asimetría (Skewness)": datos[NUMERICAS].skew(),
    "Curtosis (Kurtosis)": datos[NUMERICAS].kurtosis(),
})
stats_forma = stats_forma.sort_values(by="Asimetría (Skewness)", key=abs, ascending=False)
print(stats_forma.to_string())



print("\n" + "=" * 60)
print("3.4 MEDIDAS DE POSICIÓN (CUARTILES)")
print("=" * 60)

cuartiles = datos[NUMERICAS].quantile([0.25, 0.50, 0.75])
cuartiles.index = ["Q1 (25%)", "Q2 (50%)", "Q3 (75%)"]
print(cuartiles.to_string())



print("\n" + "=" * 60)
print("4. GRÁFICOS UNIVARIADOS")
print("=" * 60)



conteo_informal = datos["INFORMAL_LABEL"].value_counts().reset_index()
conteo_informal.columns = ["Condición", "Cantidad"]

fig = px.pie(
    conteo_informal,
    names="Condición",
    values="Cantidad",
    title="Distribución de Informalidad Laboral (10 meses GEIH)",
    color="Condición",
    color_discrete_map={"Formal": "#2ecc71", "Informal": "#e74c3c"},
    hole=0.4,
)
fig.show()



for col in NUMERICAS:
    fig = px.histogram(
        datos,
        x=col,
        nbins=50,
        title=f"Histograma de {col}",
        color="INFORMAL_LABEL",
        color_discrete_map={"Formal": "#2ecc71", "Informal": "#e74c3c"},
        barmode="overlay",
        opacity=0.7,
        labels={"INFORMAL_LABEL": "Condición"},
    )
    fig.show()



for col in NUMERICAS:
    fig = px.box(
        datos,
        y=col,
        x="INFORMAL_LABEL",
        color="INFORMAL_LABEL",
        title=f"Boxplot de {col} por Informalidad",
        color_discrete_map={"Formal": "#2ecc71", "Informal": "#e74c3c"},
        labels={"INFORMAL_LABEL": "Condición"},
    )
    fig.show()



# Sexo
fig = px.histogram(
    datos,
    x="SEXO_LABEL",
    color="INFORMAL_LABEL",
    barmode="group",
    title="Informalidad por Sexo",
    color_discrete_map={"Formal": "#2ecc71", "Informal": "#e74c3c"},
    labels={"SEXO_LABEL": "Sexo", "INFORMAL_LABEL": "Condición"},
)
fig.show()

# Nivel educativo
fig = px.histogram(
    datos,
    x="NIVEL_EDUCATIVO_LABEL",
    color="INFORMAL_LABEL",
    barmode="group",
    title="Informalidad por Nivel Educativo",
    color_discrete_map={"Formal": "#2ecc71", "Informal": "#e74c3c"},
    labels={"NIVEL_EDUCATIVO_LABEL": "Nivel Educativo", "INFORMAL_LABEL": "Condición"},
    category_orders={"NIVEL_EDUCATIVO_LABEL": [
        "Ninguno", "Preescolar", "Primaria", "Secundaria", "Media", "Superior"
    ]},
)
fig.show()

# Posición ocupacional
fig = px.histogram(
    datos,
    x="POSICION_LABEL",
    color="INFORMAL_LABEL",
    barmode="group",
    title="Informalidad por Posición Ocupacional",
    color_discrete_map={"Formal": "#2ecc71", "Informal": "#e74c3c"},
    labels={"POSICION_LABEL": "Posición Ocupacional", "INFORMAL_LABEL": "Condición"},
)
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# Estado civil
fig = px.histogram(
    datos,
    x="ESTADO_CIVIL_LABEL",
    color="INFORMAL_LABEL",
    barmode="group",
    title="Informalidad por Estado Civil",
    color_discrete_map={"Formal": "#2ecc71", "Informal": "#e74c3c"},
    labels={"ESTADO_CIVIL_LABEL": "Estado Civil", "INFORMAL_LABEL": "Condición"},
)
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# Contrato
fig = px.histogram(
    datos,
    x="CONTRATO_LABEL",
    color="INFORMAL_LABEL",
    barmode="group",
    title="Informalidad por Tenencia de Contrato",
    color_discrete_map={"Formal": "#2ecc71", "Informal": "#e74c3c"},
    labels={"CONTRATO_LABEL": "Tiene Contrato", "INFORMAL_LABEL": "Condición"},
)
fig.show()



print("\n" + "=" * 60)
print("4.5 TASAS DE INFORMALIDAD POR CATEGORÍA")
print("=" * 60)

# Tasa de informalidad por mes
tasa_mes = (
    datos.groupby("MES_PERIODO")["INFORMAL"]
    .mean()
    .reset_index()
    .rename(columns={"INFORMAL": "Tasa_Informalidad"})
)
tasa_mes["Tasa_Informalidad"] = tasa_mes["Tasa_Informalidad"] * 100

fig = px.line(
    tasa_mes.sort_values("MES_PERIODO"),
    x="MES_PERIODO",
    y="Tasa_Informalidad",
    title="Evolución de la Tasa de Informalidad por Mes (%)",
    markers=True,
    labels={"MES_PERIODO": "Mes", "Tasa_Informalidad": "Tasa Informalidad (%)"},
)
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# Tasa por sexo
tasa_sexo = (
    datos.groupby("SEXO_LABEL")["INFORMAL"]
    .mean()
    .reset_index()
    .rename(columns={"INFORMAL": "Tasa_Informalidad"})
)
tasa_sexo["Tasa_Informalidad"] = tasa_sexo["Tasa_Informalidad"] * 100

fig = px.bar(
    tasa_sexo,
    x="SEXO_LABEL",
    y="Tasa_Informalidad",
    title="Tasa de Informalidad por Sexo (%)",
    color="SEXO_LABEL",
    labels={"SEXO_LABEL": "Sexo", "Tasa_Informalidad": "Tasa Informalidad (%)"},
    text_auto=".1f",
)
fig.show()

# Tasa por nivel educativo
tasa_edu = (
    datos.groupby("NIVEL_EDUCATIVO_LABEL")["INFORMAL"]
    .mean()
    .reset_index()
    .rename(columns={"INFORMAL": "Tasa_Informalidad"})
)
tasa_edu["Tasa_Informalidad"] = tasa_edu["Tasa_Informalidad"] * 100

fig = px.bar(
    tasa_edu,
    x="NIVEL_EDUCATIVO_LABEL",
    y="Tasa_Informalidad",
    title="Tasa de Informalidad por Nivel Educativo (%)",
    color="NIVEL_EDUCATIVO_LABEL",
    labels={
        "NIVEL_EDUCATIVO_LABEL": "Nivel Educativo",
        "Tasa_Informalidad": "Tasa Informalidad (%)",
    },
    category_orders={"NIVEL_EDUCATIVO_LABEL": [
        "Ninguno", "Preescolar", "Primaria", "Secundaria", "Media", "Superior"
    ]},
    text_auto=".1f",
)
fig.show()

# Tasa por posición ocupacional
tasa_pos = (
    datos.groupby("POSICION_LABEL")["INFORMAL"]
    .mean()
    .reset_index()
    .rename(columns={"INFORMAL": "Tasa_Informalidad"})
)
tasa_pos["Tasa_Informalidad"] = tasa_pos["Tasa_Informalidad"] * 100

fig = px.bar(
    tasa_pos.sort_values("Tasa_Informalidad", ascending=True),
    x="Tasa_Informalidad",
    y="POSICION_LABEL",
    orientation="h",
    title="Tasa de Informalidad por Posición Ocupacional (%)",
    color="Tasa_Informalidad",
    color_continuous_scale="RdYlGn_r",
    labels={
        "POSICION_LABEL": "Posición Ocupacional",
        "Tasa_Informalidad": "Tasa Informalidad (%)",
    },
    text_auto=".1f",
)
fig.show()

# Tasa por rango de edad
datos["RANGO_EDAD"] = pd.cut(
    datos["EDAD"],
    bins=[0, 18, 25, 35, 45, 55, 65, 100],
    labels=["<18", "18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
)

tasa_edad = (
    datos.groupby("RANGO_EDAD", observed=True)["INFORMAL"]
    .mean()
    .reset_index()
    .rename(columns={"INFORMAL": "Tasa_Informalidad"})
)
tasa_edad["Tasa_Informalidad"] = tasa_edad["Tasa_Informalidad"] * 100

fig = px.bar(
    tasa_edad,
    x="RANGO_EDAD",
    y="Tasa_Informalidad",
    title="Tasa de Informalidad por Rango de Edad (%)",
    color="Tasa_Informalidad",
    color_continuous_scale="RdYlGn_r",
    labels={"RANGO_EDAD": "Rango de Edad", "Tasa_Informalidad": "Tasa Informalidad (%)"},
    text_auto=".1f",
)
fig.show()



print("\n" + "=" * 60)
print("5. ANÁLISIS MULTIVARIADO")
print("=" * 60)



correlation_matrix = datos[NUMERICAS + ["INFORMAL"]].corr()
print("\nMatriz de correlaciones:")
print(correlation_matrix.to_string())

fig = px.imshow(
    correlation_matrix,
    text_auto=".2f",
    title="Matriz de Correlaciones",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
)
fig.show()



fig = px.scatter(
    datos.sample(5000, random_state=42),
    x="EDAD",
    y="INGRESO_LABORAL",
    color="INFORMAL_LABEL",
    title="Edad vs Ingreso Laboral (muestra de 5,000)",
    color_discrete_map={"Formal": "#2ecc71", "Informal": "#e74c3c"},
    opacity=0.5,
    labels={
        "EDAD": "Edad",
        "INGRESO_LABORAL": "Ingreso Laboral ($)",
        "INFORMAL_LABEL": "Condición",
    },
)
fig.show()

fig = px.scatter(
    datos.sample(5000, random_state=42),
    x="HORAS_SEMANA",
    y="INGRESO_LABORAL",
    color="INFORMAL_LABEL",
    title="Horas Semanales vs Ingreso Laboral (muestra de 5,000)",
    color_discrete_map={"Formal": "#2ecc71", "Informal": "#e74c3c"},
    opacity=0.5,
    labels={
        "HORAS_SEMANA": "Horas por Semana",
        "INGRESO_LABORAL": "Ingreso Laboral ($)",
        "INFORMAL_LABEL": "Condición",
    },
)
fig.show()



ingreso_pos = (
    datos.groupby(["POSICION_LABEL", "INFORMAL_LABEL"])["INGRESO_LABORAL"]
    .mean()
    .reset_index()
)

fig = px.bar(
    ingreso_pos,
    x="POSICION_LABEL",
    y="INGRESO_LABORAL",
    color="INFORMAL_LABEL",
    barmode="group",
    title="Ingreso Laboral Promedio por Posición Ocupacional e Informalidad",
    color_discrete_map={"Formal": "#2ecc71", "Informal": "#e74c3c"},
    labels={
        "POSICION_LABEL": "Posición Ocupacional",
        "INGRESO_LABORAL": "Ingreso Promedio ($)",
        "INFORMAL_LABEL": "Condición",
    },
    text_auto=",.0f",
)
fig.update_layout(xaxis_tickangle=-45)
fig.show()



fig = px.violin(
    datos,
    y="EDAD",
    x="INFORMAL_LABEL",
    color="INFORMAL_LABEL",
    box=True,
    title="Distribución de Edad por Condición Laboral",
    color_discrete_map={"Formal": "#2ecc71", "Informal": "#e74c3c"},
    labels={"EDAD": "Edad", "INFORMAL_LABEL": "Condición"},
)
fig.show()



print("\n" + "=" * 60)
print("RESUMEN DE HALLAZGOS CLAVE")
print("=" * 60)
print(f"""
1. El dataset contiene {datos.shape[0]:,} registros de 10 meses (Ene 2024 - Feb 2026).

2. La tasa de informalidad promedio es {datos['INFORMAL'].mean()*100:.1f}%, 
   consistente con las cifras oficiales del DANE para Colombia.

3. Por sexo: los hombres ({datos[datos['SEXO']==1]['INFORMAL'].mean()*100:.1f}%) 
   tienen mayor informalidad que las mujeres ({datos[datos['SEXO']==2]['INFORMAL'].mean()*100:.1f}%).

4. Por nivel educativo: a menor nivel educativo, mayor informalidad.
   Primaria: {datos[datos['NIVEL_EDUCATIVO']==3]['INFORMAL'].mean()*100:.1f}% vs 
   Superior: {datos[datos['NIVEL_EDUCATIVO']==6]['INFORMAL'].mean()*100:.1f}%.

5. Posiciones más informales: jornaleros ({datos[datos['POSICION_OCUPACIONAL']==8]['INFORMAL'].mean()*100:.1f}%), 
   trabajadores familiares sin pago ({datos[datos['POSICION_OCUPACIONAL']==6]['INFORMAL'].mean()*100:.1f}%),
   cuenta propia ({datos[datos['POSICION_OCUPACIONAL']==4]['INFORMAL'].mean()*100:.1f}%).

6. Empleados del gobierno tienen informalidad prácticamente del 
   {datos[datos['POSICION_OCUPACIONAL']==2]['INFORMAL'].mean()*100:.1f}%.

7. El ingreso laboral promedio de los formales (${datos[datos['INFORMAL']==0]['INGRESO_LABORAL'].mean():,.0f})
   es significativamente mayor que el de los informales (${datos[datos['INFORMAL']==1]['INGRESO_LABORAL'].mean():,.0f}).

8. Los formales trabajan en promedio {datos[datos['INFORMAL']==0]['HORAS_SEMANA'].mean():.1f} horas/semana
   vs {datos[datos['INFORMAL']==1]['HORAS_SEMANA'].mean():.1f} horas/semana de los informales.
""")

print("=" * 60)
print("EDA COMPLETADO")
print("=" * 60)
