"""
==========================================================================
PROYECTO: Predicción de Informalidad Laboral en Colombia
DATOS: GEIH - DANE (Ene 2024 - Feb 2026, 10 meses)
==========================================================================
Este script lee los microdatos de la GEIH del DANE de múltiples meses,
los une, limpia y prepara para modelado de Machine Learning.

Variable objetivo: INFORMAL (1 = No cotiza a pensión, 0 = Sí cotiza)
==========================================================================
"""

import pandas as pd
import numpy as np
import os

MESES = {
    "Enero 2024": "Enero 2024/CSV",
    "Febrero 2024": "Febrero 2024/Febrero 2024/CSV",  # Carpeta doble
    "Marzo 2024": "Marzo 2024/CSV/CSV",
    "Abril 2024": "Abril 2024/CSV/CSV",
    "Enero 2025": "Enero 2025/CSV",
    "Febrero 2025": "Febrero 2025/CSV",
    "Marzo 2025": "Marzo 2025/CSV",
    "Abril 2025": "Abril 2025/CSV",
    "Enero 2026": "Enero 2026/CSV",
    "Febrero 2026": "Febrero 2026/CSV",
}

# Nombres posibles del archivo de Características Generales
# (el DANE a veces cambia ligeramente el nombre entre años)
NOMBRES_CARACTERISTICAS = [
    "Características generales, seguridad social en salud y educación.CSV",
    "Características generales, seguridad social en salud y educación.csv",
    "Características_generales__seguridad_social_en_salud_y_educación.CSV",
    "Características generales  seguridad social en salud y educación.CSV",
]

NOMBRES_OCUPADOS = [
    "Ocupados.CSV",
    "Ocupados.csv",
]


def buscar_archivo(carpeta, nombres_posibles):
    """Busca un archivo probando varios nombres posibles."""
    for nombre in nombres_posibles:
        ruta = os.path.join(carpeta, nombre)
        if os.path.exists(ruta):
            return ruta
    # Si no encuentra con nombres exactos, buscar por coincidencia parcial
    if os.path.exists(carpeta):
        for archivo in os.listdir(carpeta):
            archivo_lower = archivo.lower()
            if "caracter" in archivo_lower and archivo_lower.endswith(".csv"):
                return os.path.join(carpeta, archivo)
            if "ocupado" in archivo_lower and archivo_lower.endswith(".csv"):
                return os.path.join(carpeta, archivo)
    return None


LLAVES = ["DIRECTORIO", "SECUENCIA_P", "ORDEN"]

lista_dataframes = []
errores = []

for mes, ruta_csv in MESES.items():
    print(f"\n{'='*60}")
    print(f"Procesando: {mes}")
    print(f"{'='*60}")

    
    ruta_caract = buscar_archivo(ruta_csv, NOMBRES_CARACTERISTICAS)
    ruta_ocup = buscar_archivo(ruta_csv, NOMBRES_OCUPADOS)

    if ruta_caract is None:
        print(f"  ERROR: No se encontró archivo de Características en {ruta_csv}")
        if os.path.exists(ruta_csv):
            print(f"  Archivos disponibles: {os.listdir(ruta_csv)}")
        else:
            print(f"  LA CARPETA NO EXISTE: {ruta_csv}")
        errores.append(mes)
        continue

    if ruta_ocup is None:
        print(f"  ERROR: No se encontró archivo de Ocupados en {ruta_csv}")
        errores.append(mes)
        continue

    print(f"  Características: {ruta_caract}")
    print(f"  Ocupados: {ruta_ocup}")

    
    try:
        caract = pd.read_csv(ruta_caract, sep=";", encoding="latin-1", low_memory=False)
        ocup = pd.read_csv(ruta_ocup, sep=";", encoding="latin-1", low_memory=False)
    except Exception as e:
        print(f"  ERROR al leer: {e}")
        errores.append(mes)
        continue

    print(f"  Características: {caract.shape[0]:,} filas")
    print(f"  Ocupados: {ocup.shape[0]:,} filas")

    
    datos_mes = caract.merge(ocup, on=LLAVES, how="inner", suffixes=("_cg", "_oc"))
    print(f"  Después del merge: {datos_mes.shape[0]:,} filas")

    
    datos_mes["MES_PERIODO"] = mes

    lista_dataframes.append(datos_mes)


if not lista_dataframes:
    print("\nERROR: No se pudo leer ningún mes. Revisa las rutas.")
    exit()

datos = pd.concat(lista_dataframes, ignore_index=True)
print(f"\n{'='*60}")
print(f"TOTAL DATOS UNIDOS: {datos.shape[0]:,} filas de {len(lista_dataframes)} meses")
print(f"{'='*60}")

if errores:
    print(f"Meses con error (revisar): {errores}")



columnas_mapeo = {
    "P3271": "SEXO",
    "P6040": "EDAD",
    "P6083": "NIVEL_EDUCATIVO",
    "P6070": "ESTADO_CIVIL",
    "DPTO": "DEPARTAMENTO",
    "P6090": "AFILIADO_SALUD",
    "P6430": "POSICION_OCUPACIONAL",
    "P6440": "TIENE_CONTRATO",
    "P6500": "INGRESO_MONETARIO",
    "P6800": "HORAS_SEMANA",
    "P6920": "COTIZA_PENSION",
    "INGLABO": "INGRESO_LABORAL",
    "RAMA2D_R4": "RAMA_ACTIVIDAD",
}


def obtener_columna(df, nombre_base):
    """Busca la columna con o sin sufijo _cg o _oc."""
    if nombre_base in df.columns:
        return nombre_base
    elif f"{nombre_base}_cg" in df.columns:
        return f"{nombre_base}_cg"
    elif f"{nombre_base}_oc" in df.columns:
        return f"{nombre_base}_oc"
    return None



columnas_reales = {}
for col_dane, col_nueva in columnas_mapeo.items():
    col_real = obtener_columna(datos, col_dane)
    if col_real:
        columnas_reales[col_real] = col_nueva
    else:
        print(f"  ADVERTENCIA: No se encontró la columna {col_dane}")


columnas_reales["MES_PERIODO"] = "MES_PERIODO"


df = datos[list(columnas_reales.keys())].rename(columns=columnas_reales)

print(f"\nDataset con variables seleccionadas: {df.shape[0]:,} filas, {df.shape[1]} columnas")
print(f"Columnas: {list(df.columns)}")



df = df[df["COTIZA_PENSION"].isin([1, 2])].copy()
df["INFORMAL"] = (df["COTIZA_PENSION"] == 2).astype(int)

print(f"\n{'='*60}")
print(f"VARIABLE OBJETIVO: INFORMAL")
print(f"{'='*60}")
print(f"  0 (Formal):   {(df['INFORMAL'] == 0).sum():>10,} ({(df['INFORMAL'] == 0).mean()*100:.1f}%)")
print(f"  1 (Informal): {(df['INFORMAL'] == 1).sum():>10,} ({(df['INFORMAL'] == 1).mean()*100:.1f}%)")
print(f"  Total:        {len(df):>10,}")



print(f"\n{'='*60}")
print(f"VALORES NULOS ANTES DE LIMPIEZA")
print(f"{'='*60}")
print(df.isnull().sum())

df["INGRESO_MONETARIO"] = df["INGRESO_MONETARIO"].fillna(0)
df["INGRESO_LABORAL"] = df["INGRESO_LABORAL"].fillna(df["INGRESO_LABORAL"].median())
df["ESTADO_CIVIL"] = df["ESTADO_CIVIL"].fillna(df["ESTADO_CIVIL"].mode()[0])
df["AFILIADO_SALUD"] = df["AFILIADO_SALUD"].fillna(df["AFILIADO_SALUD"].mode()[0])

print(f"\nVALORES NULOS DESPUÉS DE LIMPIEZA")
print(df.isnull().sum())



print(f"\n{'='*60}")
print(f"RESUMEN POR MES")
print(f"{'='*60}")
resumen_mes = df.groupby("MES_PERIODO").agg(
    total=("INFORMAL", "count"),
    informales=("INFORMAL", "sum"),
    tasa_informalidad=("INFORMAL", "mean"),
).sort_index()
resumen_mes["tasa_informalidad"] = resumen_mes["tasa_informalidad"].apply(
    lambda x: f"{x*100:.1f}%"
)
print(resumen_mes.to_string())

print(f"\n{'='*60}")
print(f"INFORMALIDAD POR SEXO")
print(f"{'='*60}")
sexo_map = {1: "Hombre", 2: "Mujer"}
print(
    df.groupby("SEXO")["INFORMAL"]
    .mean()
    .rename(index=sexo_map)
    .apply(lambda x: f"{x*100:.1f}%")
)

print(f"\n{'='*60}")
print(f"INFORMALIDAD POR NIVEL EDUCATIVO")
print(f"{'='*60}")
edu_map = {
    1: "Ninguno",
    2: "Preescolar",
    3: "Primaria",
    4: "Secundaria",
    5: "Media",
    6: "Superior",
}
print(
    df.groupby("NIVEL_EDUCATIVO")["INFORMAL"]
    .mean()
    .rename(index=edu_map)
    .apply(lambda x: f"{x*100:.1f}%")
)

print(f"\n{'='*60}")
print(f"INFORMALIDAD POR POSICIÓN OCUPACIONAL")
print(f"{'='*60}")
pos_map = {
    1: "Emp. particular",
    2: "Emp. gobierno",
    3: "Emp. doméstico",
    4: "Cuenta propia",
    5: "Patrón",
    6: "Trab. familiar sin pago",
    7: "Trab. sin pago otros",
    8: "Jornalero",
}
print(
    df.groupby("POSICION_OCUPACIONAL")["INFORMAL"]
    .mean()
    .rename(index=pos_map)
    .apply(lambda x: f"{x*100:.1f}%")
)



RUTA_SALIDA = "datos_informalidad_10meses.csv"
df.to_csv(RUTA_SALIDA, index=False)
print(f"\n{'='*60}")
print(f"DATASET GUARDADO: {RUTA_SALIDA}")
print(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")
print(f"Meses incluidos: {df['MES_PERIODO'].nunique()}")
print(f"{'='*60}")
print("\n¡Listo! Usa 'datos_informalidad_10meses.csv' para tu EDA con Plotly y modelos ML.")
