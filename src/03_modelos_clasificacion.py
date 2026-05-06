import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE

print("Librerías cargadas correctamente.")


RUTA_DATOS = "data/processed/datos_informalidad_10meses.csv"
datos = pd.read_csv(RUTA_DATOS)

print(f"Dataset cargado: {datos.shape[0]:,} filas x {datos.shape[1]} columnas")
print(f"\nColumnas disponibles: {list(datos.columns)}")
print(f"\nDistribución de la variable objetivo:")
print(datos["INFORMAL"].value_counts())
print(f"\nTasa de informalidad: {datos['INFORMAL'].mean()*100:.1f}%")


# =========================================================================
# 3. DEFINIR VARIABLES DE ENTRADA (X) Y SALIDA (Y)
# =========================================================================
# Variables predictoras:
#   SEXO               - Sexo (1=Hombre, 2=Mujer)
#   EDAD               - Edad en años
#   NIVEL_EDUCATIVO    - Nivel educativo (1=Ninguno ... 6=Superior)
#   ESTADO_CIVIL       - Estado civil
#   AFILIADO_SALUD     - Afiliado a salud (1=Sí, 2=No)
#   POSICION_OCUPACIONAL - Posición ocupacional (1=Emp.particular ... 8=Jornalero)
#   TIENE_CONTRATO     - Tiene contrato (1=Sí, 2=No)
#   HORAS_SEMANA       - Horas trabajadas por semana
#   INGRESO_LABORAL    - Ingreso laboral total
#   RAMA_ACTIVIDAD     - Rama de actividad económica
#
# NOTA: No se incluyen INGRESO_MONETARIO (muy correlacionado con INGRESO_LABORAL)
#       ni DEPARTAMENTO (demasiadas categorías, requiere tratamiento especial)
#       ni COTIZA_PENSION (es la variable que define INFORMAL, sería trampa)
#       ni MES_PERIODO (no es predictora)

FEATURES = [
    "SEXO",
    "EDAD",
    "NIVEL_EDUCATIVO",
    "ESTADO_CIVIL",
    "AFILIADO_SALUD",
    "POSICION_OCUPACIONAL",
    "TIENE_CONTRATO",
    "HORAS_SEMANA",
    "INGRESO_LABORAL",
    "RAMA_ACTIVIDAD",
]

TARGET = "INFORMAL"

X = datos[FEATURES]
y = datos[TARGET]

print(f"\nVariables de entrada ({len(FEATURES)}): {FEATURES}")
print(f"Variable objetivo: {TARGET}")
print(f"\nForma de X: {X.shape}")
print(f"Forma de y: {y.shape}")


print("\n" + "=" * 60)
print("4. MATRIZ DE CORRELACIONES (verificar multicolinealidad)")
print("=" * 60)

correlation_matrix = X.corr()
print(correlation_matrix.to_string())

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="Blues", center=0, fmt=".2f")
plt.title("Matriz de Correlaciones - Variables de Entrada")
plt.tight_layout()
plt.savefig("data/processed/correlacion_features.png", dpi=150)
plt.show()
print("\nGráfico guardado: data/processed/correlacion_features.png")


print("\n" + "=" * 60)
print("5. DIVISIÓN ENTRENAMIENTO / PRUEBA")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Entrenamiento: {X_train.shape[0]:,} registros")
print(f"Prueba:        {X_test.shape[0]:,} registros")
print(f"\nDistribución en entrenamiento:")
print(y_train.value_counts())
print(f"\nDistribución en prueba:")
print(y_test.value_counts())


# =========================================================================
# BALANCEAR CLASES CON SMOTE
# =========================================================================
print("\n" + "=" * 60)
print("6. BALANCEO DE CLASES CON SMOTE")
print("=" * 60)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"\nAntes del balanceo:")
print(y_train.value_counts())
print(f"\nDespués del balanceo:")
print(y_train_bal.value_counts())


# =========================================================================
# ESCALAR DATOS (necesario para Regresión Logística)
# =========================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)


def evaluar_modelo(nombre, modelo, X_train, y_train, X_test, y_test):
    """Entrena, predice y evalúa un modelo de clasificación."""
    print(f"\n{'=' * 60}")
    print(f"MODELO: {nombre}")
    print(f"{'=' * 60}")

    # Entrenar
    modelo.fit(X_train, y_train)

    # Predecir
    y_pred = modelo.predict(X_test)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # Sensibilidad
    f1 = f1_score(y_test, y_pred)
    specificity = tn / (tn + fp)

    print(f"\n  Accuracy      : {accuracy:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Sensibilidad  : {recall:.4f}")
    print(f"  Especificidad : {specificity:.4f}")
    print(f"  F1 Score      : {f1:.4f}")

    print(f"\n  Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=["Formal", "Informal"]))

    # Visualizar matriz de confusión
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Formal", "Informal"]
    )
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Matriz de Confusión - {nombre}")
    plt.tight_layout()
    filename = f"data/processed/cm_{nombre.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"  Gráfico guardado: {filename}")

    return {
        "Modelo": nombre,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensibilidad": recall,
        "Especificidad": specificity,
        "F1 Score": f1,
    }


# =========================================================================
# MODELO 1: REGRESIÓN LOGÍSTICA
# =========================================================================
modelo_log = LogisticRegression(max_iter=1000, random_state=42)
resultado_log = evaluar_modelo(
    "Regresión Logística", modelo_log, X_train_scaled, y_train_bal, X_test_scaled, y_test
)


# =========================================================================
# MODELO 2: RANDOM FOREST
# =========================================================================
# Random Forest no necesita escalado, pero usamos datos balanceados
modelo_rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
)
resultado_rf = evaluar_modelo(
    "Random Forest",
    modelo_rf,
    X_train_bal,
    y_train_bal,
    X_test,
    y_test,
)


# =========================================================================
# IMPORTANCIA DE VARIABLES (Random Forest)
# =========================================================================
print("\n" + "=" * 60)
print("11. IMPORTANCIA DE VARIABLES (Random Forest)")
print("=" * 60)

importancias = pd.DataFrame(
    {"Variable": FEATURES, "Importancia": modelo_rf.feature_importances_}
).sort_values("Importancia", ascending=True)

print(importancias.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(importancias["Variable"], importancias["Importancia"], color="steelblue")
plt.xlabel("Importancia")
plt.title("Importancia de Variables - Random Forest")
plt.tight_layout()
plt.savefig("data/processed/importancia_variables_rf.png", dpi=150)
plt.show()
print("\nGráfico guardado: data/processed/importancia_variables_rf.png")


# =========================================================================
# CURVAS ROC
# =========================================================================
print("\n" + "=" * 60)
print("12. CURVAS ROC")
print("=" * 60)

# Probabilidades
y_prob_log = modelo_log.predict_proba(X_test_scaled)[:, 1]
y_prob_rf = modelo_rf.predict_proba(X_test)[:, 1]

# AUC
auc_log = roc_auc_score(y_test, y_prob_log)
auc_rf = roc_auc_score(y_test, y_prob_rf)

print(f"  AUC Regresión Logística : {auc_log:.4f}")
print(f"  AUC Random Forest       : {auc_rf:.4f}")

# Curvas
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f"Regresión Logística (AUC={auc_log:.4f})", linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", label="Aleatorio (AUC=0.5)")
plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
plt.title("Curvas ROC - Comparación de Modelos")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/processed/curvas_roc.png", dpi=150)
plt.show()
print("\nGráfico guardado: data/processed/curvas_roc.png")


# =========================================================================
# COMPARACIÓN DE MODELOS
# =========================================================================
print("\n" + "=" * 60)
print("13. COMPARACIÓN DE MODELOS")
print("=" * 60)

df_resultados = pd.DataFrame([resultado_log, resultado_rf])
print(df_resultados.to_string(index=False))

# Gráfico comparativo
metricas = ["Accuracy", "Precision", "Sensibilidad", "Especificidad", "F1 Score"]

fig, axes = plt.subplots(1, len(metricas), figsize=(18, 5), sharey=True)
for i, metrica in enumerate(metricas):
    axes[i].bar(
        df_resultados["Modelo"],
        df_resultados[metrica],
        color=["#3498db", "#2ecc71"],
    )
    axes[i].set_title(metrica)
    axes[i].set_ylim(0, 1)
    axes[i].tick_params(axis="x", rotation=45)
    for j, v in enumerate(df_resultados[metrica]):
        axes[i].text(j, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

plt.suptitle("Comparación de Modelos de Clasificación", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("data/processed/comparacion_modelos.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGráfico guardado: data/processed/comparacion_modelos.png")


# =========================================================================
# PREDICCIÓN DE UN NUEVO INDIVIDUO
# =========================================================================
print("\n" + "=" * 60)
print("14. PREDICCIÓN DE UN NUEVO INDIVIDUO (ejemplo)")
print("=" * 60)

# Ejemplo: Mujer, 35 años, Secundaria, Soltera, Afiliada salud,
#          Cuenta propia, Sin contrato, 48 horas/semana,
#          Ingreso $800,000, Rama 47 (Comercio)
nuevo_individuo = pd.DataFrame(
    {
        "SEXO": [2],
        "EDAD": [35],
        "NIVEL_EDUCATIVO": [4],
        "ESTADO_CIVIL": [5],
        "AFILIADO_SALUD": [1],
        "POSICION_OCUPACIONAL": [4],
        "TIENE_CONTRATO": [2],
        "HORAS_SEMANA": [48],
        "INGRESO_LABORAL": [800000],
        "RAMA_ACTIVIDAD": [47],
    }
)

print("\nDatos del nuevo individuo:")
print(nuevo_individuo.to_string(index=False))

# Predicción con Regresión Logística (necesita escalado)
nuevo_scaled = scaler.transform(nuevo_individuo)
pred_log = modelo_log.predict(nuevo_scaled)[0]
prob_log = modelo_log.predict_proba(nuevo_scaled)[0]

print(f"\n  Regresión Logística:")
print(f"    Predicción: {'INFORMAL' if pred_log == 1 else 'FORMAL'}")
print(f"    Probabilidad Formal:   {prob_log[0]:.4f}")
print(f"    Probabilidad Informal: {prob_log[1]:.4f}")

# Predicción con Random Forest (no necesita escalado)
pred_rf = modelo_rf.predict(nuevo_individuo)[0]
prob_rf = modelo_rf.predict_proba(nuevo_individuo)[0]

print(f"\n  Random Forest:")
print(f"    Predicción: {'INFORMAL' if pred_rf == 1 else 'FORMAL'}")
print(f"    Probabilidad Formal:   {prob_rf[0]:.4f}")
print(f"    Probabilidad Informal: {prob_rf[1]:.4f}")


print("\n" + "=" * 60)
print("RESUMEN DEL MODELADO")
print("=" * 60)
print(f"""
PROYECTO: Predicción de Informalidad Laboral en Colombia
DATOS: GEIH - DANE, 10 meses (Ene 2024 - Feb 2026)
REGISTROS: {datos.shape[0]:,} trabajadores ocupados

VARIABLE OBJETIVO: INFORMAL (0=Formal, 1=Informal)
TASA DE INFORMALIDAD: {datos['INFORMAL'].mean()*100:.1f}%

VARIABLES PREDICTORAS ({len(FEATURES)}):
  {', '.join(FEATURES)}

NOTA: Se excluyeron COTIZA_PENSION (define la variable objetivo),
      INGRESO_MONETARIO (alta correlación con INGRESO_LABORAL),
      DEPARTAMENTO (requiere encoding especial), y MES_PERIODO.

BALANCEO: SMOTE aplicado al conjunto de entrenamiento.

RESULTADOS:
""")
print(df_resultados.to_string(index=False))
print(f"""
AUC Regresión Logística : {auc_log:.4f}
AUC Random Forest       : {auc_rf:.4f}

CONCLUSIÓN:
  El mejor modelo es {'Random Forest' if auc_rf > auc_log else 'Regresión Logística'}
  con un AUC de {max(auc_rf, auc_log):.4f}.

  Las variables más importantes para predecir informalidad son:
""")
top_vars = importancias.tail(5).iloc[::-1]
for _, row in top_vars.iterrows():
    print(f"    - {row['Variable']}: {row['Importancia']:.4f}")

print("\n" + "=" * 60)
print("MODELADO COMPLETADO")
print("=" * 60)
