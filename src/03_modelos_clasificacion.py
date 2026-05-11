import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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

# Silenciar warnings de Optuna para que la salida sea legible
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

print("Librerías cargadas correctamente.")

# Crear carpetas para guardar resultados
os.makedirs("results/modelos", exist_ok=True)


RUTA_DATOS = "data/processed/datos_informalidad_10meses.csv"
datos = pd.read_csv(RUTA_DATOS)

print(f"Dataset cargado: {datos.shape[0]:,} filas x {datos.shape[1]} columnas")
print(f"\nDistribución de la variable objetivo:")
print(datos["INFORMAL"].value_counts())
print(f"\nTasa de informalidad: {datos['INFORMAL'].mean()*100:.1f}%")


# =========================================================================
# DEFINIR VARIABLES DE ENTRADA (X) Y SALIDA (Y)
# =========================================================================
# Variables predictoras (10):
#   SEXO, EDAD, NIVEL_EDUCATIVO, ESTADO_CIVIL, AFILIADO_SALUD,
#   POSICION_OCUPACIONAL, TIENE_CONTRATO, HORAS_SEMANA,
#   INGRESO_LABORAL, RAMA_ACTIVIDAD
#
# Excluidas:
#   COTIZA_PENSION   -> Define la variable objetivo (data leakage)
#   INGRESO_MONETARIO -> Correlación 0.67 con INGRESO_LABORAL
#   DEPARTAMENTO     -> Demasiadas categorías
#   MES_PERIODO      -> No es predictora

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
print(f"Forma de X: {X.shape}")


# =========================================================================
# VERIFICAR MULTICOLINEALIDAD
# =========================================================================
print("\n" + "=" * 60)
print("4. MATRIZ DE CORRELACIONES")
print("=" * 60)

correlation_matrix = X.corr()
print(correlation_matrix.to_string())

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="Blues", center=0, fmt=".2f")
plt.title("Matriz de Correlaciones - Variables de Entrada")
plt.tight_layout()
plt.savefig("results/modelos/correlacion_features.png", dpi=150)
plt.close()
print("Gráfico guardado: results/modelos/correlacion_features.png")


# =========================================================================
# DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# =========================================================================
print("\n" + "=" * 60)
print("5. DIVISIÓN ENTRENAMIENTO / PRUEBA (80/20)")
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
# FUNCIÓN PARA EVALUAR MODELOS
# =========================================================================
def evaluar_modelo(nombre, modelo, X_train, y_train, X_test, y_test, guardar_como):
    """Entrena, predice y evalúa un modelo de clasificación."""
    print(f"\n  --- {nombre} ---")

    # Entrenar
    modelo.fit(X_train, y_train)

    # Predecir
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # Sensibilidad
    f1 = f1_score(y_test, y_pred)
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y_test, y_prob)

    print(f"    Accuracy      : {accuracy:.4f}")
    print(f"    Precision     : {precision:.4f}")
    print(f"    Sensibilidad  : {recall:.4f}")
    print(f"    Especificidad : {specificity:.4f}")
    print(f"    F1 Score      : {f1:.4f}")
    print(f"    AUC           : {auc:.4f}")

    print(f"\n    Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=["Formal", "Informal"]))

    # Guardar matriz de confusión
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Formal", "Informal"]
    )
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Matriz de Confusión - {nombre}")
    plt.tight_layout()
    plt.savefig(f"results/modelos/{guardar_como}.png", dpi=150)
    plt.close()
    print(f"    Gráfico guardado: results/modelos/{guardar_como}.png")

    return {
        "Modelo": nombre,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensibilidad": recall,
        "Especificidad": specificity,
        "F1 Score": f1,
        "AUC": auc,
    }


# =========================================================================
# VALIDACIÓN CRUZADA (5-Fold)
# =========================================================================
print("\n" + "=" * 60)
print("7. VALIDACIÓN CRUZADA (5-Fold Stratified)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Escalar para Logistic Regression
scaler_cv = StandardScaler()
X_scaled = scaler_cv.fit_transform(X)

# Regresión Logística
scores_lr = cross_val_score(
    LogisticRegression(max_iter=1000, random_state=42),
    X_scaled, y, cv=cv, scoring="roc_auc",
)
print(f"\nRegresión Logística - AUC por fold: {scores_lr}")
print(f"  Media: {scores_lr.mean():.4f} (+/- {scores_lr.std():.4f})")

# Random Forest
scores_rf = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X, y, cv=cv, scoring="roc_auc",
)
print(f"\nRandom Forest - AUC por fold: {scores_rf}")
print(f"  Media: {scores_rf.mean():.4f} (+/- {scores_rf.std():.4f})")


# =========================================================================
# OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA
# =========================================================================
print("\n" + "=" * 60)
print("8. OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA")
print("=" * 60)

# Escalar datos de entrenamiento para Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------------------------------------------------
# Optimizar Regresión Logística
# -------------------------------------------------------------------------
def objective_lr(trial):
    """Función objetivo para optimizar Regresión Logística."""
    C = trial.suggest_float("C", 0.001, 100, log=True)
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
    penalty = "l2"

    modelo = LogisticRegression(
        C=C, solver=solver, penalty=penalty, max_iter=1000, random_state=42,
    )

    scores = cross_val_score(modelo, X_train_scaled, y_train, cv=3, scoring="roc_auc")
    return scores.mean()


print("\nOptimizando Regresión Logística (50 trials)...")
study_lr = optuna.create_study(direction="maximize")
study_lr.optimize(objective_lr, n_trials=50, show_progress_bar=False)

print(f"  Mejor AUC (CV): {study_lr.best_value:.4f}")
print(f"  Mejores hiperparámetros: {study_lr.best_params}")


# -------------------------------------------------------------------------
# Optimizar Random Forest
# -------------------------------------------------------------------------
def objective_rf(trial):
    """Función objetivo para optimizar Random Forest."""
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
    )

    scores = cross_val_score(modelo, X_train, y_train, cv=3, scoring="roc_auc")
    return scores.mean()


print("\nOptimizando Random Forest (50 trials)...")
study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=50, show_progress_bar=False)

print(f"  Mejor AUC (CV): {study_rf.best_value:.4f}")
print(f"  Mejores hiperparámetros: {study_rf.best_params}")


# =========================================================================
# EXPERIMENTO 1: DATOS DESBALANCEADOS (sin SMOTE)
# =========================================================================
print("\n" + "=" * 70)
print("9. EXPERIMENTO 1: DATOS DESBALANCEADOS (sin SMOTE)")
print("=" * 70)
print(f"  Distribución entrenamiento: {dict(y_train.value_counts())}")

resultados_desbalanceado = []

# Logistic Regression con hiperparámetros optimizados
modelo_lr_opt = LogisticRegression(
    **study_lr.best_params, max_iter=1000, random_state=42
)
res = evaluar_modelo(
    "Reg. Logística (Desbalanceado)",
    modelo_lr_opt,
    X_train_scaled, y_train,
    X_test_scaled, y_test,
    "cm_lr_desbalanceado",
)
resultados_desbalanceado.append(res)

# Random Forest con hiperparámetros optimizados
modelo_rf_opt = RandomForestClassifier(
    **study_rf.best_params, random_state=42, n_jobs=-1
)
res = evaluar_modelo(
    "Random Forest (Desbalanceado)",
    modelo_rf_opt,
    X_train, y_train,
    X_test, y_test,
    "cm_rf_desbalanceado",
)
resultados_desbalanceado.append(res)

df_desbalanceado = pd.DataFrame(resultados_desbalanceado)
print("\nResumen Experimento 1 (Desbalanceado):")
print(df_desbalanceado.to_string(index=False))


# =========================================================================
# EXPERIMENTO 2: DATOS BALANCEADOS (con SMOTE)
# =========================================================================
print("\n" + "=" * 70)
print("10. EXPERIMENTO 2: DATOS BALANCEADOS (con SMOTE)")
print("=" * 70)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"  Antes del balanceo:  {dict(y_train.value_counts())}")
print(f"  Después del balanceo: {dict(y_train_bal.value_counts())}")

# Escalar datos balanceados
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled_2 = scaler.transform(X_test)

resultados_balanceado = []

# Logistic Regression
modelo_lr_bal = LogisticRegression(
    **study_lr.best_params, max_iter=1000, random_state=42
)
res = evaluar_modelo(
    "Reg. Logística (SMOTE)",
    modelo_lr_bal,
    X_train_bal_scaled, y_train_bal,
    X_test_scaled_2, y_test,
    "cm_lr_balanceado",
)
resultados_balanceado.append(res)

# Random Forest
modelo_rf_bal = RandomForestClassifier(
    **study_rf.best_params, random_state=42, n_jobs=-1
)
res = evaluar_modelo(
    "Random Forest (SMOTE)",
    modelo_rf_bal,
    X_train_bal, y_train_bal,
    X_test, y_test,
    "cm_rf_balanceado",
)
resultados_balanceado.append(res)

df_balanceado = pd.DataFrame(resultados_balanceado)
print("\nResumen Experimento 2 (Balanceado con SMOTE):")
print(df_balanceado.to_string(index=False))


# =========================================================================
# COMPARACIÓN DE EXPERIMENTOS
# =========================================================================
print("\n" + "=" * 70)
print("11. COMPARACIÓN: DESBALANCEADO VS BALANCEADO (SMOTE)")
print("=" * 70)

df_todos = pd.concat(
    [df_desbalanceado, df_balanceado], ignore_index=True
)
print(df_todos.to_string(index=False))

# Gráfico comparativo
metricas = ["Accuracy", "Precision", "Sensibilidad", "Especificidad", "F1 Score", "AUC"]
modelos_nombres = df_todos["Modelo"].values

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
colores = ["#e74c3c", "#3498db", "#e67e22", "#2ecc71"]

for i, metrica in enumerate(metricas):
    axes[i].bar(range(len(modelos_nombres)), df_todos[metrica], color=colores)
    axes[i].set_title(metrica, fontsize=12, fontweight="bold")
    axes[i].set_ylim(0, 1.05)
    axes[i].set_xticks(range(len(modelos_nombres)))
    axes[i].set_xticklabels(
        [m.replace(" (", "\n(") for m in modelos_nombres], fontsize=8, ha="center"
    )
    for j, v in enumerate(df_todos[metrica]):
        axes[i].text(j, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

plt.suptitle(
    "Comparación de Modelos: Desbalanceado vs SMOTE", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("results/modelos/comparacion_experimentos.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nGráfico guardado: results/modelos/comparacion_experimentos.png")


# =========================================================================
# CURVAS ROC COMPARATIVAS
# =========================================================================
print("\n" + "=" * 60)
print("12. CURVAS ROC")
print("=" * 60)

# Re-entrenar para obtener probabilidades
modelo_lr_opt.fit(X_train_scaled, y_train)
modelo_rf_opt.fit(X_train, y_train)
modelo_lr_bal.fit(X_train_bal_scaled, y_train_bal)
modelo_rf_bal.fit(X_train_bal, y_train_bal)

modelos_roc = {
    "Reg. Logística (Desbal.)": (modelo_lr_opt, X_test_scaled),
    "Random Forest (Desbal.)": (modelo_rf_opt, X_test),
    "Reg. Logística (SMOTE)": (modelo_lr_bal, X_test_scaled_2),
    "Random Forest (SMOTE)": (modelo_rf_bal, X_test),
}

plt.figure(figsize=(10, 8))
colores_roc = ["#e74c3c", "#3498db", "#e67e22", "#2ecc71"]

for idx, (nombre, (modelo, X_t)) in enumerate(modelos_roc.items()):
    y_prob = modelo.predict_proba(X_t)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{nombre} (AUC={auc:.4f})", linewidth=2, color=colores_roc[idx])

plt.plot([0, 1], [0, 1], "k--", label="Aleatorio (AUC=0.5)")
plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
plt.title("Curvas ROC - Comparación de Modelos y Experimentos")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/modelos/curvas_roc.png", dpi=150)
plt.close()
print("Gráfico guardado: results/modelos/curvas_roc.png")


# =========================================================================
# IMPORTANCIA DE VARIABLES (mejor modelo Random Forest)
# =========================================================================
print("\n" + "=" * 60)
print("13. IMPORTANCIA DE VARIABLES (Random Forest SMOTE)")
print("=" * 60)

importancias = pd.DataFrame(
    {"Variable": FEATURES, "Importancia": modelo_rf_bal.feature_importances_}
).sort_values("Importancia", ascending=True)

print(importancias.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(importancias["Variable"], importancias["Importancia"], color="steelblue")
plt.xlabel("Importancia")
plt.title("Importancia de Variables - Random Forest (SMOTE)")
plt.tight_layout()
plt.savefig("results/modelos/importancia_variables_rf.png", dpi=150)
plt.close()
print("Gráfico guardado: results/modelos/importancia_variables_rf.png")


# =========================================================================
# PREDICCIÓN DE UN NUEVO INDIVIDUO
# =========================================================================
print("\n" + "=" * 60)
print("14. PREDICCIÓN DE UN NUEVO INDIVIDUO (ejemplo)")
print("=" * 60)

# Mujer, 35 años, Secundaria, Soltera, Afiliada salud,
# Cuenta propia, Sin contrato, 48h/semana, $800,000, Comercio
nuevo = pd.DataFrame(
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

print("Datos del nuevo individuo:")
print(nuevo.to_string(index=False))

# Predicción con el mejor modelo (Random Forest SMOTE)
pred = modelo_rf_bal.predict(nuevo)[0]
prob = modelo_rf_bal.predict_proba(nuevo)[0]

print(f"\n  Random Forest (SMOTE) - Mejor modelo:")
print(f"    Predicción: {'INFORMAL' if pred == 1 else 'FORMAL'}")
print(f"    Probabilidad Formal:   {prob[0]:.4f}")
print(f"    Probabilidad Informal: {prob[1]:.4f}")


print("\n" + "=" * 70)
print("RESUMEN FINAL DEL MODELADO")
print("=" * 70)
print(f"""
PROYECTO: Predicción de Informalidad Laboral en Colombia
DATOS: GEIH - DANE, 10 meses (Ene 2024 - Feb 2026)
REGISTROS: {datos.shape[0]:,} trabajadores ocupados

VARIABLE OBJETIVO: INFORMAL (0=Formal, 1=Informal)
TASA DE INFORMALIDAD: {datos['INFORMAL'].mean()*100:.1f}%

VARIABLES PREDICTORAS ({len(FEATURES)}):
  {', '.join(FEATURES)}

HIPERPARÁMETROS OPTIMIZADOS CON OPTUNA (50 trials cada uno):
  Regresión Logística: {study_lr.best_params}
  Random Forest:       {study_rf.best_params}

VALIDACIÓN CRUZADA (5-Fold):
  Regresión Logística AUC: {scores_lr.mean():.4f} (+/- {scores_lr.std():.4f})
  Random Forest AUC:       {scores_rf.mean():.4f} (+/- {scores_rf.std():.4f})

COMPARACIÓN DE EXPERIMENTOS:
""")
print(df_todos.to_string(index=False))

# Identificar mejor modelo
mejor = df_todos.loc[df_todos["AUC"].idxmax()]
print(f"""
CONCLUSIÓN:
  El mejor modelo es {mejor['Modelo']} con AUC de {mejor['AUC']:.4f}.

  Las variables más importantes para predecir informalidad son:
""")
top_vars = importancias.tail(5).iloc[::-1]
for _, row in top_vars.iterrows():
    print(f"    - {row['Variable']}: {row['Importancia']:.4f}")

print("\n" + "=" * 70)
print("GRÁFICOS GENERADOS EN results/modelos/:")
print("=" * 70)
print("  - correlacion_features.png")
print("  - cm_lr_desbalanceado.png")
print("  - cm_rf_desbalanceado.png")
print("  - cm_lr_balanceado.png")
print("  - cm_rf_balanceado.png")
print("  - comparacion_experimentos.png")
print("  - curvas_roc.png")
print("  - importancia_variables_rf.png")
print("\nMODELADO COMPLETADO")
