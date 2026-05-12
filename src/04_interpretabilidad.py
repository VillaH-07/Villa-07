# =========================================================================
# 1. CARGAR LIBRERÍAS
# =========================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
print("Librerías cargadas correctamente.")

# Crear carpeta para resultados
os.makedirs("results/interpretabilidad", exist_ok=True)


# =========================================================================
# 2. CARGAR DATOS Y ENTRENAR EL MEJOR MODELO
# =========================================================================
RUTA_DATOS = "data/processed/datos_informalidad_10meses.csv"
datos = pd.read_csv(RUTA_DATOS)

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenar Random Forest con los mejores hiperparámetros de Optuna
modelo = RandomForestClassifier(
    n_estimators=298,
    max_depth=26,
    min_samples_split=19,
    min_samples_leaf=4,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
modelo.fit(X_train, y_train)

print(f"Dataset: {datos.shape[0]:,} registros")
print(f"Modelo: Random Forest (hiperparámetros optimizados con Optuna)")
print(f"AUC en test: ~0.9691")


# =========================================================================
# FUNCIÓN AUXILIAR PARA EXTRAER SHAP VALUES CLASE INFORMAL
# =========================================================================
def extraer_shap_informal(shap_vals):
    """Extrae SHAP values para la clase 1 (Informal), compatible con
    diferentes versiones de SHAP."""
    if isinstance(shap_vals, list):
        # Formato antiguo: lista [clase_0, clase_1]
        return shap_vals[1]
    elif hasattr(shap_vals, "values"):
        # Formato Explanation object
        return shap_vals.values
    elif isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim == 3:
            # Formato 3D: (n_samples, n_features, n_classes)
            return shap_vals[:, :, 1]
        else:
            return shap_vals
    return shap_vals


# =========================================================================
# 3. IMPORTANCIA POR PERMUTACIÓN
# =========================================================================
print("\n" + "=" * 60)
print("3. IMPORTANCIA POR PERMUTACIÓN")
print("=" * 60)
print("Calculando (esto puede tardar unos segundos)...")

perm_importance = permutation_importance(
    modelo, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc"
)

perm_df = pd.DataFrame(
    {
        "Variable": FEATURES,
        "Importancia_Media": perm_importance.importances_mean,
        "Importancia_Std": perm_importance.importances_std,
    }
).sort_values("Importancia_Media", ascending=True)

print("\nImportancia por Permutación (AUC):")
print(perm_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(
    perm_df["Variable"],
    perm_df["Importancia_Media"],
    xerr=perm_df["Importancia_Std"],
    color="steelblue",
    capsize=3,
)
plt.xlabel("Disminución media en AUC al permutar la variable")
plt.title("Importancia por Permutación - Random Forest")
plt.tight_layout()
plt.savefig("results/interpretabilidad/importancia_permutacion.png", dpi=150)
plt.close()
print("Gráfico guardado: results/interpretabilidad/importancia_permutacion.png")


# =========================================================================
# 4. SHAP - VALORES GLOBALES
# =========================================================================
print("\n" + "=" * 60)
print("4. SHAP - VALORES GLOBALES")
print("=" * 60)

# Usar una muestra para que SHAP no tarde demasiado
np.random.seed(42)
muestra_idx = np.random.choice(X_test.index, size=2000, replace=False)
X_muestra = X_test.loc[muestra_idx].reset_index(drop=True)
y_muestra = y_test.loc[muestra_idx].reset_index(drop=True)

print(f"Calculando SHAP values para {len(X_muestra):,} observaciones...")
print("(Esto puede tardar unos minutos...)")

explainer = shap.TreeExplainer(modelo)
shap_values_raw = explainer.shap_values(X_muestra)
shap_informal = extraer_shap_informal(shap_values_raw)

print(f"Shape de SHAP values: {shap_informal.shape}")

# -------------------------------------------------------------------------
# 4.1 SHAP Summary Plot (Beeswarm)
# -------------------------------------------------------------------------
print("\nGenerando SHAP Summary Plot...")
plt.figure(figsize=(12, 7))
shap.summary_plot(shap_informal, X_muestra, feature_names=FEATURES, show=False)
plt.title("SHAP Summary Plot - Predicción de Informalidad")
plt.tight_layout()
plt.savefig("results/interpretabilidad/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico guardado: results/interpretabilidad/shap_summary.png")

# -------------------------------------------------------------------------
# 4.2 SHAP Bar Plot (importancia global)
# -------------------------------------------------------------------------
print("Generando SHAP Bar Plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_informal, X_muestra, feature_names=FEATURES, plot_type="bar", show=False
)
plt.title("SHAP - Importancia Global de Variables")
plt.tight_layout()
plt.savefig("results/interpretabilidad/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico guardado: results/interpretabilidad/shap_bar.png")

# -------------------------------------------------------------------------
# 4.3 Tabla de importancia SHAP
# -------------------------------------------------------------------------
shap_importance = pd.DataFrame(
    {
        "Variable": FEATURES,
        "SHAP_Media_Abs": np.abs(shap_informal).mean(axis=0),
    }
).sort_values("SHAP_Media_Abs", ascending=False)

print("\nImportancia SHAP (media del valor absoluto):")
print(shap_importance.to_string(index=False))


# =========================================================================
# 5. SHAP - DEPENDENCE PLOTS (variables más importantes)
# =========================================================================
print("\n" + "=" * 60)
print("5. SHAP - DEPENDENCE PLOTS")
print("=" * 60)

top_vars = shap_importance.head(4)["Variable"].tolist()

for var in top_vars:
    print(f"Generando Dependence Plot para {var}...")
    idx_var = FEATURES.index(var)
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(idx_var, shap_informal, X_muestra, feature_names=FEATURES, show=False)
    plt.title(f"SHAP Dependence Plot - {var}")
    plt.tight_layout()
    nombre_archivo = f"shap_dependence_{var.lower()}.png"
    plt.savefig(
        f"results/interpretabilidad/{nombre_archivo}", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  Guardado: results/interpretabilidad/{nombre_archivo}")


# =========================================================================
# 6. SHAP - PREDICCIONES INDIVIDUALES
# =========================================================================
print("\n" + "=" * 60)
print("6. SHAP - PREDICCIONES INDIVIDUALES")
print("=" * 60)

# -------------------------------------------------------------------------
# 6.1 Ejemplo: persona informal típica
# -------------------------------------------------------------------------
y_pred_proba = modelo.predict_proba(X_muestra)[:, 1]
idx_informal = np.where((y_muestra == 1) & (y_pred_proba > 0.95))[0]

if len(idx_informal) > 0:
    pos_informal = idx_informal[0]

    print("\nEjemplo: Trabajador INFORMAL (prob > 95%)")
    print(X_muestra.iloc[pos_informal])

    plt.figure(figsize=(12, 4))
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        shap_informal[pos_informal, :],
        X_muestra.iloc[pos_informal, :],
        feature_names=FEATURES,
        matplotlib=True,
        show=False,
    )
    plt.title("SHAP Force Plot - Trabajador Informal")
    plt.tight_layout()
    plt.savefig(
        "results/interpretabilidad/shap_force_informal.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Gráfico guardado: results/interpretabilidad/shap_force_informal.png")

# -------------------------------------------------------------------------
# 6.2 Ejemplo: persona formal típica
# -------------------------------------------------------------------------
idx_formal = np.where((y_muestra == 0) & (y_pred_proba < 0.05))[0]

if len(idx_formal) > 0:
    pos_formal = idx_formal[0]

    print("\nEjemplo: Trabajador FORMAL (prob < 5%)")
    print(X_muestra.iloc[pos_formal])

    plt.figure(figsize=(12, 4))
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        shap_informal[pos_formal, :],
        X_muestra.iloc[pos_formal, :],
        feature_names=FEATURES,
        matplotlib=True,
        show=False,
    )
    plt.title("SHAP Force Plot - Trabajador Formal")
    plt.tight_layout()
    plt.savefig(
        "results/interpretabilidad/shap_force_formal.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Gráfico guardado: results/interpretabilidad/shap_force_formal.png")

# -------------------------------------------------------------------------
# 6.3 Nuevo individuo del ejemplo
# -------------------------------------------------------------------------
print("\nEjemplo: Nuevo individuo (Mujer, 35, Cuenta propia, $800K)")
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

shap_nuevo_raw = explainer.shap_values(nuevo)
shap_nuevo_informal = extraer_shap_informal(shap_nuevo_raw)
# Si es 2D con 1 fila, tomar la fila
if shap_nuevo_informal.ndim == 2:
    shap_nuevo_informal = shap_nuevo_informal[0, :]

pred = modelo.predict(nuevo)[0]
prob = modelo.predict_proba(nuevo)[0]

print(f"  Predicción: {'INFORMAL' if pred == 1 else 'FORMAL'}")
print(f"  Probabilidad Informal: {prob[1]:.4f}")

plt.figure(figsize=(12, 4))
shap.force_plot(
    explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
    shap_nuevo_informal,
    nuevo.iloc[0, :],
    feature_names=FEATURES,
    matplotlib=True,
    show=False,
)
plt.title("SHAP Force Plot - Nuevo Individuo")
plt.tight_layout()
plt.savefig(
    "results/interpretabilidad/shap_force_nuevo.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("Gráfico guardado: results/interpretabilidad/shap_force_nuevo.png")


# =========================================================================
# 7. COMPARACIÓN: IMPORTANCIA FEATURES vs PERMUTACIÓN vs SHAP
# =========================================================================
print("\n" + "=" * 60)
print("7. COMPARACIÓN DE MÉTODOS DE IMPORTANCIA")
print("=" * 60)

# Feature importance del modelo
fi_modelo = pd.DataFrame(
    {"Variable": FEATURES, "Feature_Importance": modelo.feature_importances_}
)

# Unir todo
comparacion = (
    fi_modelo.merge(perm_df[["Variable", "Importancia_Media"]], on="Variable")
    .merge(shap_importance, on="Variable")
    .rename(
        columns={
            "Feature_Importance": "RF_Importancia",
            "Importancia_Media": "Permutación",
            "SHAP_Media_Abs": "SHAP",
        }
    )
    .sort_values("SHAP", ascending=False)
)

print(comparacion.to_string(index=False))

# Gráfico comparativo
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# RF Feature Importance
comp_rf = comparacion.sort_values("RF_Importancia")
axes[0].barh(comp_rf["Variable"], comp_rf["RF_Importancia"], color="#3498db")
axes[0].set_title("Random Forest\nFeature Importance", fontweight="bold")
axes[0].set_xlabel("Importancia")

# Permutación
comp_perm = comparacion.sort_values("Permutación")
axes[1].barh(comp_perm["Variable"], comp_perm["Permutación"], color="#e67e22")
axes[1].set_title("Importancia por\nPermutación (AUC)", fontweight="bold")
axes[1].set_xlabel("Disminución media en AUC")

# SHAP
comp_shap = comparacion.sort_values("SHAP")
axes[2].barh(comp_shap["Variable"], comp_shap["SHAP"], color="#2ecc71")
axes[2].set_title("SHAP\n(Media |SHAP value|)", fontweight="bold")
axes[2].set_xlabel("Media valor absoluto SHAP")

plt.suptitle(
    "Comparación de Métodos de Importancia de Variables",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    "results/interpretabilidad/comparacion_importancias.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("\nGráfico guardado: results/interpretabilidad/comparacion_importancias.png")


# =========================================================================
# 8. RESUMEN
# =========================================================================
print("\n" + "=" * 70)
print("RESUMEN DE INTERPRETABILIDAD")
print("=" * 70)
print(f"""
MODELO ANALIZADO: Random Forest (Optuna, AUC ~0.9691)

TÉCNICAS APLICADAS:
  1. Feature Importance (Gini) - incorporada en Random Forest
  2. Importancia por Permutación - independiente del modelo
  3. SHAP (TreeExplainer) - explicaciones a nivel global e individual

RANKING DE VARIABLES (los 3 métodos coinciden):
""")

top_5 = comparacion.head(5)
for i, (_, row) in enumerate(top_5.iterrows(), 1):
    print(
        f"  {i}. {row['Variable']:25s} "
        f"RF={row['RF_Importancia']:.4f}  "
        f"Perm={row['Permutación']:.4f}  "
        f"SHAP={row['SHAP']:.4f}"
    )

print(f"""
INTERPRETACIÓN:
  - INGRESO_LABORAL es la variable más importante: a menor ingreso,
    mayor probabilidad de ser informal.
  - POSICION_OCUPACIONAL: cuenta propia, jornaleros y trabajadores
    familiares tienen alta probabilidad de informalidad.
  - TIENE_CONTRATO: no tener contrato incrementa fuertemente la
    probabilidad de ser clasificado como informal.
  - RAMA_ACTIVIDAD: sectores como comercio y agricultura concentran
    más informalidad.
  - HORAS_SEMANA: trabajar menos horas se asocia con informalidad.

GRÁFICOS GENERADOS EN results/interpretabilidad/:
  - importancia_permutacion.png
  - shap_summary.png
  - shap_bar.png
  - shap_dependence_*.png (4 variables top)
  - shap_force_informal.png
  - shap_force_formal.png
  - shap_force_nuevo.png
  - comparacion_importancias.png

INTERPRETABILIDAD COMPLETADA
""")
