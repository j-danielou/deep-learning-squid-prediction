# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:23:01 2026

@author: jdanielou
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

#1. Configs
TARGET_YEAR = 2024
PATH_CSV = f"D:/deep-learning-squid-prediction/data/dataset_ML_baseline_{TARGET_YEAR}.csv"

if not os.path.exists(PATH_CSV):
    raise FileNotFoundError(f"Exécute d'abord le script 01_build_baseline_dataset.py pour générer {PATH_CSV}")

print(f"Chargement du dataset : {PATH_CSV}")
df = pd.read_csv(PATH_CSV)

#2. Preparation des donnée
FEATURES = ['thetao', 'so', 'uo', 'vo', 'eke', 'sst_grad', 'so_grad', 'elevation', 'CHL', 'chl_grad']
TARGET = 'CPUE_tonnes_par_heure'

X = df[FEATURES]
y = df[TARGET]

#80% training, 20% validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Échantillons pour l'apprentissage : {len(X_train)}")
print(f"Échantillons pour l'évaluation    : {len(X_test)}")

#3. Entrainement modele
print("\nCréation et entraînement du 'Comité d'Experts' (XGBoost)...")
model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=100,       #nombre d'arbres de décision
    learning_rate=0.1,      #prudence avec laquelle chaque arbre corrige le précédent
    max_depth=6,            #profondeur des questions (complexité des interactions)
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

#4. mode eval
y_pred = model.predict(X_test)

# Sécurité halieutique : on ne peut pas pêcher des valeurs négatives
y_pred = np.maximum(y_pred, 0)

# Calcul des scores
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "="*50)
print(" RÉSULTATS DE LA BASELINE (XGBoost)")
print("="*50)
print(f"Score R² (Précision) : {r2:.4f}")
print(f"Erreur RMSE          : {rmse:.2f} tonnes/h")
print("="*50)


#5. Explicabilité
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': FEATURES, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#2c3e50')
plt.xlabel("Importance de la variable (Score XGBoost)")
plt.title(f"Quelles variables contrôlent la présence du calmar en {TARGET_YEAR} ?")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Sauvegarde et affichage
chemin_graphique = "D:/deep-learning-squid-prediction/output/feature_importance_baseline.png"
os.makedirs(os.path.dirname(chemin_graphique), exist_ok=True)
plt.savefig(chemin_graphique, dpi=300)
print(f"\nGraphique sauvegardé dans : {chemin_graphique}")

plt.show()