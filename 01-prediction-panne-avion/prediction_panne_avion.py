# -*- coding: utf-8 -*-
# Projet : Prediction-panne-d-un-avion
# Code original extrait de Prediction_panne_d_un_avion.ipynb
# Auteur : Fatou DIOUF


# %% [markdown]
'''
# **Prédiction de panne d'un avion dataset de la NASA**
'''

# %% [markdown]
'''
# **Objectif: Prédir le temps de vie d'un moteur d'avoin (RUL)**

---
'''

# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import json

# %% [code]
from google.colab import drive
drive.mount('/content/drive')

# %% [code]
from google.colab import files
files.upload()

# %% [code]
import pandas as pd

df = pd.read_csv("train_FD001.txt", sep=" ", header=None)
df = df.dropna(axis=1)

df.head()

# %% [code]
df.shape

# %% [markdown]
'''
0=ID moteur

1=Temps (cycle)

2=Paramètre opérationnel 1

3=Paramètre opérationnel 2

4=Paramètre opérationnel 3

5=Capteur 1

6=Capteur 2

...

26=Capteur 26
'''

# %% [code]
cols=(
    ["id",
    "cycle",
    "Paramètre opérationnel 1",
    "Paramètre opérationnel 2",
    "Paramètre opérationnel 3",]
    +["Capteur "+str(i) for i in range(1,22)]
)
df.columns=cols

# %% [code]
df.columns

# %% [code]
df

# %% [code]
df.info()

# %% [code]
df['cycle'].max()

# %% [code]
df['Capteur 1'].unique()

# %% [code]
df['Paramètre opérationnel 3'].unique()

# %% [code]
df['Capteur 17'].unique()

# %% [code]
df['Capteur 18'].unique()

# %% [code]
df['Capteur 1'].unique()

# %% [code]
df['Capteur 2'].unique()

# %% [code]
import seaborn as sns
import matplotlib.pyplot as plt

for i in df.columns:
  plt.figure(figsize=(10,5))
  sns.boxplot(df[i],color='pink')
  plt.show()


# %% [markdown]
'''
**Dans ce adataset on doit prédire L'RUL Remaining Useful Life temps de vie du capteur mais dans ce cas on doit le calculé et l’ajouter dans le dataset apr ce qu'elle ne s'y touve pas encore**
'''

# %% [code]
RUL = df.groupby("id")["cycle"].max().reset_index()
RUL.columns = ["id", "max_cycle"]

df = df.merge(RUL, on="id")

df["RUL"] = df["max_cycle"] - df["cycle"]

# %% [code]
df["RUL"]

# %% [code]
df.columns

# %% [code]
df.shape

# %% [code]
df.nunique()

# %% [markdown]
'''
# **On supprime les colonne qui n'ont pas beaucoups d'informations**
'''

# %% [code]
df.drop(columns=["Paramètre opérationnel 3","Capteur 1","Capteur 5","Capteur 6","Capteur 10","Capteur 16","Capteur 18","Capteur 19"],inplace=True)

# %% [code]
df.columns

# %% [markdown]
'''
# **Renplacons les outliers par laa valeure mediane grace a la methode IQR**
'''

# %% [code]
Q1=df.quantile(0.25)
Q3=df.quantile(0.75)

IQR=Q3-Q1

Max=Q3+1.5*IQR
Min=Q1-1.5*IQR

med=df.median()
df = df.where((df >= Min) & (df <= Max), med,axis=1)


# %% [code]
sns.pairplot(df)

# %% [code]

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)


# %% [markdown]
'''
# **Random forest Regressor**
'''

# %% [code]
X=df.drop(["RUL","max_cycle","cycle","id"],axis=1)
y=df["RUL"]

# %% [code]
df_test_last = df.groupby("id").last().reset_index()


# %% [code]
X = df.drop(["RUL","max_cycle","cycle","id"], axis=1)
y = df["RUL"]

# Dernière observation de chaque moteur
df_test_last = df.groupby("id").last().reset_index()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Séparation des données pour entraîner le modèle
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Création du modèle Random Forest
rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=6,
    random_state=42
)

# Entraînement
rf.fit(X_train, y_train)

# Prédiction sur le test
y1_pred = rf.predict(X_test)

print(y1_pred)

# %% [markdown]
'''
# **chérchons les valeures optimals pour le random forest**
'''

# %% [code]
from sklearn.model_selection import GridSearchCV

model = RandomForestRegressor()

param_grid = {
    'max_depth': [1,2,3, 4,5, 6],
    'n_estimators': [100,200,300,400, 500,600],
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='r2'
)

grid.fit(X_train, y_train)

# %% [code]
print("Meilleurs paramètres :", grid.best_params_)
print("Meilleur score :", grid.best_score_)

# %% [code]
mse_rf = mean_squared_error(y_test, y1_pred)
r2_rf = r2_score(y_test, y1_pred)

print("MSE Random Forest :", mse_rf)
print("R² Random Forest :", r2_rf)

# %% [code]
X_engine_last = df_test_last.drop(
    ["RUL", "max_cycle", "cycle", "id"],
    axis=1
)

y_engine_last_real = df_test_last["RUL"].values

# %% [code]
y_rf_engine_last = rf.predict(X_engine_last)

y_rf_engine_last = np.maximum(
    y_rf_engine_last,
    0
)

# %% [code]
min_index = np.argmin(y_rf_engine_last)

print("======================================")
print("MOTEUR PRÉDIT COMME LE PLUS PROCHE DE LA FIN")
print("======================================")

print("ID moteur :", df_test_last.iloc[min_index]["id"])
print("RUL réelle :", y_engine_last_real[min_index])
print("RUL prédite :", y_rf_engine_last[min_index])

# %% [code]
resultats_rf = pd.DataFrame({
    "id": df_test_last["id"],
    "RUL_reelle": y_engine_last_real,
    "RUL_predite": y_rf_engine_last
})

print(resultats_rf.to_string(index=False))

# %% [code]
from google.colab import files
files.upload()
y_true = np.loadtxt("RUL_FD001.txt")
y_true

# %% [markdown]
'''
# **XGBoost Regressor**
'''

# %% [code]
# ============================================================
# XGBOOST - PRÉDICTION DE LA RUL
# ============================================================

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. MODÈLE DE BASE
# ============================================================

model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)


# ============================================================
# 2. GRILLE DE RECHERCHE
# ============================================================

param_grid_2 = {
    "max_depth": [1, 2, 3, 4, 5, 6],
    "learning_rate": [0.1, 0.001, 0.0001, 0.2, 0.002, 0.0002]
}


# ============================================================
# 3. GRID SEARCH
# ============================================================

grid_xgb = GridSearchCV(
    estimator=model,
    param_grid=param_grid_2,
    scoring="r2",
    cv=5,
    n_jobs=-1
)

grid_xgb.fit(X_train, y_train)


# ============================================================
# 4. MEILLEURS PARAMÈTRES
# ============================================================

print("======================================")
print("OPTIMISATION XGBOOST")
print("======================================")

print("Meilleurs paramètres :", grid_xgb.best_params_)
print("Meilleur score R² :", grid_xgb.best_score_)


# ============================================================
# 5. MEILLEUR MODÈLE
# ============================================================

xgb = grid_xgb.best_estimator_


# ============================================================
# 6. PRÉDICTION SUR LE JEU DE TEST
# ============================================================

y_xgb_pred = xgb.predict(X_test)

# Éviter les RUL négatives
y_xgb_pred = np.maximum(y_xgb_pred, 0)


# ============================================================
# 7. ÉVALUATION
# ============================================================

mse_xgb = mean_squared_error(y_test, y_xgb_pred)
r2_xgb = r2_score(y_test, y_xgb_pred)

print("\n======================================")
print("ÉVALUATION XGBOOST")
print("======================================")

print("MSE :", mse_xgb)
print("R²  :", r2_xgb)


# ============================================================
# 8. DERNIÈRE OBSERVATION DE CHAQUE MOTEUR
# ============================================================

df_test_last = df.groupby("id").last().reset_index()

X_engine_last = df_test_last.drop(
    ["RUL", "max_cycle", "cycle", "id"],
    axis=1
)

y_engine_last_real = df_test_last["RUL"].values


# ============================================================
# 9. PRÉDICTION DE LA RUL POUR CHAQUE MOTEUR
# ============================================================

y_xgb_engine_last = xgb.predict(X_engine_last)

# Éviter les RUL négatives
y_xgb_engine_last = np.maximum(
    y_xgb_engine_last,
    0
)


# ============================================================
# 10. TABLEAU DES RÉSULTATS
# ============================================================

resultats_xgb = pd.DataFrame({
    "id": df_test_last["id"],
    "RUL_reelle": y_engine_last_real,
    "RUL_predite": y_xgb_engine_last
})


print("\n======================================")
print("RUL DES DERNIÈRES OBSERVATIONS")
print("======================================")

print(
    resultats_xgb.to_string(index=False)
)


# ============================================================
# 11. ERREUR ABSOLUE PAR MOTEUR
# ============================================================

resultats_xgb["Erreur_absolue"] = abs(
    resultats_xgb["RUL_reelle"] -
    resultats_xgb["RUL_predite"]
)


# ============================================================
# 12. ERREUR MOYENNE
# ============================================================

mae_xgb_last = resultats_xgb["Erreur_absolue"].mean()

print("\nMAE XGBoost sur les dernières observations :",
      mae_xgb_last)


# ============================================================
# 13. LES 10 PLUS GRANDES ERREURS
# ============================================================

print("\n======================================")
print("10 PLUS GRANDES ERREURS")
print("======================================")

print(
    resultats_xgb
    .sort_values(
        by="Erreur_absolue",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 14. MOTEUR PRÉDIT COMME LE PLUS PROCHE DE LA FIN
# ============================================================

min_index = np.argmin(y_xgb_engine_last)

print("\n======================================")
print("MOTEUR PRÉDIT COMME LE PLUS PROCHE DE LA FIN")
print("======================================")

print(
    "ID moteur :",
    df_test_last.iloc[min_index]["id"]
)

print(
    "RUL réelle :",
    y_engine_last_real[min_index]
)

print(
    "RUL prédite :",
    y_xgb_engine_last[min_index]
)

# %% [markdown]
'''
# **Linear Regression**
'''

# %% [code]
# ============================================================
# RÉGRESSION LINÉAIRE
# ============================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd


# 1. Création du modèle
rl = LinearRegression()


# 2. Entraînement
rl.fit(X_train, y_train)


# 3. Prédiction sur les données de test
y_rl_pred = rl.predict(X_test)

# La RUL ne peut pas être négative
y_rl_pred = np.maximum(y_rl_pred, 0)


# 4. Évaluation du modèle
mse_rl = mean_squared_error(y_test, y_rl_pred)
r2_rl = r2_score(y_test, y_rl_pred)

print("======================================")
print("RÉGRESSION LINÉAIRE")
print("======================================")

print("MSE :", mse_rl)
print("R²  :", r2_rl)


# ============================================================
# 5. DERNIÈRE OBSERVATION DE CHAQUE MOTEUR
# ============================================================

df_test_last = df.groupby("id").last().reset_index()

X_engine_last = df_test_last.drop(
    ["RUL", "max_cycle", "cycle", "id"],
    axis=1
)

y_engine_last_real = df_test_last["RUL"].values


# ============================================================
# 6. PRÉDICTION DE LA RUL DE CHAQUE MOTEUR
# ============================================================

y_rl_engine_last = rl.predict(X_engine_last)

# Éviter les valeurs négatives
y_rl_engine_last = np.maximum(
    y_rl_engine_last,
    0
)


# ============================================================
# 7. TABLEAU DES RÉSULTATS
# ============================================================

resultats_rl = pd.DataFrame({
    "id": df_test_last["id"],
    "RUL_reelle": y_engine_last_real,
    "RUL_predite": y_rl_engine_last
})

print("\n======================================")
print("RUL DES DERNIÈRES OBSERVATIONS")
print("======================================")

print(resultats_rl.to_string(index=False))


# ============================================================
# 8. ERREUR ABSOLUE
# ============================================================

resultats_rl["Erreur_absolue"] = abs(
    resultats_rl["RUL_reelle"] -
    resultats_rl["RUL_predite"]
)

mae_rl = resultats_rl["Erreur_absolue"].mean()

print("\nMAE Régression Linéaire :", mae_rl)


# ============================================================
# 9. MOTEUR PRÉDIT COMME LE PLUS PROCHE DE LA FIN
# ============================================================

min_index = np.argmin(y_rl_engine_last)

print("\n======================================")
print("MOTEUR PRÉDIT COMME LE PLUS PROCHE DE LA FIN")
print("======================================")

print(
    "ID moteur :",
    df_test_last.iloc[min_index]["id"]
)

print(
    "RUL réelle :",
    y_engine_last_real[min_index]
)

print(
    "RUL prédite :",
    y_rl_engine_last[min_index]
)

# %% [code]
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================
# RÉGRESSION LINÉAIRE
# ==============================
rmse_rl = np.sqrt(mean_squared_error(y_test, y_rl_pred))
mae_rl = mean_absolute_error(y_test, y_rl_pred)
r2_rl = r2_score(y_test, y_rl_pred)


# ==============================
# RANDOM FOREST
# ==============================
rmse_rf = np.sqrt(mean_squared_error(y_test, y1_pred))
mae_rf = mean_absolute_error(y_test, y1_pred)
r2_rf = r2_score(y_test, y1_pred)


# ==============================
# XGBOOST
# ==============================
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_xgb_pred))
mae_xgb = mean_absolute_error(y_test, y_xgb_pred)
r2_xgb = r2_score(y_test, y_xgb_pred)


# ==============================
# AFFICHAGE
# ==============================
print("======================================")
print("COMPARAISON DES MODÈLES")
print("======================================")

print("\nRégression linéaire")
print("RMSE :", rmse_rl)
print("MAE  :", mae_rl)
print("R²   :", r2_rl)

print("\nRandom Forest")
print("RMSE :", rmse_rf)
print("MAE  :", mae_rf)
print("R²   :", r2_rf)

print("\nXGBoost")
print("RMSE :", rmse_xgb)
print("MAE  :", mae_xgb)
print("R²   :", r2_xgb)

# %% [code]
resultats_modeles = pd.DataFrame({
    "Modèle": [
        "Régression linéaire",
        "Random Forest",
        "XGBoost"
    ],
    "RMSE": [
        rmse_rl,
        rmse_rf,
        rmse_xgb
    ],
    "MAE": [
        mae_rl,
        mae_rf,
        mae_xgb
    ],
    "R²": [
        r2_rl,
        r2_rf,
        r2_xgb
    ]
})

print(resultats_modeles)
