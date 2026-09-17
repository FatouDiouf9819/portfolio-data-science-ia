# -*- coding: utf-8 -*-
# Projet : Prediction-panne-d-un-avion
# Code original extrait de Prediction_panne_d_un_avion.ipynb
# Auteur : Fatima DIOUF


# %% [markdown]
# # **Prédiction de panne d'un avion dataset de la NASA**

# %% [markdown]
# # **Objectif: Prédir le temps de vie d'un moteur d'avoin (RUL)**
# 
# ---

# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
drive.mount('/content/drive')
import json

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
# 0=ID moteur
# 
# 1=Temps (cycle)
# 
# 2=Paramètre opérationnel 1
# 
# 3=Paramètre opérationnel 2
# 
# 4=Paramètre opérationnel 3
# 
# 5=Capteur 1
# 
# 6=Capteur 2
# 
# ...
# 
# 26=Capteur 26

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
# **Dans ce adataset on doit prédire L'RUL Remaining Useful Life temps de vie du capteur mais dans ce cas on doit le calculé et l’ajouter dans le dataset apr ce qu'elle ne s'y touve pas encore**

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
# # **On supprime les colonne qui n'ont pas beaucoups d'informations**

# %% [code]
df.drop(columns=["Paramètre opérationnel 3","Capteur 1","Capteur 5","Capteur 6","Capteur 10","Capteur 16","Capteur 18","Capteur 19"],inplace=True)

# %% [code]
df.columns

# %% [markdown]
# # **Renplacons les outliers par laa valeure mediane grace a la methode IQR**

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
# # **Random forest Regressor**

# %% [code]
X=df.drop(["RUL","max_cycle","cycle","id"],axis=1)
y=df["RUL"]

# %% [code]
df_test_last = df.groupby("id").last().reset_index()
X_test = df_test_last.drop(["RUL","max_cycle","cycle","id"], axis=1)
y_test = df_test_last["RUL"]

# %% [code]
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
#decoupage du dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#Definition du model
rf=RandomForestRegressor(n_estimators=500, max_depth=6)
#entrainement du model
rf.fit(X_train,y_train)
# 3. prédiction
y1_pred = rf.predict(X_test)
y1_pred

# %% [markdown]
# # **chérchons les valeures optimals pour le random forest**

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
print('MSE EST ',mean_squared_error(y_test, y1_pred))
print('R2 EST ',r2_score(y_test, y1_pred))

# %% [code]
print('Le temps de vie pour la premiere moteur est :', y2_pred_df_last[0])

# %% [code]
from google.colab import files
files.upload()
y_true = np.loadtxt("RUL_FD001.txt")
y_true

# %% [code]
print('Le temps de vie pour la premiere moteur est :', y2_pred_df_last[0])

# %% [markdown]
# # **XGBoost Regressor**

# %% [code]
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# %% [code]
from sklearn.model_selection import GridSearchCV

model = XGBRegressor()

param_grid_1 = {
    'max_depth': [1,2,3, 4,5, 6],
    'learning_rate': [0.1,0.001,0.0001,0.2,0.002,0.0002],
}

grid_1 = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='r2'
)

grid_1.fit(X_train, y_train)

# %% [code]
print("Meilleurs paramètres :", grid_1.best_params_)
print("Meilleur score :", grid_1.best_score_)

# %% [code]
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

xgb = XGBRegressor(learning_rate=0.1,max_depth=4,random_state=42)
xgb.fit(X_train, y_train)
y2_pred = xgb.predict(X_test)
y2_pred = np.maximum(y2_pred, 0)
y2_pred

# %% [code]
print('MSE EST ',mean_squared_error(y_test, y2_pred))
print('R2 EST ',r2_score(y_test, y2_pred))


# %% [code]
print('Le temps de vie pour la premiere moteur est :', y2_pred_df_last[0])

# %% [code]
y2_pred_df_last.min()

# %% [code]
y2_pred_df_last.max()

# %% [markdown]
# # **Ici on affiche l'index , l'id et le temps de vie restant pour le moteur le plus proche de la panne**

# %% [code]
min_index = np.argmin(y2_pred_df_last)

print("Index :", min_index)

print("ID moteur :", df_test_last.iloc[min_index]["id"])

print("RUL prédit :", y2_pred_df_last[min_index])

# %% [markdown]
# # **Ici on affiche l'index , l'id et le temps de vie restant pour le moteur le ,moins proche de la panne**

# %% [code]
max_index = np.argmax(y2_pred_df_last)

print("Index :", max_index)

print("ID moteur :", df_test_last.iloc[max_index]["id"])

print("RUL prédit :", y2_pred_df_last[max_index])

# %% [markdown]
# # **Linear Regression**

# %% [code]
from sklearn.linear_model import  LinearRegression
rl=LinearRegression()
rl.fit(X_train,y_train)
y3_pred=rl.predict(X_test)


# %% [code]
print('Le temps de vie pour la premiere moteur est :', y3_pred[0])

# %% [code]
print('MSE EST ',mean_squared_error(y_test, y3_pred))
print('R2 EST ',r2_score(y_test, y3_pred))
