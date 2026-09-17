# -*- coding: utf-8 -*-
# Projet : Telco-Customer-Churn
# Code original extrait de telco_customer_churn.ipynb
# Auteur : Fatima DIOUF


# %% [markdown]
# # **Objectif** :
# **Prédire ou comprendre le churn (départ client)**

# %% [code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from google.colab import drive\

drive.mount('/content/drive')


# %% [code]
df=pd.read_excel('/content/drive/MyDrive/Formation_IA_ForceN/Telco_customer_churn.xlsx')
df.head()

# %% [markdown]
# # **Liste des variables et leurs significations**
# 
# **1. Identification du client**
# 
# **CustomerID**: identifiant unique du client
# Count : compteur (souvent = 1 pour chaque ligne, peu utile)
# 
# **2. Informations géographiques**
# 
# **Country**: pays du client
# 
# **State**: état / région (ex : California)
# 
# **City** : ville
# 
# **Zip Code**: code postal
# 
# **Lat Long** : coordonnées géographiques (latitude, longitude combinées)
# 
# **Latitude** : latitude
# 
# **Longitude**: longitude
# 
# **3. Informations démographiques**
# 
# **Gender** : sexe (Male / Female)
# 
# **Senior Citizen**: personne âgée (1 = oui, 0 = non)
# 
# **Partner :** a un partenaire (Yes / No)
# 
# **Dependents**: a des personnes à charge (Yes / No)
# 
# *Utilité : comprendre quels profils quittent le service*
# 
# **4. Services souscrits**
# 
# **Phone Service**: a un service téléphonique
# 
# **Multiple Lines**: plusieurs lignes téléphoniques
# 
# **Internet**
# 
# **Internet Service**: type d’internet (DSL, Fiber optic, None)
# 
# **Online Security** : sécurité en ligne
# 
# **Online Backup** : sauvegarde en ligne
# 
# **Device Protection** : protection des appareils
# 
# **Tech Support**: assistance technique
# 
# **Streaming TV**: TV en streaming
# 
# **Streaming Movies** : films en streaming
# 
# *Utilité : voir quels services fidélisent ou font fuir*
# 
# **5. Informations sur le contrat**
# 
# **Contract** : type de contrat
# 
# -Month-to-month (mensuel)
# 
# -One year
# 
# -Two year
# 
# **Paperless Billing**: facture électronique (Yes / No)
# 
# **Payment Method**: méthode de paiement
# 
# -Electronic check
# 
# -Mailed check
# 
# -Bank transfer
# 
# -Credit card
# 
# *Très important pour analyser le churn*
# 
# **6. Données financières**
# 
# **Monthly Charges**: montant payé par mois
# 
# **Total Charges**: montant total payé depuis le début
# 
# *Permet de voir*:
# 
# clients chers
# 
# clients rentables
# 
# **7. Variables liées au churn (OBJECTIF)**
# 
# **Churn Label :**
# 
#  Yes = client parti
# 
#  No = client resté
# 
# **Churn Value** :
# 
# 1 = churn
# 
# 0 = pas churn
# 
# **Churn Score**: score de probabilité de départ (souvent calculé par un modèle)
# 
# **Churn Reason** : raison du départ
# 
# Exemples :
# 
# -Competitor made better offer
# 
# -Moved
# 
# -Price too high
# 
# **8. Valeur client**
# 
# CLTV (Customer Lifetime Value) : valeur totale estimée du client pour
# l’entreprise
# 
# *Très important pour le business* :
# savoir quels clients sont les plus rentables

# %% [code]
df['Phone Service']

# %% [code]
df['Internet Service']

# %% [code]
df['Tenure Months']

# %% [code]
df.info()

# %% [code]
df['CLTV']

# %% [code]
l=['CustomerID','Count','Country','State','City','Zip Code','Lat Long','Latitude','Longitude','Churn Label','Churn Score','Churn Reason','CLTV']

# %% [code]
df1=df.drop(columns=l)
df1

# %% [code]
df1['Total Charges']

# %% [markdown]
# # **Corriger le type de Total Charges**

# %% [code]
df1['Total Charges'].dtypes

# %% [code]
df1['Total Charges'] = pd.to_numeric(df1['Total Charges'], errors='coerce')

# %% [code]
df1['Total Charges']=df1['Total Charges'].astype(float)

# %% [code]
df1.info()

# %% [code]
df1.isnull().sum()

# %% [code]
df1.duplicated()

# %% [code]
for i in df1.columns:
  plt.figure(figsize=(10,5))
  sns.boxplot(df1[i],color='skyblue')
  plt.show()

# %% [code]
col_Categories=df1.select_dtypes(include='object').columns
col_Categories

# %% [code]
df1_Encoding=pd.get_dummies(df1,columns=col_Categories,drop_first=True)
df1_Encoding = df1_Encoding.fillna(df1_Encoding.mean())
df1_Encoding

# %% [code]
X=df1_Encoding.drop(columns='Churn Value')
y=df1_Encoding['Churn Value']

# %% [markdown]
# # **Random Forest**

# %% [code]
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
#separation des variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#Creation du model
rf=RandomForestClassifier(n_estimators=100, max_depth=5)
#Entrainement du model
rf.fit(X_train,y_train)
#prediction
y1_pred=rf.predict(X_test)


# %% [code]
y1_pred

# %% [code]
from sklearn.metrics import accuracy_score
# Évaluation
print("Accuracy:", accuracy_score(y_test, y1_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y1_pred))
print("Rpport de classification:\n", classification_report(y_test, y1_pred))

# %% [code]
cm1=confusion_matrix(y_test,y1_pred)
nom_classe=['non churn = 0','churn = 1']
sns.heatmap(cm1,cmap='Reds',annot=True,fmt='d',
            xticklabels=nom_classe,
            yticklabels=nom_classe,

            )
plt.xlabel('Valeur prédite')
plt.ylabel('Valeur réelle')
plt.title('Matrice de confusion Random Forest Classifier')
plt.show()

# %% [markdown]
# # **Logistic Regression**

# %% [code]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#separation des variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# definition du model
lr=LogisticRegression()
# Entrainement du model
lr.fit(X_train,y_train)


# %% [code]
y2_pred=lr.predict(X_test)

# %% [code]
y2_pred

# %% [code]
# Évaluation
print("Accuracy:", accuracy_score(y_test, y2_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test,y2_pred))

# %% [code]
cm=confusion_matrix(y_test,y2_pred)

# %% [code]
cm

# %% [code]
cm=confusion_matrix(y_test,y2_pred)
nom_classe=['non churn = 0','churn = 1']
sns.heatmap(cm,cmap='pink',annot=True,fmt='d',
            xticklabels=nom_classe,
            yticklabels=nom_classe,

            )
plt.xlabel('Valeur prédite')
plt.ylabel('Valeur réelle')
plt.title('Matrice de confusion Logistic Regression')
plt.show()

# %% [markdown]
# Pour la methode RandomForest,on a une accuracy de 78,7 pourcent alors que pour le regression Logistiaue on a 79,8 pourcent de plus pour la regression logistic les FN et les FP sont moins importante que pour la methode de Random Forest alors on peut en deduire que la regression est legerement plus precise que Random

# %% [markdown]
# # **XGBoost Classifier**

# %% [code]
from xgboost import XGBClassifier
xbg=XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
xbg.fit(X_train,y_train)
y4_pred=xbg.predict(X_test)



# %% [code]
y4_pred

# %% [code]
print("Accuracy:", accuracy_score(y_test, y4_pred))

# %% [code]
cm=confusion_matrix(y_test,y4_pred)
nom_classe=['non churn = 0','churn = 1']
sns.heatmap(cm,cmap='grey',annot=True,fmt='d',
            xticklabels=nom_classe,
            yticklabels=nom_classe,

            )
plt.xlabel('Valeur prédite')
plt.ylabel('Valeur reelle')
plt.title('Matrice de confusion XGBoost Classifier')
plt.show()
