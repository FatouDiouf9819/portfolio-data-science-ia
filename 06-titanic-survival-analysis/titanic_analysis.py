# -*- coding: utf-8 -*-
# Projet : Titanic - Analyse Exploratoire & Prétraitement de Données (EDA)
# Code original extrait de exo_titanic.ipynb
# Auteur : Fatima DIOUF


# %% [code] - Cellule 0
#Importations des bibliotheques necessaires
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import missingno as msno
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import MissingIndicator, KNNImputer,SimpleImputer
from sklearn.impute import IterativeImputer
df = pd.read_csv('/content/Titanic.csv').set_index('PassengerId')

# %% [code] - Cellule 1
#Affissage des 5 premieres lignes
df.head()

# %% [code] - Cellule 2
# Vue d'enssemble sur les colonnes
df.info()

# %% [code] - Cellule 3
df['Pclass'].value_counts()

# %% [code] - Cellule 4
df['Survived'].value_counts()

# %% [code] - Cellule 5
df['SibSp'].value_counts()

# %% [code] - Cellule 6
df['Embarked'].value_counts()

# %% [code] - Cellule 7
df['Cabin'].value_counts()

# %% [code] - Cellule 8
#Affissage des 5 dernieres lignes
df.tail()

# %% [code] - Cellule 9
#Transformation des variable suivantes en var Categorielles
df['Survived']=df['Survived'].astype('category')
df['Sex']=df['Sex'].astype('category')
df['Embarked']=df['Embarked'].astype('category')
df['Pclass']=df['Pclass'].astype('category')

# %% [code] - Cellule 10
#Pour avoir une vue d'enssemble des colones de notre dataset et verifié si la modification fontionne
df.info()

# %% [code] - Cellule 11
#Stat descriptive des variables numerique ou quantitatives
df.describe()

# %% [code] - Cellule 12
#stat descriptives des variable categoriels ou quanlitatives
df.describe(include='category')

# %% [code] - Cellule 13
df

# %% [code] - Cellule 14
#cherchons si il y'a de valeures manquantes dans notre dataset
df.isnull().sum()

# %% [code] - Cellule 15
# Visualiser les valeurs manquantes avec missingno
msno.matrix(df)
plt.show()

# %% [code] - Cellule 16
#Diagramme en bar pour identifier les valeures manquantes
msno.bar(df)
plt.show()

# %% [code] - Cellule 17
#Matrice de correlation des valeures manquantes
msno.heatmap(df)
plt.show()

# %% [code] - Cellule 18
# Tableau des valeurs manquantes par colonne
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_table = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
print(missing_table)

# %% [code] - Cellule 19
# Analyser la corrélation entre les valeurs manquantes
msno.dendrogram(df)
plt.show()

# %% [code] - Cellule 20
#Traitement des valeurs manquantes des variable quantitatives en les remplacant par la moyenne
imputer_median = SimpleImputer(strategy='median')
df['Age'] = imputer_median.fit_transform(df[['Age']])
df.isnull().mean()

# %% [code] - Cellule 21
#Traitement des valeurs manquantes des variable qualitatives en les remplacant pas le mode
var_quali=['Cabin','Embarked']
imputer_mode = SimpleImputer(strategy='most_frequent')
df[var_quali]=imputer_mode.fit_transform(df[var_quali])
df.isnull().mode()

# %% [code] - Cellule 22
#Verifions si les valeurs manquante sont bien traités
msno.bar(df)
plt.show()

# %% [code] - Cellule 23
n=(df.dtypes == 'int64') | (df.dtypes == 'float64')
n

# %% [code] - Cellule 24
# Grouper le DataFrame 'df' par la colonne 'Age'
# et calculer la moyenne de toutes les autres colonnes *numériques*.
df.groupby('Age').mean(numeric_only=True)

# %% [code] - Cellule 25
x=df.loc[:,['Pclass','Survived','Sex']]

# %% [code] - Cellule 26
ax=sns.countplot(x='Pclass',data=df,label=['Premiere classe','Deuxieme class','Troisieme classe'])
plt.title('Repartition des classe')
plt.xlabel('Class')
plt.ylabel('effectif')
ax.patches[0].set_color('pink')
ax.patches[1].set_color('lime')
ax.patches[2].set_color('blue')
plt.legend()
plt.show()



# %% [code] - Cellule 27
df['Pclass'].value_counts().plot(kind='pie', autopct='%1.1f%%', title='Répartition des classes')
label=['Premiere classe','Deuxieme class','Troisieme classe']
plt.legend(label)
plt.show()

# %% [code] - Cellule 28
sns.histplot(x='Age',data=df, bins=20,palette='Set2',kde=True)
plt.title('Répartition des ages')
plt.xlabel('Age')
plt.ylabel('Effectif')
plt.title('Répartition des ages')
plt.show()

# %% [code] - Cellule 29
sns.catplot(x='Pclass',y='Age',data=df,hue='Sex')

# %% [code] - Cellule 30
sns.countplot(x='Embarked',data=df,hue='Fare')
plt.title('Répartition des prix')
plt.xlabel('Prix')
plt.ylabel('Effectif')
plt.show()

# %% [code] - Cellule 32
w=df['Age'].max()
i=df['Age'].idxmax()
df.loc[i,'Sex']
