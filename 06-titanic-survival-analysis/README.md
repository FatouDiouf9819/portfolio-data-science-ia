# 🚢 Titanic - Analyse Exploratoire de Données (EDA) & Prétraitement

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge)](https://seaborn.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

Ce projet réalise une **Analyse Exploratoire des Données (EDA)** approfondie et un pipeline complet de **prétraitement / imputation des valeurs manquantes** sur le jeu de données historique du **Titanic**.

---

## 🎯 Objectifs du Projet

1. **Exploration & Compréhension des Données** :
   - Analyse structurelle des variables quantitatives (`Age`, `Fare`, `SibSp`, `Parch`) et qualitatives (`Pclass`, `Sex`, `Embarked`, `Survived`).
   - Typage optimal des variables catégorielles avec Pandas.
2. **Gestion Avancée des Valeurs Manquantes** :
   - Cartographie visuelle et corrélation des données manquantes via **`missingno`** (*matrix*, *bar plot*, *heatmap*, *dendrogram*).
   - Imputation statistique ciblée :
     - **Variables quantitatives (`Age`)** : Remplacement par la médiane (`SimpleImputer(strategy='median')`).
     - **Variables qualitatives (`Cabin`, `Embarked`)** : Remplacement par le mode / classe la plus fréquente (`SimpleImputer(strategy='most_frequent')`).
3. **Analyses Statistiques & Visualisations Multidimensionnelles** :
   - Répartition des passagers par classe (`Pclass`) en diagrammes à barres et camemberts (*pie charts*).
   - Distribution des âges et densités (KDE avec Seaborn).
   - Relations croisées entre la survie, le sexe, la classe et les tarifs payés (`catplot`, `countplot`).

---

## 📁 Structure du Projet

```text
├── Titanic.csv                # Dataset original du Titanic
├── exo_titanic.ipynb          # Notebook Jupyter interactif complet
├── titanic_analysis.py        # Script Python pur (exécutable directement)
├── requirements.txt           # Dépendances et bibliothèques requises
├── .gitignore                 # Fichiers ignorés par Git
└── README.md                  # Documentation du projet
```

---

## 🛠️ Technologies & Bibliothèques Utilisées

- **Langage** : Python 3
- **Manipulation de Données** : `pandas`, `numpy`
- **Prétraitement & Imputation** : `scikit-learn` (`SimpleImputer`, `KNNImputer`, `IterativeImputer`)
- **Visualisation & Analyse Graphique** : `matplotlib`, `seaborn`, `missingno`

---

## 🚀 Installation & Exécution

### 1. Cloner le dépôt
```bash
git clone https://github.com/fatimadiouf/Titanic-Survival-Analysis-EDA.git
cd Titanic-Survival-Analysis-EDA
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Exécuter l'analyse

- **Option A : Via Jupyter Notebook**
```bash
jupyter notebook exo_titanic.ipynb
```

- **Option B : Via le Script Python**
```bash
python titanic_analysis.py
```

---

## 👩‍💻 Auteur

* **Fatima DIOUF** - [GitHub @fatimadiouf](https://github.com/fatimadiouf)
