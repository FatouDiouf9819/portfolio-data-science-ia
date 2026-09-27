#  Prédiction de la durée de vie utile d’un moteur d’avion — NASA C-MAPSS

##  Présentation du projet

Ce projet porte sur la **prédiction de la durée de vie utile restante d’un moteur aéronautique**, appelée **RUL (Remaining Useful Life)**, à partir de données de dégradation issues du jeu de données **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**.

L’objectif est d’exploiter les données provenant des cycles de fonctionnement et des capteurs des moteurs afin de construire des modèles de **Machine Learning capables d’estimer le nombre de cycles restant avant la fin de vie utile d’un moteur**.

Ce projet s’inscrit dans le domaine de la **maintenance prédictive**, où les modèles de Machine Learning peuvent être utilisés pour analyser la dégradation des équipements et contribuer à la planification des opérations de maintenance.

---

##  Objectifs du projet

Le projet a pour objectifs de :

1. **Préparer et nettoyer les données** issues du dataset NASA C-MAPSS.
2. **Construire la variable cible RUL** pour chaque moteur.
3. **Analyser les données des capteurs** et leur évolution au cours des cycles.
4. **Prétraiter les données** afin de les rendre adaptées aux modèles de Machine Learning.
5. **Construire plusieurs modèles de régression** pour prédire le RUL.
6. **Comparer les performances** des différents modèles.
7. **Sélectionner le modèle présentant les meilleures performances** dans la configuration expérimentale utilisée.
8. Utiliser les prédictions du RUL comme **indicateur potentiel pour la maintenance prédictive**.

---

##  Dataset — NASA C-MAPSS

Le projet utilise le jeu de données :

**NASA Turbofan Jet Engine Degradation Simulation — C-MAPSS**

Ce dataset simule la dégradation progressive de moteurs aéronautiques au cours de leur fonctionnement.

Les données permettent de suivre plusieurs moteurs à travers leurs différents cycles opérationnels et d’observer l’évolution de leurs paramètres de fonctionnement.

### Principales informations utilisées

Le dataset contient notamment :

- **Identifiant du moteur** : permet de distinguer les différentes unités moteur.
- **Cycle de fonctionnement** : indique l’évolution du moteur au cours du temps.
- **Paramètres opérationnels** : décrivent les conditions de fonctionnement.
- **Capteurs** : mesures issues des différents capteurs du moteur.
- **RUL** : durée de vie restante du moteur, utilisée comme variable cible.

Le projet exploite notamment les données provenant de **21 capteurs**.

---

##  Problématique

La problématique étudiée peut être formulée ainsi :

> **Étant donné l’historique de fonctionnement et les mesures des capteurs d’un moteur, peut-on estimer sa durée de vie utile restante (RUL) ?**

Il s’agit donc d’un **problème de régression supervisée**, puisque la variable cible RUL est une valeur numérique continue.

---

##  Construction de la variable cible — RUL

La variable **RUL (Remaining Useful Life)** représente le nombre de cycles restant avant la fin de vie utile du moteur.

Pour chaque moteur, le RUL est calculé à partir de son dernier cycle observé et du cycle courant :

```text
RUL = dernier cycle du moteur - cycle courant
```

Ainsi :

- au début de la vie du moteur, le RUL est élevé ;
- au fur et à mesure que les cycles augmentent, le RUL diminue ;
- lorsque le moteur atteint son dernier cycle observé, le RUL devient égal à `0`.

Cette variable constitue la **variable cible** utilisée pour entraîner les modèles de Machine Learning.

---

##  Prétraitement et nettoyage des données

Avant l’entraînement des modèles, plusieurs étapes de préparation des données ont été réalisées.

###  Analyse des variables

Les différentes variables du dataset ont été analysées afin d’identifier les informations pertinentes pour la prédiction du RUL.

###  Sélection des variables

Les capteurs présentant une variance très faible ou une information limitée ont été identifiés afin de réduire les variables peu informatives pour la modélisation.

###  Détection des valeurs aberrantes

Les valeurs aberrantes (*outliers*) ont été détectées à l’aide de la méthode de l’**IQR (Interquartile Range)**.

L’IQR permet de déterminer une plage statistique à partir du premier et du troisième quartile afin d’identifier les observations situées en dehors des limites considérées.

###  Traitement des valeurs aberrantes

Les valeurs identifiées comme aberrantes ont été remplacées par la **médiane** de la variable concernée.

Cette étape permet de limiter l’influence des valeurs extrêmes sur l’apprentissage des modèles.

###  Mise à l’échelle des données

Les variables utilisées pour la modélisation sont préparées et mises à l’échelle afin de faciliter leur utilisation par les algorithmes de Machine Learning.

---

##  Analyse exploratoire des données

Une analyse exploratoire a été réalisée afin de mieux comprendre :

- la structure du dataset ;
- l’évolution des moteurs au cours des cycles ;
- les caractéristiques des capteurs ;
- la distribution des variables ;
- la relation entre les variables explicatives et le RUL ;
- la présence éventuelle de valeurs aberrantes.

Des outils de visualisation tels que **Matplotlib** et **Seaborn** ont été utilisés pour représenter et analyser les données.

---

##  Modèles de Machine Learning

Trois modèles de régression ont été étudiés dans le cadre de ce projet.

### 1. Régression linéaire

La **Régression linéaire** est utilisée comme modèle de référence (*baseline*).

Elle permet d’établir un premier niveau de performance et de disposer d’un point de comparaison avec les modèles plus complexes.

### 2. Random Forest

Le **Random Forest Regressor** est un modèle d’ensemble reposant sur plusieurs arbres de décision.

Il permet de modéliser des relations non linéaires entre les caractéristiques des moteurs et leur durée de vie restante.

Une recherche d’hyperparamètres a été réalisée afin d’identifier une configuration adaptée au modèle.

### 3. XGBoost

**XGBoost Regressor** est un modèle de gradient boosting basé sur des arbres de décision.

Il permet de modéliser des relations complexes et non linéaires entre les données des capteurs et la durée de vie restante du moteur.

Une recherche d’hyperparamètres a également été utilisée dans le processus de modélisation.

---

## ⚙️ Optimisation des modèles

Afin d’améliorer les performances des modèles d’ensemble, une recherche d’hyperparamètres a été réalisée.

### Random Forest

La recherche d’hyperparamètres du Random Forest a permis de retenir notamment la configuration suivante :

- `n_estimators = 400`
- `max_depth = 6`

### XGBoost

Une recherche par **GridSearchCV** a également été utilisée pour rechercher une configuration adaptée au modèle XGBoost.

Le modèle final utilisé pour les prédictions correspond au **meilleur estimateur obtenu à partir de la recherche d’hyperparamètres**.

---

##  Méthodes d’évaluation

Les modèles sont évalués à l’aide de trois métriques principales.

### RMSE — Root Mean Squared Error

Le **RMSE** mesure l’écart entre les valeurs réelles et les valeurs prédites.

Il pénalise davantage les erreurs importantes.

**Plus le RMSE est faible, meilleure est la performance du modèle.**

### MAE — Mean Absolute Error

Le **MAE** mesure l’erreur absolue moyenne entre les valeurs réelles et les valeurs prédites.

**Plus le MAE est faible, meilleure est la performance du modèle.**

### R² — Coefficient de détermination

Le **R²** mesure la proportion de la variance de la variable cible expliquée par le modèle.

**Plus le R² est élevé, meilleure est la performance du modèle.**

---

##  Résultats et comparaison des modèles

Les trois modèles ont été évalués sur le même jeu de test à partir de trois métriques de régression :

- **RMSE (Root Mean Squared Error)** : mesure l’écart entre les valeurs réelles et prédites, avec une pénalisation plus forte des grandes erreurs. Plus il est faible, meilleur est le modèle.
- **MAE (Mean Absolute Error)** : mesure l’erreur absolue moyenne entre les valeurs réelles et prédites. Plus il est faible, meilleur est le modèle.
- **R² (coefficient de détermination)** : mesure la proportion de la variance de la variable cible expliquée par le modèle. Plus il est élevé, meilleur est le modèle.

| Modèle | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---:|---:|---:|
| Régression linéaire | 42.181 | 32.254 | 0.593 |
| Random Forest | 41.077 | 30.440 | 0.614 |
| **XGBoost** | **40.089** | **29.283** | **0.633** |

###  Modèle retenu : XGBoost

À l’issue de la comparaison, **XGBoost a été retenu comme modèle final** pour la prédiction du RUL.

Ce choix repose sur les résultats obtenus sur le jeu de test :

- il présente le **RMSE le plus faible (40.089)** ;
- il présente le **MAE le plus faible (29.283)** ;
- il obtient le **R² le plus élevé (0.633)**.

Ainsi, dans la configuration expérimentale utilisée dans ce projet, **XGBoost fournit les estimations de RUL les plus proches des valeurs observées parmi les trois modèles étudiés**.

Le choix de XGBoost est donc fondé sur les performances prédictives mesurées par les métriques retenues, et non sur une supériorité générale du modèle dans tous les contextes.

---

## 🔎 Analyse des prédictions

Après l’entraînement, les modèles sont utilisés pour produire des estimations de RUL.

Une analyse complémentaire est réalisée sur les observations correspondant aux derniers cycles disponibles pour chaque moteur.

Cette étape permet d’observer les estimations produites par les modèles lorsque les moteurs se trouvent à un stade avancé de leur fonctionnement.

Les prédictions peuvent ainsi être comparées aux valeurs réelles de RUL afin d’analyser le comportement du modèle dans différentes situations.

---

##  Analyse des erreurs

L’analyse des erreurs permet d’étudier la différence entre :

- le **RUL réel** ;
- le **RUL prédit** par le modèle.

L’erreur de prédiction peut être analysée afin d’identifier les observations pour lesquelles le modèle présente les écarts les plus importants.

Cette analyse complète l’évaluation globale réalisée à partir du RMSE, du MAE et du R².

---

##  Application à la maintenance prédictive

La prédiction du **RUL** peut être utilisée comme un indicateur de l’état de dégradation d’un moteur.

Une estimation faible du RUL peut indiquer qu’un moteur se rapproche de la fin de sa durée de vie utile.

Dans un contexte industriel, ce type de prédiction peut contribuer à :

- anticiper les opérations de maintenance ;
- identifier les équipements nécessitant une attention particulière ;
- mieux planifier les interventions ;
- réduire les risques liés aux défaillances imprévues ;
- exploiter les données issues des capteurs pour améliorer la surveillance des équipements.

Les prédictions constituent ainsi un **indicateur d’aide à la décision** dans une démarche de maintenance prédictive.

---

##  Workflow du projet

Le workflow général du projet peut être résumé ainsi :

```text
NASA C-MAPSS Dataset
        │
        ▼
Analyse exploratoire des données
        │
        ▼
Nettoyage et prétraitement
        │
        ▼
Construction du RUL
        │
        ▼
Préparation des variables
        │
        ▼
┌──────────────────────────────┐
│       Modèles testés         │
├──────────────────────────────┤
│ Régression linéaire          │
│ Random Forest                │
│ XGBoost                      │
└──────────────┬───────────────┘
               │
               ▼
     Évaluation des modèles
       RMSE / MAE / R²
               │
               ▼
    Comparaison des performances
               │
               ▼
        XGBoost retenu
               │
               ▼
          Prédiction RUL
               │
               ▼
     Maintenance prédictive
```

---

##  Technologies et bibliothèques utilisées

### Langage

- **Python 3**

### Manipulation des données

- `pandas`
- `numpy`

### Visualisation

- `matplotlib`
- `seaborn`

### Machine Learning

- `scikit-learn`
- `xgboost`

### Environnement

- **Jupyter Notebook**

---

## 📁 Structure du projet

```text
Prediction-panne-d-un-avion/
│
├── README.md
├── Prediction_panne_d_un_avion.ipynb
├── requirements.txt
└── ...
```

---

##  Installation et utilisation

### 1. Cloner le dépôt

```bash
git clone https://github.com/fatoudiouf/Prediction-panne-d-un-avion.git
cd Prediction-panne-d-un-avion
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer le notebook

```bash
jupyter notebook Prediction_panne_d_un_avion.ipynb
```

Le notebook contient les différentes étapes du projet :

1. chargement des données ;
2. analyse exploratoire ;
3. nettoyage des données ;
4. traitement des valeurs aberrantes ;
5. construction du RUL ;
6. préparation des variables ;
7. entraînement des modèles ;
8. recherche d’hyperparamètres ;
9. prédictions ;
10. évaluation des performances ;
11. comparaison des modèles ;
12. sélection du modèle final.

---

##  Résumé du projet

| Élément | Résultat |
|---|---|
| **Problématique** | Prédiction de la durée de vie restante d’un moteur |
| **Domaine** | Maintenance prédictive |
| **Dataset** | NASA C-MAPSS |
| **Type de problème** | Régression supervisée |
| **Variable cible** | RUL — Remaining Useful Life |
| **Nombre de capteurs exploités** | 21 |
| **Modèles étudiés** | Régression linéaire, Random Forest, XGBoost |
| **Modèle retenu** | **XGBoost** |
| **RMSE XGBoost** | **40.089** |
| **MAE XGBoost** | **29.283** |
| **R² XGBoost** | **0.633** |
| **Langage** | Python |
| **Domaine d’application** | Maintenance prédictive |

---

##  Compétences mises en pratique

Ce projet permet de mettre en pratique plusieurs compétences en **Data Science, Machine Learning et modélisation** :

- analyse exploratoire des données ;
- nettoyage et préparation des données ;
- traitement des valeurs aberrantes ;
- analyse de données de capteurs ;
- ingénierie de variables ;
- construction d’une variable cible ;
- modélisation de problèmes de régression ;
- entraînement de modèles de Machine Learning ;
- Régression linéaire ;
- Random Forest ;
- XGBoost ;
- recherche d’hyperparamètres ;
- comparaison de modèles ;
- évaluation avec RMSE, MAE et R² ;
- analyse des erreurs ;
- interprétation des résultats ;
- application du Machine Learning à la **maintenance prédictive**.

Ce projet illustre également le lien entre **modélisation mathématique, programmation, analyse de données et intelligence artificielle**.

---

##  Intérêt académique et professionnel

Ce projet s’inscrit dans mon parcours en **Modélisation Mathématique et Simulation Numérique** et dans mon orientation progressive vers la **Data Science et l’Intelligence Artificielle**.

Il m’a permis de mettre en pratique des connaissances en :

- mathématiques appliquées ;
- programmation Python ;
- analyse de données ;
- Machine Learning ;
- modélisation prédictive ;
- optimisation de modèles ;
- interprétation des résultats.

Le projet constitue ainsi une application concrète de la modélisation mathématique à une problématique industrielle de **maintenance prédictive**.

---

##  Limites et perspectives

Les résultats présentés correspondent à la **configuration expérimentale et au jeu de test utilisés dans ce projet**.

Ils ne permettent pas de conclure à une supériorité générale de XGBoost dans tous les contextes de prédiction de RUL.

Plusieurs pistes pourraient être envisagées pour approfondir le projet :

- utiliser une stratégie de séparation des données basée sur les moteurs afin d’éviter qu’un même moteur soit présent dans les ensembles d’entraînement et de test ;
- améliorer le traitement des données temporelles ;
- étudier davantage l’évolution des capteurs au cours des cycles ;
- tester d’autres modèles de Machine Learning ;
- explorer les approches de Deep Learning adaptées aux séries temporelles ;
- expérimenter des architectures telles que les réseaux récurrents ou les Transformers ;
- améliorer l’interprétabilité des prédictions ;
- développer une interface permettant de visualiser les estimations de RUL.

---

##  Auteur

**Fatou DIOUF**

Étudiante en **Master 1 Modélisation Mathématique et Simulation Numérique**

### 🔗 GitHub

[![GitHub](https://img.shields.io/badge/GitHub-fatoudiouf-black?logo=github)](https://github.com/fatoudiouf)
