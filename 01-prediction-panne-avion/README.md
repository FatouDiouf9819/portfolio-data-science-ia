# ✈️ Prédiction de Panne de Moteur d'Avion (NASA C-MAPSS) - Estimation du RUL

Ce projet applique des techniques de **Machine Learning** et de **Maintenance Prédictive** sur le jeu de données **NASA Turbofan Jet Engine Degradation Simulation (C-MAPSS)** pour prédire la durée de vie restante (**RUL - Remaining Useful Life**) d'un moteur d'avion en fonction de ses cycles d'utilisation et des relevés de ses 21 capteurs de bord.

---

## 🎯 Objectifs du Projet

1. **Calcul et Ingénierie de la variable cible (RUL)** : Calcul du nombre de cycles restants avant la défaillance critique de chaque unité motrice.
2. **Prétraitement & Nettoyage des Données** :
   - Sélection et élimination des capteurs à variance quasi-nulle (absence de signal pertinent).
   - Traitement et imputation des valeurs aberrantes (*outliers*) via la méthode de l'intervalle interquartile (**IQR**) et remplacement par la médiane.
   - Normalisation et mise à l'échelle des données de capteurs.
3. **Modélisation Prédictive & Comparaison d'Algorithmes** :
   - **Régression Linéaire (Baseline)**
   - **Random Forest Regressor** (avec recherche d'hyperparamètres optimaux)
   - **XGBoost Regressor**
4. **Prise de Décision Opérationnelle** :
   - Détection automatique et identification des moteurs prioritaires les plus proches de la panne pour la maintenance d'urgence.

---

## 📊 Jeu de Données (NASA C-MAPSS)

Le dataset comprend les variables suivantes :
- **ID Moteur (`unit_number`)** : Identifiant unique de chaque turboréacteur.
- **Temps en cycles (`time_in_cycles`)** : Nombre d'heures de vol / cycles opérationnels.
- **Paramètres opérationnels (1 à 3)** : Réglages d'altitude, vitesse et conditions de vol.
- **Relevés capteurs (Capteurs 1 à 21)** : Température totale, pression statique, vitesse du compresseur, rapport de pression, débit de carburant, etc.

---

## 🛠️ Technologies et Librairies Utilisées

- **Langage** : Python 3
- **Manipulation de Données** : `pandas`, `numpy`
- **Visualisation de Données** : `matplotlib`, `seaborn`
- **Machine Learning** : `scikit-learn`, `xgboost`

---

## 🚀 Installation et Utilisation

### 1. Cloner le dépôt
```bash
git clone https://github.com/fatimadiouf/Prediction-panne-d-un-avion.git
cd Prediction-panne-d-un-avion
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer le Notebook Jupyter
```bash
jupyter notebook Prediction_panne_d_un_avion.ipynb
```

---

## 👩‍💻 Auteur

* **Fatou DIOUF** - [GitHub @fatimadiouf](https://github.com/fatimadiouf)
