#  Analyse et Profilage du Catalogue E-Commerce & Gestion des Stocks (Product Sales & Inventory Analysis)

Ce projet d'**Analyse Exploratoire des Données (EDA)** et de **Business Intelligence** porte sur l'étude d'un catalogue de produits e-commerce multi-catégories. L'objectif est d'analyser la structure des prix, la répartition des stocks, les caractéristiques des produits (marques, catégories, tailles, coloris, disponibilités) et d'identifier des insights décisionnels pour la gestion d'inventaire et la stratégie commerciale.

---

##  Résumé du Projet

| Élément | Description |
|---|---|
| **Problématique** | Analyse exploratoire, politique de prix et optimisation de l'inventaire e-commerce |
| **Domaine** | E-Commerce, Retail Analytics & Gestion de Stock |
| **Type d'Analyse** | Statistique descriptive univariée & bivariée, Profilage de catalogue |
| **Variables Clés** | `Price`, `Stock`, `Category`, `Brand`, `Size`, `Color`, `Availability` |
| **Outils & Librairies** | Python, Pandas, NumPy, Seaborn, Matplotlib |
| **Auteur** | **Fatou DIOUF** — Étudiante en Master Modélisation Mathématique & Simulation Numérique |

---

##  Objectifs de l'Étude

1. **Structuration & Typage des Données** :
   - Encodage optimal des variables catégorielles (`Brand`, `Category`, `Size`, `Color`, `Availability`, `Currency`).
   - Audit de la qualité des données (détection des doublons, valeurs manquantes et valeurs aberrantes).
2. **Analyse Statistique & Distributionnelle** :
   - Étude de l'hétérogénéité des prix (prix moyen : ~451 $ USD, écart-type : ~281 $ USD, dispersion de 1 $ à 999 $ USD).
   - Analyse des niveaux de stock (médiane à 576 unités, dispersion de 10 à 998 unités).
3. **Exploration Bivariée & Croisements Métier** :
   - Relations entre catégories de produits et tarification (catégories premium vs accessibles).
   - Distribution des niveaux de stock par catégorie, coloris et statut de disponibilité (*In stock*, *Limited stock*, *Backorder*, *Pre-order*).
   - Analyse de la corrélation entre prix unitaire et volume en stock.

---

##  Principaux Résultats & Insights

- **Diversité du catalogue** : Le catalogue comprend 100 marques et fournisseurs distincts, avec une forte représentation de catégories telles que *Automotive*, *Cleaning supplies*, *Health & wellness*, et *Kid's clothing*.
- **Dispersion des Prix** :
  - 50 % des produits sont proposés à un prix inférieur ou égal à 408 $ USD.
  - 25 % des articles haut de gamme dépassent 679 $ USD (notamment dans les équipements automobiles et mobilier).
  - Les accessoires et produits d'entrée de gamme démarrent à 1 $ USD.
- **Gestion des Stocks & Disponibilité** :
  - Les articles à forte valeur unitaire conservent des niveaux de stock significatifs, reflétant une politique d'anticipation ou une rotation plus lente.
  - Les produits en *Pre-order* ou *Out of stock* sont concentrés sur des gammes de prix spécifiques.

---

##  Stack Technique

- **Langage** : Python 3
- **Manipulation & Traitement de Données** : `pandas`, `numpy`
- **Visualisation Graphique** : `seaborn`, `matplotlib`
- **Environnement** : Jupyter Notebook

---

##  Structure du Projet

```text
Product-Sales-Analysis/
│
├── README.md                          <- Présentation et rapport d'analyse
├── requirements.txt                   <- Dépendances Python
├── product_sales_analysis.ipynb       <- Notebook complet et interactif
└── product_sales_analysis.py          <- Script Python nettoyé et exécutable
```

---

##  Installation et Utilisation

### 1. Cloner le dépôt
```bash
git clone https://github.com/FatouDiouf9819/Product-Sales-Analysis.git
cd Product-Sales-Analysis
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer l'analyse

* **Via Jupyter Notebook** :
```bash
jupyter notebook product_sales_analysis.ipynb
```

* **Ou via le script Python** :
```bash
python product_sales_analysis.py
```

---

##  Auteur

**Fatou DIOUF**  
Étudiante en **Master 1 Modélisation Mathématique et Simulation Numérique**  
*Profil GitHub :* [github.com/FatouDiouf9819](https://github.com/FatouDiouf9819)
