# Prédiction du churn bancaire

Projet réalisé dans le cadre du cours de MLOps en M2 SIAD Data sciences. Notre objectif est ici de prédire le churn bancaire en développant un projet qui respecte les bonnes pratiques du MLOps. Les consignes précisant explicitement que c'est principalement ce second point que ce projet évalue et moins la qualité/originalité de la modélisation, nous travaillerons ici sur une variante d'une problématique assez connue en DS : la modélisation d'un churn, ici dans le secteur bancaire. 

## Objectif

L'objectif de ce projet est donc de construire un modèle, en appliquant les bonnes pratiques du MLOps, capable de prédire si oui ou non un client va quitter la banque. Nous travaillerons donc ici à la construction d'un classifieur binaire. 
Comme souvent avec les problématiques de churn, nous avons un déséquilibre au niveau de la variable à expliquer : le churn n'est observé que pour 20% du dataset. 
Dans ce cadre les métriques d'évaluation des performances du modèle que nous allons juger seront principalement l'accuracy, le recall, la précision, le F1-score, l'AUC et le PR-AUC. 

## Données
- **Source** : https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset 
- **Date de téléchargement** : 15/10/2026
- **Volumétrie** : 10 000 lignes et 14 variables
- **Emplacement** : `data/raw/churn_modelling.csv` 

### Dictionnaire des variables

- RowNumber: Le numéro attribué à chaque ligne du jeu de données.
- CustomerId: Identifiant unique attribué à chaque client.
- Surname: Le nom de famille du client.
- CreditScore: Le score de crédit du client.
- Geography: La localisation géographique du client.
- Gender: Le sexe du client.
- Age: L'âge du client.
- Tenure: Le nombre d'années depuis lesquelles l'individu est client de la banque.
- Balance: Le solde du compte bancaire du client.
- NumOfProducts: Le nombre de produits bancaires détenus par le client.
- HasCrCard: Indique si le client possède une carte de crédit ou non.
- IsActiveMember: Indique si le client est un membre actif de la banque ou non.
- EstimatedSalary: Salaire estimé du client.
- Exited: Indique si le client a quitté la banque (notre variable cible).

## Installation
**Prérequis** : Python 3.12 (développé et testé avec Python 3.12.10) et Git.

### 1. Cloner le dépôt
```bash
git clone https://github.com/RomanDelbare/mlops_roman_delbare.git
cd mlops_roman_delbare
```

### 2. Créer et activer l'environnement virtuel
**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
**Linux / macOS**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances
```bash
python -m pip install --upgrade pip
pip install -r envi_dev.txt
```

(Pour une exécution seule, sans les outils de développement faire : `pip install -r requirements.txt`).

## Structure du dépôt
```
data/raw/            données brutes (non modifiées)
data/processed/      données générées par le code (ignorées par Git)
notebooks/           un notebook par personne, sorties vidées avant commit
src/donnees.py       chargement des données et découpage train/test communs
tests/               tests automatiques (pytest)
```

## Utilisation
Le CSV utilise `;` comme séparateur et commence par un BOM : il faut toujours le charger avec la fonction commune, qui applique ces réglages.

```python
from src.donnees import charger_donnees, retirer_identifiants, separer_train_test

df = retirer_identifiants(charger_donnees())
X_train, X_test, y_train, y_test = separer_train_test(df)  # stratifié, graine 42
```

Tous les modèles sont entraînés et évalués sur ce même découpage.

Depuis un notebook rangé dans `notebooks/`, ajouter d'abord la racine du dépôt au chemin :
```python
import sys; sys.path.append("..")
```

## Vérifications
```bash
ruff check .   # style du code
pytest         # tests
```
Ces deux commandes tournent aussi automatiquement sur GitHub à chaque pull request.

Avant de commiter un notebook, vider ses sorties :
```bash
jupyter nbconvert --clear-output --inplace notebooks/mon_notebook.ipynb
```

## État d'avancement
RAF : EDA + Feature eng + Modélisation + Évaluation des perfs