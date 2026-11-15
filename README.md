# Prédiction du churn bancaire

L'objectif est ici de construire un modèle en développant un projet qui respecte les bonnes pratiques du MLOps, avec tracking MLFlow et processus d'intégration continue dans le cadre de notre cours de MLOps en M2 SIAD DS. Les consignes précisant explicitement que c'est principalement l'aspect MLOps que ce projet évalue et moins la qualité/originalité de la modélisation, nous travaillerons ici sur une variante d'une problématique assez connue en DS : la modélisation d'un churn.

## Objectif

L'objectif de ce projet est donc de construire un modèle, en appliquant les bonnes pratiques du MLOps, capable de prédire si oui ou non un client va quitter la banque. Nous travaillerons donc ici à la construction d'un classifieur binaire. 
Nos différents tests de modèles seront enregistrés via MLFlow
Comme souvent avec les problématiques de churn, nous avons un déséquilibre au niveau de la variable à expliquer : le churn n'est observé que pour 20% du dataset. 
Ici la métrique de performance qu'on va principalement juger est le PR-AUC. En effet comme on est sur une problématique de churn on peut avoir un relativement bon AUC 'classique", un bon accuracy etc même si on rate quasi tout les churners. Avec le PR-AUC comme métrique à optimiser principalement on va méchaniquement se concentrer sur la détection des churners et donc orienter le modèle dans le bon sens. 
Nous conserverons néanmoins l'AUC "standard" (cad issu de la courbe ROC classique), le rappel, la précision et le F1-score comme métriques complémentaires afin d'obtenir une évaluation plus complète des performances.

## Données
- **Source** : https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset
- **Date de téléchargement** : 15/10/2026
- **Volumétrie** : 10 000 lignes et 14 variables
- **Emplacement** : `data/raw/churn_modelling.csv` (fichier brut, jamais modifié)
- **Colonnes non utilisées par le modèle** : `RowNumber`, `CustomerId`, `Surname` (identifiants)

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

### 1. Cloner le dépôt :

```bash
git clone https://github.com/RomanDelbare/mlops_roman_delbare.git
cd mlops_roman_delbare
```

### 2. Créer l'environnement virtuel (une seule fois) :

**Windows (PowerShell ou Git Bash)**
```bash
py -3.12 -m venv .venv
```

**Linux / macOS**
```bash
python3.12 -m venv .venv
```

### 3. Activer l'environnement (à refaire dans chaque nouveau terminal) :

**Windows, PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```
Si PowerShell refuse avec une erreur de « politique d'exécution », lancer d'abord
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (valable pour la fenêtre en cours).

**Windows, Git Bash**
```bash
source .venv/Scripts/activate
```
**Linux / macOS**
```bash
source .venv/bin/activate
```

### 4. Installer les dépendances :

```bash
python -m pip install --upgrade pip
pip install -r envi_dev.txt
pip install -e .
```
(Pour une exécution seule, sans les outils de développement : `pip install -r requirements.txt` puis `pip install -e .`.)

### 5. Vérifier l'installation

```bash
python -m ruff check .       # style du code : doit afficher « All checks passed! »
python -m pytest tests -q    # tests : environ une minute
```
## Structure du dépôt
```
.github/workflows/    intégration continue (ruff et pytest)
data/raw/             données brutes, jamais modifiées
data/train/, test/    copie du découpage (via une fonction de découpage de données qu'on réutilise partout)
base_commune/         fonctions communes : les ressources qu'on réutilise/se partage
modele_final/         modèle retenu : paramètres, préparation, entraînement, prédiction
notebooks/            EDA, comparaison des modèles, choix du seuil de décision
tests/                tests automatiques (avec pytest)
reports/              suivi MLflow en CSV (un fichier par auteur + une synthèse)
```

## Modèle final
Le modèle retenu est un XGBoost (300 arbres de profondeur 3, taux d'apprentissage de 0,05)
avec un seuil de décision de 0,30, choisi pour maximiser le F1. Ses paramètres sont dans
`modele_final/params.yaml`. 
Le choix du modèle est justifié dans `notebooks/comparaison_modeles/04_choix_du_modele.md`.

### Entraîner
```bash
python -m modele_final.entrainement_final
```
Cette commande contrôle les données, choisit le seuil sur des prédictions « hors pli » du jeu
d'entraînement, évalue le modèle une seule fois sur le jeu de test, puis l'enregistre dans
MLflow : modèle `churn_xgboost` du registre, alias `modele_final`. 

### Prédire
```python
from base_commune.outils import charger_donnees, decouper_donnees
from modele_final.prediction import predire_clients

_, clients, _, _ = decouper_donnees(charger_donnees())  # clients du jeu de test
predictions = predire_clients(clients.head(5))
print(predictions)  # tableau de 0 et de 1 : 1 = le client part, 0 = il reste
```
`predire_clients` vérifie d'abord les données (colonnes attendues, modalités autorisées,
bornes) : la colonne cible `Exited` ne doit pas être fournie. Il faut avoir lancé
l'entraînement avant. On passe par ce module plutôt que par le `predict()` du modèle qui
coupe a 0.5 au lieu de notre seuil.

### Résultats
| Évaluation | Précision | Rappel | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Validation croisée, seuil 0,5 | 0,771 | 0,457 | 0,573 | 0,699 | 0,864 |
| Hors pli (entraînement), seuil 0,30 | 0,613 | 0,658 | 0,635 | | |
| **Jeu de test, seuil 0,30 (utilisé une fois)** | **0,613** | **0,646** | **0,629** | **0,716** | **0,869** |

Sur le jeu de test (2 000 clients, environ 407 départs), le modèle retrouve environ 65 % des
départs avec une précision d'environ trois fois le taux de base (20 %). Le résultat est proche
de l'estimation hors pli, ce qui indique que le seuil se généralise bien. 

## Suivi des expériences
Tous les essais sont suivis avec MLflow (experience `churn_bancaire`) : paramètres, métriques de
validation croisée etc... Pour le modèle final, le modèle lui-même (signature, exemple d'entrée) 
est inscrit dans le registre.

La base MLflow (`mlflow.db`) est locale et n'est pas versionnée. Après un clone, elle ne contient
donc que l'essai du modèle final, créé par `python -m modele_final.entrainement_final`. Les essais
d'exploration (KNN, Random Forest, SVM, régression logistique, XGBoost) sont conservés dans
`reports/` : un fichier CSV par auteur (`runs_<auteur>.csv`) et une synthèse fusionnée.

Pourr consulter la base locale, depuis la racine du projet :

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
puis ouvrir http://127.0.0.1:5000 

```bash
python -m base_commune.fusion_rapports   # écrit reports/synthese_runs.csv

```

## Organisation du travail
- Une branche par fonctionnalité ; aucun commit direct sur `main` (branche protégée).
- Chaque modification passe par une pull request, relue et approuvée par l'autre auteur, avec la
  CI (ruff et pytest) exécutée sur chaque PR, puis fusionnée en « squash ».
- Les tests couvrent le chargement des données, le prétraitement, l'entraînement, la prédiction
  et la fusion des rapports de suivi.


## Vérifications

Nous avons mis en place des tests (unitaires et d'intégration) et une verif ruff pour respecter 
une convention de style de code que nous avons définis entre nous

```bash
ruff check .   # style du code
pytest         # tests
```
Ces deux commandes tournent aussi automatiquement sur GitHub à chaque pull request.

Avant de commiter un notebook, on vide ses sorties :

```bash
jupyter nbconvert --clear-output --inplace notebooks/mon_notebook.ipynb
```

## État d'avancement
Version 1.0 : EDA, base commune, tests et intégration continue, comparaison de cinq modèles
(KNN, Random Forest, SVM, régression logistique, XGBoost), choix du seuil de décision, modèle
final (préparation, entraînement, prédiction), suivi MLflow exporté et fusionné.
Tout semble bon 
