# Prédiction du churn bancaire

Projet réalisé dans le cadre du cours de MLOps en M2 SIAD Data sciences. Notre objectif est ici de prédire le churn bancaire en développant un projet qui respecte les bonnes pratiques du MLOps et en essayant de faire un bon tracking MLFlow. Les consignes précisant explicitement que c'est principalement ce second point que ce projet évalue et moins la qualité/originalité de la modélisation, nous travaillerons ici sur une variante d'une problématique assez connue en DS : la modélisation d'un churn, ici dans le secteur bancaire, pour ne pas complexifier inutilement les choses. 

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
pip install -e .
```

(Pour une exécution seule, sans les outils de développement, faire : `pip install -r requirements.txt` et `pip install -e .`).

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
Le choix du modèle est justifié dans `notebooks/comparaison_modeles/choix_du_modele.md`.

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
Tous les essais sont enregistrés dans MLflow, sous `churn_bancaire`, dans
`mlflow.db` et pour le modèle final, le modèle lui-même (signature, exemple d'entrée). 
Pour les consulter, depuis la racine du projet :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Comme `mlflow.db` n'est pas dans le dépôt, on à un fichier csv de suivi par auteur. 
Pour remmette tt ça au propre on fusionne le tout avec :

```bash
python -m base_commune.fusion_rapports   # écrit reports/synthese_runs.csv
```

## Vérifications
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
