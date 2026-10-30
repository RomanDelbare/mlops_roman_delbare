# Explication sur base_commune

Nous allons chercher à vraiment jouer le jeu du travail collaboratif pour pratiquer le MLOps de façon plus concrète.
Pour ça nous avons fait le choix de tester chacun de notre côté plusieurs classifieurs binaires et les transformations associées à ces derniers.

Notre but est de gagner du temps et potentiellement d'optimiser nos résultats en pouvant passer plus de temps sur chaque modèle, puis de mettre en commun les résultats
pour sélectionner le meilleur modèle. C'est sur ce dernier qu'on basera le reste du projet. Pour ça on va :

1) Uniformiser l'importation et le découpage du dataset + le calcul d'un certain nombre de métriques de performance.
2) Essayer de mettre en place un tracking avec MLflow.
3) Utiliser un petit csv qui compile les résumés de nos différents tests.

Cela va nous permettre de :
- Éviter de biaiser la comparaison entre nos modèles (par exemple en n'utilisant pas la même seed d'aléatoire ou en oubliant la stratification).
- Faciliter la comparaison des modèles : on peut tous les deux calculer toutes les métriques qu'on veut, mais on est sûrs d'avoir au moins une base de comparaison.
- Visualiser les résultats de chaque modèle en attaquant directement le csv plutôt que de relancer nos codes, pour gagner du temps entre nous dans la comparaison des modèles.
- Nous entraîner à utiliser MLflow, ce qui est nouveau pour nous deux (on va donc s'aider de l'IA générative sur cette partie).

Ainsi on a créé `outils.py`, dont les fonctions sont détaillées ci-dessous.

## Utilisation de `base_commune`

Le module `base_commune/outils.py` regroupe les fonctions communes du projet pour aller dans le sens de ce qu'on vient d'expliquer.
Prérequis : `pip install -e .` (voir Installation dans le README), puis dans le notebook ou le script :

```python
from base_commune.outils import (
    charger_donnees,
    configurer_mlflow,
    decouper_donnees,
    evaluer_modele,
    evaluer_test,
    exporter_runs,
    tags_reproductibilite,
)
```

### `charger_donnees()`
Charge le fichier brut `data/raw/churn_modelling.csv` et retire les colonnes identifiantes, inutiles pour tout type de modèle (`RowNumber`, `CustomerId`, `Surname`).
- Argument optionnel : `chemin` (par défaut le CSV du projet).
- Retourne un `DataFrame` de 11 colonnes (10 variables explicatives + `Exited`).

```python
df = charger_donnees()
```

### `decouper_donnees(df, test_size=0.2)`
Sépare les données en jeu d'entraînement et jeu de test, de façon stratifiée sur `Exited` et avec `random_state=42` (comme ça le découpage est identique pour tout le monde).
- Retourne `X_train, X_test, y_train, y_test`.

```python
X_train, X_test, y_train, y_test = decouper_donnees(df)
```

### `evaluer_modele(modele, X_train, y_train, n_splits=5)`
Évalue un `Pipeline` scikit-learn par validation croisée stratifiée (5 plis, `shuffle=True`, `random_state=42`) sur le jeu d'entraînement uniquement.
- Geography et Gender sont des variables texte : un modèle seul plante, il faut un `Pipeline` qui contient l'encodage (et la standardisation si besoin). Le prétraitement est alors recalculé dans chaque pli, ce qui évite les fuites d'information. Voir l'exemple complet plus bas.
- Retourne un dictionnaire de 18 métriques : pour chacune de `pr_auc`, `roc_auc`, `f1`, `precision`, `recall`, `accuracy`, les clés `cv_<métrique>_mean` et `cv_<métrique>_std` (moyenne et écart-type sur les plis) et `train_<métrique>_mean` (score sur les données d'entraînement).

```python
metriques = evaluer_modele(pipeline, X_train, y_train)
print(metriques["cv_pr_auc_mean"])
```

### `evaluer_test(modele, X_train, y_train, X_test, y_test, seuil=0.5)`
Entraîne le modèle sur tout le jeu d'entraînement puis le score sur le jeu de test.
- À n'utiliser qu'une seule fois, sur le modèle retenu à la fin, pour ne pas biaiser le score final.
- Retourne les 6 métriques avec le préfixe `test_` ; f1, précision, recall et accuracy sont calculés avec le seuil indiqué.

```python
resultats = evaluer_test(pipeline, X_train, y_train, X_test, y_test)
```

### `configurer_mlflow(nom_experience)`
À appeler une fois au début du notebook, avant tout `mlflow.start_run()`. Connecte MLflow à la base locale du projet (`mlflow.db`) et sélectionne (ou crée) l'expérience indiquée, pour que tous les essais soient enregistrés au même endroit.
Nb : celle-ci est générée avec l'IA comme on débute sur MLflow, fiabilité donc non garantie.

```python
configurer_mlflow("churn_bancaire")
```

### `tags_reproductibilite()`
Retourne un dictionnaire `{"git_commit", "git_dirty", "data_md5"}` : commit Git courant, présence de modifications non commitées, empreinte (MD5) du fichier de données brut. À enregistrer avec chaque run, avec l'auteur et la famille de modèle.
Nb : cette fonction est une suggestion de l'IA.

```python
mlflow.set_tags({"author": "david", "model_family": "logistic_regression", **tags_reproductibilite()})
```

### `exporter_runs(nom_experience, auteur)`
Exporte dans `reports/runs_<auteur>.csv` les runs terminés (statut `FINISHED`) de l'auteur indiqué (tag `author`) : paramètres, métriques, nom du run et tags de reproductibilité.
- Retourne le chemin du fichier créé.
- À relancer après chaque série d'essais, avant de commiter le CSV.
Nb : pareil, suggestion de l'IA.

```python
chemin = exporter_runs("churn_bancaire", "david")
```

### Exemple complet

```python
import mlflow
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from base_commune.outils import (
    charger_donnees,
    configurer_mlflow,
    decouper_donnees,
    evaluer_modele,
    exporter_runs,
    tags_reproductibilite,
)

df = charger_donnees()
X_train, X_test, y_train, y_test = decouper_donnees(df)

pretraitement = make_column_transformer(
    (OneHotEncoder(), ["Geography", "Gender"]),
    remainder=StandardScaler(),
)
pipeline = make_pipeline(pretraitement, LogisticRegression(max_iter=1000))

configurer_mlflow("churn_bancaire")
with mlflow.start_run(run_name="regression_logistique_base"):
    mlflow.set_tags({"author": "david", "model_family": "logistic_regression"})
    mlflow.set_tags(tags_reproductibilite())
    mlflow.log_params({"max_iter": 1000, "class_weight": None})
    mlflow.log_metrics(evaluer_modele(pipeline, X_train, y_train))

exporter_runs("churn_bancaire", auteur="david")
```

### Visualiser les essais
Depuis la racine du dépôt, avec l'environnement activé :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Puis ouvrir http://127.0.0.1:5000.
