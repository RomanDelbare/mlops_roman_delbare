# Explication sur base_commune : 

Nous allons chercher à vraiment jouer le jeu du travail collaboratif pour pratiquer le MLOps de facon plus concrète. 
Pour ça nous avons fait le choix de tester chacun de notre côté plusieurs classifiers binaires et les transformations associés à ces derniers.

Notre but est de gagner du temps et potentiellement d'optimiser nos résultats en pouvant passer plus de temps sur chaque modèle; Puis de mettre en commun les résultats
pour sélectionner le meilleur modèle. C'est sur ce dernier qu'on basera le reste du projet. Pour ça on va : 

1) Uniformiser l'importation et le découpage du dataset + le calcul d'un certains nombres de métriques de performances. 
2) Essayer de mettre en place un tracking avec MLFlow
3) Utiliser un petit csv qui compile les résumés de nos différents tests 

Cela va nous permettre de : 
- Eviter de biaiser la comparaison entre nos modèles (par exemple en n'utilisant pas la même seed d'aléatoire ou en oubliant la stratification)
- Faciliter la comparaison des modèles : on peut tout les deux calculer toutes les métriques qu'on veut mais on est sûr d'avoir au moins une base de comparaison 
- Visualiser les résultats de chaque modèle en attaquant direct le csv plutot que de compiler nos codes pour gagner du temps + gagner du temps entre nous dans la comparaison des modèles. 
- Nous entraîner à utiliser MLFlow ce qui est nouveau pour nous deux (on va donc s'aider de l'ia gen sur cette partie)

Ainsi on à créer outils.py qui va nous permettre d'importer quelques fonctions pour faire tout ça : 
charger_donnees 
, decouper_donnees et evaluer_modele

## Utilisation de `base_commune`

Le module `base_commune/outils.py` regroupe les fonctions communes du projet pour aller dans le sens de ce qu'on vient d'expliquer.
Prérequis : `pip install -e .` (voir Installation), puis dans le notebook ou le script :

"from base_commune.outils import (charger_donnees, decouper_donnees, evaluer_modele, evaluer_test, configurer_mlflow, tags_reproductibilite, exporter_runs)"

### `charger_donnees()`
Charge le fichier brut `data/raw/churn_modelling.csv` et retire les colonnes identifiantes car inutiles pour tout type de modèle (`RowNumber`, `CustomerId`, `Surname`).
- Argument optionnel : `chemin` (par défaut le CSV du projet).
- Retourne un `DataFrame` de 11 colonnes (10 variables explicatives + `Exited`).

Faire : 
"df = charger_donnees()" puis travailler sur df. 


### `decouper_donnees(df, test_size=0.2)`
Sépare les données en jeu d'entraînement et jeu de test, de façon stratifiée sur `Exited` et avec `random_state=42` (comme ca découpage identique pour tout le monde).

- Retourne `X_train, X_test, y_train, y_test`.

Faire : 
"X_train, X_test, y_train, y_test = decouper_donnees(df)"

### `evaluer_modele(modele, X_train, y_train, n_splits=5)`
Évalue un `Pipeline` scikit-learn (prétraitement + modèle) par validation croisée stratifiée(5 plis, `shuffle=True`, `random_state=42`) sur le jeu d'entraînement uniquement.
- Geography et Gender sont en texte : un modèle seul plante, il faut que l'encodage soit dans le Pipeline (comme ça il est refait dans chaque pli, sans fuite).
- Retourne un dictionnaire de 18 métriques : pour chacune de `pr_auc`, `roc_auc`, `f1`, `precision`, `recall`, `accuracy`, les clés `cv_<métrique>_mean` `cv_<métrique>_std` (moyenne et écart-type sur les plis) et `train_<métrique>_mean` (score sur les données d'entraînement).

Faire: 
"metriques = evaluer_modele(pipeline, X_train, y_train)
print(metriques["cv_pr_auc_mean"])"

### `evaluer_test(modele, X_train, y_train, X_test, y_test, seuil=0.5)`
Entraîne le modèle sur tout le jeu d'entraînement puis le score sur le jeu de test. À n'utiliser qu'une seule fois, à la fin, sur le modèle qu'on a retenu : si on s'en sert pour comparer ou régler nos modèles, le score final ne veut plus rien dire.
- Retourne les 6 métriques communes avec le préfixe `test_` (`test_pr_auc`, `test_roc_auc`, `test_f1`, `test_precision`, `test_recall`, `test_accuracy`).
- `f1`, `precision`, `recall` et `accuracy` dépendent du seuil de décision : 0,5 par défaut, à remplacer par le seuil optimisé si on en a choisi un.

Faire :
"resultats = evaluer_test(pipeline, X_train, y_train, X_test, y_test)"

### `configurer_mlflow(nom_experience)`
À appeler une fois au début du notebook, avant tout `mlflow.start_run()`. Connecte MLflow à la base locale du projet (`mlflow.db`) et sélectionne (ou crée) l'experiment indiqué, pour que tous les essais soient enregistrés au même endroit.
Nb : celle ci est générée comme on débute sur MLFlow, fiabilité donc non garantie

Faire : "configurer_mlflow("churn_bancaire")"


### `tags_reproductibilite()`
Retourne un dictionnaire `{"git_commit", "git_dirty", "data_md5"}` : commit Git courant, présence de modifications non commitées, empreinte (MD5) du fichier de données brut. À enregistrer avec chaque run.
Nb : Cette fonction est une suggestion de l'ia 

Faire : 
"mlflow.set_tags({"author": "[roman ou david ou autre user]" , **tags_reproductibilite()})"


### `exporter_runs(nom_experience, auteur)`
Exporte dans `reports/runs_<auteur>.csv` les runs terminés (statut `FINISHED`) de l'auteur indiqué (tag `author`) : paramètres, métriques, nom du run et tags de reproductibilité.
- Retourne le chemin du fichier créé.
- À relancer après chaque série d'essais, avant de commiter le CSV.
Nb : pareil suggestion de l'ia 
Faire : 
"chemin = exporter_runs("churn_bancaire", "[nom de l'auteur]")"

### Visualiser les essais
Depuis la racine du dépôt, avec l'environnement activé :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Puis ouvrir http://127.0.0.1:5000.
