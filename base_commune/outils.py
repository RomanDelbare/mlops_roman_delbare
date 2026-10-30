""" CE .py va nous servir à appeler des Fonctions communes à nos tests pour travailler en collab mais en uniformisant les points qui vont être cruciaux pour 
comparer nos modèles (chargement et découpage du dataset , évaluation des perfs + suivi MLflow)."""

import hashlib
import subprocess
from pathlib import Path
import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

RANDOM_STATE = 42 # Pour qu'on utilise tjr la meme graine 
TARGET = "Exited" # Pour l'évaluation 
COLONNES_IDENTIFIANTES = ["RowNumber", "CustomerId", "Surname"] # Colonne qu'on va tjr exclure indépendament du modèle elles serviront tjr à rien
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1", "precision": "precision", "recall": "recall", "accuracy": "accuracy"} # Métriques qu'on va utiliser pour scorer les modèles
RACINE = Path(__file__).resolve().parents[1] 
CHEMIN_DONNEES = RACINE / "data" / "raw" / "churn_modelling.csv" # Pour automatiser le chargement des données comme on va faire une structure de projet propre
MLFLOW_TRACKING_URI = f"sqlite:///{(RACINE / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = RACINE / "mlartifacts" # Uniformiser le tracking via MLFlow (en tout cas on va essayer)
DOSSIER_REPORTS = RACINE / "reports" # Comme ca si besoin on va pouvoir faire une visualisation sur les différents résultats en tapant direct un csv au lieu de compiler des notebooks


def charger_donnees(chemin: Path = CHEMIN_DONNEES) -> pd.DataFrame:
    return pd.read_csv(chemin, sep=";").drop(columns=COLONNES_IDENTIFIANTES) # On charge les données en auto et vire tjr les colonnes inutiles


def decouper_donnees(df: pd.DataFrame, test_size: float = 0.2):
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=RANDOM_STATE) # Comme ca en appelant cette fonction on est sûr que personne se trompe dans le découpage 


def evaluer_modele(modele, X_train, y_train, n_splits: int = 5) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE) # On fait dla validation croisée que sur le train pour pas biaiser par défaut à 5 plis
    res = cross_validate(modele, X_train, y_train, cv=cv, scoring=SCORING,
                         return_train_score=True, n_jobs=1)
    metriques = {}
    for nom in SCORING:
        metriques[f"cv_{nom}_mean"] = float(res[f"test_{nom}"].mean())
        metriques[f"cv_{nom}_std"] = float(res[f"test_{nom}"].std())
        metriques[f"train_{nom}_mean"] = float(res[f"train_{nom}"].mean())
    return metriques # Automatise/uniformise le scoring de la perf des modèles pour qu'on ai au moins une base tjr comparable sur les différents modèles. 


def configurer_mlflow(nom_experience: str) -> None: # Config MLFlow en partie generé avec IA comme on est débutants dessus 
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(nom_experience) is None:
        mlflow.create_experiment(nom_experience,
                                 artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(nom_experience)


def tags_reproductibilite() -> dict: # Suggestion de l'IA pour améliorer notre code actuel.
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=RACINE, text=True).strip()
        modifie = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                               cwd=RACINE, text=True).strip())
    except Exception:
        commit, modifie = "inconnu", True
    md5 = hashlib.md5(CHEMIN_DONNEES.read_bytes()).hexdigest()
    return {"git_commit": commit, "git_dirty": str(modifie), "data_md5": md5}


def exporter_runs(nom_experience: str, auteur: str) -> Path: # Avec ca on exporte tjr les resultats dans le fichier csv 
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[nom_experience],
                              filter_string=f"tags.author = '{auteur}'")
    runs = runs[runs["status"] == "FINISHED"]
    tags = ["tags.mlflow.runName", "tags.model_family", "tags.author",
            "tags.git_commit", "tags.git_dirty", "tags.data_md5"]
    colonnes = (["run_id", "start_time"] + [c for c in tags if c in runs.columns]
                + sorted(c for c in runs.columns if c.startswith("params."))
                + sorted(c for c in runs.columns if c.startswith("metrics.")))
    DOSSIER_REPORTS.mkdir(exist_ok=True)
    chemin = DOSSIER_REPORTS / f"runs_{auteur}.csv"
    runs[colonnes].sort_values("start_time").to_csv(chemin, index=False)
    return chemin