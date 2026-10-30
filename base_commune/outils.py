"""Fonctions communes à nos tests de modèles.

Elles uniformisent les points cruciaux pour comparer nos modèles : chargement et
découpage du dataset, évaluation des performances et suivi MLflow.
"""

import hashlib
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

# Pour qu'on utilise toujours la même graine
RANDOM_STATE = 42
# Variable cible
TARGET = "Exited"
# Colonnes toujours exclues, quel que soit le modèle : elles ne servent à rien
COLONNES_IDENTIFIANTES = ["RowNumber", "CustomerId", "Surname"]
# Métriques communes utilisées pour scorer les modèles
SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
}
RACINE = Path(__file__).resolve().parents[1]
# Chemin des données, calculé depuis la racine du projet
CHEMIN_DONNEES = RACINE / "data" / "raw" / "churn_modelling.csv"
# Tracking MLflow commun
MLFLOW_TRACKING_URI = f"sqlite:///{(RACINE / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = RACINE / "mlartifacts"
# Résumés des runs en CSV, pour comparer les modèles sans relancer les notebooks
DOSSIER_REPORTS = RACINE / "reports"


def charger_donnees(chemin: Path = CHEMIN_DONNEES) -> pd.DataFrame:
    """Charge le CSV brut et retire les colonnes identifiantes."""
    return pd.read_csv(chemin, sep=";").drop(columns=COLONNES_IDENTIFIANTES)


def decouper_donnees(df: pd.DataFrame, test_size: float = 0.2):
    """Découpe en train et test, stratifié sur la cible, avec la graine commune.

    Renvoie X_train, X_test, y_train, y_test. Comme ça, personne ne se trompe
    dans le découpage.
    """
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def evaluer_modele(modele, X_train, y_train, n_splits: int = 5) -> dict:
    """Validation croisée stratifiée, uniquement sur le train (5 plis par défaut).

    Le modèle doit être un Pipeline qui contient le prétraitement (encodage de
    Geography et Gender, standardisation...). Le prétraitement est alors refait
    dans chaque pli, sans fuite d'information. Exemple dans utilisation.md.

    Renvoie la moyenne et l'écart-type de chaque métrique en validation, et la
    moyenne sur le train pour repérer le surapprentissage.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    res = cross_validate(
        modele,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=1,
    )
    metriques = {}
    for nom in SCORING:
        metriques[f"cv_{nom}_mean"] = float(res[f"test_{nom}"].mean())
        metriques[f"cv_{nom}_std"] = float(res[f"test_{nom}"].std())
        metriques[f"train_{nom}_mean"] = float(res[f"train_{nom}"].mean())
    return metriques


def evaluer_test(modele, X_train, y_train, X_test, y_test, seuil: float = 0.5) -> dict:
    """Entraîne le modèle sur tout le train et le score sur le test.

    À utiliser une seule fois, sur le modèle retenu à la fin. Les métriques qui
    dépendent d'un seuil (f1, précision, recall, accuracy) utilisent le seuil donné.
    """
    modele.fit(X_train, y_train)
    proba = modele.predict_proba(X_test)[:, 1]
    pred = (proba >= seuil).astype(int)
    return {
        "test_pr_auc": float(average_precision_score(y_test, proba)),
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "test_f1": float(f1_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred)),
        "test_accuracy": float(accuracy_score(y_test, pred)),
    }


# Config MLflow en partie générée avec l'IA, comme on débute dessus
def configurer_mlflow(nom_experience: str) -> None:
    """Pointe MLflow vers la base locale et crée l'expérience si besoin."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(nom_experience) is None:
        mlflow.create_experiment(
            nom_experience, artifact_location=MLFLOW_ARTIFACTS.as_uri()
        )
    mlflow.set_experiment(nom_experience)


# Suggestion de l'IA pour améliorer notre code
def tags_reproductibilite() -> dict:
    """Commit Git courant, modifications non commitées et empreinte des données."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=RACINE, text=True
        ).strip()
        modifie = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=RACINE, text=True
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, modifie = "inconnu", True
    md5 = hashlib.md5(CHEMIN_DONNEES.read_bytes()).hexdigest()
    return {"git_commit": commit, "git_dirty": str(modifie), "data_md5": md5}


def exporter_runs(nom_experience: str, auteur: str) -> Path:
    """Exporte les runs terminés d'un auteur dans reports/runs_<auteur>.csv."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(
        experiment_names=[nom_experience], filter_string=f"tags.author = '{auteur}'"
    )
    runs = runs[runs["status"] == "FINISHED"]
    tags = [
        "tags.mlflow.runName",
        "tags.model_family",
        "tags.author",
        "tags.git_commit",
        "tags.git_dirty",
        "tags.data_md5",
    ]
    colonnes = (
        ["run_id", "start_time"]
        + [c for c in tags if c in runs.columns]
        + sorted(c for c in runs.columns if c.startswith("params."))
        + sorted(c for c in runs.columns if c.startswith("metrics."))
    )
    DOSSIER_REPORTS.mkdir(exist_ok=True)
    chemin = DOSSIER_REPORTS / f"runs_{auteur}.csv"
    runs[colonnes].sort_values("start_time").to_csv(chemin, index=False)
    return chemin
