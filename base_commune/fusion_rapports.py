"""
Fusionne nos csv de suivi MLflow en un seul fichier.

Lancement : python -m base_commune.fusion_rapports
"""

import pandas as pd

from base_commune.outils import DOSSIER_REPORTS

FICHIER_SYNTHESE = "synthese_runs.csv"
COLONNES_DE_TRI = ["tags.model_family", "metrics.cv_pr_auc_mean"]
COLONNES_EN_TETE = [
    "tags.author",
    "tags.model_family",
    "tags.mlflow.runName",
    "metrics.cv_pr_auc_mean",
    "metrics.cv_pr_auc_std",
    "metrics.cv_roc_auc_mean",
    "metrics.cv_precision_mean",
    "metrics.cv_recall_mean",
    "metrics.cv_f1_mean",
]


def fusionner_rapports(dossier=DOSSIER_REPORTS):
    """Rassemble les fichiers runs_<auteur>.csv en reports/synthese_runs.csv.

    Un même essai n'apparaît qu'une fois. Les essais sont triés par famille de
    modèle, puis par PR-AUC décroissante. Retourne le chemin du fichier créé.
    """
    fichiers = sorted(dossier.glob("runs_*.csv"))
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier runs_*.csv dans {dossier}")

    synthese = pd.concat([pd.read_csv(fichier) for fichier in fichiers])
    synthese = synthese.drop_duplicates(subset="run_id").reset_index(drop=True)

    colonnes_en_tete = [c for c in COLONNES_EN_TETE if c in synthese.columns]
    autres_colonnes = [c for c in synthese.columns if c not in colonnes_en_tete]
    synthese = synthese[colonnes_en_tete + autres_colonnes]

    if all(colonne in synthese.columns for colonne in COLONNES_DE_TRI):
        # famille de modèle (ordre alphabétique), puis PR-AUC décroissante
        synthese = synthese.sort_values(COLONNES_DE_TRI, ascending=[True, False])

    chemin = dossier / FICHIER_SYNTHESE
    synthese.to_csv(chemin, index=False)
    return chemin


if __name__ == "__main__":
    chemin_de_la_synthese = fusionner_rapports()
    nombre_d_essais = len(pd.read_csv(chemin_de_la_synthese))
    print(f"{nombre_d_essais} essais écrits dans {chemin_de_la_synthese}")