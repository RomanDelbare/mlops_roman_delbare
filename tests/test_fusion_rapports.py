"""
Tests de la fusion des rapports de suivi MLflow.
Ce test est une suggestion de l'ia, je n'aurai pas
pensé utiliser pytest pour ça de moi même.
"""

import pandas as pd
import pytest

from base_commune.fusion_rapports import FICHIER_SYNTHESE, fusionner_rapports


def ecrire_rapport(dossier, auteur, essais):
    """Écrit un faux fichier runs_<auteur>.csv à partir de (id, famille, pr_auc)."""
    lignes = [
        {
            "run_id": identifiant,
            "tags.author": auteur,
            "tags.model_family": famille,
            "tags.mlflow.runName": f"essai_{identifiant}",
            "metrics.cv_pr_auc_mean": pr_auc,
        }
        for identifiant, famille, pr_auc in essais
    ]
    pd.DataFrame(lignes).to_csv(dossier / f"runs_{auteur}.csv", index=False)


def test_la_fusion_rassemble_les_rapports_de_chaque_auteur(tmp_path):
    ecrire_rapport(tmp_path, "roman", [("a1", "knn", 0.62), ("a2", "svm", 0.66)])
    ecrire_rapport(tmp_path, "david", [("b1", "xgboost", 0.70)])
    synthese = pd.read_csv(fusionner_rapports(tmp_path))
    assert set(synthese["run_id"]) == {"a1", "a2", "b1"}
    assert set(synthese["tags.author"]) == {"roman", "david"}


def test_un_meme_essai_n_apparait_qu_une_fois(tmp_path):
    ecrire_rapport(tmp_path, "roman", [("a1", "knn", 0.62)])
    ecrire_rapport(tmp_path, "david", [("a1", "knn", 0.62), ("b1", "xgboost", 0.70)])
    synthese = pd.read_csv(fusionner_rapports(tmp_path))
    assert sorted(synthese["run_id"]) == ["a1", "b1"]


def test_les_essais_sont_tries_par_famille_puis_par_pr_auc(tmp_path):
    ecrire_rapport(tmp_path, "roman", [("a1", "knn", 0.50), ("a2", "knn", 0.62)])
    ecrire_rapport(tmp_path, "david", [("b1", "xgboost", 0.70)])
    synthese = pd.read_csv(fusionner_rapports(tmp_path))
    assert list(synthese["run_id"]) == ["a2", "a1", "b1"]


def test_la_synthese_n_est_pas_relue_comme_un_rapport(tmp_path):
    ecrire_rapport(tmp_path, "roman", [("a1", "knn", 0.62)])
    fusionner_rapports(tmp_path)
    synthese = pd.read_csv(fusionner_rapports(tmp_path))  # deuxième fusion
    assert len(synthese) == 1
    assert (tmp_path / FICHIER_SYNTHESE).exists()


def test_la_fusion_signale_l_absence_de_rapports(tmp_path):
    with pytest.raises(FileNotFoundError, match="runs_"):
        fusionner_rapports(tmp_path)