# tests pr lentrainement

import mlflow
import numpy as np
import pytest
from modele_final.entrainement_final import (
    choisir_seuil_f1_max,
    entrainer,
    mesurer_au_seuil,
    verifier_strategie_seuil,
)

from base_commune.outils import TARGET
from modele_final.prepa_donnees import charger_parametres


def test_le_seuil_choisi_maximise_le_f1():
    cible = np.array([0, 0, 0, 0, 1, 1])
    probabilites = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.9])
    assert choisir_seuil_f1_max(cible, probabilites) == 0.6


def test_le_seuil_est_toujours_entre_0_et_1():
    generateur = np.random.default_rng(0)
    cible = generateur.integers(0, 2, 500)
    probabilites = generateur.random(500)
    assert 0 <= choisir_seuil_f1_max(cible, probabilites) <= 1


def test_mesurer_au_seuil_sur_un_cas_simple():
    cible = np.array([0, 0, 1, 1])
    probabilites = np.array([0.2, 0.6, 0.4, 0.9])  # prédictions à 0,5 : 0, 1, 0, 1
    mesures = mesurer_au_seuil(cible, probabilites, 0.5)
    assert mesures["precision"] == pytest.approx(0.5)
    assert mesures["rappel"] == pytest.approx(0.5)
    assert mesures["f1"] == pytest.approx(0.5)


def test_une_strategie_de_seuil_inconnue_est_refusee():
    parametres = charger_parametres()
    parametres["seuil"]["strategie"] = "rappel_minimum"
    with pytest.raises(ValueError, match="rappel_minimum"):
        verifier_strategie_seuil(parametres)


@pytest.fixture
def mlflow_temporaire(tmp_path, monkeypatch):
# Redirige MLflow vers un dossier temporaire pour ne pas polluer la vraie base
# (suggestion de l'ia je n'y aurai honnêtement pas pensé)
    ancienne_adresse = mlflow.get_tracking_uri()
    adresse_de_test = f"sqlite:///{(tmp_path / 'mlflow_test.db').as_posix()}"
    monkeypatch.setattr("base_commune.outils.MLFLOW_TRACKING_URI", adresse_de_test)
    monkeypatch.setattr("base_commune.outils.MLFLOW_ARTIFACTS", tmp_path / "artefacts")
    yield
    mlflow.set_tracking_uri(ancienne_adresse)


def test_l_entrainement_complet_enregistre_un_modele_utilisable(
    mlflow_temporaire, parametres, clients_fictifs
):
    clients, cible = clients_fictifs
    donnees = clients.assign(**{TARGET: cible.values})
    parametres["validation_croisee"]["nombre_de_plis"] = 3
    parametres["mlflow"]["nom_experience"] = "test_entrainement"
    parametres["mlflow"]["nom_du_modele"] = "modele_de_test"

    resume = entrainer(donnees_brutes=donnees, parametres=parametres)

    assert 0 <= resume["seuil"] <= 1
    assert 0 <= resume["metriques_test"]["pr_auc"] <= 1
    modele = mlflow.sklearn.load_model("models:/modele_de_test@modele_final")
    assert modele.predict_proba(clients.head(5)).shape == (5, 2)
    essai = mlflow.get_run(resume["identifiant_de_l_essai"])
    assert float(essai.data.params["decision_threshold"]) == resume["seuil"]