# données et paras utilisés pr les tests comme ca sur les derniers test on va appeler 
# direct depuis ici ca sera plus simple

import mlflow
import numpy as np
import pandas as pd
import pytest

from modele_final.prepa_donnees import charger_parametres


@pytest.fixture
def parametres():
    parametres = charger_parametres()
    parametres["modele"]["n_estimators"] = 20  # entraînement rapide pour les tests
    return parametres


@pytest.fixture
def clients_fictifs():
    generateur = np.random.default_rng(0)
    nombre_de_clients = 300
    clients = pd.DataFrame(
        {
            "CreditScore": generateur.integers(350, 850, nombre_de_clients),
            "Age": generateur.integers(18, 90, nombre_de_clients),
            "Tenure": generateur.integers(0, 11, nombre_de_clients),
            "Balance": generateur.random(nombre_de_clients) * 200_000,
            "NumOfProducts": generateur.integers(1, 5, nombre_de_clients),
            "EstimatedSalary": generateur.random(nombre_de_clients) * 200_000,
            "HasCrCard": generateur.integers(0, 2, nombre_de_clients),
            "IsActiveMember": generateur.integers(0, 2, nombre_de_clients),
            "Geography": generateur.choice(
                ["France", "Spain", "Germany"], nombre_de_clients
            ),
            "Gender": generateur.choice(["Male", "Female"], nombre_de_clients),
        }
    )
    bruit = generateur.random(nombre_de_clients) < 0.1
    cible = pd.Series((clients["Age"] > 50).astype(int) ^ bruit)
    return clients, cible


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
