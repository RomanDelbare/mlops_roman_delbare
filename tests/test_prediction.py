"""Tests des prédictions avec le seuil de décision."""

import numpy as np
import pytest

from base_commune.outils import TARGET
from modele_final.entrainement_final import entrainer
from modele_final.prediction import charger_modele_final, predire, predire_clients


class ModeleFictif:
    """Modèle qui renvoie des probabilités fixées à l'avance."""

    def __init__(self, probabilites):
        self.probabilites = np.array(probabilites)

    def predict_proba(self, donnees):
        return np.column_stack([1 - self.probabilites, self.probabilites])


def test_predire_applique_le_seuil():
    modele = ModeleFictif([0.1, 0.25, 0.3, 0.6])
    assert list(predire(modele, None, 0.3)) == [0, 0, 1, 1]


def test_un_seuil_plus_bas_detecte_plus_de_churners():
    modele = ModeleFictif([0.1, 0.25, 0.3, 0.6])
    assert predire(modele, None, 0.2).sum() > predire(modele, None, 0.5).sum()


def test_le_modele_charge_garde_le_seuil_de_l_entrainement(
    mlflow_temporaire, parametres, clients_fictifs
):
    clients, cible = clients_fictifs
    parametres["validation_croisee"]["nombre_de_plis"] = 3
    parametres["mlflow"]["nom_experience"] = "test_prediction"
    parametres["mlflow"]["nom_du_modele"] = "modele_de_test_prediction"

    resume = entrainer(donnees_brutes=clients.assign(**{TARGET: cible.values}),
                       parametres=parametres)
    modele, seuil = charger_modele_final(parametres)

    assert seuil == resume["seuil"]
    predictions = predire_clients(clients.head(10), parametres)
    assert set(predictions) <= {0, 1}
    attendu = (modele.predict_proba(clients.head(10))[:, 1] >= seuil).astype(int)
    assert (predictions == attendu).all()


def test_des_donnees_invalides_sont_refusees(parametres, clients_fictifs):
    clients, _ = clients_fictifs
    clients = clients.head(5).assign(Geography="Italy")
    with pytest.raises(ValueError, match="Geography"):
        predire_clients(clients, parametres)
