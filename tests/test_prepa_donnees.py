# comme on l'a mentionné en cours, on verif que pour le modèle finalement retenus
# nos données dentrée sont ok et que le modèle est bien construit, tourne correctement
# à les comportements attendus etc

import numpy as np
import pandas as pd
import pytest

from base_commune.outils import TARGET, charger_donnees
from modele_final.prepa_donnees import (
    charger_parametres,
    construire_modele,
    verifier_colonnes,
    verifier_qualite_donnees,
)


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
            "Age": generateur.integers(18, 120, nombre_de_clients),
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


def test_les_parametres_contiennent_les_sections_attendues(parametres):
    sections = [
        "variables",
        "modele",
        "validation_croisee",
        "seuil",
        "controles_qualite",
        "mlflow",
    ]
    for section in sections:
        assert section in parametres


def test_verifier_colonnes_accepte_les_bonnes_colonnes(parametres, clients_fictifs):
    clients, _ = clients_fictifs
    verifier_colonnes(clients, parametres)  # ne doit pas lever d'erreur


def test_verifier_colonnes_signale_une_colonne_manquante(parametres, clients_fictifs):
    clients, _ = clients_fictifs
    with pytest.raises(ValueError, match="Age"):
        verifier_colonnes(clients.drop(columns="Age"), parametres)


def test_verifier_colonnes_signale_une_colonne_en_trop(parametres, clients_fictifs):
    clients, _ = clients_fictifs
    clients["Surname"] = "Dupont"
    with pytest.raises(ValueError, match="Surname"):
        verifier_colonnes(clients, parametres)


def test_le_modele_reprend_les_hyperparametres_du_fichier(parametres):
    modele = construire_modele(parametres)
    xgboost = modele.named_steps["xgb"]
    assert xgboost.max_depth == parametres["modele"]["max_depth"]
    assert xgboost.learning_rate == parametres["modele"]["learning_rate"]
    assert xgboost.n_estimators == parametres["modele"]["n_estimators"]


def test_le_modele_predit_des_probabilites_valides(parametres, clients_fictifs):
    clients, cible = clients_fictifs
    modele = construire_modele(parametres).fit(clients, cible)
    probabilites = modele.predict_proba(clients)
    assert probabilites.shape == (len(clients), 2)
    assert ((probabilites >= 0) & (probabilites <= 1)).all()
    assert np.allclose(probabilites.sum(axis=1), 1)


def test_le_modele_accepte_une_modalite_inconnue(parametres, clients_fictifs):
    clients, cible = clients_fictifs
    modele = construire_modele(parametres).fit(clients, cible)
    nouveau_client = clients.head(1).copy()
    nouveau_client["Geography"] = "Italy"
    # pays absent de l'entraînement , normalement on à fait en sorte que les pays absent
    # de lentrainement soient ejectés et un test le verif mais au cas ou on met ca aussi
    assert modele.predict_proba(nouveau_client).shape == (1, 2)


def test_deux_entrainements_donnent_les_memes_probabilites(parametres, clients_fictifs):
    clients, cible = clients_fictifs
    premier = construire_modele(parametres).fit(clients, cible).predict_proba(clients)
    second = construire_modele(parametres).fit(clients, cible).predict_proba(clients)
    assert np.allclose(premier, second)


def test_des_donnees_valides_passent_le_controle_qualite(parametres, clients_fictifs):
    clients, _ = clients_fictifs
    verifier_qualite_donnees(clients, parametres)  # ne doit pas lever d'erreur


def test_le_jeu_d_entrainement_respecte_les_controles(parametres):
    donnees = charger_donnees().drop(columns=TARGET)
    verifier_qualite_donnees(donnees, parametres)


def test_un_pays_inconnu_est_refuse(parametres, clients_fictifs):
    clients, _ = clients_fictifs
    clients.loc[0, "Geography"] = "Japon"
    with pytest.raises(ValueError, match="Japon"):
        verifier_qualite_donnees(clients, parametres)


def test_un_genre_inconnu_est_refuse(parametres, clients_fictifs):
    clients, _ = clients_fictifs
    clients.loc[0, "Gender"] = "Autre"
    with pytest.raises(ValueError, match="Autre"):
        verifier_qualite_donnees(clients, parametres)


@pytest.mark.parametrize(
    "colonne, valeur",
    [("Age", -1), ("Age", 121), ("EstimatedSalary", -0.01), ("NumOfProducts", 5)],
)
def test_une_valeur_hors_bornes_est_refusee(
    parametres, clients_fictifs, colonne, valeur
):
    clients, _ = clients_fictifs
    clients[colonne] = clients[colonne].astype(float)
    clients.loc[0, colonne] = valeur
    with pytest.raises(ValueError, match=colonne):
        verifier_qualite_donnees(clients, parametres)


@pytest.mark.parametrize(
    "colonne, valeur",
    [("Age", 0), ("Age", 120), ("EstimatedSalary", 0), ("NumOfProducts", 4)],
)
def test_les_bornes_sont_incluses(parametres, clients_fictifs, colonne, valeur):
    clients, _ = clients_fictifs
    clients[colonne] = clients[colonne].astype(float)
    clients.loc[0, colonne] = valeur
    verifier_qualite_donnees(clients, parametres)  # ne doit pas lever d'erreur


def test_une_valeur_manquante_est_refusee(parametres, clients_fictifs):
    clients, _ = clients_fictifs
    clients["Balance"] = clients["Balance"].astype(float)
    clients.loc[0, "Balance"] = np.nan
    with pytest.raises(ValueError, match="Balance : 1 valeur"):
        verifier_qualite_donnees(clients, parametres)


def test_une_variable_non_numerique_est_refusee(parametres, clients_fictifs):
    clients, _ = clients_fictifs
    clients["Age"] = clients["Age"].astype(object)
    clients.loc[0, "Age"] = "vingt"
    with pytest.raises(ValueError, match="numérique"):
        verifier_qualite_donnees(clients, parametres)


def test_tous_les_problemes_sont_signales_ensemble(parametres, clients_fictifs):
    clients, _ = clients_fictifs
    clients.loc[0, "Geography"] = "Japon"
    clients.loc[1, "Age"] = 200
    with pytest.raises(ValueError) as erreur:
        verifier_qualite_donnees(clients, parametres)
    assert "Japon" in str(erreur.value)
    assert "Age" in str(erreur.value)
