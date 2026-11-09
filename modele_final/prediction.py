"""Prédictions du modèle final avec le seuil de décision retenu.

Le predict() du modèle enregistré dans MLflow coupe à 0,5. Notre seuil est choisi
dans entrainement_final.py (max du F1), donc pour prédire on passe par ici.
"""

import mlflow
from mlflow import MlflowClient

from base_commune.outils import configurer_mlflow
from modele_final.entrainement_final import ALIAS_DU_MODELE_RETENU
from modele_final.prepa_donnees import charger_parametres, verifier_qualite_donnees


def predire(modele, donnees, seuil):
    """Renvoie 1 (le client part) ou 0 (il reste) pour chaque client."""
    probabilites = modele.predict_proba(donnees)[:, 1]
    return (probabilites >= seuil).astype(int)


def charger_modele_final(parametres=None):
    """Charge le modèle final depuis MLflow et le seuil enregistré avec."""
    parametres = parametres or charger_parametres()
    nom_du_modele = parametres["mlflow"]["nom_du_modele"]
    configurer_mlflow(parametres["mlflow"]["nom_experience"])

    adresse = f"models:/{nom_du_modele}@{ALIAS_DU_MODELE_RETENU}"
    modele = mlflow.sklearn.load_model(adresse)
    version = MlflowClient().get_model_version_by_alias(
        nom_du_modele, ALIAS_DU_MODELE_RETENU
    )
    seuil = float(mlflow.get_run(version.run_id).data.params["decision_threshold"])
    return modele, seuil


def predire_clients(donnees, parametres=None):
    """Vérifie les données, charge le modèle final et renvoie les prédictions."""
    parametres = parametres or charger_parametres()
    verifier_qualite_donnees(donnees, parametres)
    modele, seuil = charger_modele_final(parametres)
    return predire(modele, donnees, seuil)
