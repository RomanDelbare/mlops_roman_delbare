"""Entraînement du modèle final.

Ici on va contrôler les données + validation croisee et choix du seuil de décision
(max du F1 sur les prédictions hors pli) comm vu (cf notebook) 
Après : entraînement final + évaluation sur le test .
On va enregistrer le modèle final dans MLflow .

lancement en faisant python -m modele_final.entrainement_final
"""

import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from base_commune.outils import RANDOM_STATE as GRAINE_ALEATOIRE
from base_commune.outils import (
    charger_donnees,
    configurer_mlflow,
    decouper_donnees,
    evaluer_modele,
    tags_reproductibilite,
)
from modele_final.prepa_donnees import (
    charger_parametres,
    construire_modele,
    verifier_qualite_donnees,
)

# MLflow refuse par défaut de sauvegarder un modèle XGBoost : on autorise ces types.
TYPES_XGBOOST_AUTORISES = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
ALIAS_DU_MODELE_RETENU = "modele_final"


def verifier_strategie_seuil(parametres):
# Verif qu'on applique bien notre strat de seuil
    strategie = parametres["seuil"]["strategie"]
    if strategie != "f1_max":
        raise ValueError(
            f"strat de seuil non gérée : {strategie!r}"
        )


def choisir_seuil_f1_max(cible, probabilites):
# Cherche le seuil qui max le f1
    precisions, rappels, seuils = precision_recall_curve(cible, probabilites)
    scores_f1 = 2 * precisions * rappels / (precisions + rappels + 1e-12)
    indice_du_meilleur = np.argmax(scores_f1[:-1])
    return round(float(seuils[indice_du_meilleur]), 2)


def mesurer_au_seuil(cible, probabilites, seuil):
# preci rappel et f1 sur nos pred avec ce seuil
    predictions = (probabilites >= seuil).astype(int)
    return {
        "precision": precision_score(cible, predictions, zero_division=0),
        "rappel": recall_score(cible, predictions, zero_division=0),
        "f1": f1_score(cible, predictions, zero_division=0),
    }


def entrainer(donnees_brutes=None, parametres=None):
# entraine le modele final + valide sur le test + sauvegarde mlflow
    parametres = parametres or charger_parametres()
    verifier_strategie_seuil(parametres)
    if donnees_brutes is None:
        donnees_brutes = charger_donnees()

    variables_entrainement, variables_test, cible_entrainement, cible_test = (
        decouper_donnees(donnees_brutes)
    )
    verifier_qualite_donnees(variables_entrainement, parametres)
    verifier_qualite_donnees(variables_test, parametres)

    nombre_de_plis = parametres["validation_croisee"]["nombre_de_plis"]
    modele = construire_modele(parametres)
    metriques_validation_croisee = evaluer_modele(
        modele, variables_entrainement, cible_entrainement, 
        n_splits=nombre_de_plis)

    decoupage_en_plis = StratifiedKFold(
        n_splits=nombre_de_plis, shuffle=True, random_state=GRAINE_ALEATOIRE
    )
    probabilites_hors_pli = cross_val_predict(
        modele,
        variables_entrainement,
        cible_entrainement,
        cv=decoupage_en_plis,
        method="predict_proba",
    )[:, 1]
    seuil = choisir_seuil_f1_max(cible_entrainement, probabilites_hors_pli)
    mesures_hors_pli = mesurer_au_seuil(
        cible_entrainement, probabilites_hors_pli, seuil
    )

    modele.fit(variables_entrainement, cible_entrainement)
    probabilites_test = modele.predict_proba(variables_test)[:, 1]
    mesures_test = mesurer_au_seuil(cible_test, probabilites_test, seuil)
    mesures_test["pr_auc"] = average_precision_score(cible_test, probabilites_test)
    mesures_test["roc_auc"] = roc_auc_score(cible_test, probabilites_test)
# pr sauvegarder ds Mlflow (aide ia + forte 
# pour cette partie comme avant car on est moins à laise avec mlflow)
    nom_du_modele = parametres["mlflow"]["nom_du_modele"]
    configurer_mlflow(parametres["mlflow"]["nom_experience"])
    with mlflow.start_run(run_name="xgb_modele_final") as essai:
        mlflow.set_tags(
            {"model_family": "xgboost", "stage": "final", **tags_reproductibilite()}
        )
        mlflow.log_params(
            {
                **parametres["modele"],
                "decision_threshold": seuil,
                "threshold_rule": parametres["seuil"]["strategie"],
                "cv_folds": nombre_de_plis,
                "random_state": GRAINE_ALEATOIRE,
                "test_size": round(len(variables_test) / len(donnees_brutes), 2),
                "n_train": len(variables_entrainement),
                "n_test": len(variables_test),
            }
        )
        mlflow.log_metrics(metriques_validation_croisee)
        mlflow.log_metrics({f"oof_{nom}": v for nom, v in mesures_hors_pli.items()})
        mlflow.log_metrics({f"test_{nom}": v for nom, v in mesures_test.items()})
        mlflow.log_dict(parametres, "params_utilises.json")

        signature = infer_signature(
            variables_entrainement, modele.predict(variables_entrainement)
        )
        informations_du_modele = mlflow.sklearn.log_model(
            modele,
            name="model",
            signature=signature,
            input_example=variables_entrainement.head(3),
            registered_model_name=nom_du_modele,
            skops_trusted_types=TYPES_XGBOOST_AUTORISES,
        )

    # Le modèle qui vient d'être entraîné devient le modèle de référence.
    version_du_modele = informations_du_modele.registered_model_version
    MlflowClient().set_registered_model_alias(
        nom_du_modele, ALIAS_DU_MODELE_RETENU, version_du_modele
    )

    return {
        "seuil": seuil,
        "metriques_hors_pli": mesures_hors_pli,
        "metriques_test": mesures_test,
        "identifiant_de_l_essai": essai.info.run_id,
        "version_du_modele": version_du_modele,}

if __name__ == "__main__":
    resume = entrainer()
    print("Seuil de décision retenu =", resume["seuil"])
    for nom, valeur in resume["metriques_test"].items():
        print(f"  test {nom} : {valeur:.3f}")
    print("Modèle enregistré sa version=", resume["version_du_modele"])