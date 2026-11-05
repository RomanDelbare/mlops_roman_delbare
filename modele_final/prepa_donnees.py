# Ici on fait la prépa des données uniquement

from pathlib import Path

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from base_commune.outils import RANDOM_STATE as GRAINE_ALEATOIRE

CHEMIN_PARAMETRES = Path(__file__).resolve().parent / "params.yaml"


def charger_parametres(chemin_du_fichier=CHEMIN_PARAMETRES):
    with open(chemin_du_fichier, encoding="utf-8") as fichier:
        return yaml.safe_load(fichier)


def verifier_colonnes(
    variables, parametres
):  # Verifie qu'on recup bien les bonnes cols
    colonnes_attendues = set(parametres["variables"]["numeriques"]) | set(
        parametres["variables"]["categorielles"]
    )
    colonnes_recues = set(variables.columns)
    colonnes_manquantes = colonnes_attendues - colonnes_recues
    colonnes_en_trop = colonnes_recues - colonnes_attendues
    if colonnes_manquantes or colonnes_en_trop:
        raise ValueError(
            f"Colonnes manquantes : {sorted(colonnes_manquantes)} ; "
            f"colonnes en trop : {sorted(colonnes_en_trop)}"
        )


def verifier_qualite_donnees(variables, parametres):
    # Comme on l'avait fait en tp, on fait une fonction qui verif que les données 
    # en entrée sont logiques et cohérentes avec le dataset dentrainement
    verifier_colonnes(variables, parametres)
    controles = parametres["controles_qualite"]
    problemes = []

    # 1 Valeurs manquantes
    valeurs_manquantes = variables.isna().sum()
    for colonne, nombre in valeurs_manquantes[valeurs_manquantes > 0].items():
        problemes.append(f"{colonne} : {nombre} valeurs manquantes")

    # 2 verif que les modalités sont les mm qie celles du jeu d'entraînement
    for colonne, modalites_autorisees in controles["modalites_autorisees"].items():
        modalites_presentes = set(variables[colonne].dropna().unique())
        modalites_inconnues = modalites_presentes - set(modalites_autorisees)
        if modalites_inconnues:
            problemes.append(
                f"{colonne} : modalité nn autorisée "
                f"{sorted(map(str, modalites_inconnues))} "
                f"(autorisées : {sorted(modalites_autorisees)})"
            )

    # 3 Les variables numériques doivent rester dans leurs bornes
    for colonne, bornes in controles["bornes"].items():
        valeurs = variables[colonne].dropna()
        if not pd.api.types.is_numeric_dtype(valeurs):
            problemes.append(f"{colonne} : la variable doit être numérique")
            continue
        if "min" in bornes:
            nombre_trop_petit = int((valeurs < bornes["min"]).sum())
            if nombre_trop_petit:
                problemes.append(
                    f"{colonne} : {nombre_trop_petit} valeur inférieure "
                    f"à {bornes['min']}"
                )
        if "max" in bornes:
            nombre_trop_grand = int((valeurs > bornes["max"]).sum())
            if nombre_trop_grand:
                problemes.append(
                    f"{colonne} : {nombre_trop_grand} valeur supérieure "
                    f"à {bornes['max']}"
                )
    if problemes:
        raise ValueError("Données invalides : " + " ; ".join(problemes))


def construire_pretraitement(parametres):
# Encode juste les catés, comme on est sur un modèle a base darbre rien dautre a faire
    variables = parametres["variables"]
    return ColumnTransformer(
        transformers=[
            (
                "variables_categorielles",
                OneHotEncoder(handle_unknown="ignore"),
                variables["categorielles"],
            ),
            ("variables_numeriques", "passthrough", variables["numeriques"]),
        ],
        remainder="drop",
    )


def construire_modele(parametres):
    hyperparametres = parametres["modele"]
    classifieur_xgboost = XGBClassifier(
        n_estimators=hyperparametres["n_estimators"],
        max_depth=hyperparametres["max_depth"],
        learning_rate=hyperparametres["learning_rate"],
        scale_pos_weight=hyperparametres["scale_pos_weight"],
        random_state=GRAINE_ALEATOIRE,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("preprocesseur", construire_pretraitement(parametres)),
            ("xgb", classifieur_xgboost),
        ]
    )
