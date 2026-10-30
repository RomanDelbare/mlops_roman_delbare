"""Chargement des données et découpage train/test communs à tous les modèles.

Tous les notebooks et scripts passent par ces fonctions. Ainsi, chaque modèle
est entraîné et évalué sur exactement les mêmes lignes, ce qui rend la
comparaison entre modèles honnête.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Racine du dépôt, calculée à partir de ce fichier : le chemin fonctionne
# quel que soit le dossier depuis lequel on lance le code (notebook, tests...).
RACINE = Path(__file__).resolve().parent.parent
CHEMIN_DONNEES_BRUTES = RACINE / "data" / "raw" / "churn_modelling.csv"
DOSSIER_PROCESSED = RACINE / "data" / "processed"

CIBLE = "Exited"

# Identifiants sans valeur prédictive. Surname est en plus une donnée personnelle.
COLONNES_IDENTIFIANTS = ["RowNumber", "CustomerId", "Surname"]

# Valeurs par défaut du découpage. Les changer impose de réentraîner tous les modèles.
TAILLE_TEST = 0.2
GRAINE = 42


def charger_donnees(chemin: Path = CHEMIN_DONNEES_BRUTES) -> pd.DataFrame:
    """Charge le jeu de données brut.

    Le fichier utilise « ; » comme séparateur et commence par un BOM
    (caractère invisible ajouté par Excel). Sans ces deux réglages, pandas
    lit une seule colonne ou renomme la première en « \\ufeffRowNumber ».

    Args:
        chemin: chemin du CSV. Par défaut, data/raw/churn_modelling.csv.

    Returns:
        Le DataFrame brut (10 000 lignes, 14 colonnes).
    """
    return pd.read_csv(chemin, sep=";", encoding="utf-8-sig")


def retirer_identifiants(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les colonnes d'identifiants, inutiles pour la modélisation.

    Args:
        df: DataFrame contenant les colonnes de COLONNES_IDENTIFIANTS.

    Returns:
        Une copie du DataFrame sans ces colonnes.
    """
    return df.drop(columns=COLONNES_IDENTIFIANTS)


def separer_train_test(
    df: pd.DataFrame,
    taille_test: float = TAILLE_TEST,
    graine: int = GRAINE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Découpe les données en jeu d'entraînement et jeu de test.

    - stratify : garde la même proportion de churners (environ 20 %) dans
      les deux jeux, sinon le hasard peut déséquilibrer le test.
    - graine fixe : le découpage est identique à chaque exécution et pour
      chaque membre du binôme.

    Args:
        df: DataFrame contenant la colonne cible.
        taille_test: part des lignes réservée au test (0.2 = 20 %).
        graine: graine du générateur aléatoire.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=CIBLE)
    y = df[CIBLE]
    return train_test_split(
        X, y, test_size=taille_test, stratify=y, random_state=graine
    )


def sauvegarder_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    dossier: Path = DOSSIER_PROCESSED,
) -> None:
    """Enregistre le découpage dans data/processed/ (dossier ignoré par Git).

    Utile pour relire le même découpage sans le recalculer. Les fichiers
    restent locaux : chacun les régénère avec separer_train_test().

    Args:
        X_train, X_test, y_train, y_test: sorties de separer_train_test().
        dossier: dossier de destination.
    """
    dossier.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(dossier / "train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(dossier / "test.csv", index=False)
