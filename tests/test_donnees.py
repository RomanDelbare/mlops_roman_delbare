"""Tests du chargement et du découpage des données (lancer : pytest)."""

import pandas as pd

from src.donnees import (
    CIBLE,
    COLONNES_IDENTIFIANTS,
    charger_donnees,
    retirer_identifiants,
    separer_train_test,
)


def test_chargement_dimensions():
    """Le fichier doit être lu avec ses 14 colonnes, pas une seule."""
    df = charger_donnees()
    assert df.shape == (10_000, 14)


def test_chargement_premiere_colonne_sans_bom():
    """Le BOM ne doit pas se coller au nom de la première colonne."""
    df = charger_donnees()
    assert df.columns[0] == "RowNumber"


def test_cible_binaire():
    """La cible ne contient que 0 et 1."""
    df = charger_donnees()
    assert set(df[CIBLE].unique()) == {0, 1}


def test_retirer_identifiants():
    """Les identifiants disparaissent, la cible reste."""
    df = retirer_identifiants(charger_donnees())
    assert not set(COLONNES_IDENTIFIANTS) & set(df.columns)
    assert CIBLE in df.columns


def test_split_tailles():
    """80 % en train, 20 % en test, sans perte de lignes."""
    X_train, X_test, y_train, y_test = separer_train_test(charger_donnees())
    assert len(X_train) == 8_000
    assert len(X_test) == 2_000
    assert CIBLE not in X_train.columns


def test_split_stratifie():
    """La part de churners est la même (à 1 point près) dans les deux jeux."""
    _, _, y_train, y_test = separer_train_test(charger_donnees())
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_split_reproductible():
    """Deux appels avec la même graine donnent exactement le même test."""
    df = charger_donnees()
    _, X_test_1, _, _ = separer_train_test(df)
    _, X_test_2, _, _ = separer_train_test(df)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)
