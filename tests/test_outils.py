"""Tests des fonctions communes de base_commune/outils.py (lancer : pytest)."""

import pandas as pd
import pytest
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from base_commune.outils import (
    COLONNES_IDENTIFIANTES,
    SCORING,
    TARGET,
    charger_donnees,
    decouper_donnees,
    evaluer_modele,
    evaluer_test,
    tags_reproductibilite,
)


@pytest.fixture(scope="module")
def df():
    return charger_donnees()


@pytest.fixture(scope="module")
def modele():
    pretraitement = make_column_transformer(
        (OneHotEncoder(), ["Geography", "Gender"]),
        remainder=StandardScaler(),
    )
    return make_pipeline(pretraitement, LogisticRegression(max_iter=1000))


def test_chargement(df):
    """10 000 lignes, 14 colonnes moins les 3 identifiantes."""
    assert df.shape == (10_000, 11)
    assert not set(COLONNES_IDENTIFIANTES) & set(df.columns)
    assert set(df[TARGET].unique()) == {0, 1}


def test_decoupage_tailles(df):
    X_train, X_test, y_train, y_test = decouper_donnees(df)
    assert (len(X_train), len(X_test)) == (8_000, 2_000)
    assert TARGET not in X_train.columns


def test_decoupage_stratifie(df):
    """Même part de churners (à 1 point près) dans le train et le test."""
    _, _, y_train, y_test = decouper_donnees(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_decoupage_reproductible(df):
    """Deux appels donnent exactement le même test."""
    _, X_test_1, _, _ = decouper_donnees(df)
    _, X_test_2, _, _ = decouper_donnees(df)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)


def test_evaluer_modele(df, modele):
    """Toutes les métriques communes sont renvoyées, entre 0 et 1."""
    X_train, _, y_train, _ = decouper_donnees(df)
    metriques = evaluer_modele(modele, X_train, y_train, n_splits=3)
    for nom in SCORING:
        assert 0 <= metriques[f"cv_{nom}_mean"] <= 1
        assert f"cv_{nom}_std" in metriques
        assert f"train_{nom}_mean" in metriques


def test_evaluer_test(df, modele):
    X_train, X_test, y_train, y_test = decouper_donnees(df)
    metriques = evaluer_test(modele, X_train, y_train, X_test, y_test)
    assert set(metriques) == {f"test_{nom}" for nom in SCORING}
    assert all(0 <= v <= 1 for v in metriques.values())


def test_tags_reproductibilite():
    tags = tags_reproductibilite()
    assert set(tags) == {"git_commit", "git_dirty", "data_md5"}
    assert len(tags["data_md5"]) == 32
