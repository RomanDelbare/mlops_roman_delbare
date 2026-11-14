"""Vérifie que data/train et data/test correspondent à decouper_donnees."""

import pandas as pd

from base_commune.outils import RACINE, TARGET, charger_donnees, decouper_donnees


def normaliser(tableau):
    """Trie les lignes pour comparer deux tableaux sans tenir compte de l'ordre."""
    return tableau.sort_values(by=list(tableau.columns)).reset_index(drop=True)


def test_les_csv_train_et_test_correspondent_au_decoupage():
    variables_entrainement, variables_test, cible_entrainement, cible_test = (
        decouper_donnees(charger_donnees())
    )
    attendus = {
        "train": variables_entrainement.assign(**{TARGET: cible_entrainement}),
        "test": variables_test.assign(**{TARGET: cible_test}),
    }
    for nom, attendu in attendus.items():
        fichier = RACINE / "data" / nom / f"{nom}.csv"
        lu = pd.read_csv(fichier)
        pd.testing.assert_frame_equal(
            normaliser(lu), normaliser(attendu), check_dtype=False
        )