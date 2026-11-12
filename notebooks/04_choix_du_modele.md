# Choix du modèle final

On a donc comparé les modèles en utilisant les mm règles. 
Pour chacun voici la meilleur config : 

| Modèle (meilleure configuration) | PR-AUC (moyenne ± écart-type) | ROC-AUC |
|---|---|---|
| Ligne de base (prédit toujours « reste ») | 0,204 | 0,500 |
| KNN (k = 51, pondéré par la distance, sans Tenure ni EstimatedSalary) | 0,621 | 0,838 |
| SVM (noyau RBF, C = 1, gamma = « scale ») | 0,667| 0,832 |
| Régression logistique (meilleure configuration) | 0,668  | |
| Random Forest (profondeur 8, feuille minimale 5) | 0,681  | 0,856 |
| XGBoost (300 arbres, profondeur 3, taux d'apprentissage 0,05) | 0,699  | 0,864 |

## Décision

XGBoost est retenu : c'est le meilleur score observé sur les métriques qui nous interessent. 
Parmis les quelques meilleurs config :
On garde la config la config la plus stable (écart le + faible entre entrainement et valdaition).

## Seuil de décision

Le seuil par défaut (0,5) donne un rappel de 0,46 seulement. On choisit le seuil qui maximise
le F1 sur les prédictions hors pli du jeu d'entraînement : 0,30. Passer à 0,25 (rappel de
0,70) permettrait de détecter environ 68 churners de plus sur 8 000 clients, mais obligerait
à contacter environ 300 clients de plus, soit environ 3,4 contacts inutiles par départ évité.
