# Explication sur base_commune : 

Nous allons chercher à vraiment jouer le jeu du travail collaboratif pour pratiquer le MLOps de facon plus concrète. 
Pour ça nous avons fait le choix de tester chacun de notre côté plusieurs classifiers binaires et les transformations associés à ces derniers.

Notre but est de gagner du temps et potentiellement d'optimiser nos résultats en pouvant passer plus de temps sur chaque modèle; Puis de mettre en commun les résultats
pour sélectionner le meilleur modèle. C'est sur ce dernier qu'on basera le reste du projet. Pour ça on va : 

1) Uniformiser l'importation et le découpage du dataset + le calcul d'un certains nombres de métriques de performances. 
2) Essayer de mettre en place un tracking avec MLFlow
3) Utiliser un petit csv qui compile les résumés de nos différents tests 

Cela va nous permettre de : 
- Eviter de biaiser la comparaison entre nos modèles (par exemple en n'utilisant pas la même seed d'aléatoire ou en oubliant la stratification)
- Faciliter la comparaison des modèles : on peut tout les deux calculer toutes les métriques qu'on veut mais on est sûr d'avoir au moins une base de comparaison 
- Visualiser les résultats de chaque modèle en attaquant direct le csv plutot que de compiler nos codes pour gagner du temps + gagner du temps entre nous dans la comparaison des modèles. 
- Nous entraîner à utiliser MLFlow ce qui est nouveau pour nous deux (on va donc s'aider de l'ia gen sur cette partie)

Ainsi on à créer outils.py qui va nous permettre d'importer quelques fonctions pour faire tout ça : 
charger_donnees 
, decouper_donnees et evaluer_modele