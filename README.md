### Projet 7 : Élaboration d'un modèle de scoring

Nous travaillons pour la société financière, nommée "Prêt à dépenser", qui propose à ses clients des crédits à la consommation.

Celle-ci souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. 
Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées.

Pour aider l'entreprise dans la construction de cet outil, nous allons :
1) Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
2) Analyser les features qui contribuent le plus au modèle, de manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin, de mieux comprendre le score attribué par le modèle.
3) Mettre en production le modèle de scoring de prédiction à l’aide d’une API et réaliser une interface de test de cette API.
4) Mettre en œuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à l’analyse en production du data drift.

Pour ce faire :
- Nous testons trois modèles de classification différents (régression linéaire, random forest, light gbm) et retenons celui qui obtient les meilleurs performances (Light GBM).
- Nous faisons attention aux déséquilibres entres les classes et créons un score métier qui va permettre de produire un modèle qui répond au mieux aux besoins de l'entreprise (minimiser les faux négatifs = minimiser les "mauvais" clients prédits comme "bons").
- Nous créons de nouvelles variables métier et polynomiales (feature engineering) et faisons une sélection de variables avec la méthode Probe.
- Nous analysons l'importance locale et globale des features avec SHAP et LIME.
- Nous mettons en production notre modèle à l'aide d'une API que nous testons localement dans un premier temps (application Streamlit).
- Nous convevons des tests unitaires que l'on exécute automatiquement avec Github Actions.
- Nous déployons ensuite notre API (avec RENDER) et notre application (avec Streamlit Cloud) sur le Cloud, après avoir stocké notre modèle sur Google Cloud.
- Nous utilisons la librairie evidently pour étudier un éventuel data drift sur nos données.

Lien vers l'application : https://creditprediction.streamlit.app/ 

Pour l'analyse exploratoire des données, nous nous inspirons du kernel suivant : https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction/notebook
