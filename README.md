# Fast_API_Openclassrooms_P7

Modèle de Scoring Client 

* 1 Traitement des valeurs manquantes par simpleimputer en remplaçant par la médiane & des corrélations via V de cramer   
* 2 Equilibrage des données avec undersampling
* 3 Algorithme: Lightgbm optimisé avec randomizersearch + seuil optimal avec Hyperopt
* 4 Creation d'un dashboard avec la librairie Streamlit + Déploiement sur Heroku 

Code de l'API - Utilisation du framework FastAPI

* 1 Import du model en format pickle via joblib
* 2 Test du bon fonctionnement de l'API sur la page : https://fastapi-p7.herokuapp.com
* 3 Le endpoint pour la requête de type POST a été dépriorisé au profit d'un GET nous donnat directement les prédictions du modèle. 
* 4 Creation du endpoint GET fourissant l'intégralité des prédictions de score clients dans un dictionnaire au format {'id_client':'score'}

lien vers le Git : https://github.com/MartinIzabel/Fast_API_Openclassrooms_P7
lien vers l'API contenant toutes les prédictions clients : https://fastapi-p7.herokuapp.com/score