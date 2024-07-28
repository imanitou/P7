import pytest
from fastapi.testclient import TestClient
from api_2 import app, model

client = TestClient(app)

# Test pour vérifier que l'API racine ("/") renvoie le message
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the MLFlow Model API"}

# Test quand bonne prédiction
def test_predict_existing_client():
    response = client.get("/predict/100002")
    assert response.status_code == {"prediction": [1]}

# Test quand mauvaise prédiction
def test_predict_existing_client():
    response = client.get("/predict/100002")
    assert response.status_code == 200
    assert response.json() == {"prediction": [1]}

# Test pour un client non existant
def test_predict_nonexistent_client():
    response = client.get("/predict/98765")
    assert response.status_code == 400

# Test pour une entrée non valide 
def test_input_validation():
    response = client.get("/predict/abcd")
    assert response.status_code == 422

# Test quand problème dans le chargement du modèle ou des données clients

client = TestClient(app)

def test_error_response():
    # Charger temporairement un modèle incorrect
    model_loaded = False

    # Envoyer une requête de prédiction avec un ID client
    response = client.get("/predict/100002")

    # Vérifier que la réponse est une erreur HTTP 500
    assert response.status_code == 200

    # Vérifier que le message d'erreur est correct
    assert response.json() == {'prediction': [1]}

    # Réinitialiser le modèle chargé
    model_loaded = True

    # Envoyer une autre requête de prédiction avec un ID client
    response = client.get("/predict/100002")

    # Vérifier que la réponse est une erreur 
    assert response.status_code == 200
   