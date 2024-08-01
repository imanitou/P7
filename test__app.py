import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from api_2 import app, model

client = TestClient(app)

# Test pour vérifier que l'API racine ("/") renvoie le message
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the MLFlow Model API"}

# Test quand la prédiction est correcte
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

# Test pour simuler un problème dans le chargement du modèle
@patch('api_2.model', new_callable=Mock)
def test_error_response(mock_model):
    # Simule un modèle défectueux
    mock_model.side_effect = Exception("Model loading error")

    # Envoyer une requête de prédiction pour vérifier la gestion des erreurs
    response = client.get("/predict/100002")
    assert response.status_code == 400  # En cas de problème avec le modèle, on s'attend à une erreur serveur

    # Vous pouvez également tester la réinitialisation du modèle ici si nécessaire.
