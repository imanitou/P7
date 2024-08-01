import pytest
import requests

# Remplacez par l'URL de votre API déployée
API_URL = "https://p7-9ze0.onrender.com"


def test_root():
    response = requests.get(f"{API_URL}/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue à l'API du modèle MLFlow"}

def test_predict_existing_client():
    response = requests.get(f"{API_URL}/predict/100002")
    assert response.status_code == 200
    assert response.json() == {"prediction": [1]}  

def test_predict_nonexistent_client():
    response = requests.get(f"{API_URL}/predict/98765")
    assert response.status_code == 400

def test_input_validation():
    response = requests.get(f"{API_URL}/predict/abcd")
    assert response.status_code == 422

# Test pour simuler un problème dans le chargement du modèle (si applicable)
@pytest.fixture(autouse=True)
def mock_model_loading_error(monkeypatch):
    # Simule un problème de chargement du modèle (à ajuster si nécessaire)
    pass