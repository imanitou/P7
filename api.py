from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
import pandas as pd
import logging
import os
from google.cloud import storage
import tempfile

app = FastAPI()

# Configurer la journalisation
logging.basicConfig(level=logging.INFO)

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

# Remplacez par le nom de votre bucket et le chemin de votre modèle
bucket_name = 'bucket_mlflow_model'
model_blob_name = 'mlflow_model_/model.pkl'
model_local_path = '/tmp/model.pkl'

try:
    download_blob(bucket_name, model_blob_name, model_local_path)
    model = mlflow.sklearn.load_model(model_local_path)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Error loading model")

# Charger les données des clients
data_path = os.getenv('DATA_PATH', 'https://raw.githubusercontent.com/imanitou/P7/main/app_train_with_feature_selection_subset.csv')
try:
    clients_df = pd.read_csv(data_path)
    logging.info("Clients data loaded successfully.")
    logging.info(f"Clients DataFrame head:\n{clients_df.head()}")
except Exception as e:
    logging.error(f"Error loading clients data: {e}")
    raise HTTPException(status_code=500, detail="Error loading clients data")

@app.get("/")
def read_root():
    return {"message": "Welcome to the MLFlow Model API"}

@app.get("/predict/{client_id}")
def predict(client_id: int):
    try:
        # Rechercher le client par ID
        logging.info(f"Searching for client ID {client_id}")
        client_data = clients_df[clients_df['SK_ID_CURR'] == client_id]
        if client_data.empty:
            logging.warning(f"Client ID {client_id} not found.")
            raise HTTPException(status_code=404, detail="Client not found")

        # Supprimer la colonne ID pour la prédiction
        client_features = client_data.drop(columns=['SK_ID_CURR']).values
        prediction = model.predict(client_features)
        logging.info(f"Prediction for client ID {client_id}: {prediction[0]}")
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Dans le terminal lancer : uvicorn api:app --reload

# Test pour faire une requête GET à l'API avec un ID de client existant : http://127.0.0.1:8000/predict/100006