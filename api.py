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

# Lire le contenu JSON des credentials depuis la variable d'environnement
key_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')

if key_json is None:
    raise ValueError("La variable d'environnement GOOGLE_APPLICATION_CREDENTIALS_JSON n'est pas définie.")

# Créer un fichier temporaire pour les credentials JSON
with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as temp_file:
    temp_file.write(key_json)
    temp_file_path = temp_file.name

# Définir la variable d'environnement pour le chemin du fichier temporaire
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Télécharge un blob depuis le bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logging.info(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

# Configurez les chemins
bucket_name = 'bucket_mlflow_model'
model_blob_name = 'mlflow_model_'
model_local_path = '/tmp/mlflow_model_'

# Télécharger le modèle
try:
    download_blob(bucket_name, model_blob_name, model_local_path)
    logging.info("Model downloaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Error loading model")

# Charger le modèle sauvegardé
model_path = model_local_path
model = mlflow.sklearn.load_model(model_path)

# Charger les données des clients
data_path = 'https://raw.githubusercontent.com/imanitou/P7/main/app_train_with_feature_selection_subset.csv'
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
    finally:
        # Nettoyer le fichier temporaire après utilisation
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# Dans le terminal lancer : uvicorn api:app --reload

# Test pour faire une requête GET à l'API avec un ID de client existant : http://127.0.0.1:8000/predict/100006