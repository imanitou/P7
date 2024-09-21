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
    logging.info(f"Blob {source_blob_name} téléchargé vers {destination_file_name}.")

# Configurer les chemins
bucket_name = 'bucket_mlflow_model'
model_blob_name = 'mlflow_model_/'
model_local_path = 'C:/Users/guill/Imane/P7/mlflow_model_/'

# Assurez-vous que le chemin local existe
os.makedirs(model_local_path, exist_ok=True)

# Télécharger chaque fichier du répertoire du modèle
files_to_download = ['conda.yaml', 'MLmodel', 'model.pkl', 'python_env.yaml', 'requirements.txt']  # Ajoutez d'autres fichiers si nécessaire

for file_name in files_to_download:
    download_blob(bucket_name, model_blob_name + file_name, os.path.join(model_local_path, file_name))

# Charger le modèle sauvegardé
model_path = os.path.abspath(model_local_path)

try:
    model = mlflow.sklearn.load_model(model_path)
    logging.info("Modèle chargé avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle: {e}")
    raise HTTPException(status_code=500, detail="Erreur lors du chargement du modèle")

# Charger les données des clients
data_path = 'https://raw.githubusercontent.com/imanitou/P7/main/app_train_with_feature_selection_subset.csv'
try:
    clients_df = pd.read_csv(data_path)
    logging.info("Données des clients chargées avec succès.")
    logging.info(f"En-tête du DataFrame des clients :\n{clients_df.head()}")
except Exception as e:
    logging.error(f"Erreur lors du chargement des données des clients : {e}")
    raise HTTPException(status_code=500, detail="Erreur lors du chargement des données des clients")

@app.get("/")
def read_root():
    return {"message": "Bienvenue à l'API du modèle MLFlow"}

@app.get("/predict/{client_id}")
def predict(client_id: int):
    try:
        # Rechercher le client par ID
        logging.info(f"Recherche du client ID {client_id}")
        client_data = clients_df[clients_df['SK_ID_CURR'] == client_id]
        if client_data.empty:
            logging.warning(f"Client ID {client_id} non trouvé.")
            raise HTTPException(status_code=404, detail="Client non trouvé")

        
        client_features = client_data.values
        prediction = model.predict(client_features)
        logging.info(f"Prédiction pour le client ID {client_id} : {prediction[0]}")
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Nettoyer le fichier temporaire après utilisation
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# Dans le terminal lancer : uvicorn api:app --reload

# Test pour faire une requête GET à l'API avec un ID de client existant : http://127.0.0.1:8000/predict/100006