from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
import pandas as pd
import logging

app = FastAPI()

# Configurer la journalisation
logging.basicConfig(level=logging.INFO)

# Charger le modèle sauvegardé
model_path = "https://github.com/imanitou/P7/tree/main/application/mlflow_model_"
model = mlflow.sklearn.load_model(model_path)

# Charger les données des clients
try:
    clients_df = pd.read_csv("https://github.com/imanitou/P7/blob/main/app_train_with_feature_selection_subset.csv")
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
        # client_features = client_data.drop(columns=['SK_ID_CURR']).values
        client_features = client_data.values
        prediction = model.predict(client_features)
        logging.info(f"Prediction for client ID {client_id}: {prediction[0]}")
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Dans le terminal lancer : uvicorn api:app --reload

# Test pour faire une requête GET à l'API avec un ID de client existant : http://127.0.0.1:8000/predict/100006