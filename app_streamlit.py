import streamlit as st
import requests

# URL de l'API
api_url = "http://127.0.0.1:8000/predict"

st.title("MLFlow Model Deployment with Streamlit")
st.write("This is a simple web app to predict using the MLFlow model via an API.")

# Entrée de l'ID du client
client_id = st.number_input("Enter Client ID", min_value=1, step=1)

if st.button("Predict"):
    try:
        # Envoyer la requête GET à l'API
        response = requests.get(f"{api_url}/{client_id}")
        response.raise_for_status()
        prediction = response.json()["prediction"]
        st.write(f"Prediction for Client ID {client_id}: {prediction[0]}")
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        st.error(f"Other error occurred: {err}")

# Lancer l'application Streamlit dans le terminal : streamlit run app_streamlit.py