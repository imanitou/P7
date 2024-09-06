import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from babel.numbers import format_decimal
import requests

# Charger les données des clients
data_url = 'https://raw.githubusercontent.com/imanitou/P7/main/app_train_with_feature_selection_subset.csv'
clients_df = pd.read_csv(data_url)

# Fonction pour obtenir les informations d'un client
def get_client_info(client_id):
    client_info = clients_df[clients_df['SK_ID_CURR'] == client_id]
    return client_info

# Fonction pour formater les nombres en utilisant un séparateur de millier français
def format_number(number):
    try:
        return format_decimal(number, locale='fr_FR')
    except:
        return number

# Interface utilisateur Streamlit
st.title("Prédiction de Remboursement de Crédit")

# Ajouter du CSS pour changer la police
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,700;1,400;1,700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Source Serif 4', serif;
    }
    
    .centered {
        text-align: center;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# Entrée pour l'ID du client
client_id = st.number_input("Entrez le SK_ID_CURR du client :", min_value=int(clients_df['SK_ID_CURR'].min()), max_value=int(clients_df['SK_ID_CURR'].max()))

# Sélectionner les colonnes spécifiques à afficher
columns_to_display = ['SK_ID_CURR', 'AMT_ANNUITY', 'AMT_CREDIT', 'ANNUITY_INCOME_PERCENT', 'DAYS_EMPLOYED', 'CREDIT_INCOME_PERCENT', 'OWN_CAR_AGE', 'previous_loan_counts',
'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT', 'NAME_FAMILY_STATUS_Married']

# Afficher les informations du client
if client_id:
    client_info = get_client_info(client_id)
    if not client_info.empty:
        st.write("Informations concernant le client :")
        formatted_info = client_info[columns_to_display].applymap(lambda x: format_number(x) if isinstance(x, (int, float)) else x)
        st.dataframe(formatted_info)
        
        # Envoyer la requête à l'API pour obtenir la prédiction
        response = requests.get(f"https://p7-9ze0.onrender.com/predict/{client_id}")
        if response.status_code == 200:
            prediction = response.json()['prediction'][0]
            if prediction == 1:
                st.write("**Prédiction : BON CLIENT ! Le client devrait rembourser son crédit.**")
            else:
                st.write("**Prédiction : ATTENTION ! Le client risque de ne pas rembourser son crédit.**")
        else:
            st.error("Erreur lors de la prédiction")
            
        # Comparaison des caractéristiques du client avec la moyenne des autres clients
        st.markdown("<p class='centered'><u>Analyse univariée</u></p>", unsafe_allow_html=True)
        features = ['AMT_CREDIT', 'AMT_ANNUITY', 'ANNUITY_INCOME_PERCENT', 'CREDIT_INCOME_PERCENT', 'CREDIT_TERM']
        
                # Menu déroulant pour sélectionner la caractéristique à afficher
        selected_feature = st.selectbox("Choisissez une caractéristique à afficher : ", features)
        
        # Compute statistics
        feature_value = client_info[selected_feature].values[0]
        feature_mean = clients_df[selected_feature].mean()
        feature_median = clients_df[selected_feature].median()

        # Display the description of the selected feature
        st.write(f"**Client** : {format_number(feature_value)}")
        st.write(f"**Moyenne** : {format_number(feature_mean)}")
        st.write(f"**Médiane** : {format_number(feature_median)}")

        # Display the graph for the selected feature
        plt.figure(figsize=(10, 4))
        sns.kdeplot(clients_df[selected_feature], label='Distribution générale', color='#1E2D2F')
        plt.axvline(feature_value, color='red', linestyle='--', label='Client actuel')
        plt.title(f"Comparaison de la caractéristique {selected_feature}", fontsize=15)
        plt.legend(fontsize=12)
        st.pyplot(plt)

        # Bivariate Analysis
        st.markdown("<p class='centered'><u>Analyse Bivariée</u></p>", unsafe_allow_html=True)
        
        # Dropdown menus for selecting features for bivariate analysis
        feature1 = st.selectbox("Choisissez la première caractéristique :", features)
        feature2 = st.selectbox("Choisissez la seconde caractéristique :", features)
        
        if feature1 and feature2:
            plt.figure(figsize=(12, 6))
            # Scatter plot
            sns.scatterplot(data=clients_df, x=feature1, y=feature2, alpha=0.4, color='#1E2D2F')
            
            # Plot the client point
            client_x = client_info[feature1].values[0]
            client_y = client_info[feature2].values[0]
            plt.scatter(client_x, client_y, color='red', s=100, label='Client actuel')

            plt.title(f"Analyse Bivariée entre {feature1} et {feature2}", fontsize=15)
            plt.xlabel(feature1, fontsize=12)
            plt.ylabel(feature2, fontsize=12)
            plt.legend(fontsize=12)
            st.pyplot(plt)
    else:
        st.error("Client non trouvé")

# à écrire dans l'invite de commande : streamlit run app_streamlit.py