import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from babel.numbers import format_decimal
import requests
import plotly.graph_objects as go
import plotly.express as px

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
# columns_to_display = ['SK_ID_CURR', 'AMT_ANNUITY', 'AMT_CREDIT', 'ANNUITY_INCOME_PERCENT', 'DAYS_EMPLOYED', 'CREDIT_INCOME_PERCENT', 'OWN_CAR_AGE', 'previous_loan_counts',
# 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT', 'NAME_FAMILY_STATUS_Married']

# Afficher les informations du client
if client_id:
    client_info = get_client_info(client_id)
    if not client_info.empty:
        st.write("Informations concernant le client :")
        formatted_info = client_info.applymap(lambda x: format_number(x) if isinstance(x, (int, float)) else x)
        st.dataframe(formatted_info)
        
        # Envoyer la requête à l'API pour obtenir la prédiction
        response = requests.get(f"https://p7-9ze0.onrender.com/predict/{client_id}")
        if response.status_code == 200:
            prediction = response.json()['prediction'][0]
            if prediction == 0:
                st.write("**Prédiction : BON CLIENT ! Le client devrait rembourser son crédit.**")
            else:
                st.write("**Prédiction : ATTENTION ! Le client risque de ne pas rembourser son crédit.**")
        else:
            st.error("Erreur lors de la prédiction")

# Importance globale des caractéristiques

# Charger les importances globales
global_importance_df = pd.read_csv('global_shap_importance.csv')
# Il faudra changer le chemin ici (mettre le fichier dans GIT)

# Charger les descriptions
description = pd.read_csv('description.csv')
# Il faudra changer le chemin ici (mettre le fichier dans GIT)

# Titre
st.markdown("""<p style='text-align: center; font-size: 24px; text-decoration: underline;'>
            II. Importance globale des caractéristiques
            </p>
            <div style='height: 30px;'></div>  <!-- Ajouter un espace de 30 pixels -->
            """, 
            unsafe_allow_html=True)

# Dataframe
st.dataframe(global_importance_df, height=250)

st.markdown("""
            <div style='height: 30px;'></div>  <!-- Ajouter un espace de 30 pixels -->
            """, 
            unsafe_allow_html=True)

# On retient les 10 principales features
global_importance_df_10 = global_importance_df[:10].sort_values(by='Importance', ascending=True)


# Créer un graphique en barres avec Plotly
fig = px.bar(
    global_importance_df_10,
    x='Importance',
    y='Feature',
    orientation='h',  # Pour un graphique horizontal
    #title="Importance des caractéristiques pour la prédiction",
    text=global_importance_df_10['Importance'].map('{:.3f}'.format),  # Ajouter des labels pour chaque barre
    color='Importance',  # Gradient de couleur basé sur l'importance
    color_continuous_scale='brwnyl',  # Choisir une palette de couleurs
)

# Personnalisation du graphique
fig.update_layout(
    title={
        'text': "Top 10 des caractéristiques qui ont le plus influencé<br>globalement les prédictions du modèle",
        'y': 0.95,  # Position verticale
        'x': 0.5,  # Centrer le titre
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=18, color='black')
    },
    xaxis_title="Valeur SHAP moyenne",
    xaxis=dict(
        title_font=dict(size=16, color='black'),  # Couleur du texte du titre de l'axe des x
        tickfont=dict(size=14, color='black')  # Couleur du texte des ticks de l'axe des x
    ),
    yaxis_title=None,
    yaxis=dict(
        title_font=dict(size=14, color='black'),  # Couleur du texte du titre de l'axe des y
        tickfont=dict(size=14, color='black')  # Couleur du texte des ticks de l'axe des y
    ),
    font=dict(size=12, color='black'),
    plot_bgcolor='rgba(0,0,0,0)',  # Fond transparent
    paper_bgcolor='rgba(0,0,0,0)',
    showlegend=False,  # Cacher la légende
    coloraxis_colorbar=dict(
        title='Valeur SHAP moyenne',  # Titre de la barre de couleur
        title_font=dict(size=14, color='black'),  # Couleur du titre de la barre de couleur
        tickfont=dict(size=12, color='black')  # Couleur du texte des ticks de la barre de couleur
    ),
)

# Ajouter une ligne horizontale pour l'axe des abscisses
fig.update_xaxes(
    # zeroline=True,  # Afficher la ligne de l'axe des abscisses
    # zerolinecolor='white',  # Couleur de la ligne de l'axe des abscisses
    # zerolinewidth=2,  # Épaisseur de la ligne de l'axe des abscisses
    showline=True,  # Afficher la ligne de l'axe
    linecolor='black',  # Couleur de la ligne de l'axe
    linewidth=0.5  # Épaisseur de la ligne de l'axe
)

# Ajustez la taille de la figure
fig.update_layout(
    width=1200,  # Largeur en pixels
    height=600,  # Hauteur en pixels
)

# Afficher le graphique avec Streamlit
st.plotly_chart(fig)

# Description des caractéristiques

st.markdown("""<div style='height: 30px;'></div>  <!-- Ajouter un espace de 30 pixels -->
            <p style='text-align: center; font-size: 24px; text-decoration: underline;'>
            III. Description des caractéristiques
            </p>
            <div style='height: 30px;'></div>  <!-- Ajouter un espace de 30 pixels -->
            """,  
            unsafe_allow_html=True)

# Fusionner les DataFrames sur la colonne "Feature" 
merged_df = global_importance_df.merge(description, on="Feature", how="left")

# Remplacer les valeurs manquantes par "non renseigné"
merged_df = merged_df.fillna('Description non disponible.')

# Ajouter une fonctionnalité de sélection pour afficher la description
selected_feature = st.selectbox("Sélectionnez une caractéristique :", merged_df["Feature"].tolist())

# Afficher la description correspondante
if selected_feature:
    description = merged_df[merged_df["Feature"] == selected_feature]["Description"].values[0]
    st.write(f"**Description :** {description}")

# Comparaison des caractéristiques du client avec la moyenne des autres clients
st.markdown("""<p style='text-align: center; font-size: 24px; text-decoration: underline;'>
            IV. Analyse univariée
            </p>
            <div style='height: 30px;'></div>  <!-- Ajouter un espace de 30 pixels -->
            """,  
            unsafe_allow_html=True)

# features = ['AMT_CREDIT', 'AMT_ANNUITY', 'ANNUITY_INCOME_PERCENT', 'CREDIT_INCOME_PERCENT', 'CREDIT_TERM']
features = [col for col in clients_df.columns if col != "SK_ID_CURR"]

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
fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(clients_df[selected_feature], label='Distribution générale', color='#1E2D2F')
plt.axvline(feature_value, color='red', linestyle='--', label='Client actuel')
# Modifiez la couleur de fond du graphique
ax.set_facecolor('#d6ccc3')
# Modifiez la couleur des contours des axes
ax.spines['top'].set_edgecolor('#eeeae7')  # Couleur du contour du haut
ax.spines['bottom'].set_edgecolor('#eeeae7')  # Couleur du contour du bas
ax.spines['right'].set_edgecolor('#eeeae7')  # Couleur du contour de droite
ax.spines['left'].set_edgecolor('#eeeae7')  # Couleur du contour de gauche

# Modifiez la couleur du fond de la figure
fig.patch.set_facecolor('#eeeae7')  # Couleur de fond de la figure
plt.title(f"Comparaison de la caractéristique {selected_feature}", fontsize=17)
plt.legend(fontsize=14)
plt.xlabel(xlabel=selected_feature, fontsize=14)
plt.ylabel(ylabel="Density", fontsize=14)
st.pyplot(plt)

# Bivariate Analysis
st.markdown("""<div style='height: 30px;'></div>  <!-- Ajouter un espace de 30 pixels -->
            <p style='text-align: center; font-size: 24px; text-decoration: underline;'>
            V. Analyse bivariée
            </p>
            <div style='height: 30px;'></div>  <!-- Ajouter un espace de 30 pixels -->
            """, 
            unsafe_allow_html=True)


# Dropdown menus for selecting features for bivariate analysis
feature1 = st.selectbox("Choisissez la première caractéristique :", features)
feature2 = st.selectbox("Choisissez la seconde caractéristique :", features)

if feature1 and feature2:
    fig, ax = plt.subplots(figsize=(12, 6))
    # Scatter plot
    sns.scatterplot(data=clients_df, x=feature1, y=feature2, alpha=0.4, color='#1E2D2F')
    
    # Plot the client point
    client_x = client_info[feature1].values[0]
    client_y = client_info[feature2].values[0]
    plt.scatter(client_x, client_y, color='red', s=100, label='Client actuel')

    # Modifiez la couleur de fond du graphique
    ax.set_facecolor('#d6ccc3')
    # Modifiez la couleur des contours des axes
    ax.spines['top'].set_edgecolor('#eeeae7')  # Couleur du contour du haut
    ax.spines['bottom'].set_edgecolor('#eeeae7')  # Couleur du contour du bas
    ax.spines['right'].set_edgecolor('#eeeae7')  # Couleur du contour de droite
    ax.spines['left'].set_edgecolor('#eeeae7')  # Couleur du contour de gauche

    # Modifiez la couleur du fond de la figure
    fig.patch.set_facecolor('#eeeae7')  # Couleur de fond de la figure

    plt.title(f"Analyse Bivariée entre {feature1} et {feature2}", fontsize=17)
    plt.xlabel(feature1, fontsize=14)
    plt.ylabel(feature2, fontsize=14)
    plt.legend(fontsize=14)
    st.pyplot(plt)
else:
    st.error("Client non trouvé")

# à écrire dans l'invite de commande : streamlit run app_streamlit.py

            
#         Comparaison des caractéristiques du client avec la moyenne des autres clients
#         st.markdown("<p class='centered'><u>Analyse univariée</u></p>", unsafe_allow_html=True)
#         features = ['AMT_CREDIT', 'AMT_ANNUITY', 'ANNUITY_INCOME_PERCENT', 'CREDIT_INCOME_PERCENT', 'CREDIT_TERM']
        
#                 Menu déroulant pour sélectionner la caractéristique à afficher
#         selected_feature = st.selectbox("Choisissez une caractéristique à afficher : ", features)
        
#         Compute statistics
#         feature_value = client_info[selected_feature].values[0]
#         feature_mean = clients_df[selected_feature].mean()
#         feature_median = clients_df[selected_feature].median()

#         Display the description of the selected feature
#         st.write(f"**Client** : {format_number(feature_value)}")
#         st.write(f"**Moyenne** : {format_number(feature_mean)}")
#         st.write(f"**Médiane** : {format_number(feature_median)}")

#         Display the graph for the selected feature
#         plt.figure(figsize=(10, 4))
#         sns.kdeplot(clients_df[selected_feature], label='Distribution générale', color='#1E2D2F')
#         plt.axvline(feature_value, color='red', linestyle='--', label='Client actuel')
#         plt.title(f"Comparaison de la caractéristique {selected_feature}", fontsize=15)
#         plt.legend(fontsize=12)
#         st.pyplot(plt)

#         Bivariate Analysis
#         st.markdown("<p class='centered'><u>Analyse Bivariée</u></p>", unsafe_allow_html=True)
        
#         Dropdown menus for selecting features for bivariate analysis
#         feature1 = st.selectbox("Choisissez la première caractéristique :", features)
#         feature2 = st.selectbox("Choisissez la seconde caractéristique :", features)
        
#         if feature1 and feature2:
#             plt.figure(figsize=(12, 6))
#             Scatter plot
#             sns.scatterplot(data=clients_df, x=feature1, y=feature2, alpha=0.4, color='#1E2D2F')
            
#             Plot the client point
#             client_x = client_info[feature1].values[0]
#             client_y = client_info[feature2].values[0]
#             plt.scatter(client_x, client_y, color='red', s=100, label='Client actuel')

#             plt.title(f"Analyse Bivariée entre {feature1} et {feature2}", fontsize=15)
#             plt.xlabel(feature1, fontsize=12)
#             plt.ylabel(feature2, fontsize=12)
#             plt.legend(fontsize=12)
#             st.pyplot(plt)
#     else:
#         st.error("Client non trouvé")

# à écrire dans l'invite de commande : streamlit run app_streamlit.py