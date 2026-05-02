import streamlit as st
import pandas as pd
import os

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="TrueReach AI | Connexion Créateur", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0e0e0e; color: white; }
    .accent-red { color: #e31837; font-family: 'Arial Black'; }
    div.stButton > button {
        background-color: #e31837; color: white; border: none;
        font-weight: bold; width: 100%; border-radius: 0;
    }
    .stTextInput input, .stNumberInput input, .stSelectbox div {
        background-color: #1a1a1a !important; color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FICHIER DE STOCKAGE ---
DB_FILE = "database_plateforme.csv"

# Fonction pour sauvegarder les données
def sauvegarder_donnees(nouvelle_entree):
    if not os.path.isfile(DB_FILE):
        df = pd.DataFrame([nouvelle_entree])
        df.to_csv(DB_FILE, index=False)
    else:
        df = pd.read_csv(DB_FILE)
        df = pd.concat([df, pd.DataFrame([nouvelle_entree])], ignore_index=True)
        df.to_csv(DB_FILE, index=False)

# --- INTERFACE DE CONNEXION ---
st.markdown("<h1>CONNEXION <span class='accent-red'>CRÉATEUR</span></h1>", unsafe_allow_html=True)

if 'connecte' not in st.session_state:
    st.session_state.connecte = False

if not st.session_state.connecte:
    with st.form("login_form"):
        nom_user = st.text_input("Votre Nom d'influenceur (ex: Sarra B.)")
        id_user = st.text_input("Votre ID unique (ex: INF_999)")
        submit_login = st.form_submit_button("SE CONNECTER")
        
        if submit_login and nom_user and id_user:
            st.session_state.connecte = True
            st.session_state.nom = nom_user
            st.session_state.id = id_user
            st.rerun()
else:
    # --- FORMULAIRE DE QUESTIONS (CRÉATION DE DONNÉES) ---
    st.success(f"Connecté en tant que : {st.session_state.nom}")
    st.write("### Répondez aux questions pour enregistrer votre collaboration")

    with st.form("questions_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            platform = st.selectbox("Sur quelle plateforme était la collab ?", ["Instagram", "TikTok", "YouTube"])
            niche = st.selectbox("Quelle est votre thématique ?", ["Beauté", "Food", "Humour", "Sport", "Tech"])
            followers = st.number_input("Nombre d'abonnés actuel", min_value=0)
            category = st.text_input("Catégorie du produit promu (ex: Électronique)")
            
        with col2:
            engagement = st.number_input("Taux d'engagement (%)", format="%.2f")
            reels = st.number_input("Nombre de Reels postés", min_value=0)
            stories = st.number_input("Nombre de Stories postées", min_value=0)
            price = st.number_input("Prix payé par la marque (TND)", min_value=0.0)

        # Bouton d'enregistrement
        submit_data = st.form_submit_button("ENREGISTRER SUR LA PLATEFORME")

        if submit_data:
            # Création du dictionnaire selon la structure du CSV
            nouvelle_collab = {
                "influencer_id": st.session_state.id,
                "influencer_name": st.session_state.nom,
                "platform": platform,
                "niche": niche,
                "followers_count": followers,
                "engagement_rate": engagement,
                "product_category": category,
                "nb_reels": reels,
                "nb_stories": stories,
                "price_paid": price
            }
            
            sauvegarder_donnees(nouvelle_collab)
            st.balloons()
            st.success("✅ Données stockées avec succès dans la base de données de la plateforme !")

    # Bouton pour voir la base de données actuelle
    if st.checkbox("Voir l'historique de la plateforme"):
        if os.path.isfile(DB_FILE):
            st.dataframe(pd.read_csv(DB_FILE))
        else:
            st.info("La base de données est vide.")

    if st.button("Se déconnecter"):
        st.session_state.connecte = False
        st.rerun()