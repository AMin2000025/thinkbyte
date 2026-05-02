import streamlit as st

# --- CONFIGURATION GLOBALE ---
# IMPORTANT : C'est ici qu'on définit la config pour toute l'application
st.set_page_config(page_title="TrueReach AI | Gateway", layout="wide", initial_sidebar_state="collapsed")

# --- STYLE CSS (DESIGN 3SG) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@900&display=swap');
    .stApp { background-color: #0e0e0e; color: white; }
    .main-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 60px;
        text-align: center;
        margin-bottom: 10px;
        line-height: 1;
    }
    .accent-red { color: #e31837; }
    .subtitle { text-align: center; color: #888; font-size: 20px; margin-bottom: 50px; }
    
    /* Style des cartes de choix */
    .option-card {
        background-color: #1a1a1a;
        padding: 40px;
        border-radius: 15px;
        border: 2px solid #333;
        text-align: center;
        transition: 0.3s;
    }
    .option-card:hover {
        border-color: #e31837;
        transform: translateY(-10px);
    }
    h2 { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE DE NAVIGATION ---
def login_page():
    st.markdown("<p class='main-title'>TRUEREACH <span class='accent-red'>AI</span></p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Sélectionnez votre portail d'accès</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class='option-card'>
                <h1 style='font-size: 50px;'>👤</h1>
                <h2>INFLUENCEUR</h2>
                <p>Gérez vos données, analysez vos posts et créez votre profil certifié.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("ACCÉDER À MON ESPACE", key="btn_inf", use_container_width=True):
            st.switch_page("app_v2.py")

    with col2:
        st.markdown("""
            <div class='option-card'>
                <h1 style='font-size: 50px;'>🛡️</h1>
                <h2>ADMIN / MARQUE</h2>
                <p>Analysez le marché, comparez les profils et optimisez vos budgets.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("ACCÉDER AU DASHBOARD AGÉNCE", key="btn_admin", use_container_width=True):
            st.switch_page("espace_marque.py")

# --- DÉFINITION DES PAGES ---
# On déclare les fichiers existants comme faisant partie de la navigation
pg = st.navigation({
    "Accueil": [st.Page(login_page, title="Connexion", icon="🏠")],
    "Espaces": [
        st.Page("app_v2.py", title="Espace Influenceur", icon="👤"),
        st.Page("espace_marque.py", title="Espace Administrateur", icon="📊")
    ]
})

# Lancement de la navigation
pg.run()