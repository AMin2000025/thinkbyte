import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- CONFIGURATION ET STYLE ---
st.set_page_config(page_title="TrueReach AI | Espace Marque", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e0e0e; color: white; }
    h1, h2, h3 { font-family: 'Arial Black'; text-transform: uppercase; }
    .accent-red { color: #e31837; }
    .metric-box { background-color: #1a1a1a; padding: 20px; border-left: 5px solid #e31837; border-radius: 5px; }
    div.stButton > button { background-color: #e31837; color: white; border: none; font-weight: bold; width: 100%; height: 50px;}
    div.stButton > button:hover { background-color: white; color: #e31837; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>ESPACE <span class='accent-red'>AGENCE / MARQUE</span></h1>", unsafe_allow_html=True)
st.write("Trouvez les meilleurs influenceurs en fonction de votre budget et de votre domaine.")
st.markdown("---")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def charger_donnees():
    # On essaie de lire la base de données remplie par les influenceurs
    if os.path.isfile("database_plateforme.csv"):
        df = pd.read_csv("database_plateforme.csv")
    # Sinon, on utilise le gros fichier par défaut pour avoir de la donnée
    elif os.path.isfile("influencer_collabs.csv"):
        df = pd.read_csv("influencer_collabs.csv")
    else:
        return None
    return df

df = charger_donnees()

if df is None or df.empty:
    st.error("⚠️ Aucune donnée disponible. Demandez aux influenceurs de remplir le formulaire d'abord, ou placez le fichier 'influencer_collabs.csv' dans le dossier.")
else:
    # --- FORMULAIRE DE RECHERCHE (INPUTS) ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎯 Vos Critères")
        budget_max = st.number_input("Budget Total (TND) *Obligatoire*", min_value=100.0, value=5000.0, step=500.0)
        
        # Liste des domaines disponibles dans la base
        domaines_dispos = df['product_category'].dropna().unique().tolist()
        domaine_choisi = st.selectbox("Domaine / Produit ciblé", domaines_dispos)
        
        lancer_recherche = st.button("LANCER LE MATCHMAKING IA")

    with col2:
        if lancer_recherche:
            with st.spinner("L'IA calcule le meilleur Retour sur Investissement (ROI)..."):
                
                # 1. Filtrer les influenceurs qui font ce domaine
                df_filtre = df[df['product_category'] == domaine_choisi].copy()
                
                if df_filtre.empty:
                    st.warning(f"Aucun influenceur trouvé pour le domaine : {domaine_choisi}")
                else:
                    # 2. Regrouper par influenceur pour avoir leur moyenne
                    stats = df_filtre.groupby('influencer_name').agg({
                        'price_paid': 'mean',
                        'engagement_rate': 'mean',
                        'followers_count': 'max',
                        'platform': 'first'
                    }).reset_index()

                    # 3. L'ALGORITHME : Calcul du score ROI (Engagement / Prix)
                    # Plus l'engagement est haut et le prix bas, meilleur est le score
                    stats['roi_score'] = (stats['engagement_rate'] * stats['followers_count']) / (stats['price_paid'] + 1)
                    
                    # Trier du meilleur au moins bon
                    stats = stats.sort_values(by='roi_score', ascending=False)

                    # 4. Sélection intelligente selon le budget
                    influenceurs_choisis = []
                    budget_depense = 0.0
                    reach_total = 0

                    for index, row in stats.iterrows():
                        if budget_depense + row['price_paid'] <= budget_max:
                            influenceurs_choisis.append(row)
                            budget_depense += row['price_paid']
                            reach_total += row['followers_count']

                    # --- AFFICHAGE DES RÉSULTATS ---
                    if not influenceurs_choisis:
                        st.error("Votre budget est trop faible pour les influenceurs de cette catégorie.")
                    else:
                        df_resultats = pd.DataFrame(influenceurs_choisis)
                        budget_restant = budget_max - budget_depense
                        
                        st.markdown("### 🏆 L'Équipe Recommandée")
                        
                        # Métriques Globales
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Budget Utilisé", f"{budget_depense:,.0f} TND", f"Reste: {budget_restant:,.0f} TND")
                        m2.metric("Influenceurs Retenus", len(df_resultats))
                        m3.metric("Audience Potentielle", f"{reach_total:,.0f} personnes")

                        # Graphique de comparaison
                        st.markdown("#### Comparaison des profils retenus")
                        fig = px.scatter(df_resultats, x='price_paid', y='engagement_rate', 
                                         size='followers_count', color='influencer_name',
                                         hover_name='influencer_name', text='influencer_name',
                                         labels={'price_paid': 'Prix Moyen (TND)', 'engagement_rate': 'Taux d\'Engagement (%)'},
                                         template='plotly_dark')
                        fig.update_traces(textposition='top center')
                        st.plotly_chart(fig, use_container_width=True)

                        # Tableau détaillé
                        st.markdown("#### Détail des coûts")
                        st.dataframe(df_resultats[['influencer_name', 'platform', 'engagement_rate', 'price_paid']].rename(
                            columns={'influencer_name': 'Nom', 'platform': 'Plateforme', 'engagement_rate': 'Engagement (%)', 'price_paid': 'Prix Estimé (TND)'}
                        ), use_container_width=True)
                        
                        st.success("💡 **Stratégie validée :** L'algorithme a maximisé votre portée en combinant ces profils sans dépasser votre budget.")