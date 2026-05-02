import streamlit as st
import pandas as pd

from database import init_db, load_collaborations, export_csv
from agent import research_influencer

init_db()

st.set_page_config(
    page_title="Influencer Research Bot",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Influencer Brand Collaboration Research Bot")

st.write("""
This chatbot searches the web, detects possible brand collaborations,
classifies evidence, saves results into a dataset, and scores each collaboration.
""")

with st.sidebar:
    st.header("Research Settings")

    influencer_name = st.text_input("Influencer name")
    username = st.text_input("Username", placeholder="@username")
    influencer_niche = st.text_input("Influencer niche", placeholder="beauty, fitness, food, lifestyle...")

    max_queries = st.slider("Number of search queries", 3, 20, 8)
    results_per_query = st.slider("Results per query", 2, 10, 4)

    start_button = st.button("Start research")

if start_button:
    if not influencer_name:
        st.error("Please enter the influencer name.")
    else:
        with st.spinner("Searching and analyzing... This may take a few minutes."):
            rows = research_influencer(
                influencer_name=influencer_name,
                username=username,
                influencer_niche=influencer_niche,
                max_queries=max_queries,
                results_per_query=results_per_query
            )

        st.success(f"Research finished. Found {len(rows)} possible collaboration rows.")

st.header("📊 Brand Collaborations Dataset")

df = load_collaborations()

if df.empty:
    st.info("No data yet. Start a research task from the sidebar.")
else:
    st.dataframe(df, use_container_width=True)

    confirmed_df = df[df["is_confirmed_collaboration"] == "yes"]
    maybe_df = df[df["is_confirmed_collaboration"] == "maybe"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Total rows", len(df))
    col2.metric("Confirmed collaborations", len(confirmed_df))
    col3.metric("Maybe collaborations", len(maybe_df))

    csv_path = export_csv()

    with open(csv_path, "rb") as file:
        st.download_button(
            label="Download CSV",
            data=file,
            file_name="brand_collaborations.csv",
            mime="text/csv"
        )