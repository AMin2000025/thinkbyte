import sqlite3
import pandas as pd
from config import DATABASE_PATH


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS brand_collaborations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        influencer_name TEXT,
        username TEXT,
        platform TEXT,
        brand_name TEXT,
        brand_industry TEXT,
        campaign_type TEXT,
        content_type TEXT,
        collaboration_date TEXT,
        evidence_url TEXT,
        evidence_type TEXT,
        confidence_score INTEGER,
        sentiment TEXT,
        is_confirmed_collaboration TEXT,
        collaboration_score INTEGER,
        source_title TEXT,
        source_snippet TEXT,
        notes TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_collaboration(row):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO brand_collaborations (
        influencer_name,
        username,
        platform,
        brand_name,
        brand_industry,
        campaign_type,
        content_type,
        collaboration_date,
        evidence_url,
        evidence_type,
        confidence_score,
        sentiment,
        is_confirmed_collaboration,
        collaboration_score,
        source_title,
        source_snippet,
        notes
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row.get("influencer_name", ""),
        row.get("username", ""),
        row.get("platform", ""),
        row.get("brand_name", ""),
        row.get("brand_industry", ""),
        row.get("campaign_type", ""),
        row.get("content_type", ""),
        row.get("collaboration_date", ""),
        row.get("evidence_url", ""),
        row.get("evidence_type", ""),
        row.get("confidence_score", 0),
        row.get("sentiment", ""),
        row.get("is_confirmed_collaboration", ""),
        row.get("collaboration_score", 0),
        row.get("source_title", ""),
        row.get("source_snippet", ""),
        row.get("notes", "")
    ))

    conn.commit()
    conn.close()


def load_collaborations():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM brand_collaborations", conn)
    conn.close()
    return df


def export_csv(path="brand_collaborations.csv"):
    df = load_collaborations()
    df.to_csv(path, index=False)
    return path