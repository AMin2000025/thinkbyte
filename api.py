from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import research_influencer
from database import init_db, load_collaborations

app = FastAPI(title="Influencer Research Bot API")

# --- STEP 1: ALLOW THE BROWSER TO ACCESS THE API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your HTML file to connect regardless of how it's opened
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    influencer_name: str = Field(..., min_length=1)
    username: str = ""
    influencer_niche: str = ""
    max_queries: int = Field(8, ge=1, le=25)
    results_per_query: int = Field(4, ge=1, le=10)

def _dataframe_to_records(df):
    if df.empty:
        return []
    # Replace NaN values with None so JSON doesn't break
    cleaned = df.where(df.notnull(), None)
    return cleaned.to_dict(orient="records")

def _build_summary(df):
    if df.empty:
        return {"total": 0, "confirmed": 0, "maybe": 0, "no": 0}

    confirmed = (df["is_confirmed_collaboration"] == "yes").sum()
    maybe = (df["is_confirmed_collaboration"] == "maybe").sum()
    no = (df["is_confirmed_collaboration"] == "no").sum()

    return {
        "total": int(len(df)),
        "confirmed": int(confirmed),
        "maybe": int(maybe),
        "no": int(no),
    }

@app.on_event("startup")
def _startup():
    init_db()

@app.get("/api/health")
def health():
    return {"status": "ok"}

# --- GET ALL EXISTING DATA ---
@app.get("/api/collaborations")
def get_collaborations():
    df = load_collaborations()
    return {
        "rows": _dataframe_to_records(df),
        "summary": _build_summary(df),
    }

# --- TRIGGER NEW RESEARCH ---
@app.post("/api/research")
def run_research(payload: ResearchRequest):
    # Runs the AI agent logic
    rows = research_influencer(
        influencer_name=payload.influencer_name,
        username=payload.username,
        influencer_niche=payload.influencer_niche,
        max_queries=payload.max_queries,
        results_per_query=payload.results_per_query,
    )

    # Reload the full database to send back the updated list
    df = load_collaborations()
    return {
        "saved_rows": rows,
        "rows": _dataframe_to_records(df),
        "summary": _build_summary(df),
    }