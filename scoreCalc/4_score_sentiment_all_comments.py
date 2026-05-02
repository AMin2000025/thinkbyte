"""
4_score_sentiment_all_comments.py
===================================
Runs every comment through the sentiment engine from scoring.py
and prints an overall report: what % of comments are positive,
negative, or neutral.

Uses the same two-step logic as scoring.py:
  1. Tunisian dialect dictionary (instant, no internet needed)
  2. VADER + Google Translate for everything else

Usage
-----
    python 4_score_sentiment_all_comments.py
    python 4_score_sentiment_all_comments.py --input comments_20260501_230423.csv
    python 4_score_sentiment_all_comments.py --input comments_20260501_230423.csv --output sentiment_results.csv
"""

import argparse
import sys
import time

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator


# ── Tunisian dialect dictionaries (from scoring.py) ───────────────────────────

MOTS_POSITIFS_TN = [
    'tayara', '7lou', 'hlou', 'behi', 'yhabal', 'yhebel',
    'top', 'raw3a', 'mrigel', 'tahfoun', 'mhabel',
]
MOTS_NEGATIFS_TN = [
    'khayeb', '5ayeb', 'maset', 'zibla', 'kavi', 'arnaque',
    'nul', 'bhim', 'ka3ba la', 'madhroub', 'madhrouba', 'chleka',
]

analyseur_vader = SentimentIntensityAnalyzer()

# Fix emojis that VADER misreads in social media context
# 🔥 = hype/fire (positive), 😭 = crying-laughing in Arab/TN dialect (positive)
analyseur_vader.lexicon.update({
    "fire":   2.0,   # 🔥 → "fire" = hype in social media (was -1.4)
    "crying": 1.8,   # 😭 → "loudly crying" = laughing in TN/Arab dialect (was -2.1)
    "skull":  1.5,   # 💀 → "skull" = dead from laughing — positive slang
})


# ── Single-comment analyser ───────────────────────────────────────────────────

def analyze_sentiment(comment_text: str) -> dict:
    """
    Analyse one comment and return its sentiment.

    Returns
    -------
    dict
        sentiment   : "positive" | "negative" | "neutral"
        method      : "tunisian_dict" | "vader" | "error"
        translated  : str — English translation used by VADER (or "")
        score       : float — VADER compound score (or 0.0 for dict method)
    """
    text = str(comment_text).strip() if comment_text and str(comment_text).strip() else ""

    if not text:
        return {"sentiment": "neutral", "method": "empty", "translated": "", "score": 0.0}

    lower = text.lower()

    # ── Step 1: Tunisian dictionary ───────────────────────────────────────
    if any(mot in lower for mot in MOTS_POSITIFS_TN):
        return {"sentiment": "positive", "method": "tunisian_dict", "translated": "", "score": 1.0}

    if any(mot in lower for mot in MOTS_NEGATIFS_TN):
        return {"sentiment": "negative", "method": "tunisian_dict", "translated": "", "score": -1.0}

    # ── Step 2: Translate → VADER ─────────────────────────────────────────
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        score = analyseur_vader.polarity_scores(translated)['compound']

        if score >= 0.05:
            sentiment = "positive"
        elif score <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {"sentiment": sentiment, "method": "vader", "translated": translated, "score": round(score, 4)}

    except Exception:
        # No internet / translation failed — fall back to VADER on raw text
        try:
            score = analyseur_vader.polarity_scores(text)['compound']
            if score >= 0.05:
                sentiment = "positive"
            elif score <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            return {"sentiment": sentiment, "method": "vader_raw", "translated": "", "score": round(score, 4)}
        except Exception:
            return {"sentiment": "neutral", "method": "error", "translated": "", "score": 0.0}


# ── Batch scorer ──────────────────────────────────────────────────────────────

def score_sentiment_all(
    input_path: str = "comments.csv",
    output_path: str = "comments_sentiment.csv",
) -> None:

    print(f"\n{'='*60}")
    print("  Comment Sentiment Scorer")
    print(f"{'='*60}")
    print(f"  Loading : {input_path}\n")

    df = pd.read_csv(input_path, encoding="utf-8")
    if "comment_text" not in df.columns:
        print(f"Error: 'comment_text' column not found in {input_path}", file=sys.stderr)
        sys.exit(1)

    total = len(df)
    print(f"  {total} comments to analyse...\n")

    results = []
    for i, row in df.iterrows():
        result = analyze_sentiment(row["comment_text"])
        results.append(result)

        # Progress every 25 comments
        done = i + 1
        if done % 25 == 0 or done == total:
            pct = done / total * 100
            print(f"  [{done:>3}/{total}]  {pct:5.1f}% done...", end="\r")

        # Small delay to avoid Google Translate rate-limiting
        if result["method"] == "vader":
            time.sleep(0.15)

    print()  # newline after progress

    # Attach results to dataframe
    results_df = pd.DataFrame(results)
    df["sentiment"]   = results_df["sentiment"].values
    df["vader_score"] = results_df["score"].values
    df["method"]      = results_df["method"].values
    df["translated"]  = results_df["translated"].values

    # ── Summary ───────────────────────────────────────────────────────────
    positive = (df["sentiment"] == "positive").sum()
    negative = (df["sentiment"] == "negative").sum()
    neutral  = (df["sentiment"] == "neutral").sum()

    pct_pos = positive / total * 100
    pct_neg = negative / total * 100
    pct_neu = neutral  / total * 100

    print(f"\n{'─'*60}")
    print(f"  SENTIMENT SUMMARY")
    print(f"{'─'*60}")
    print(f"  Total comments    : {total}")
    print(f"")
    print(f"  😊 Positive       : {positive:>4}  ({pct_pos:5.1f}%)")
    print(f"  😐 Neutral        : {neutral:>4}  ({pct_neu:5.1f}%)")
    print(f"  😠 Negative       : {negative:>4}  ({pct_neg:5.1f}%)")
    print(f"{'─'*60}")

    # Overall verdict
    if pct_pos >= 60:
        verdict = "🟢 Overall: comments are MOSTLY POSITIVE"
    elif pct_neg >= 40:
        verdict = "🔴 Overall: comments are MOSTLY NEGATIVE"
    elif pct_pos > pct_neg:
        verdict = "🟡 Overall: comments lean POSITIVE but mixed"
    elif pct_neg > pct_pos:
        verdict = "🟠 Overall: comments lean NEGATIVE but mixed"
    else:
        verdict = "⚪ Overall: comments are MIXED / NEUTRAL"

    print(f"  {verdict}")
    print(f"{'─'*60}\n")

    # ── Sample negatives ──────────────────────────────────────────────────
    negatives = df[df["sentiment"] == "negative"][["comment_text", "vader_score"]].head(8)
    if not negatives.empty:
        print("  Sample negative comments:")
        print(f"  {'─'*55}")
        for _, r in negatives.iterrows():
            txt = str(r["comment_text"])[:55]
            print(f"  [{r['vader_score']:+.3f}]  {repr(txt)}")
        print()

    # ── Save output ───────────────────────────────────────────────────────
    keep = ["commenter", "comment_text", "sentiment", "vader_score", "method"]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  Per-comment results saved to: {output_path}")
    print(f"\n{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score all comments as positive / neutral / negative."
    )
    parser.add_argument(
        "--input",  "-i",
        default="comments_20260501_230423.csv",
        help="Input CSV path  (default: comments_20260501_230423.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="comments_sentiment.csv",
        help="Output CSV path (default: comments_sentiment.csv)",
    )
    args = parser.parse_args()

    try:
        score_sentiment_all(args.input, args.output)
    except FileNotFoundError:
        print(f"Error: file not found — '{args.input}'", file=sys.stderr)
        sys.exit(1)
