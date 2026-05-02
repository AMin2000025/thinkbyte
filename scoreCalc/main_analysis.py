"""
main_analysis.py
=================
Runs both analyses on your comments CSV and displays a combined
dashboard showing:
  - % of comments that are NOT bots
  - % of comments that are POSITIVE
  - A breakdown of each category

Usage
-----
    python main_analysis.py
    python main_analysis.py --input comments_20260501_230423.csv
"""

import argparse
import sys
import re
import time
import json
import argparse

import joblib
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

try:
    import emoji as emoji_lib
    _HAS_EMOJI = True
except ImportError:
    _HAS_EMOJI = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "comment_authenticity_model.pkl"

MOTS_POSITIFS_TN = [
    'tayara', '7lou', 'hlou', 'behi', 'yhabal', 'yhebel',
    'top', 'raw3a', 'mrigel', 'tahfoun', 'mhabel',
]
MOTS_NEGATIFS_TN = [
    'khayeb', '5ayeb', 'maset', 'zibla', 'kavi', 'arnaque',
    'nul', 'bhim', 'ka3ba la', 'madhroub', 'madhrouba', 'chleka',
]

analyseur_vader = SentimentIntensityAnalyzer()
# Fix emojis VADER misreads in Arab/TN social media context
# 🔥 → "fire" = -1.4 in VADER, but means hype/excitement here
# 😭 → "crying" = -2.1 in VADER, but means laughing in TN/Arab dialect
analyseur_vader.lexicon.update({
    "fire":   2.0,
    "crying": 1.8,
    "skull":  1.5,
})


# ══════════════════════════════════════════════════════════════════════════════
#  BOT DETECTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _count_emojis(text):
    if _HAS_EMOJI:
        return sum(1 for ch in text if ch in emoji_lib.EMOJI_DATA)
    return len(re.findall(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        r"\U00002700-\U000027BF\U0001FA00-\U0001FA6F"
        r"\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+", text))

def _is_emoji_only(text, emoji_count):
    if _HAS_EMOJI:
        stripped = "".join(ch for ch in text if ch not in emoji_lib.EMOJI_DATA).strip()
    else:
        stripped = re.sub(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
            r"\U00002700-\U000027BF\U0001FA00-\U0001FA6F"
            r"\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+", "", text).strip()
    return len(stripped) == 0 and emoji_count > 0

def _has_repeated_chars(text, threshold=5):
    return bool(re.search(r"(.)\1{" + str(threshold - 1) + r",}", text))

def build_bot_features(df):
    if "likes_on_comment" not in df.columns:
        df["likes_on_comment"] = 0
    if "commenter" not in df.columns:
        df["commenter"] = "unknown"
    if "post_shortcode" not in df.columns:
        df["post_shortcode"] = "unknown"

    df["comment_text"] = df["comment_text"].fillna("")
    df["comment_length"] = df["comment_text"].str.len()
    df["emoji_count"] = df["comment_text"].apply(_count_emojis)
    df["is_emoji_only_or_mostly"] = df.apply(
        lambda r: int(_is_emoji_only(r["comment_text"], r["emoji_count"])), axis=1)
    df["has_repeated_chars"] = df["comment_text"].apply(
        lambda t: int(_has_repeated_chars(t)))
    df["text_for_model"] = df["comment_text"]

    dup_counts = df["comment_text"].map(df["comment_text"].value_counts())
    df["duplicate_comment_count"] = dup_counts.fillna(1).astype(int)
    user_counts = df["commenter"].map(df["commenter"].value_counts())
    df["comments_by_same_user"] = user_counts.fillna(1).astype(int)

    if df["post_shortcode"].nunique() > 1:
        multi = (df.groupby(["commenter", "comment_text"])["post_shortcode"]
                 .nunique().reset_index(name="n_posts"))
        multi["same_comment_on_multiple_posts"] = (multi["n_posts"] > 1).astype(int)
        df = df.merge(multi[["commenter", "comment_text", "same_comment_on_multiple_posts"]],
                      on=["commenter", "comment_text"], how="left")
    else:
        df["same_comment_on_multiple_posts"] = 0

    df["followers_count"]        = 500
    df["following_count"]        = 400
    df["posts_count"]            = 20
    df["has_link_in_bio"]        = 0
    df["has_promo_words_in_bio"] = 0
    df["is_giveaway_post"]       = 0
    df["post_likes_count"]       = 1000
    df["post_comments_count"]    = 50
    return df

BOT_FEATURES = [
    "text_for_model", "followers_count", "following_count", "posts_count",
    "likes_on_comment", "comment_length", "emoji_count", "is_emoji_only_or_mostly",
    "has_repeated_chars", "has_link_in_bio", "has_promo_words_in_bio",
    "duplicate_comment_count", "comments_by_same_user",
    "same_comment_on_multiple_posts", "is_giveaway_post",
    "post_likes_count", "post_comments_count",
]


# ══════════════════════════════════════════════════════════════════════════════
#  SENTIMENT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_sentiment(comment_text):
    text = str(comment_text).strip() if comment_text and str(comment_text).strip() else ""
    if not text:
        return "neutral"

    lower = text.lower()
    if any(mot in lower for mot in MOTS_POSITIFS_TN):
        return "positive"
    if any(mot in lower for mot in MOTS_NEGATIFS_TN):
        return "negative"

    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        score = analyseur_vader.polarity_scores(translated)['compound']
        time.sleep(0.15)
    except Exception:
        try:
            score = analyseur_vader.polarity_scores(text)['compound']
        except Exception:
            return "neutral"

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def progress_bar(value, total, width=30):
    filled = int(width * value / total) if total > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]  {value/total*100:5.1f}%  ({value}/{total})"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(input_path="comments_20260501_230423.csv", output_path="main_results.csv"):

    print(f"\n{'═'*62}")
    print("   INSTAGRAM COMMENT ANALYSER")
    print(f"{'═'*62}")
    print(f"   File : {input_path}\n")

    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv(input_path, encoding="utf-8")
    total = len(df)
    print(f"   {total} comments loaded.\n")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 1 — BOT DETECTION
    # ══════════════════════════════════════════════════════════════════════
    print(f"{'─'*62}")
    print("   STEP 1/2 — Bot Detection")
    print(f"{'─'*62}")

    model = joblib.load(MODEL_PATH)
    df = build_bot_features(df)
    probs = model.predict_proba(df[BOT_FEATURES])
    df["bot_score"] = probs[:, 1].round(4)
    df["bot_decision"] = df["bot_score"].apply(
        lambda s: "likely authentic" if s < 0.35
        else "uncertain"             if s < 0.70
        else "likely bot-like"
    )

    bot_authentic = (df["bot_decision"] == "likely authentic").sum()
    bot_uncertain = (df["bot_decision"] == "uncertain").sum()
    bot_flagged   = (df["bot_decision"] == "likely bot-like").sum()
    not_bot       = bot_authentic + bot_uncertain

    print(f"   ✅ Likely authentic  : {bot_authentic:>4}")
    print(f"   ❓ Uncertain         : {bot_uncertain:>4}")
    print(f"   🤖 Likely bot-like   : {bot_flagged:>4}")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2 — SENTIMENT ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print("   STEP 2/2 — Sentiment Analysis")
    print(f"{'─'*62}")
    print("   (translating & scoring comments...)\n")

    sentiments = []
    for i, row in df.iterrows():
        s = analyze_sentiment(row["comment_text"])
        sentiments.append(s)
        done = i + 1
        if done % 25 == 0 or done == total:
            print(f"   [{done:>3}/{total}]  {done/total*100:5.1f}%...", end="\r")

    print()
    df["sentiment"] = sentiments

    sent_positive = (df["sentiment"] == "positive").sum()
    sent_neutral  = (df["sentiment"] == "neutral").sum()
    sent_negative = (df["sentiment"] == "negative").sum()

    print(f"   😊 Positive  : {sent_positive:>4}")
    print(f"   😐 Neutral   : {sent_neutral:>4}")
    print(f"   😠 Negative  : {sent_negative:>4}")

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL DASHBOARD
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*62}")
    print("   FINAL RESULTS DASHBOARD")
    print(f"{'═'*62}\n")

    print(f"   📊 Total comments analysed : {total}\n")

    print(f"   🤖 NOT BOT SCORE")
    print(f"   {progress_bar(not_bot, total)}\n")

    print(f"   😊 POSITIVE SCORE")
    print(f"   {progress_bar(sent_positive, total)}\n")

    print(f"{'─'*62}")

    # Verdict
    not_bot_pct  = not_bot      / total * 100
    positive_pct = sent_positive / total * 100

    if not_bot_pct >= 80:
        bot_verdict = "✅ Audience looks REAL"
    elif not_bot_pct >= 50:
        bot_verdict = "⚠️  Audience is MIXED"
    else:
        bot_verdict = "🚨 HIGH BOT ACTIVITY detected"

    if positive_pct >= 60:
        sent_verdict = "🟢 Audience reaction is VERY POSITIVE"
    elif positive_pct >= 40:
        sent_verdict = "🟡 Audience reaction is MOSTLY POSITIVE"
    elif positive_pct >= 20:
        sent_verdict = "🟠 Audience reaction is MIXED"
    else:
        sent_verdict = "🔴 Audience reaction is NEGATIVE"

    print(f"\n   {bot_verdict}")
    print(f"   {sent_verdict}")
    print(f"\n{'═'*62}\n")

    # ── Save results ───────────────────────────────────────────────────────
    keep = ["commenter", "comment_text", "likes_on_comment",
            "bot_score", "bot_decision", "sentiment"]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"   Full results saved to: {output_path}\n")

def analyze_engagement(json_path: str = "full_dump_20260501_230423.json") -> dict:
    """
    Load a full_dump JSON file and calculate engagement & real-follower metrics.

    Parameters
    ----------
    json_path : str — path to the full_dump JSON file

    Returns
    -------
    dict with keys:
        username            str
        followers           int
        total_posts_scraped int
        avg_likes           float   average likes across all scraped posts
        engagement_rate_pct float   (avg_likes / followers) * 100
        real_followers_pct  float   estimated % of real/genuine followers
        real_followers_est  int     estimated number of real followers
        engagement_level    str     "Very High" | "High" | "Average" | "Low" | "Very Low"
        verdict             str     plain-language summary
    """

    # ── Load JSON ──────────────────────────────────────────────────────────
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profile = data.get("profile", {})
    posts   = data.get("posts",   [])

    username  = profile.get("username",  "unknown")
    followers = profile.get("followers", 0)

    if not posts:
        raise ValueError("No posts found in the JSON file.")
    if followers == 0:
        raise ValueError("Follower count is 0 — cannot calculate engagement rate.")

    # ── Average likes ──────────────────────────────────────────────────────
    total_likes = sum(p.get("likes", 0) for p in posts)
    avg_likes   = total_likes / len(posts)

    # ── Engagement rate ────────────────────────────────────────────────────
    engagement_rate_pct = (avg_likes / followers) * 100

    # ── Estimate real followers % from engagement benchmarks ──────────────
    # Industry standard benchmarks for Instagram (2024–2026):
    #   > 6%    → exceptional  (micro/niche creators, very loyal audience)
    #   3–6%    → strong       (engaged community, mostly real)
    #   1–3%    → average      (typical for mid-size creators)
    #   0.5–1%  → below avg    (large accounts, some inflated following)
    #   < 0.5%  → poor         (likely bought followers / ghost followers)
    if engagement_rate_pct >= 6.0:
        engagement_level   = "Very High"
        real_followers_pct = 95.0
    elif engagement_rate_pct >= 3.0:
        engagement_level   = "High"
        # Linear scale: 3% → 75%, 6% → 95%
        real_followers_pct = 75.0 + (engagement_rate_pct - 3.0) / 3.0 * 20.0
    elif engagement_rate_pct >= 1.0:
        engagement_level   = "Average"
        # Linear scale: 1% → 55%, 3% → 75%
        real_followers_pct = 55.0 + (engagement_rate_pct - 1.0) / 2.0 * 20.0
    elif engagement_rate_pct >= 0.5:
        engagement_level   = "Low"
        # Linear scale: 0.5% → 35%, 1% → 55%
        real_followers_pct = 35.0 + (engagement_rate_pct - 0.5) / 0.5 * 20.0
    else:
        engagement_level   = "Very Low"
        # Linear scale: 0% → 10%, 0.5% → 35%
        real_followers_pct = max(10.0, engagement_rate_pct / 0.5 * 25.0)

    real_followers_pct = round(min(real_followers_pct, 99.0), 1)
    real_followers_est = int(followers * real_followers_pct / 100)

    # ── Verdict ────────────────────────────────────────────────────────────
    if engagement_level == "Very High":
        verdict = "Excellent engagement — audience is highly loyal and very likely real."
    elif engagement_level == "High":
        verdict = "Strong engagement — audience is mostly real with genuine interest."
    elif engagement_level == "Average":
        verdict = "Typical engagement for this follower range — some ghost followers likely."
    elif engagement_level == "Low":
        verdict = "Below-average engagement — a notable portion of followers may be inactive or fake."
    else:
        verdict = "Very low engagement — high probability of purchased or ghost followers."

    return {
        "username":            username,
        "followers":           followers,
        "total_posts_scraped": len(posts),
        "avg_likes":           round(avg_likes, 1),
        "engagement_rate_pct": round(engagement_rate_pct, 3),
        "real_followers_pct":  real_followers_pct,
        "real_followers_est":  real_followers_est,
        "engagement_level":    engagement_level,
        "verdict":             verdict,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def progress_bar(value, total, width=30, color_char="█"):
    filled = int(width * value / total) if total > 0 else 0
    bar = color_char * filled + "░" * (width - filled)
    return f"[{bar}]  {value/total*100:5.1f}%"


def display_results(r: dict) -> None:
    W = 62
    print(f"\n{'═'*W}")
    print(f"   ENGAGEMENT & AUDIENCE QUALITY REPORT")
    print(f"{'═'*W}")
    print(f"   👤 @{r['username']}")
    print(f"   👥 Followers        : {r['followers']:,}")
    print(f"   📸 Posts analysed   : {r['total_posts_scraped']}")
    print(f"{'─'*W}")

    print(f"\n   ❤️  AVG LIKES PER POST   :  {r['avg_likes']:,.1f} likes")
    print()

    # Engagement rate bar
    # Cap visual at 10% for display purposes
    eng_display = min(r['engagement_rate_pct'], 10.0)
    eng_bar = progress_bar(eng_display, 10.0)
    print(f"   📈 ENGAGEMENT RATE")
    print(f"   {eng_bar}  ({r['engagement_rate_pct']:.3f}%)")
    print(f"   Level : {r['engagement_level']}")
    print()

    # Real followers bar
    real_bar = progress_bar(r['real_followers_pct'], 100.0)
    print(f"   ✅ ESTIMATED REAL FOLLOWERS")
    print(f"   {real_bar}")
    print(f"   ~{r['real_followers_est']:,} out of {r['followers']:,} followers are genuine")
    print()

    print(f"{'─'*W}")
    print(f"   💬 {r['verdict']}")
    print(f"{'═'*W}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run bot detection + sentiment analysis on all comments."
    )
    parser.add_argument("--input",  "-i", default="comments_20260501_230423.csv")
    parser.add_argument("--output", "-o", default="main_results.csv")
    args = parser.parse_args()

    try:
        run_analysis(args.input, args.output)
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
