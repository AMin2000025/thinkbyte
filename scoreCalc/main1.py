"""
main_analysis.py
=================
Runs three analyses on your Instagram data and displays a combined
dashboard showing:
  - % of comments that are POSITIVE          → x
  - % of comments that are NOT bots          → y
  - % of followers that are REAL             → z

Usage
-----
    python main_analysis.py
    python main_analysis.py --csv comments_20260501_230423.csv --json full_dump_20260501_230423.json
"""

import argparse
import sys
import re
import time
import json

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
#  REAL FOLLOWERS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_engagement(json_path: str) -> dict:
    """Calculate engagement rate and estimate real followers % from JSON dump."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profile   = data.get("profile", {})
    posts     = data.get("posts",   [])
    username  = profile.get("username",  "unknown")
    followers = profile.get("followers", 0)

    if not posts or followers == 0:
        return {"username": username, "followers": followers,
                "posts_analysed": 0, "avg_likes": 0,
                "engagement_rate_pct": 0, "real_followers_pct": 50.0,
                "engagement_level": "Unknown"}

    avg_likes           = sum(p.get("likes", 0) for p in posts) / len(posts)
    engagement_rate_pct = (avg_likes / followers) * 100

    if engagement_rate_pct >= 6.0:
        level = "Very High";  real_pct = 95.0
    elif engagement_rate_pct >= 3.0:
        level = "High";       real_pct = 75.0 + (engagement_rate_pct - 3.0) / 3.0 * 20.0
    elif engagement_rate_pct >= 1.0:
        level = "Average";    real_pct = 55.0 + (engagement_rate_pct - 1.0) / 2.0 * 20.0
    elif engagement_rate_pct >= 0.5:
        level = "Low";        real_pct = 35.0 + (engagement_rate_pct - 0.5) / 0.5 * 20.0
    else:
        level = "Very Low";   real_pct = max(10.0, engagement_rate_pct / 0.5 * 25.0)

    return {
        "username":            username,
        "followers":           followers,
        "posts_analysed":      len(posts),
        "avg_likes":           round(avg_likes, 1),
        "engagement_rate_pct": round(engagement_rate_pct, 3),
        "real_followers_pct":  round(min(real_pct, 99.0), 1),
        "engagement_level":    level,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPER
# ══════════════════════════════════════════════════════════════════════════════

def progress_bar(pct, width=30):
    filled = int(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]  {pct:5.1f}%"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(
    csv_path    = "comments_20260501_230423.csv",
    json_path   = "full_dump_20260501_230423.json",
    output_path = "main_results.csv",
):
    print(f"\n{'═'*62}")
    print("   INSTAGRAM FULL ANALYSIS")
    print(f"{'═'*62}")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 1 — BOT DETECTION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print("   STEP 1/3 — Bot Detection")
    print(f"{'─'*62}")

    df    = pd.read_csv(csv_path, encoding="utf-8")
    total = len(df)
    print(f"   {total} comments loaded.\n")

    model = joblib.load(MODEL_PATH)
    df    = build_bot_features(df)
    probs = model.predict_proba(df[BOT_FEATURES])
    df["bot_score"]    = probs[:, 1].round(4)
    df["bot_decision"] = df["bot_score"].apply(
        lambda s: "likely authentic" if s < 0.35
        else "uncertain"             if s < 0.70
        else "likely bot-like"
    )

    bot_authentic = (df["bot_decision"] == "likely authentic").sum()
    bot_uncertain = (df["bot_decision"] == "uncertain").sum()
    bot_flagged   = (df["bot_decision"] == "likely bot-like").sum()
    not_bot       = bot_authentic + bot_uncertain

    print(f"   ✅ Likely authentic : {bot_authentic:>4}")
    print(f"   ❓ Uncertain        : {bot_uncertain:>4}")
    print(f"   🤖 Likely bot-like  : {bot_flagged:>4}")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2 — SENTIMENT ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print("   STEP 2/3 — Sentiment Analysis")
    print(f"{'─'*62}")
    print("   (translating & scoring comments...)\n")

    sentiments = []
    for i, row in df.iterrows():
        sentiments.append(analyze_sentiment(row["comment_text"]))
        done = i + 1
        if done % 25 == 0 or done == total:
            print(f"   [{done:>3}/{total}]  {done/total*100:5.1f}%...", end="\r")

    print()
    df["sentiment"] = sentiments

    sent_positive = (df["sentiment"] == "positive").sum()
    sent_neutral  = (df["sentiment"] == "neutral").sum()
    sent_negative = (df["sentiment"] == "negative").sum()

    print(f"   😊 Positive : {sent_positive:>4}")
    print(f"   😐 Neutral  : {sent_neutral:>4}")
    print(f"   😠 Negative : {sent_negative:>4}")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 3 — REAL FOLLOWERS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print("   STEP 3/3 — Real Followers Estimation")
    print(f"{'─'*62}")

    eng = analyze_engagement(json_path)

    print(f"   👤 @{eng['username']}")
    print(f"   👥 Followers       : {eng['followers']:,}")
    print(f"   📸 Posts analysed  : {eng['posts_analysed']}")
    print(f"   ❤️  Avg likes/post  : {eng['avg_likes']:,.1f}")
    print(f"   📈 Engagement rate : {eng['engagement_rate_pct']:.3f}%  ({eng['engagement_level']})")

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL DASHBOARD
    # ══════════════════════════════════════════════════════════════════════
    not_bot_pct      = round(not_bot       / total * 100, 1)
    positive_pct     = round(sent_positive / total * 100, 1)
    real_followers_z = eng["real_followers_pct"]

    print(f"\n{'═'*62}")
    print("   FINAL RESULTS DASHBOARD")
    print(f"{'═'*62}\n")

    print(f"   😊 POSITIVE COMMENTS")
    print(f"   {progress_bar(positive_pct)}\n")

    print(f"   🤖 NON-BOT COMMENTS")
    print(f"   {progress_bar(not_bot_pct)}\n")

    print(f"   ✅ REAL FOLLOWERS")
    print(f"   {progress_bar(real_followers_z)}")
    print(f"   (~{int(eng['followers'] * real_followers_z / 100):,} out of {eng['followers']:,})\n")

    print(f"{'─'*62}")

    # Verdicts
    if not_bot_pct >= 80:
        bot_v = "✅ Comments look REAL"
    elif not_bot_pct >= 50:
        bot_v = "⚠️  Comments are MIXED"
    else:
        bot_v = "🚨 HIGH BOT ACTIVITY in comments"

    if positive_pct >= 60:
        sent_v = "🟢 Audience reaction is VERY POSITIVE"
    elif positive_pct >= 40:
        sent_v = "🟡 Audience reaction is MOSTLY POSITIVE"
    elif positive_pct >= 20:
        sent_v = "🟠 Audience reaction is MIXED"
    else:
        sent_v = "🔴 Audience reaction is NEGATIVE"

    if real_followers_z >= 80:
        follow_v = "✅ Follower base looks GENUINE"
    elif real_followers_z >= 50:
        follow_v = "⚠️  Follower base is PARTIALLY genuine"
    else:
        follow_v = "🚨 HIGH number of FAKE followers detected"

    print(f"\n   {sent_v}")
    print(f"   {bot_v}")
    print(f"   {follow_v}")
    print(f"\n{'═'*62}\n")

    # ── Save results ───────────────────────────────────────────────────────
    keep = ["commenter", "comment_text", "likes_on_comment",
            "bot_score", "bot_decision", "sentiment"]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"   Full results saved to: {output_path}\n")

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL SCORE VARIABLES
    #  x = % positive comments   (int)
    #  y = % non-bot comments    (int)
    #  z = % real followers      (int)
    # ══════════════════════════════════════════════════════════════════════
    x = int(positive_pct)
    y = int(not_bot_pct)
    z = int(real_followers_z)

    print(f"{'─'*62}")
    print(f"   x (positive comments %) = {x}")
    print(f"   y (non-bot comments  %) = {y}")
    print(f"   z (real followers    %) = {z}")
    print(f"{'─'*62}\n")

    return x, y, z


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run bot detection + sentiment + real followers analysis."
    )
    parser.add_argument("--csv",    "-c", default="comments.csv",
                        help="Path to the comments CSV file")
    parser.add_argument("--json",   "-j", default="full_dump.json",
                        help="Path to the full_dump JSON file")
    parser.add_argument("--output", "-o", default="main_results.csv",
                        help="Path for the scored output CSV")
    args = parser.parse_args()

    try:
        x, y, z = run_analysis(args.csv, args.json, args.output)
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    finalScore=(x+y+z)/3
    print("*********************************")
    print ("final score :"+str(finalScore)+"*")
    print("********************************")