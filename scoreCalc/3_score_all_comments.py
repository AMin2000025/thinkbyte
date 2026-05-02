"""
3_score_all_comments.py
========================
Loads the comments CSV, runs every comment through the trained model,
and prints a final report showing what percentage of comments are
NOT bots (i.e. are authentic).

Also saves a detailed CSV with per-comment scores so you can drill down.

Usage
-----
    python 3_score_all_comments.py
    python 3_score_all_comments.py --input comments_20260501_230423.csv
"""

import argparse
import sys
import re

import joblib
import pandas as pd

try:
    import emoji as emoji_lib
    _HAS_EMOJI = True
except ImportError:
    _HAS_EMOJI = False


# ── Model loading ─────────────────────────────────────────────────────────────

MODEL_PATH = "comment_authenticity_model.pkl"


def _load_model():
    return joblib.load(MODEL_PATH)


# ── Feature helpers (same logic as file 1) ────────────────────────────────────

def _count_emojis(text: str) -> int:
    if _HAS_EMOJI:
        return sum(1 for ch in text if ch in emoji_lib.EMOJI_DATA)
    return len(re.findall(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        r"\U00002700-\U000027BF\U0001FA00-\U0001FA6F"
        r"\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+",
        text
    ))


def _is_emoji_only(text: str, emoji_count: int) -> bool:
    if _HAS_EMOJI:
        stripped = "".join(ch for ch in text if ch not in emoji_lib.EMOJI_DATA).strip()
    else:
        stripped = re.sub(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
            r"\U00002700-\U000027BF\U0001FA00-\U0001FA6F"
            r"\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+",
            "", text
        ).strip()
    return len(stripped) == 0 and emoji_count > 0


def _has_repeated_chars(text: str, threshold: int = 5) -> bool:
    return bool(re.search(r"(.)\1{" + str(threshold - 1) + r",}", text))


# ── Dataset-level feature engineering ────────────────────────────────────────
# Features like duplicate_comment_count and comments_by_same_user
# are derived from the full dataset, which makes them much more accurate
# than using a fixed default of 1 for every row.

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 17 model features from the raw comments dataframe.
    Columns expected: comment_text, likes_on_comment (optional),
                      commenter (optional), post_shortcode (optional).
    """
    # Normalise column presence
    if "likes_on_comment" not in df.columns:
        df["likes_on_comment"] = 0
    if "commenter" not in df.columns:
        df["commenter"] = "unknown"
    if "post_shortcode" not in df.columns:
        df["post_shortcode"] = "unknown"

    df["comment_text"] = df["comment_text"].fillna("")

    # ── Text-derived features ──────────────────────────────────────────────
    df["comment_length"] = df["comment_text"].str.len()

    df["emoji_count"] = df["comment_text"].apply(_count_emojis)

    df["is_emoji_only_or_mostly"] = df.apply(
        lambda r: int(_is_emoji_only(r["comment_text"], r["emoji_count"])), axis=1
    )

    df["has_repeated_chars"] = df["comment_text"].apply(
        lambda t: int(_has_repeated_chars(t))
    )

    # Combined text field the TF-IDF vectoriser was trained on
    df["text_for_model"] = df["comment_text"]

    # ── Dataset-derived engagement features ────────────────────────────────
    # How many times this exact comment text appears across ALL rows
    dup_counts = df["comment_text"].map(df["comment_text"].value_counts())
    df["duplicate_comment_count"] = dup_counts.fillna(1).astype(int)

    # How many comments each user posted in this dataset
    user_counts = df["commenter"].map(df["commenter"].value_counts())
    df["comments_by_same_user"] = user_counts.fillna(1).astype(int)

    # Did the same user post the same comment on multiple posts?
    if df["post_shortcode"].nunique() > 1:
        multi = (
            df.groupby(["commenter", "comment_text"])["post_shortcode"]
            .nunique()
            .reset_index(name="n_posts")
        )
        multi["same_comment_on_multiple_posts"] = (multi["n_posts"] > 1).astype(int)
        df = df.merge(multi[["commenter", "comment_text", "same_comment_on_multiple_posts"]],
                      on=["commenter", "comment_text"], how="left")
    else:
        df["same_comment_on_multiple_posts"] = 0

    # ── Account-level defaults (not in CSV — use realistic averages) ──────
    df["followers_count"]        = 500
    df["following_count"]        = 400
    df["posts_count"]            = 20
    df["has_link_in_bio"]        = 0
    df["has_promo_words_in_bio"] = 0
    df["is_giveaway_post"]       = 0
    df["post_likes_count"]       = 1000
    df["post_comments_count"]    = 50

    return df


# ── Scoring ───────────────────────────────────────────────────────────────────

FEATURES = [
    "text_for_model",
    "followers_count",
    "following_count",
    "posts_count",
    "likes_on_comment",
    "comment_length",
    "emoji_count",
    "is_emoji_only_or_mostly",
    "has_repeated_chars",
    "has_link_in_bio",
    "has_promo_words_in_bio",
    "duplicate_comment_count",
    "comments_by_same_user",
    "same_comment_on_multiple_posts",
    "is_giveaway_post",
    "post_likes_count",
    "post_comments_count",
]


def score_comments(
    input_path: str = "comments.csv",
    output_path: str = "comments_scored.csv",
) -> None:
    """
    Run every comment through the model and print an authenticity summary.

    Parameters
    ----------
    input_path  : str — path to the source CSV file
    output_path : str — where to save the per-comment scored CSV
    """
    print(f"\n{'='*60}")
    print("  Comment Authenticity Scorer")
    print(f"{'='*60}")
    print(f"  Loading model : {MODEL_PATH}")
    model = _load_model()
    print(f"  Loading data  : {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8")
    total = len(df)
    print(f"  Comments found: {total}\n")

    # Build features
    df = build_features(df)

    # Predict
    probs = model.predict_proba(df[FEATURES])
    df["bot_score"]          = probs[:, 1].round(4)
    df["authenticity_score"] = (1 - probs[:, 1]).round(4)

    df["decision"] = df["bot_score"].apply(
        lambda s: "likely authentic" if s < 0.35
        else "uncertain"             if s < 0.70
        else "likely bot-like"
    )

    # ── Summary stats ─────────────────────────────────────────────────────
    authentic = (df["decision"] == "likely authentic").sum()
    uncertain = (df["decision"] == "uncertain").sum()
    bot_like  = (df["decision"] == "likely bot-like").sum()

    pct_authentic = authentic / total * 100
    pct_uncertain = uncertain / total * 100
    pct_bot       = bot_like  / total * 100

    # "Not bot" = authentic + uncertain (below 0.70 threshold)
    not_bot     = authentic + uncertain
    pct_not_bot = not_bot / total * 100

    print(f"{'─'*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'─'*60}")
    print(f"  Total comments       : {total}")
    print(f"")
    print(f"  ✅ Likely authentic  : {authentic:>4}  ({pct_authentic:5.1f}%)")
    print(f"  ❓ Uncertain         : {uncertain:>4}  ({pct_uncertain:5.1f}%)")
    print(f"  🤖 Likely bot-like   : {bot_like:>4}  ({pct_bot:5.1f}%)")
    print(f"{'─'*60}")
    print(f"  📊 NOT BOT (auth + uncertain) : {not_bot} / {total}  →  {pct_not_bot:.1f}%")
    print(f"{'─'*60}\n")

    # ── Save detailed output ──────────────────────────────────────────────
    keep_cols = ["commenter", "comment_text", "likes_on_comment",
                 "bot_score", "authenticity_score", "decision"]
    # Only keep columns that actually exist
    keep_cols = [c for c in keep_cols if c in df.columns]

    df[keep_cols].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  Per-comment scores saved to: {output_path}\n")

    # ── Sample of bot-like comments ───────────────────────────────────────
    bots_sample = df[df["decision"] == "likely bot-like"][
        ["commenter", "comment_text", "bot_score"]
    ].head(10) if "commenter" in df.columns else df[df["decision"] == "likely bot-like"][
        ["comment_text", "bot_score"]
    ].head(10)

    if not bots_sample.empty:
        print("  Sample of bot-like comments:")
        print(f"  {'─'*55}")
        for _, row in bots_sample.iterrows():
            txt = str(row["comment_text"])[:45]
            usr = row.get("commenter", "")
            print(f"  [{row['bot_score']:.3f}] @{usr:<20} {repr(txt)}")
    else:
        print("  No comments classified as bot-like.")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score all comments and report the % that are not bots."
    )
    parser.add_argument(
        "--input",  "-i",
        default="comments_20260501_230423.csv",
        help="Path to the input CSV  (default: comments_20260501_230423.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="comments_scored.csv",
        help="Path for the scored output CSV (default: comments_scored.csv)",
    )
    args = parser.parse_args()

    try:
        score_comments(args.input, args.output)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
