"""
1_analyze_single_comment.py
============================
Provides analyze_comment() — analyses one comment using the trained
comment_authenticity_model.pkl and returns a bot score + decision.

Usage
-----
    from analyze_single_comment import analyze_comment

    result = analyze_comment(
        comment_text   = "🔥🔥🔥🔥🔥",
        likes_on_comment = 0,
    )
    print(result)
    # {'bot_score': 0.33, 'authenticity_score': 0.67,
    #  'decision': 'uncertain', 'comment_text': '🔥🔥🔥🔥🔥'}

Or run directly:
    python 1_analyze_single_comment.py
"""

import re
import joblib
import pandas as pd

try:
    import emoji as emoji_lib
    _HAS_EMOJI = True
except ImportError:
    _HAS_EMOJI = False


# ── Load model once at import time ────────────────────────────────────────────
MODEL_PATH = "comment_authenticity_model.pkl"
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


# ── Feature engineering helpers ───────────────────────────────────────────────

def _count_emojis(text: str) -> int:
    if _HAS_EMOJI:
        return sum(1 for ch in text if ch in emoji_lib.EMOJI_DATA)
    # fallback: count Unicode emoji ranges
    return len(re.findall(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        r"\U00002700-\U000027BF\U0001FA00-\U0001FA6F"
        r"\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+",
        text
    ))


def _is_emoji_only(text: str, emoji_count: int) -> bool:
    """True when the comment contains only emojis and whitespace."""
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
    """True if any character repeats ≥ threshold times consecutively."""
    return bool(re.search(r"(.)\1{" + str(threshold - 1) + r",}", text))


def _build_feature_row(
    comment_text: str,
    likes_on_comment: int = 0,
    # account-level fields (optional — use realistic defaults when unknown)
    followers_count: int = 500,
    following_count: int = 400,
    posts_count: int = 20,
    has_link_in_bio: int = 0,
    has_promo_words_in_bio: int = 0,
    # engagement context fields (optional)
    duplicate_comment_count: int = 1,
    comments_by_same_user: int = 1,
    same_comment_on_multiple_posts: int = 0,
    is_giveaway_post: int = 0,
    post_likes_count: int = 1000,
    post_comments_count: int = 50,
) -> dict:
    """Build the feature dict the model expects from a single comment."""
    text = str(comment_text).strip() if comment_text and str(comment_text).strip() else ""
    emoji_count = _count_emojis(text)

    return {
        "text_for_model":               text,
        "followers_count":              followers_count,
        "following_count":              following_count,
        "posts_count":                  posts_count,
        "likes_on_comment":             likes_on_comment,
        "comment_length":               len(text),
        "emoji_count":                  emoji_count,
        "is_emoji_only_or_mostly":      int(_is_emoji_only(text, emoji_count)),
        "has_repeated_chars":           int(_has_repeated_chars(text)),
        "has_link_in_bio":              has_link_in_bio,
        "has_promo_words_in_bio":       has_promo_words_in_bio,
        "duplicate_comment_count":      duplicate_comment_count,
        "comments_by_same_user":        comments_by_same_user,
        "same_comment_on_multiple_posts": same_comment_on_multiple_posts,
        "is_giveaway_post":             is_giveaway_post,
        "post_likes_count":             post_likes_count,
        "post_comments_count":          post_comments_count,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_comment(
    comment_text: str,
    likes_on_comment: int = 0,
    **kwargs,
) -> dict:
    """
    Analyse a single comment and return a bot-likelihood assessment.

    Parameters
    ----------
    comment_text     : str  — the raw comment string
    likes_on_comment : int  — likes received on this comment (default 0)
    **kwargs         : any extra feature overrides accepted by
                       _build_feature_row() (e.g. followers_count,
                       duplicate_comment_count, etc.)

    Returns
    -------
    dict
        bot_score          float  0.0–1.0  (higher = more bot-like)
        authenticity_score float  0.0–1.0
        decision           str    "likely authentic" | "uncertain" | "likely bot-like"
        comment_text       str    the input comment (trimmed)
    """
    model = _get_model()
    row = _build_feature_row(comment_text, likes_on_comment, **kwargs)
    df = pd.DataFrame([row])

    bot_score = round(float(model.predict_proba(df)[0][1]), 4)
    authenticity_score = round(1 - bot_score, 4)

    if bot_score < 0.35:
        decision = "likely authentic"
    elif bot_score < 0.70:
        decision = "uncertain"
    else:
        decision = "likely bot-like"

    return {
        "bot_score":          bot_score,
        "authenticity_score": authenticity_score,
        "decision":           decision,
        "comment_text":       row["text_for_model"],
    }


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_comments = [
        ("🔥🔥🔥🔥🔥🔥🔥🔥",                          0),
        ("LSE",                                        0),
        ("",                                           0),
        ("Bravo @taaritdorra beauty and brains 😍",    1),
        ("Great video, learned a lot from this!",      10),
        ("Follow me! Check my bio link 🎁 giveaway",   0),
        ("واو توغوموري 100% 😍😍😂",                    7),
    ]

    print(f"\n{'Comment':<55} {'Bot':>6}  {'Auth':>6}  Decision")
    print("-" * 90)
    for text, likes in test_comments:
        r = analyze_comment(text, likes_on_comment=likes)
        preview = repr(text[:50])
        print(f"{preview:<55} {r['bot_score']:>6.3f}  {r['authenticity_score']:>6.3f}  {r['decision']}")
