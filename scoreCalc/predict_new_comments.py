import pandas as pd
import joblib

model = joblib.load("comment_authenticity_model.pkl")

df = pd.read_csv("your_new_comments.csv")

df["text_for_model"] = (
    df["comment_text"].fillna("") + " " +
    df["post_caption"].fillna("") + " " +
    df["account_description"].fillna("")
)

features = [
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
    "post_comments_count"
]

df["bot_score"] = model.predict_proba(df[features])[:, 1]
df["comment_authenticity_score"] = 1 - df["bot_score"]

df["decision"] = df["bot_score"].apply(
    lambda score: "likely authentic" if score < 0.35
    else "uncertain" if score < 0.70
    else "likely bot-like"
)

df.to_csv("new_comments_with_scores.csv", index=False, encoding="utf-8-sig")

print("Done. Results saved to new_comments_with_scores.csv")