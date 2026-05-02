"""
Instagram Profile Scraper using Apify
======================================
Scrapes: followers, likes, shares (video views), and comments
from a given Instagram influencer profile.

Requirements:
    pip install apify-client pandas

Usage:
    Set your APIFY_API_TOKEN in the config section below,
    then run: python instagram_scraper.py
"""

import json
import csv
import os
from datetime import datetime
from apify_client import ApifyClient

# ─────────────────────────────────────────────
#  CONFIGURATION — edit these values
# ─────────────────────────────────────────────
APIFY_API_TOKEN = ""   # ← paste your Apify token here
INSTAGRAM_USERNAME = input("donner le nom de l'instagrammeur : ")           # ← target influencer username (no @)
MAX_POSTS = 20                             # number of posts to scrape
MAX_COMMENTS_PER_POST = 50                # max comments to collect per post
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))           # folder where results are saved
# ─────────────────────────────────────────────


def scrape_profile(client: ApifyClient, username: str) -> dict:
    """Run the Instagram Profile Scraper actor and return profile data."""
    print(f"\n[1/3] Scraping profile info for @{username} ...")

    run_input = {
        "usernames": [username],
    }

    run = client.actor("apify/instagram-profile-scraper").call(run_input=run_input)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())

    if not items:
        raise ValueError(f"No profile data returned for @{username}. Check the username.")

    profile = items[0]
    return profile


def scrape_posts(client: ApifyClient, username: str, max_posts: int) -> list[dict]:
    """Run the Instagram Post Scraper actor and return a list of posts."""
    print(f"[2/3] Scraping up to {max_posts} posts for @{username} ...")

    run_input = {
        # The actor expects a profile URL, not a plain username
        "directUrls": [f"https://www.instagram.com/{username}/"],
        "resultsType": "posts",
        "resultsLimit": max_posts,
    }

    run = client.actor("apify/instagram-scraper").call(run_input=run_input)
    posts = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    print(f"      → {len(posts)} posts retrieved.")
    return posts


def scrape_comments(client: ApifyClient, post_shortcodes: list[str], max_per_post: int) -> dict[str, list]:
    """Run the Instagram Comment Scraper actor and return comments grouped by post."""
    print(f"[3/3] Scraping comments (up to {max_per_post} per post) ...")

    if not post_shortcodes:
        print("      → No post shortcodes available, skipping comments.")
        return {}

    run_input = {
        "directUrls": [f"https://www.instagram.com/p/{code}/" for code in post_shortcodes],
        "resultsLimit": max_per_post,
    }

    run = client.actor("apify/instagram-comment-scraper").call(run_input=run_input)
    raw_comments = list(client.dataset(run["defaultDatasetId"]).iterate_items())

    # Group by post shortcode
    grouped: dict[str, list] = {}
    for comment in raw_comments:
        code = comment.get("postShortCode") or comment.get("shortCode", "unknown")
        grouped.setdefault(code, []).append(comment)

    total = sum(len(v) for v in grouped.values())
    print(f"      → {total} comments retrieved across {len(grouped)} posts.")
    return grouped


def extract_post_metrics(post: dict) -> dict:
    """Pull the key metrics out of a raw post dict."""
    return {
        "shortcode": post.get("shortCode") or post.get("id", ""),
        "url": post.get("url", ""),
        "type": post.get("type", ""),
        "timestamp": post.get("timestamp", ""),
        "caption": (post.get("caption") or "")[:300],   # truncate long captions
        "likes": post.get("likesCount", 0),
        # Instagram calls video plays "videoViewCount"; images have 0 shares
        "shares_or_views": post.get("videoViewCount", 0),
        "comments_count": post.get("commentsCount", 0),
    }


def save_results(profile: dict, posts: list[dict], comments_by_post: dict[str, list], output_dir: str):
    """Save everything to disk as JSON + CSV."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 1. Profile summary ──────────────────────────────────────────────
    profile_summary = {
        "username": profile.get("username", ""),
        "full_name": profile.get("fullName", ""),
        "biography": profile.get("biography", ""),
        "followers": profile.get("followersCount", 0),
        "following": profile.get("followsCount", 0),
        "total_posts": profile.get("postsCount", 0),
        "is_verified": profile.get("verified", False),
        "profile_pic_url": profile.get("profilePicUrl", ""),
        "scraped_at": timestamp,
    }

    profile_path = os.path.join(output_dir, f"profile_{timestamp}.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile_summary, f, ensure_ascii=False, indent=2)
    print(f"\n   Profile saved → {profile_path}")

    # ── 2. Posts CSV ────────────────────────────────────────────────────
    post_rows = [extract_post_metrics(p) for p in posts]
    posts_csv_path = os.path.join(output_dir, f"posts_{timestamp}.csv")
    if post_rows:
        fieldnames = list(post_rows[0].keys())
        with open(posts_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(post_rows)
        print(f"   Posts saved   → {posts_csv_path}")

    # ── 3. Comments CSV ─────────────────────────────────────────────────
    comment_rows = []
    for shortcode, comments in comments_by_post.items():
        for c in comments:
            comment_rows.append({
                "post_shortcode": shortcode,
                "post_url": f"https://www.instagram.com/p/{shortcode}/",
                "commenter": c.get("ownerUsername", ""),
                "comment_text": c.get("text", ""),
                "likes_on_comment": c.get("likesCount", 0),
                "timestamp": c.get("timestamp", ""),
            })

    comments_csv_path = os.path.join(output_dir, f"comments.csv")
    if comment_rows:
        fieldnames = list(comment_rows[0].keys())
        with open(comments_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comment_rows)
        print(f"   Comments saved → {comments_csv_path}")

    # ── 4. Full dump (JSON) ─────────────────────────────────────────────
    full_dump = {
        "profile": profile_summary,
        "posts": post_rows,
        "comments": comment_rows,
    }
    dump_path = os.path.join(output_dir, f"full_dump.json")
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(full_dump, f, ensure_ascii=False, indent=2)
    print(f"   Full dump saved → {dump_path}")

    return profile_summary, post_rows, comment_rows


def print_summary(profile_summary: dict, post_rows: list[dict], comment_rows: list[dict]):
    """Print a human-readable summary to the console."""
    sep = "─" * 50
    print(f"\n{sep}")
    print("  SCRAPING SUMMARY")
    print(sep)
    print(f"  Profile   : @{profile_summary['username']} ({profile_summary['full_name']})")
    print(f"  Verified  : {'✓' if profile_summary['is_verified'] else '✗'}")
    print(f"  Followers : {profile_summary['followers']:,}")
    print(f"  Following : {profile_summary['following']:,}")
    print(f"  Posts     : {profile_summary['total_posts']:,}")
    print(sep)

    if post_rows:
        total_likes = sum(p["likes"] for p in post_rows)
        total_views = sum(p["shares_or_views"] for p in post_rows)
        total_comments = sum(p["comments_count"] for p in post_rows)
        print(f"  Posts scraped        : {len(post_rows)}")
        print(f"  Total likes          : {total_likes:,}")
        print(f"  Total video views    : {total_views:,}")
        print(f"  Total comments count : {total_comments:,}")
        print(f"  Comments collected   : {len(comment_rows)}")

    print(sep)







def main():
    
    client = ApifyClient(APIFY_API_TOKEN)

    # Step 1 — Profile
    profile = scrape_profile(client, INSTAGRAM_USERNAME)

    # Step 2 — Posts
    posts = scrape_posts(client, INSTAGRAM_USERNAME, MAX_POSTS)

    # Step 3 — Comments
    shortcodes = [
        p.get("shortCode") or p.get("id", "")
        for p in posts
        if p.get("shortCode") or p.get("id")
    ]
    comments_by_post = scrape_comments(client, shortcodes, MAX_COMMENTS_PER_POST)

    # Save & summarise
    print("\n[Saving results...]")
    profile_summary, post_rows, comment_rows = save_results(
        profile, posts, comments_by_post, OUTPUT_DIR
    )
    print_summary(profile_summary, post_rows, comment_rows)


if __name__ == "__main__":
    main()
