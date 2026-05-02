"""
engagement_analysis.py
=======================
Calculates:
  1. Average likes per post
  2. Engagement rate  = avg_likes / followers  (as a %)
  3. Real followers % — estimated from industry benchmarks:
       engagement rate >= 6%   → very high  → ~90–100% real
       engagement rate 3–6%    → high       → ~70–89%  real
       engagement rate 1–3%    → average    → ~50–69%  real
       engagement rate 0.5–1%  → low        → ~30–49%  real
       engagement rate < 0.5%  → very low   → ~10–29%  real

Usage
-----
    python engagement_analysis.py
    python engagement_analysis.py --input full_dump_20260501_230423.json
"""

import json
import argparse
import sys


# ══════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

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
        description="Calculate engagement rate and estimate real followers %."
    )
    parser.add_argument(
        "--input", "-i",
        default="full_dump_20260501_230423.json",
        help="Path to the full_dump JSON file",
    )
    args = parser.parse_args()

    try:
        results = analyze_engagement(args.input)
        display_results(results)
    except FileNotFoundError:
        print(f"\nError: file not found — '{args.input}'", file=sys.stderr)
        sys.exit(1)
    except (ValueError, KeyError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
