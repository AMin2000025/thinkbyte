def score_sentiment(sentiment):
    sentiment = str(sentiment).lower()

    if sentiment == "positive":
        return 20
    if sentiment == "neutral":
        return 12
    if sentiment == "mixed":
        return 8
    if sentiment == "negative":
        return 2
    return 5


def score_evidence(confidence_score):
    confidence_score = int(confidence_score or 0)

    if confidence_score >= 90:
        return 25
    if confidence_score >= 75:
        return 20
    if confidence_score >= 60:
        return 14
    if confidence_score >= 40:
        return 8
    return 2


def score_brand_fit(brand_industry, influencer_niche=""):
    if not influencer_niche:
        return 12

    brand = str(brand_industry).lower()
    niche = str(influencer_niche).lower()

    if brand and brand in niche:
        return 20

    related_groups = {
        "beauty": ["beauty", "skincare", "makeup", "fashion", "lifestyle"],
        "fashion": ["fashion", "beauty", "lifestyle"],
        "fitness": ["fitness", "health", "sports", "nutrition"],
        "food": ["food", "restaurant", "beverage", "lifestyle"],
        "tech": ["technology", "gaming", "electronics"],
        "travel": ["travel", "hotel", "tourism", "lifestyle"]
    }

    for group, keywords in related_groups.items():
        if any(word in brand for word in keywords) and any(word in niche for word in keywords):
            return 18

    return 10


def score_collaboration(row, influencer_niche=""):
    evidence_score = score_evidence(row.get("confidence_score", 0))
    sentiment_score = score_sentiment(row.get("sentiment", "unknown"))
    brand_fit_score = score_brand_fit(row.get("brand_industry", ""), influencer_niche)

    # Default simple scores. Later you can replace these with real metrics.
    engagement_score = 15
    content_quality_score = 8

    total = (
        evidence_score
        + engagement_score
        + sentiment_score
        + brand_fit_score
        + content_quality_score
    )

    return min(total, 100)