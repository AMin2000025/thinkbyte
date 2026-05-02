from ollama_client import ask_ollama_json


COLLAB_SCHEMA = {
    "type": "object",
    "properties": {
        "is_collaboration": {
            "type": "string",
            "enum": ["yes", "no", "maybe"]
        },
        "brand_name": {"type": "string"},
        "brand_industry": {"type": "string"},
        "platform": {"type": "string"},
        "campaign_type": {"type": "string"},
        "content_type": {"type": "string"},
        "collaboration_date": {"type": "string"},
        "evidence_type": {"type": "string"},
        "confidence_score": {"type": "integer"},
        "sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative", "mixed", "unknown"]
        },
        "notes": {"type": "string"}
    },
    "required": [
        "is_collaboration",
        "brand_name",
        "brand_industry",
        "platform",
        "campaign_type",
        "content_type",
        "collaboration_date",
        "evidence_type",
        "confidence_score",
        "sentiment",
        "notes"
    ]
}


def extract_collaboration_from_page(
    influencer_name,
    username,
    page_title,
    page_url,
    page_snippet,
    page_text
):
    system_prompt = """
You are an influencer marketing research assistant.

Your job is to analyze web page text and decide whether it proves a brand collaboration.

Return only valid JSON matching the schema.

Rules:
- Do not invent missing facts.
- If there is no clear brand collaboration, set is_collaboration to "no".
- If the evidence is weak, set is_collaboration to "maybe".
- Strong evidence includes: paid partnership, #ad, #sponsored, discount code, affiliate link, official campaign page, agency case study, brand ambassador announcement.
- Medium evidence includes: brand tag plus promotional caption, brand repost, event invitation.
- Weak evidence includes: casual brand mention, wearing a product without disclosure, fan speculation.
- Confidence score must be 0 to 100.
- If no brand is clearly identified, brand_name should be empty.
"""

    user_prompt = f"""
Influencer name: {influencer_name}
Username: {username}

Source title: {page_title}
Source URL: {page_url}
Search snippet: {page_snippet}

Page text:
{page_text}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    result = ask_ollama_json(messages, COLLAB_SCHEMA)

    result["influencer_name"] = influencer_name
    result["username"] = username
    result["evidence_url"] = page_url
    result["source_title"] = page_title
    result["source_snippet"] = page_snippet
    result["is_confirmed_collaboration"] = (
        "yes" if result.get("confidence_score", 0) >= 70 and result.get("is_collaboration") == "yes"
        else "maybe" if result.get("is_collaboration") == "maybe"
        else "no"
    )

    return result