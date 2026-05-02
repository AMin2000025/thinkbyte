import requests
import trafilatura
from bs4 import BeautifulSoup
from tavily import TavilyClient
from config import TAVILY_API_KEY

def generate_search_queries(influencer_name, username):
    base_terms = [
        "brand collaboration",
        "paid partnership",
        "#ad",
        "#sponsored",
        "sponsored by",
        "ambassador",
        "campaign",
        "promo code",
        "discount code",
        "giveaway",
        "marketing agency",
        "talent agency",
        "represented by",
        "managed by"
    ]

    queries = []

    for term in base_terms:
        queries.append(f'"{influencer_name}" "{term}"')

    if username:
        clean_username = username.replace("@", "")
        for term in base_terms:
            queries.append(f'"@{clean_username}" "{term}"')

        queries.extend([
            f'site:instagram.com "{clean_username}" "#ad"',
            f'site:instagram.com "{clean_username}" "paid partnership"',
            f'site:tiktok.com "@{clean_username}" "ad"',
            f'site:youtube.com "{clean_username}" "sponsored by"',
            f'site:linkedin.com "{influencer_name}" "agency"'
        ])

    return queries[:20]


from tavily import TavilyClient
from config import TAVILY_API_KEY

def brave_search(query, count=5):  # Keep the name 'brave_search' so you don't have to change agent.py
    if not TAVILY_API_KEY:
        raise ValueError("Missing TAVILY_API_KEY in .env")

    client = TavilyClient(api_key=TAVILY_API_KEY)
    
    # We use search instead of brave's endpoint
    response = client.search(query=query, max_results=count)
    results = response.get("results", [])

    cleaned = []
    for item in results:
        cleaned.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", "") # Tavily uses 'content' for snippets
        })

    return cleaned

def fetch_page_text(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                return text[:12000]

        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:12000]

    except Exception as error:
        return f"ERROR FETCHING PAGE: {error}"