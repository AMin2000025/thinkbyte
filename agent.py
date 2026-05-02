from search_tools import generate_search_queries, brave_search, fetch_page_text
from extractor import extract_collaboration_from_page
from scoring import score_collaboration
from database import save_collaboration


def research_influencer(influencer_name, username, influencer_niche="", max_queries=8, results_per_query=4):
    queries = generate_search_queries(influencer_name, username)
    queries = queries[:max_queries]

    saved_rows = []
    reviewed_urls = set()

    for query in queries:
        search_results = brave_search(query, count=results_per_query)

        for result in search_results:
            url = result.get("url")

            if not url or url in reviewed_urls:
                continue

            reviewed_urls.add(url)

            page_text = fetch_page_text(url)

            if not page_text or page_text.startswith("ERROR FETCHING PAGE"):
                continue

            extracted = extract_collaboration_from_page(
                influencer_name=influencer_name,
                username=username,
                page_title=result.get("title", ""),
                page_url=url,
                page_snippet=result.get("snippet", ""),
                page_text=page_text
            )

            if extracted.get("is_collaboration") in ["yes", "maybe"]:
                extracted["collaboration_score"] = score_collaboration(
                    extracted,
                    influencer_niche=influencer_niche
                )

                # Save confirmed and maybe results. You can filter later in the UI.
                save_collaboration(extracted)
                saved_rows.append(extracted)

    return saved_rows