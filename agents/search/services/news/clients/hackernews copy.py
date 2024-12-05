# agents/search/services/news/clients/hackernews.py

import aiohttp
from typing import List, Dict
from datetime import datetime
from agents.search.services.news.models import NewsArticle
import logging

logger = logging.getLogger(__name__)

class HackerNewsClient:
    """Cliente para integração com a API do Hacker News"""

    def __init__(self, api_url: str):
        self.api_url = api_url

    async def search(self, topic: str, session: aiohttp.ClientSession) -> List[Dict]:
        """Busca artigos no Hacker News usando a API do Algolia"""
        params = {"query": topic, "hitsPerPage": 50}
        try:
            async with session.get(self.api_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Erro ao buscar do Hacker News: {response.status}")
                    return []
                return await response.json()
        except Exception as e:
            logger.error(f"Erro ao buscar do Hacker News: {str(e)}")
            return []

    def to_news_article(self, data: Dict) -> NewsArticle:
        """Converte o formato da API do Hacker News para o modelo NewsArticle"""
        return NewsArticle(
            title=data.get("title", "Sem título"),
            url=data.get("url", f"https://news.ycombinator.com/item?id={data['objectID']}"),
            source="Hacker News",
            published_date=datetime.fromtimestamp(data["created_at_i"]),
            summary=data.get("story_text", ""),
            relevance_score=0.0,
            category="technology",
            metadata={
                "points": data.get("points", 0),
                "comments_count": data.get("num_comments", 0)
            }
        )