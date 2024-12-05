# agents/search/services/news_integration.py

import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import logging
from pathlib import Path
from agents.search.services.news.config import NewsApiConfig
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)

class NewsArticle(BaseModel):
    """Modelo para artigos de notícias"""
    title: str
    url: str
    source: str
    published_date: datetime
    summary: str
    relevance_score: float
    category: str
    points: Optional[int] = None  # Pontuação do Hacker News
    comments: Optional[int] = None  # Número de comentários do Hacker News

class NewsIntegrationService:
    """Serviço de integração com APIs de notícias técnicas"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.news_apis = {
            "dev_to": {
                "url": "https://dev.to/api/",
                "key": self.config.DEVTO_API_KEY
            },
            "hacker_news": {
                "url": "http://hn.algolia.com/api/v1/search",
                "key": None  # Hacker News API não requer chave
            }
        }

    async def initialize(self):
        """Inicializa sessão HTTP"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Fecha sessão HTTP"""
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_recent_news(self, topic: str, days: int = 30) -> List[NewsArticle]:
        """Busca notícias recentes sobre um tópico"""
        if not self.session:
            await self.initialize()

        news_items = []
        tasks = [
            self.fetch_hacker_news(topic),
            self.fetch_dev_to(topic, days)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                news_items.extend(result)

        # Ordenar por relevância e data
        return sorted(
            news_items,
            key=lambda x: (x.relevance_score, x.published_date),
            reverse=True
        )

    async def fetch_hacker_news(self, topic: str) -> List[NewsArticle]:
        """Busca artigos no Algolia Hacker News API"""
        url = self.news_apis["hacker_news"]["url"]
        try:
            if topic in self.cache:
                logger.info(f"Cache hit para o tópico: {topic}")
                return self.cache[topic]

            async with self.session.get(url, params={"query": topic, "hitsPerPage": 50}) as response:
                if response.status != 200:
                    logger.error(f"Erro ao buscar do Hacker News: {response.status}")
                    return []

                data = await response.json()
                articles = [
                    NewsArticle(
                        title=hit.get("title", "Sem título"),
                        url=hit.get("url", f"https://news.ycombinator.com/item?id={hit['objectID']}"),
                        source="Hacker News",
                        published_date=datetime.fromtimestamp(hit["created_at_i"]),
                        summary=hit.get("story_text", ""),
                        relevance_score=self._calculate_relevance(hit, topic),
                        category="technology",
                        points=hit.get("points", 0),
                        comments=hit.get("num_comments", 0)
                    )
                    for hit in data.get("hits", [])
                ]
                self.cache[topic] = articles
                return articles
        except Exception as e:
            logger.error(f"Erro ao buscar do Hacker News: {str(e)}")
            return []

    async def fetch_dev_to(self, topic: str, days: int) -> List[NewsArticle]:
        """Busca artigos do Dev.to"""
        url = self.news_apis["dev_to"]["url"]
        try:
            async with self.session.get(
                url,
                params={"tag": topic, "top": "30"},
                headers={"api-key": self.news_apis["dev_to"]["key"]}
            ) as response:
                if response.status != 200:
                    logger.error(f"Erro ao buscar do Dev.to: {response.status}")
                    return []

                articles = await response.json()
                return [
                    NewsArticle(
                        title=article["title"],
                        url=article["url"],
                        source="Dev.to",
                        published_date=datetime.fromisoformat(article["published_at"]),
                        summary=article["description"],
                        relevance_score=self._calculate_relevance(article, topic),
                        category="development"
                    )
                    for article in articles
                    if (datetime.now() - datetime.fromisoformat(article["published_at"])).days <= days
                ]
        except Exception as e:
            logger.error(f"Erro ao buscar do Dev.to: {str(e)}")
            return []

    def _calculate_relevance(self, article: Dict, topic: str) -> float:
        """Calcula relevância de um artigo"""
        relevance = 0.0
        if topic.lower() in article.get("title", "").lower():
            relevance += 0.4
        if article.get("points", 0) > 50:
            relevance += 0.2
        if article.get("num_comments", 0) > 10:
            relevance += 0.1
        return min(relevance, 1.0)
