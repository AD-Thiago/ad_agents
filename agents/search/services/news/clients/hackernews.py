import aiohttp
from typing import List, Dict
from datetime import datetime
from ..models import NewsArticle
import logging

logger = logging.getLogger(__name__)

class HackerNewsClient:
    """Cliente para integração com a API do Hacker News"""

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session = None

    async def initialize(self, session: aiohttp.ClientSession):
        """Inicializa sessão HTTP"""
        self.session = session
        logger.info("Sessão HTTP inicializada para o cliente Hacker News")

    async def close(self):
        """Fecha sessão HTTP"""
        if self.session:
            logger.info("Sessão HTTP fechada para o cliente Hacker News")
            self.session = None  # Sessão é compartilhada e não deve ser encerrada aqui

    async def search(self, topic: str) -> List[NewsArticle]:
        """Busca artigos no Hacker News usando a API do Algolia"""
        params = {"query": topic, "hitsPerPage": 50}
        try:
            async with self.session.get(self.api_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Erro ao buscar do Hacker News: {response.status}")
                    return []
                data = await response.json()
                return [self.to_news_article(item) for item in data["hits"]]
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
