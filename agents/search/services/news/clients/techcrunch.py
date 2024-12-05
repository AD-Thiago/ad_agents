import asyncio
import aiohttp
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
from ..models import NewsArticle
from ...utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class TechCrunchClient:
    """Cliente para a API do TechCrunch"""

    def __init__(self, api_url: str, api_key: Optional[str], session: aiohttp.ClientSession):
        self.api_url = api_url
        self.api_key = api_key
        self.session = session
        self.rate_limiter = RateLimiter(max_calls=3000, period=3600)

    async def search_articles(self, search_term: str) -> List[NewsArticle]:
        """Busca artigos relacionados a um tópico"""
        try:
            logger.info(f"Buscando artigos no TechCrunch com o termo: {search_term}")
            async with self.rate_limiter:
                async with self.session.get(
                    f"{self.api_url}/search",
                    params={
                        "q": search_term,
                        "from_date": (datetime.now() - timedelta(days=30)).isoformat()
                    },
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Erro ao buscar no TechCrunch: {response.status}")
                    data = await response.json()
                    logger.debug(f"Dados recebidos da API do TechCrunch: {data}")
                    return [
                        NewsArticle(
                            title=article["title"],
                            url=article["url"],
                            author=article.get("author", "Desconhecido"),
                            source="tech_crunch",
                            published_date=datetime.fromisoformat(article["published_at"]),
                            summary=article.get("summary", ""),
                            tags=["technology"],
                            metadata={},
                            relevance_score=0.0  # A relevância será calculada posteriormente
                        )
                        for article in data.get("articles", [])
                    ]
        except Exception as e:
            logger.error(f"Erro ao buscar no TechCrunch: {str(e)}")
            return []
