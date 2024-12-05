import asyncio
import aiohttp
from typing import List, Dict, Optional
from datetime import datetime, timezone
from ..models import NewsArticle
from ...utils.rate_limiter import RateLimiter
import logging

logger = logging.getLogger(__name__)

class DevToClient:
    """Cliente para a API do Dev.to"""

    def __init__(self, api_url: str, api_key: Optional[str], session: aiohttp.ClientSession):
        self.api_url = api_url
        self.api_key = api_key
        self.session = session
        self.rate_limiter = RateLimiter(max_calls=3000, period=3600)

    async def search_articles(self, search_term: str, tag: Optional[str] = None) -> List[NewsArticle]:
        """
        Busca artigos no Dev.to com base no termo de pesquisa e tags opcionais.
        """
        try:
            logger.info(f"Buscando artigos no Dev.to com o termo: {search_term}")
            params = {"per_page": 30}
            if tag:
                params["tag"] = tag

            headers = {"api-key": self.api_key} if self.api_key else {}

            async with self.rate_limiter:
                async with self.session.get(f"{self.api_url}/articles", params=params, headers=headers) as response:
                    if response.status != 200:
                        raise Exception(f"Erro ao buscar no Dev.to: {response.status}")
                    data = await response.json()
                    logger.debug(f"Dados recebidos da API do Dev.to: {data}")
                    return [self.to_news_article(article) for article in data]
        except Exception as e:
            logger.error(f"Erro ao buscar no Dev.to: {str(e)}")
            return []

    def to_news_article(self, article: Dict) -> NewsArticle:
        """
        Converte um artigo da API Dev.to para o modelo NewsArticle.
        """
        try:
            published_at = datetime.fromisoformat(article["published_at"])
            return NewsArticle(
                title=article["title"],
                url=article["url"],
                author=article.get("user", {}).get("name", "Desconhecido"),
                source="dev.to",
                published_date=published_at,
                summary=article.get("description", ""),
                tags=article.get("tag_list", []),
                metadata={
                    "reading_time": article.get("reading_time_minutes", 0),
                    "comments_count": article.get("comments_count", 0),
                    "reactions_count": article.get("public_reactions_count", 0)
                },
                relevance_score=0.0
            )
        except Exception as e:
            logger.error(f"Erro ao processar artigo do Dev.to: {str(e)}")
            raise