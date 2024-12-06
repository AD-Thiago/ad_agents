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

    # agents/search/services/news/clients/devto.py

    async def search_articles(self, topic: str) -> List[NewsArticle]:
        """Busca artigos no Dev.to"""
        try:
            logger.info(f"Buscando artigos no Dev.to com o termo: {topic}")
            
            # Buscar por tag python + termo de busca
            params = {
                "tag": "python",
                "per_page": 30,
                "search": topic
            }
            
            headers = {"api-key": self.api_key} if self.api_key else {}
            
            async with self.session.get(f"{self.api_url}/articles", params=params, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Erro ao buscar do Dev.to: {response.status}")
                    return []
                    
                articles = await response.json()
                results = []
                
                for article in articles:
                    try:
                        # Adicionar tags relevantes
                        article['tag_list'].append('python')  # Garantir que python está nas tags
                        if any(kw in article.get('body_markdown', '').lower() for kw in ['async', 'asyncio']):
                            article['tag_list'].append('asyncio')
                            
                        results.append(self._convert_to_article(article))
                    except Exception as e:
                        logger.error(f"Erro ao processar artigo: {str(e)}")
                        continue
                
                return results
                
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
    def _convert_to_article(self, article: Dict) -> NewsArticle:
        try:
            published_at = datetime.fromisoformat(article["published_at"].replace("Z", "+00:00"))
            
            # Processar tags
            tags = article.get("tag_list", [])
            if isinstance(tags, str):
                tags = [tags]
            elif isinstance(tags, (list, tuple)):
                tags = [str(tag) for tag in tags if tag]
            else:
                tags = []
                
            # Processar resumo
            summary = article.get("description", "")
            if not summary and article.get("body_markdown"):
                summary = article["body_markdown"][:500] + "..."
                
            return NewsArticle(
                title=article["title"],
                url=article["url"],
                author=article.get("user", {}).get("name", "Unknown"),
                source="dev.to",
                published_date=published_at,
                summary=summary,
                tags=tags,
                metadata={
                    "reading_time": article.get("reading_time_minutes", 0),
                    "comments_count": article.get("comments_count", 0),
                    "reactions_count": article.get("public_reactions_count", 0)
                }
            )
        except Exception as e:
            logger.error(f"Erro ao processar artigo do Dev.to: {str(e)}")
            raise