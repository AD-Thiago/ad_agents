# agents/search/services/news/clients/devto.py

from typing import List, Optional
import aiohttp
from datetime import datetime, timezone
from pydantic import BaseModel, HttpUrl
import logging
from ...utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class DevToArticle(BaseModel):
    """Modelo para artigos do Dev.to"""
    id: int
    title: str
    description: Optional[str]
    url: HttpUrl
    published_at: datetime
    tag_list: List[str]
    user: dict
    reading_time_minutes: Optional[int]
    comments_count: Optional[int]
    public_reactions_count: Optional[int]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DevToClient:
    """Cliente para a API do Dev.to"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://dev.to/api"
        self.api_key = api_key
        # Dev.to tem limite de 3000 requests/hora
        self.rate_limiter = RateLimiter(max_calls=3000, period=3600)
        
    def _ensure_timezone(self, dt: datetime) -> datetime:
        """Garante que a data tem timezone (UTC se não especificado)"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    async def get_articles(
        self,
        search_term: Optional[str] = None,
        tag: Optional[str] = None,
        username: Optional[str] = None,
        page: int = 1,
        per_page: int = 30
    ) -> List[DevToArticle]:
        """Busca artigos no Dev.to"""
        params = {
            "page": page,
            "per_page": min(per_page, 1000)
        }
        
        if tag:
            params["tag"] = tag
        if username:
            params["username"] = username
            
        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    if search_term:
                        url = f"{self.base_url}/articles/search"
                        params["q"] = search_term
                    else:
                        url = f"{self.base_url}/articles"
                        
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status == 429:  # Too Many Requests
                            retry_after = int(response.headers.get("Retry-After", 60))
                            logger.warning(f"Rate limit hit. Retry after {retry_after} seconds")
                            return []
                            
                        response.raise_for_status()
                        data = await response.json()
                        
                        # Processar as datas antes de criar os objetos
                        for article in data:
                            if "published_at" in article:
                                dt = datetime.fromisoformat(article["published_at"].replace("Z", "+00:00"))
                                article["published_at"] = self._ensure_timezone(dt)
                        
                        return [DevToArticle(**article) for article in data]
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching from Dev.to: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return []
    
    def to_news_article(self, article: DevToArticle) -> "NewsArticle":
        """Converte DevToArticle para NewsArticle"""
        from ..models import NewsArticle
        
        return NewsArticle(
            title=article.title,
            url=str(article.url),
            source="dev.to",
            author=article.user.get("name"),
            published_date=self._ensure_timezone(article.published_at),
            summary=article.description or "",
            content=None,
            tags=article.tag_list,
            metadata={
                "reading_time": article.reading_time_minutes,
                "comments_count": article.comments_count,
                "reactions_count": article.public_reactions_count,
                "author_username": article.user.get("username")
            },
            relevance_score=0.0
        )