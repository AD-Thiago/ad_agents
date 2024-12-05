# agents/search/services/news/service.py

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
from .models import NewsArticle, NewsSearchQuery
from .config import NewsApiConfig
from .cache import NewsCache
from .metrics import NewsMetrics
from .clients.devto import DevToClient

logger = logging.getLogger(__name__)

def ensure_timezone(dt: Optional[datetime]) -> Optional[datetime]:
    """Garante que a data tem timezone (UTC se não especificado)"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

class NewsIntegrationService:
    """Serviço principal de integração de notícias"""

    def __init__(self):
        self.config = NewsApiConfig()
        self.cache = NewsCache(self.config)
        self.metrics = NewsMetrics()
        
        # Inicializar clientes
        self.devto_client = DevToClient(api_key=self.config.DEVTO_API_KEY)
        
        self.session = None

    async def initialize(self):
        """Inicializa o serviço"""
        logger.info("Initializing News Integration Service")
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Fecha conexões"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("News Integration Service shut down")

    async def search_news(self, query: NewsSearchQuery) -> List[NewsArticle]:
        """Busca notícias com base nos parâmetros fornecidos"""
        try:
            # Garantir que as datas têm timezone
            query.start_date = ensure_timezone(query.start_date)
            query.end_date = ensure_timezone(query.end_date)

            # Verificar cache
            cache_key = str(query.dict())
            cached_results = self.cache.get(cache_key)
            if cached_results:
                logger.info(f"Cache hit for query: {query.topic}")
                return cached_results

            # Buscar em todas as fontes configuradas
            results = []
            
            # Busca no Dev.to
            async with self.metrics.track_request("dev.to"):
                devto_articles = await self.devto_client.get_articles(
                    search_term=query.topic,
                    tag=query.keywords[0] if query.keywords else None
                )
                for article in devto_articles:
                    results.append(self.devto_client.to_news_article(article))
                    self.metrics.record_processed_article("dev.to")

            # Filtrar e ordenar resultados
            filtered_results = self._filter_and_sort_results(results, query)

            # Armazenar no cache
            self.cache.set(cache_key, filtered_results)

            return filtered_results

        except Exception as e:
            logger.error(f"Error in search_news: {str(e)}")
            return []

    def _filter_and_sort_results(self, articles: List[NewsArticle], query: NewsSearchQuery) -> List[NewsArticle]:
        """Filtra e ordena os resultados"""
        filtered = []
        for article in articles:
            # Garantir que a data do artigo tem timezone
            article.published_date = ensure_timezone(article.published_date)
            
            # Filtrar por data
            if query.start_date and article.published_date < query.start_date:
                continue
            if query.end_date and article.published_date > query.end_date:
                continue

            # Filtrar por relevância
            if self._calculate_relevance(article, query) >= query.min_relevance:
                filtered.append(article)

        # Ordenar por data de publicação (mais recente primeiro)
        filtered.sort(key=lambda x: x.published_date, reverse=True)

        # Limitar número de resultados
        return filtered[:query.max_results] if query.max_results else filtered

    def _calculate_relevance(self, article: NewsArticle, query: NewsSearchQuery) -> float:
        """Calcula pontuação de relevância para um artigo"""
        score = 0.0
        
        # Relevância do título
        if query.topic.lower() in article.title.lower():
            score += 0.4
        
        # Relevância das tags
        if article.tags and any(keyword.lower() in tag.lower() for keyword in query.keywords for tag in article.tags):
            score += 0.3
        
        # Relevância do resumo
        if query.topic.lower() in article.summary.lower():
            score += 0.2
        
        # Bônus por engajamento
        if article.metadata:
            reactions = article.metadata.get("reactions_count", 0)
            comments = article.metadata.get("comments_count", 0)
            if reactions > 50 or comments > 10:
                score += 0.1
        
        return min(score, 1.0)