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
from .clients.hackernews import HackerNewsClient
from .clients.techcrunch import TechCrunchClient
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
        self.session = None
        self.hacker_news_client = None
        self.tech_crunch_client = None
        self.dev_to_client = None

    async def initialize(self):
        """Inicializa o serviço e os clientes"""
        logger.info("Initializing News Integration Service")
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        self.hacker_news_client = HackerNewsClient(self.config.HACKER_NEWS_API_URL, self.session)
        self.tech_crunch_client = TechCrunchClient(
            self.config.TECH_CRUNCH_API_URL,
            self.config.TECH_CRUNCH_API_KEY,
            self.session
        )
        self.dev_to_client = DevToClient(
            self.config.DEVTO_API_URL,
            self.config.DEVTO_API_KEY,
            self.session
        )

    async def close(self):
        """Fecha conexões"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("News Integration Service shut down")

    async def search_news(self, query: NewsSearchQuery) -> List[NewsArticle]:
        """Busca notícias com base nos parâmetros fornecidos"""
        try:
            # Verificar datas e timezone
            query.start_date = ensure_timezone(query.start_date)
            query.end_date = ensure_timezone(query.end_date)

            # Limitar período de busca
            max_period = self.config.get_max_search_period_timedelta()
            if query.start_date and query.end_date:
                if (query.end_date - query.start_date) > max_period:
                    logger.warning(f"Período de busca maior que o máximo permitido. Limitando a {max_period.days} dias")
                    query.start_date = query.end_date - max_period

            # Verificar cache
            cache_key = str(query.dict())
            cached_results = self.cache.get(cache_key)
            if cached_results:
                logger.info(f"Cache hit for query: {query.topic}")
                return cached_results

            # Buscar em todas as fontes
            results = await asyncio.gather(
                self.fetch_hacker_news(query.topic),
                self.fetch_tech_crunch(query.topic),
                self.fetch_dev_to(query.topic),
                return_exceptions=True
            )

            # Processar resultados
            all_articles = []
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Erro ao buscar notícias: {result}")

            # Filtrar e ordenar
            filtered_results = self._filter_and_sort_results(all_articles, query)

            # Atualizar cache
            self.cache.set(cache_key, filtered_results)

            return filtered_results
        except Exception as e:
            logger.error(f"Erro inesperado no método search_news: {str(e)}")
            return []

    # ... resto do código continua igual ...

    async def fetch_hacker_news(self, topic: str) -> List[NewsArticle]:
        """Busca artigos no Hacker News"""
        try:
            logger.info("Fetching articles from Hacker News")
            return await self.hacker_news_client.search(topic)
        except Exception as e:
            logger.error(f"Erro no Hacker News: {str(e)}")
            return []

    async def fetch_tech_crunch(self, topic: str) -> List[NewsArticle]:
        """Busca artigos no TechCrunch"""
        try:
            logger.info("Fetching articles from TechCrunch")
            return await self.tech_crunch_client.search_articles(topic)
        except Exception as e:
            logger.error(f"Erro no TechCrunch: {str(e)}")
            return []

    async def fetch_dev_to(self, topic: str) -> List[NewsArticle]:
        """Busca artigos no Dev.to"""
        try:
            logger.info("Fetching articles from Dev.to")
            return await self.dev_to_client.search_articles(topic)
        except Exception as e:
            logger.error(f"Erro no Dev.to: {str(e)}")
            return []

    def _filter_and_sort_results(self, articles: List[NewsArticle], query: NewsSearchQuery) -> List[NewsArticle]:
        """Filtra e ordena os resultados"""
        filtered = []
        for article in articles:
            # Garantir que a data do artigo tem timezone
            if article.published_date:
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
        filtered.sort(key=lambda x: x.published_date or datetime.min, reverse=True)

        # Limitar número de resultados
        return filtered[:query.max_results] if query.max_results else filtered

    # agents/search/services/news/service.py

    # agents/search/services/news/service.py

    def _calculate_relevance(self, article: NewsArticle, query: str) -> float:
        """Calcula pontuação de relevância para um artigo"""
        score = 0.0
        
        # Termos importantes para busca
        important_terms = {
            'python': 0.4,
            'async': 0.3,
            'asyncio': 0.4,
            'await': 0.2,
            'development': 0.1,
            'programming': 0.1
        }
        
        # Verificar título
        title_lower = article.title.lower()
        for term, weight in important_terms.items():
            if term in title_lower:
                score += weight
                
        # Verificar tags
        for tag in article.tags:
            tag_lower = tag.lower()
            for term, weight in important_terms.items():
                if term in tag_lower:
                    score += weight * 0.5  # Metade do peso para tags
                    
        # Verificar sumário
        summary_lower = article.summary.lower()
        for term, weight in important_terms.items():
            if term in summary_lower:
                score += weight * 0.3  # 30% do peso para sumário
                
        # Bônus por engajamento
        if article.metadata:
            points = article.metadata.get('points', 0)
            comments = article.metadata.get('comments_count', 0)
            
            if points > 100 or comments > 50:
                score += 0.2
            elif points > 50 or comments > 25:
                score += 0.1
                
        return min(score, 1.0)