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

        # Inicializar clientes como None para configurar após criar a sessão
        self.hacker_news_client = None
        self.tech_crunch_client = None
        self.dev_to_client = None

    async def initialize(self):
        """Inicializa o serviço e os clientes"""
        logger.info("Initializing News Integration Service")
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Passar a sessão ao inicializar os clientes
        self.hacker_news_client = HackerNewsClient(self.config.HACKER_NEWS_API_URL, self.session)
        self.tech_crunch_client = TechCrunchClient(self.config.TECH_CRUNCH_API_URL, self.config.TECH_CRUNCH_API_KEY, self.session)
        self.dev_to_client = DevToClient(self.config.DEVTO_API_URL, self.config.DEVTO_API_KEY, self.session)

    async def close(self):
        """Fecha conexões"""
        logger.info("Shutting down News Integration Service")
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
            results = await asyncio.gather(
                self.fetch_hacker_news(query.topic),
                self.fetch_tech_crunch(query.topic),
                self.fetch_dev_to(query.topic),
                return_exceptions=True
            )

            # Processar resultados e lidar com erros
            all_articles = []
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Erro ao buscar notícias: {result}")

            # Filtrar e ordenar resultados
            filtered_results = self._filter_and_sort_results(all_articles, query)

            # Armazenar no cache
            self.cache.set(cache_key, filtered_results)

            return filtered_results
        except Exception as e:
            logger.error(f"Erro inesperado no método search_news: {str(e)}")
            return []

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