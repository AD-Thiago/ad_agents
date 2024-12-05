# agents/search/services/news/service.py

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from .models import NewsArticle, NewsSearchQuery
from .config import NewsApiConfig
from .cache import NewsCache
from .metrics import NewsMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsIntegrationService:
    """Serviço principal de integração de notícias"""

    def __init__(self):
        self.config = NewsApiConfig()
        self.cache = NewsCache(self.config)
        self.metrics = NewsMetrics()
        self.session = None

        # Download NLTK data se necessário
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    async def initialize(self):
        """Inicializa o serviço"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        logger.info("News Integration Service initialized")

    async def close(self):
        """Fecha conexões"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("News Integration Service shut down")

    async def search_news(self, query: NewsSearchQuery) -> List[NewsArticle]:
        """
        Busca notícias baseado nos parâmetros fornecidos
        """
        try:
            # Verifica cache
            cache_key = str(query.dict())
            cached_results = self.cache.get(cache_key)
            if cached_results:
                logger.info(f"Cache hit for query: {query.topic}")
                return cached_results

            # Busca em todas as fontes configuradas
            results = []
            for source_id, config in self.config.NEWS_SOURCES.items():
                try:
                    source_results = await self._search_source(source_id, query)
                    results.extend(source_results)
                except Exception as e:
                    logger.error(f"Error searching source {source_id}: {str(e)}")

            # Filtra e ordena resultados
            filtered_results = self._filter_and_sort_results(results, query)

            # Armazena no cache
            self.cache.set(cache_key, filtered_results)

            return filtered_results

        except Exception as e:
            logger.error(f"Error in search_news: {str(e)}")
            return []

    async def _search_source(self, source_id: str, query: NewsSearchQuery) -> List[NewsArticle]:
        """Busca notícias em uma fonte específica"""
        source_config = self.config.NEWS_SOURCES.get(source_id)
        if not source_config:
            return []

        try:
            async with self.session.get(
                f"{source_config['base_url']}search",
                params={
                    "q": query.topic,
                    "from": query.start_date.isoformat() if query.start_date else None,
                    "to": query.end_date.isoformat() if query.end_date else None,
                }
            ) as response:
                data = await response.json()
                return [
                    await self._process_article(article, source_id)
                    for article in data.get("articles", [])
                ]
        except Exception as e:
            logger.error(f"Error fetching from {source_id}: {str(e)}")
            return []

    async def _process_article(self, raw_article: Dict, source_id: str) -> NewsArticle:
        """Processa um artigo de notícia"""
        try:
            content = await self._fetch_article_content(raw_article.get("url", ""))
            return NewsArticle(
                title=raw_article.get("title", ""),
                url=raw_article.get("url", ""),
                source=source_id,
                author=raw_article.get("author"),
                published_date=datetime.fromisoformat(raw_article["published_at"]),
                summary=self._generate_summary(content) if content else raw_article.get("summary", ""),
                content=content,
                category=raw_article.get("category", "technology"),
                tags=raw_article.get("tags", [])
            )
        except Exception as e:
            logger.error(f"Error processing article: {str(e)}")
            raise

    def _filter_and_sort_results(self, articles: List[NewsArticle], query: NewsSearchQuery) -> List[NewsArticle]:
        """Filtra e ordena os resultados"""
        filtered = []
        for article in articles:
            if query.start_date and article.published_date < query.start_date:
                continue
            if query.end_date and article.published_date > query.end_date:
                continue
            filtered.append(article)

        # Ordena por data de publicação (mais recente primeiro)
        return sorted(filtered, key=lambda x: x.published_date, reverse=True)

    async def _fetch_article_content(self, url: str) -> Optional[str]:
        """Busca e extrai conteúdo de um artigo"""
        if not url:
            return None

        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove elementos indesejados
                for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
                    element.decompose()
                
                # Encontra o conteúdo principal
                article = soup.find('article') or soup.find(class_=['post-content', 'article-content'])
                if article:
                    return article.get_text(strip=True)
                return None
        except Exception as e:
            logger.error(f"Error fetching article content: {str(e)}")
            return None

    def _generate_summary(self, content: str) -> str:
        """Gera um resumo do conteúdo"""
        if not content:
            return ""
            
        sentences = sent_tokenize(content)
        if len(sentences) <= 3:
            return content
            
        # Usa as primeiras 3 sentenças como resumo
        summary = " ".join(sentences[:3])
        
        # Limita o tamanho
        max_length = 500
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
            
        return summary