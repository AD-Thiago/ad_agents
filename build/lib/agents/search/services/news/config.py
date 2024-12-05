# agents/search/services/news/config.py

from typing import Dict, List
from pydantic import BaseSettings, HttpUrl, Field
from datetime import timedelta

class NewsApiConfig(BaseSettings):
    """Configurações para integrações com APIs de notícias"""
    
    TECH_NEWS_SOURCES: Dict[str, Dict] = {
        "tech_crunch": {
            "name": "TechCrunch",
            "base_url": "https://api.techcrunch.com/v1/",
            "priority": 1,
            "categories": ["technology", "startups", "ai", "cloud"]
        },
        "hacker_news": {
            "name": "Hacker News",
            "base_url": "https://hacker-news.firebaseio.com/v0/",
            "priority": 2,
            "categories": ["technology", "programming", "data-science"]
        },
        "dev_to": {
            "name": "Dev.to",
            "base_url": "https://dev.to/api/",
            "priority": 3,
            "categories": ["development", "programming", "webdev"]
        },
        "the_verge": {
            "name": "The Verge",
            "base_url": "https://www.theverge.com/api/v1/",
            "priority": 4,
            "categories": ["technology", "gadgets", "ai"]
        },
        "reuters_tech": {
            "name": "Reuters Technology",
            "base_url": "https://api.reuters.com/tech/v2/",
            "priority": 5,
            "categories": ["technology", "business", "innovation"]
        }
    }

    # Cache settings
    CACHE_TTL: int = 3600  # 1 hora
    MAX_CACHE_ITEMS: int = 10000
    
    # Rate limiting
    DEFAULT_RATE_LIMIT: int = 100  # requisições por minuto
    RATE_LIMIT_WINDOW: int = 60  # segundos
    
    # Retry settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1  # segundos
    
    # Timeout settings
    REQUEST_TIMEOUT: int = 30  # segundos
    
    # Content settings
    MIN_ARTICLE_LENGTH: int = 100  # palavras
    MAX_SUMMARY_LENGTH: int = 500  # caracteres
    
    # Relevance calculation weights
    RELEVANCE_WEIGHTS: Dict[str, float] = {
        "title_match": 0.4,
        "content_match": 0.3,
        "recency": 0.2,
        "source_priority": 0.1
    }
    
    # Date range settings
    DEFAULT_DATE_RANGE: timedelta = timedelta(days=30)
    MAX_DATE_RANGE: timedelta = timedelta(days=365)
    
    # Language settings
    SUPPORTED_LANGUAGES: List[str] = ["en", "pt-br", "es"]
    DEFAULT_LANGUAGE: str = "en"
    
    # Search settings
    MIN_RELEVANCE_SCORE: float = 0.5
    DEFAULT_MAX_RESULTS: int = 50
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PREFIX: str = "news_integration"
    
    class Config:
        env_prefix = "NEWS_"