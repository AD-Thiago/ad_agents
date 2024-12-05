   
# agents/search/services/news/config.py

from pydantic import BaseSettings, Field
from typing import Dict, List, Optional
from datetime import timedelta
from pathlib import Path
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente do .env
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class NewsApiConfig(BaseSettings):
    """Configurações para integrações com APIs de notícias"""
    
    # Dev.to
    DEVTO_API_URL: str = Field("https://dev.to/api", env='DEVTO_API_URL')
    DEVTO_API_KEY: Optional[str] = Field(None, env='DEVTO_API_KEY')
    DEVTO_MAX_RESULTS: int = Field(100, env='DEVTO_MAX_RESULTS')
    DEVTO_RATE_LIMIT: int = Field(3000, env='DEVTO_RATE_LIMIT')
    
    # TechCrunch
    TECH_CRUNCH_API_URL: str = Field("https://api.techcrunch.com/v1", env='TECH_CRUNCH_API_URL')
    TECH_CRUNCH_API_KEY: Optional[str] = Field(None, env='TECH_CRUNCH_API_KEY')
    TECH_CRUNCH_MAX_RESULTS: int = Field(100, env='TECH_CRUNCH_MAX_RESULTS')
    TECH_CRUNCH_RATE_LIMIT: int = Field(3000, env='TECH_CRUNCH_RATE_LIMIT')

    # Hacker News
    HACKER_NEWS_API_URL: str = Field("http://hn.algolia.com/api/v1", env='HACKER_NEWS_API_URL')
    HACKER_NEWS_API_KEY: Optional[str] = Field(None, env='HACKER_NEWS_API_KEY')
    HACKER_NEWS_MAX_RESULTS: int = Field(100, env='HACKER_NEWS_MAX_RESULTS')
    HACKER_NEWS_RATE_LIMIT: int = Field(3000, env='HACKER_NEWS_RATE_LIMIT')

    # Cache
    CACHE_TTL: int = Field(3600, env='NEWS_CACHE_TTL')
    MAX_CACHE_ITEMS: int = Field(10000, env='NEWS_MAX_CACHE_ITEMS')
    
    # Relevância
    MIN_RELEVANCE_SCORE: float = Field(0.3, env='NEWS_MIN_RELEVANCE')
    RELEVANCE_WEIGHTS: Dict[str, float] = {
        "title_match": 0.4,
        "tag_match": 0.3,
        "content_match": 0.2,
        "engagement": 0.1
    }
    
    # Limites
    DEFAULT_MAX_RESULTS: int = Field(50, env='NEWS_DEFAULT_MAX_RESULTS')
    MAX_SEARCH_PERIOD: int = Field(
        default=30,  # 30 dias por padrão
        env='NEWS_MAX_SEARCH_PERIOD'
    )
    
    # Configurações de fontes
    ENABLED_SOURCES: List[str] = Field(
        default=["dev.to", "tech_crunch", "hacker_news"],
        env='NEWS_ENABLED_SOURCES'
    )
    
    # Rate Limiting Global
    RATE_LIMIT_WINDOW: int = Field(60, env='NEWS_RATE_LIMIT_WINDOW')
    MAX_REQUESTS_PER_WINDOW: int = Field(1000, env='NEWS_MAX_REQUESTS_PER_WINDOW')
    
    class Config:
        env_prefix = "NEWS_"
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = 'utf-8'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Configurações de API carregadas")
        logger.debug(f"Configurações: {self.dict()}")

    def get_max_search_period_timedelta(self) -> timedelta:
        """Retorna o período máximo de busca como timedelta"""
        return timedelta(days=self.MAX_SEARCH_PERIOD)