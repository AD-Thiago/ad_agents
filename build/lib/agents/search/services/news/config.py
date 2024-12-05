# agents/search/services/news/config.py

from pydantic import BaseSettings, Field
from typing import Dict, List, Optional
from datetime import timedelta
from pathlib import Path
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente do .env
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class NewsApiConfig(BaseSettings):
    """Configurações para integrações com APIs de notícias"""
    
    # Dev.to
    DEVTO_API_KEY: Optional[str] = Field(None, env='NEWS_DEVTO_API_KEY')
    DEVTO_MAX_RESULTS: int = Field(100, env='NEWS_DEVTO_MAX_RESULTS')
    DEVTO_RATE_LIMIT: int = Field(3000, env='NEWS_DEVTO_RATE_LIMIT')
    
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
    MAX_SEARCH_PERIOD: timedelta = Field(
        default_factory=lambda: timedelta(days=int(os.getenv('NEWS_MAX_SEARCH_PERIOD', '30'))),
    )
    
    # Configurações de fontes
    ENABLED_SOURCES: List[str] = Field(
        default=["dev.to"],
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