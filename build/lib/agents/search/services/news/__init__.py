# agents/search/services/news/__init__.py

from .service import NewsIntegrationService
from .models import NewsArticle, NewsSearchQuery
from .config import NewsApiConfig

__all__ = [
    'NewsIntegrationService',
    'NewsArticle',
    'NewsSearchQuery',
    'NewsApiConfig'
]