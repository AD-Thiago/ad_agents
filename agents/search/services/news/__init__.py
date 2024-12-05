# agents/search/services/news/__init__.py
from .service import NewsIntegrationService
from .models import NewsArticle, NewsSearchQuery

__all__ = ['NewsIntegrationService', 'NewsArticle', 'NewsSearchQuery']