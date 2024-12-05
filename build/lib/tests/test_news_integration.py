# tests/test_news_integration.py

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
from agents.search.services.news.service import NewsIntegrationService
from agents.search.services.news.models import NewsSearchQuery
from agents.search.services.news.clients.devto import DevToArticle

pytestmark = pytest.mark.asyncio

@pytest.fixture
async def news_service():
    """Fixture para o serviço de notícias"""
    service = NewsIntegrationService()
    await service.initialize()
    try:
        yield service
    finally:
        await service.close()

@pytest.fixture
def mock_devto_response():
    """Mock para resposta do Dev.to"""
    return [
        {
            "id": 1,
            "title": "Understanding Python Async",
            "description": "A guide to async/await in Python",
            "url": "https://dev.to/test/python-async",
            "published_at": datetime.now().isoformat(),
            "tag_list": ["python", "programming"],
            "user": {
                "name": "Test Author",
                "username": "testuser"
            },
            "reading_time_minutes": 5,
            "comments_count": 10,
            "public_reactions_count": 20
        },
        {
            "id": 2,
            "title": "JavaScript Best Practices",
            "description": "Writing better JavaScript code",
            "url": "https://dev.to/test/javascript",
            "published_at": (datetime.now() - timedelta(days=2)).isoformat(),
            "tag_list": ["javascript", "webdev"],
            "user": {
                "name": "Another Author",
                "username": "jsdev"
            },
            "reading_time_minutes": 8,
            "comments_count": 15,
            "public_reactions_count": 30
        }
    ]

async def test_search_integration(news_service, mock_devto_response):
    """Testa integração completa da busca"""
    # Criar query
    query = NewsSearchQuery(
        topic="python",
        keywords=["programming"],
        start_date=datetime.now() - timedelta(days=7),
        min_relevance=0.3,
        max_results=10
    )
    
    # Mock do cliente Dev.to
    with patch.object(news_service.devto_client, 'get_articles', return_value=[
        DevToArticle.parse_obj(article) for article in mock_devto_response
    ]):
        results = await news_service.search_news(query)
        
        assert len(results) > 0
        assert any("Python" in article.title for article in results)
        assert all(article.source == "dev.to" for article in results)

async def test_cache_integration(news_service, mock_devto_response):
    """Testa integração com cache"""
    query = NewsSearchQuery(topic="python")
    
    # Mock do cliente Dev.to
    with patch.object(news_service.devto_client, 'get_articles', return_value=[
        DevToArticle.parse_obj(mock_devto_response[0])
    ]) as mock_get:
        # Primeira chamada
        results1 = await news_service.search_news(query)
        assert len(results1) > 0
        assert mock_get.called
        
        # Segunda chamada (deve vir do cache)
        mock_get.reset_mock()
        results2 = await news_service.search_news(query)
        assert len(results2) > 0
        assert not mock_get.called
        
        # Verificar se resultados são iguais
        assert results1 == results2

async def test_error_handling(news_service):
    """Testa tratamento de erros na integração"""
    query = NewsSearchQuery(topic="python")
    
    # Simular erro no cliente
    with patch.object(news_service.devto_client, 'get_articles', side_effect=Exception("API Error")):
        results = await news_service.search_news(query)
        assert len(results) == 0

async def test_relevance_filtering(news_service, mock_devto_response):
    """Testa filtro de relevância"""
    query = NewsSearchQuery(
        topic="python",
        min_relevance=0.7  # Relevância alta
    )
    
    # Mock do cliente Dev.to
    with patch.object(news_service.devto_client, 'get_articles', return_value=[
        DevToArticle.parse_obj(article) for article in mock_devto_response
    ]):
        results = await news_service.search_news(query)
        
        # Apenas artigos muito relevantes devem passar
        assert all(
            "Python" in article.title or 
            "python" in article.tags 
            for article in results
        )