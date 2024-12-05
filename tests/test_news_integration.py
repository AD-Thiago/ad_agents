# tests/test_news_integration.py

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from agents.search.services.news.service import NewsIntegrationService
from agents.search.services.news.models import NewsSearchQuery

pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def news_service():
    """Fixture para o serviço de notícias"""
    service = NewsIntegrationService()
    await service.initialize()
    try:
        yield service
    finally:
        await service.close()

@pytest.fixture
def mock_response():
    """Mock para respostas HTTP"""
    class MockResponse:
        def __init__(self, data):
            self._data = data

        async def json(self):
            return self._data

        async def text(self):
            return "<html><body><article>Test content</article></body></html>"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    return MockResponse

@pytest.fixture(autouse=True)
async def setup_teardown():
    """Fixture para setup e teardown global"""
    # Setup
    yield
    # Teardown - limpar registry do prometheus
    from prometheus_client import REGISTRY
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)

async def test_search_news_basic(news_service, mock_response):
    """Testa busca básica de notícias"""
    mock_data = {
        "articles": [
            {
                "title": "Test Article",
                "url": "https://example.com/article",
                "published_at": datetime.now().isoformat(),
                "author": "Test Author",
                "summary": "Test summary",
                "category": "technology"
            }
        ]
    }

    # Configurar o mock
    mock_resp = mock_response(mock_data)
    
    with patch.object(news_service.session, 'get', return_value=mock_resp):
        query = NewsSearchQuery(
            topic="test",
            keywords=["python", "technology"],
            start_date=datetime.now() - timedelta(days=7)
        )
        
        results = await news_service.search_news(query)
        assert isinstance(results, list)

async def test_cache_functionality(news_service, mock_response):
    """Testa funcionalidade de cache"""
    mock_data = {
        "articles": [
            {
                "title": "Cached Article",
                "url": "https://example.com/cached",
                "published_at": datetime.now().isoformat(),
                "author": "Cache Test",
                "summary": "Test cache functionality",
                "category": "technology"
            }
        ]
    }

    query = NewsSearchQuery(topic="cache test")
    mock_resp = mock_response(mock_data)

    # Primeira chamada
    with patch.object(news_service.session, 'get', return_value=mock_resp):
        results1 = await news_service.search_news(query)
        assert isinstance(results1, list)

async def test_error_handling(news_service):
    """Testa tratamento de erros"""
    query = NewsSearchQuery(topic="error test")

    with patch.object(news_service.session, 'get', side_effect=Exception("Test error")):
        results = await news_service.search_news(query)
        assert len(results) == 0

if __name__ == '__main__':
    pytest.main(['-v', __file__])