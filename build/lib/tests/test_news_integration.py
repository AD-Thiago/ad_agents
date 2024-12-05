# tests/test_news_integration.py

import pytest
from unittest.mock import patch
from datetime import datetime, timedelta
from agents.search.services.news.service import NewsIntegrationService
from agents.search.services.news.models import NewsArticle, NewsSearchQuery

pytestmark = pytest.mark.asyncio

@pytest.fixture
async def news_service():
    """Fixture para o serviço de notícias"""
    service = NewsIntegrationService()
    await service.initialize()
    yield service
    await service.close()

@pytest.fixture
def mock_devto_response():
    """Mock para resposta do Dev.to"""
    return [
        {
            "title": "Understanding Python Async",
            "url": "https://dev.to/test/python-async",
            "published_at": datetime.now().isoformat(),
            "description": "A guide to async/await in Python",
            "tag_list": ["python", "programming"]
        },
        {
            "title": "JavaScript Best Practices",
            "url": "https://dev.to/test/javascript",
            "published_at": (datetime.now() - timedelta(days=2)).isoformat(),
            "description": "Writing better JavaScript code",
            "tag_list": ["javascript", "webdev"]
        }
    ]

@pytest.fixture
def mock_hacker_news_response():
    """Mock para resposta do Hacker News"""
    return {
        "hits": [
            {
                "objectID": "123",
                "title": "Exploring Python Asyncio",
                "url": "https://news.ycombinator.com/item?id=123",
                "created_at_i": int((datetime.now() - timedelta(days=1)).timestamp()),
                "points": 150,
                "num_comments": 45,
                "story_text": "Asyncio is a powerful tool for concurrency in Python."
            },
            {
                "objectID": "456",
                "title": "New AI advancements",
                "url": "https://news.ycombinator.com/item?id=456",
                "created_at_i": int((datetime.now() - timedelta(days=2)).timestamp()),
                "points": 200,
                "num_comments": 30,
                "story_text": "Latest trends in artificial intelligence."
            }
        ]
    }

async def test_search_integration(news_service, mock_devto_response, mock_hacker_news_response):
    """Teste de integração completo com múltiplas fontes"""
    query = NewsSearchQuery(
        topic="python",
        keywords=["programming"],
        start_date=datetime.now() - timedelta(days=7),
        min_relevance=0.3,
        max_results=10
    )
    
    with patch.object(news_service, 'fetch_dev_to', return_value=[
        NewsArticle(
            title=article["title"],
            url=article["url"],
            source="Dev.to",
            published_date=datetime.fromisoformat(article["published_at"]),
            summary=article["description"],
            relevance_score=0.8,
            category="development"
        ) for article in mock_devto_response
    ]), patch.object(news_service, 'fetch_hacker_news', return_value=[
        NewsArticle(
            title=hit["title"],
            url=hit["url"],
            source="Hacker News",
            published_date=datetime.fromtimestamp(hit["created_at_i"]),
            summary=hit.get("story_text", ""),
            relevance_score=0.9,
            category="technology",
            points=hit["points"],
            comments=hit["num_comments"]
        ) for hit in mock_hacker_news_response["hits"]
    ]):
        results = await news_service.fetch_recent_news(query.topic)
        
        assert len(results) > 0
        assert any(article.source == "Dev.to" for article in results)
        assert any(article.source == "Hacker News" for article in results)

async def test_cache_integration(news_service, mock_devto_response):
    """Teste de integração com cache"""
    query = NewsSearchQuery(topic="python")
    
    with patch.object(news_service, 'fetch_dev_to', return_value=[
        NewsArticle(
            title=article["title"],
            url=article["url"],
            source="Dev.to",
            published_date=datetime.fromisoformat(article["published_at"]),
            summary=article["description"],
            relevance_score=0.8,
            category="development"
        ) for article in mock_devto_response
    ]) as mock_fetch:
        # Primeira chamada
        results1 = await news_service.fetch_recent_news(query.topic)
        assert len(results1) > 0
        assert mock_fetch.called

        # Segunda chamada (deve vir do cache)
        mock_fetch.reset_mock()
        results2 = await news_service.fetch_recent_news(query.topic)
        assert len(results2) > 0
        assert not mock_fetch.called
        
        # Verificar que os resultados são iguais
        assert results1 == results2

async def test_error_handling(news_service):
    """Teste de tratamento de erros na integração"""
    query = NewsSearchQuery(topic="python")
    
    with patch.object(news_service, 'fetch_dev_to', side_effect=Exception("API Error")), \
         patch.object(news_service, 'fetch_hacker_news', side_effect=Exception("API Error")):
        results = await news_service.fetch_recent_news(query.topic)
        assert len(results) == 0

async def test_relevance_filtering(news_service, mock_devto_response, mock_hacker_news_response):
    """Teste de filtro por relevância"""
    query = NewsSearchQuery(
        topic="python",
        min_relevance=0.7  # Apenas resultados muito relevantes
    )
    
    with patch.object(news_service, 'fetch_dev_to', return_value=[
        NewsArticle(
            title=article["title"],
            url=article["url"],
            source="Dev.to",
            published_date=datetime.fromisoformat(article["published_at"]),
            summary=article["description"],
            relevance_score=0.6,
            category="development"
        ) for article in mock_devto_response
    ]), patch.object(news_service, 'fetch_hacker_news', return_value=[
        NewsArticle(
            title=hit["title"],
            url=hit["url"],
            source="Hacker News",
            published_date=datetime.fromtimestamp(hit["created_at_i"]),
            summary=hit.get("story_text", ""),
            relevance_score=0.8,
            category="technology"
        ) for hit in mock_hacker_news_response["hits"]
    ]):
        results = await news_service.fetch_recent_news(query.topic)
        
        assert len(results) > 0
        assert all(article.relevance_score >= 0.7 for article in results)
