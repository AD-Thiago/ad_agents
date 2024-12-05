# tests/test_devto_client.py

import pytest
from datetime import datetime
from unittest.mock import patch, Mock
from agents.search.services.news.clients.devto import DevToClient, DevToArticle

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_response():
    """Mock para resposta da API"""
    return [
        {
            "id": 1,
            "title": "Test Article",
            "description": "Test Description",
            "url": "https://dev.to/test/article",
            "published_at": datetime.now().isoformat(),
            "tag_list": ["python", "testing"],
            "user": {
                "name": "Test User",
                "username": "testuser"
            },
            "reading_time_minutes": 5,
            "comments_count": 10,
            "public_reactions_count": 20
        }
    ]

async def test_get_articles_success(mock_response):
    """Testa busca de artigos com sucesso"""
    client = DevToClient()
    
    # Mock da sessão HTTP
    mock_session = Mock()
    mock_session.get.return_value.__aenter__.return_value.status = 200
    mock_session.get.return_value.__aenter__.return_value.json = Mock(
        return_value=mock_response
    )
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        articles = await client.get_articles(search_term="python")
        
        assert len(articles) == 1
        article = articles[0]
        assert article.title == "Test Article"
        assert article.tag_list == ["python", "testing"]
        
        # Testar conversão para NewsArticle
        news_article = client.to_news_article(article)
        assert news_article.title == article.title
        assert news_article.source == "dev.to"
        assert "reading_time" in news_article.metadata

async def test_get_articles_rate_limit(mock_response):
    """Testa comportamento quando rate limit é atingido"""
    client = DevToClient()
    
    # Mock da sessão HTTP com rate limit
    mock_session = Mock()
    mock_session.get.return_value.__aenter__.return_value.status = 429
    mock_session.get.return_value.__aenter__.return_value.headers = {"Retry-After": "60"}
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        articles = await client.get_articles(search_term="python")
        assert len(articles) == 0

async def test_get_articles_error():
    """Testa tratamento de erro na API"""
    client = DevToClient()
    
    # Mock da sessão HTTP com erro
    mock_session = Mock()
    mock_session.get.side_effect = Exception("API Error")
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        articles = await client.get_articles(search_term="python")
        assert len(articles) == 0

async def test_article_conversion(mock_response):
    """Testa conversão de DevToArticle para NewsArticle"""
    client = DevToClient()
    devto_article = DevToArticle.parse_obj(mock_response[0])
    news_article = client.to_news_article(devto_article)
    
    assert news_article.title == devto_article.title
    assert news_article.url == str(devto_article.url)
    assert news_article.source == "dev.to"
    assert news_article.tags == devto_article.tag_list
    assert news_article.metadata["reading_time"] == devto_article.reading_time_minutes