# agents/search/services/news_integration.py

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import aiohttp
from pydantic import BaseModel
import json

class NewsArticle(BaseModel):
    """Modelo para artigos de notícias"""
    title: str
    url: str
    source: str
    published_date: datetime
    summary: str
    relevance_score: float
    category: str

class NewsIntegrationService:
    """Serviço de integração com APIs de notícias técnicas"""

    def __init__(self):
        # Configurar APIs - em produção, mover para variáveis de ambiente
        self.news_apis = {
            "tech_crunch": {
                "url": "https://api.techcrunch.com/v1/",
                "key": "your_api_key"
            },
            "hacker_news": {
                "url": "https://hacker-news.firebaseio.com/v0/",
                "key": None
            },
            "dev_to": {
                "url": "https://dev.to/api/",
                "key": "your_api_key"
            }
        }
        self.session = None

    async def initialize(self):
        """Inicializa sessão HTTP"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Fecha sessão HTTP"""
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_recent_news(self, topic: str, days: int = 30) -> List[NewsArticle]:
        """Busca notícias recentes sobre um tópico"""
        if not self.session:
            await self.initialize()

        news_items = []
        tasks = [
            self._fetch_tech_crunch(topic, days),
            self._fetch_hacker_news(topic, days),
            self._fetch_dev_to(topic, days)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                news_items.extend(result)

        # Ordenar por relevância e data
        return sorted(
            news_items,
            key=lambda x: (x.relevance_score, x.published_date),
            reverse=True
        )

    async def _fetch_tech_crunch(self, topic: str, days: int) -> List[NewsArticle]:
        """Busca notícias do TechCrunch"""
        try:
            async with self.session.get(
                f"{self.news_apis['tech_crunch']['url']}search",
                params={
                    "q": topic,
                    "from_date": (datetime.now() - timedelta(days=days)).isoformat()
                },
                headers={"Authorization": f"Bearer {self.news_apis['tech_crunch']['key']}"}
            ) as response:
                data = await response.json()
                return [
                    NewsArticle(
                        title=article["title"],
                        url=article["url"],
                        source="TechCrunch",
                        published_date=datetime.fromisoformat(article["published_at"]),
                        summary=article["summary"],
                        relevance_score=self._calculate_relevance(article, topic),
                        category="technology"
                    )
                    for article in data.get("articles", [])
                ]
        except Exception as e:
            print(f"Erro ao buscar do TechCrunch: {str(e)}")
            return []

    async def _fetch_hacker_news(self, topic: str, days: int) -> List[NewsArticle]:
        """Busca notícias do Hacker News"""
        try:
            async with self.session.get(
                f"{self.news_apis['hacker_news']['url']}topstories.json"
            ) as response:
                story_ids = await response.json()
                stories = []
                
                # Limitar a 50 histórias para não sobrecarregar
                for story_id in story_ids[:50]:
                    async with self.session.get(
                        f"{self.news_apis['hacker_news']['url']}item/{story_id}.json"
                    ) as story_response:
                        story = await story_response.json()
                        if topic.lower() in story.get("title", "").lower():
                            stories.append(
                                NewsArticle(
                                    title=story["title"],
                                    url=story.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                                    source="Hacker News",
                                    published_date=datetime.fromtimestamp(story["time"]),
                                    summary=story.get("text", "No summary available"),
                                    relevance_score=self._calculate_relevance(story, topic),
                                    category="technology"
                                )
                            )
                return stories
        except Exception as e:
            print(f"Erro ao buscar do Hacker News: {str(e)}")
            return []

    async def _fetch_dev_to(self, topic: str, days: int) -> List[NewsArticle]:
        """Busca artigos do Dev.to"""
        try:
            async with self.session.get(
                f"{self.news_apis['dev_to']['url']}articles",
                params={"tag": topic, "top": "30"},
                headers={"api-key": self.news_apis['dev_to']['key']}
            ) as response:
                articles = await response.json()
                return [
                    NewsArticle(
                        title=article["title"],
                        url=article["url"],
                        source="Dev.to",
                        published_date=datetime.fromisoformat(article["published_at"]),
                        summary=article["description"],
                        relevance_score=self._calculate_relevance(article, topic),
                        category="technology"
                    )
                    for article in articles
                    if (datetime.now() - datetime.fromisoformat(article["published_at"])).days <= days
                ]
        except Exception as e:
            print(f"Erro ao buscar do Dev.to: {str(e)}")
            return []

    def _calculate_relevance(self, article: Dict, topic: str) -> float:
        """Calcula pontuação de relevância para um artigo"""
        relevance = 0.0
        
        # Relevância do título
        if topic.lower() in article.get("title", "").lower():
            relevance += 0.4
            
        # Relevância do conteúdo/resumo
        if topic.lower() in article.get("summary", "").lower() or \
           topic.lower() in article.get("description", "").lower() or \
           topic.lower() in article.get("text", "").lower():
            relevance += 0.3
            
        # Engajamento (se disponível)
        points = article.get("points", 0)
        comments = article.get("num_comments", 0) or article.get("comment_count", 0)
        
        if points > 100:
            relevance += 0.2
        elif points > 50:
            relevance += 0.1
            
        if comments > 50:
            relevance += 0.1
            
        return min(relevance, 1.0)