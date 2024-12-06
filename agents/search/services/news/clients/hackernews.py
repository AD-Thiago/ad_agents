 # agents/search/services/news/clients/hackernews.py

import aiohttp
from typing import List, Dict
from datetime import datetime
from ..models import NewsArticle
import logging

logger = logging.getLogger(__name__)

# agents/search/services/news/clients/hackernews.py

class HackerNewsClient:
    """Cliente para integração com a API do Hacker News"""

    def __init__(self, api_url: str, session: aiohttp.ClientSession):
        self.api_url = api_url
        self.session = session

    async def search(self, topic: str) -> List[NewsArticle]:
        """
        Busca artigos no Hacker News usando a API do Algolia
        """
        params = {
            "query": topic,
            "tags": "story",  # Apenas stories, não comentários
            "numericFilters": "points>1",  # Filtra por pontos para garantir qualidade
            "hitsPerPage": 50
        }
        
        try:
            async with self.session.get(self.api_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Erro ao buscar do Hacker News: {response.status}")
                    return []
                    
                data = await response.json()
                return [self._convert_to_article(hit) for hit in data.get("hits", [])]
                
        except Exception as e:
            logger.error(f"Erro ao buscar do Hacker News: {str(e)}")
            return []

    def _convert_to_article(self, hit: Dict) -> NewsArticle:
        """Converte um resultado da API em um NewsArticle"""
        # Extrair tags relevantes do HN
        tags = []
        if "story" in hit.get("_tags", []):
            if hit.get("title", "").lower().find("python") >= 0:
                tags.append("python")
            if "show_hn" in hit.get("_tags", []):
                tags.append("show")
            if "ask_hn" in hit.get("_tags", []):
                tags.append("ask")
            
        return NewsArticle(
            title=hit.get("title", "Sem título"),
            url=hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
            author=hit.get("author", "Unknown"),
            source="Hacker News",
            published_date=datetime.fromtimestamp(hit.get("created_at_i", 0)),
            summary=hit.get("story_text", "")[:500],  # Limitando tamanho do resumo
            tags=tags,
            metadata={
                "points": hit.get("points", 0),
                "comments_count": hit.get("num_comments", 0)
            },
            relevance_score=0.0
        )