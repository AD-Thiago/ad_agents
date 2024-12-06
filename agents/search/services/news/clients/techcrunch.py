# agents/search/services/news/clients/techcrunch.py

import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List
import logging
from ..models import NewsArticle
from urllib.parse import quote

logger = logging.getLogger(__name__)

class TechCrunchClient:
    def __init__(self, base_url: str, session: aiohttp.ClientSession):
        self.base_url = "https://techcrunch.com"
        self.search_url = f"{self.base_url}/search"
        self.session = session

    async def search_articles(self, query: str) -> List[NewsArticle]:
        """
        Busca artigos no TechCrunch usando web scraping
        """
        try:
            logger.info(f"Buscando artigos no TechCrunch com o termo: {query}")
            
            # URL de busca do TechCrunch
            encoded_query = quote(query)
            url = f"{self.search_url}/{encoded_query}"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Erro ao acessar TechCrunch: {response.status}")
                    return []
                
                html = await response.text()
                return await self._parse_search_results(html)
                
        except Exception as e:
            logger.error(f"Erro ao buscar no TechCrunch: {str(e)}")
            return []

    async def _parse_search_results(self, html: str) -> List[NewsArticle]:
        """
        Parse dos resultados da busca do TechCrunch
        """
        articles = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Encontrar os artigos na página
            # Ajuste os seletores conforme a estrutura atual do site
            article_elements = soup.select('article.post-block')
            
            for element in article_elements:
                try:
                    # Extrair informações do artigo
                    title = element.select_one('h2.post-block__title a')
                    link = title.get('href') if title else None
                    title_text = title.text.strip() if title else None
                    
                    # Extrair autor
                    author = element.select_one('span.post-block__author a')
                    author_name = author.text.strip() if author else "Unknown"
                    
                    # Extrair data
                    date_element = element.select_one('time.post-block__time')
                    date_str = date_element.get('datetime') if date_element else None
                    published_date = datetime.fromisoformat(date_str) if date_str else datetime.now()
                    
                    # Extrair resumo
                    summary = element.select_one('div.post-block__content')
                    summary_text = summary.text.strip() if summary else ""
                    
                    if title_text and link:
                        articles.append(NewsArticle(
                            title=title_text,
                            url=link if link.startswith('http') else f"{self.base_url}{link}",
                            author=author_name,
                            source="TechCrunch",
                            published_date=published_date,
                            summary=summary_text[:500] if summary_text else "",
                            tags=["technology", "startup", "tech-news"],  # Tags padrão para TechCrunch
                        ))
                except Exception as e:
                    logger.warning(f"Erro ao processar artigo individual: {str(e)}")
                    continue
                    
            return articles
            
        except Exception as e:
            logger.error(f"Erro ao fazer parse dos resultados do TechCrunch: {str(e)}")
            return []

    async def close(self):
        """Método placeholder para compatibilidade com interface"""
        pass