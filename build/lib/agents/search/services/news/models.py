# agents/search/services/news/models.py

from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl

class NewsSource(BaseModel):
    """Configuração de fonte de notícias"""
    name: str
    base_url: HttpUrl
    api_key: Optional[str]
    priority: int = Field(default=1, ge=1, le=10)
    rate_limit: int = Field(default=100)  # requisições por minuto
    categories: List[str] = []
    enabled: bool = True

class NewsArticle(BaseModel):
    """Modelo detalhado para artigos de notícias"""
    title: str
    url: HttpUrl
    source: str
    author: Optional[str]
    published_date: datetime
    updated_date: Optional[datetime]
    summary: str
    content: Optional[str]
    relevance_score: float = Field(ge=0.0, le=1.0)
    category: str
    tags: List[str] = []
    language: str = "en"
    read_time: Optional[int]  # minutos
    metadata: Dict = Field(default_factory=dict)

class NewsSearchQuery(BaseModel):
    """Modelo para consultas de notícias"""
    topic: str
    keywords: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    language: str = "en"
    min_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    max_results: int = Field(default=50, ge=1, le=100)
    categories: List[str] = Field(default_factory=list)