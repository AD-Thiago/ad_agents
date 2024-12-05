from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl

class NewsSource(BaseModel):
    """Configuração de fonte de notícias"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=10)
    enabled: bool = True
    categories: List[str] = Field(default_factory=list)
    rate_limit: Optional[int] = None

class NewsSearchQuery(BaseModel):
    """Modelo para consultas de busca de notícias"""
    topic: str
    keywords: List[str] = Field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_relevance: float = 0.3
    max_results: Optional[int] = None

class NewsArticle(BaseModel):
    """Modelo para artigos de notícias"""
    title: str
    url: str
    source: str
    author: Optional[str]
    published_date: datetime
    summary: str
    content: Optional[str]
    tags: List[str] = Field(default_factory=list)
    category: str = "technology"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: Optional[float] = Field(default=0.0)