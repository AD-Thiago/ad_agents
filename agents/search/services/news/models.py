from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field

class NewsArticle(BaseModel):
    """Modelo para artigos de notícias"""
    title: str
    url: str
    author: Optional[str] = "Unknown"
    source: str
    published_date: datetime
    summary: str = ""
    tags: List[str] = Field(default_factory=list)  # Simplificando para lista de strings
    category: str = "technology"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = 0.0

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