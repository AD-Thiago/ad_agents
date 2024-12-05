from typing import List, Dict, Union
from pydantic import BaseModel, Field
from datetime import datetime

# Modelo para mensagens consumidas pela fila planning.generated
class PlanningGenerated(BaseModel):
    topic: str
    keywords: List[str]
    target_audience: str
    content_type: str
    priority: int
    estimated_impact: float

# Modelo para mensagens publicadas na fila content.generated
class ContentGenerated(BaseModel):
    content: str
    title: str
    meta_description: str
    keywords: List[str]
    seo_score: float
    readability_score: float
    created_at: datetime = Field(default_factory=datetime.now)

# Modelo para mensagens consumidas pela fila content.improved
class ContentImproved(BaseModel):
    content_id: str
    content: str
    suggestions: List[str]