# File: agents/planning/models.py

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from core.domains.definitions import ContentType, ExpertiseLevel, DomainType
from core.domains.market_segments import MarketPriority, Industry

class PlanningRequest(BaseModel):
    """Solicitação de planejamento"""
    trigger_id: str
    workflow_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    parameters: Dict[str, Any] = Field(default_factory=dict)

class MarketInsight(BaseModel):
    """Insight de mercado"""
    segment: str
    industry: Industry
    priority: MarketPriority
    metrics: Dict[str, float]
    opportunities: List[str]
    key_technologies: List[str]
    target_audience: List[str]

class TechInsight(BaseModel):
    """Insight tecnológico"""
    technology: str
    adoption_rate: float
    growth_rate: float
    maturity_level: str
    market_impact: float
    recommendations: List[str]

class SeasonalInsight(BaseModel):
    """Insight sazonal"""
    event_name: str
    impact_level: float
    key_themes: List[str]
    time_frame: str
    action_items: List[str]

class ContentStrategy(BaseModel):
    """Estratégia de conteúdo"""
    content_type: ContentType
    target_expertise: ExpertiseLevel
    domain_type: DomainType
    estimated_impact: float = Field(ge=0.0, le=1.0)
    priority: MarketPriority
    key_topics: List[str]
    content_structure: Dict[str, Any]
    success_metrics: List[str]

class PlanningInsights(BaseModel):
    """Conjunto de insights coletados"""
    market: List[MarketInsight]
    tech: List[TechInsight]
    seasonal: List[SeasonalInsight]
    timestamp: datetime = Field(default_factory=datetime.now)

class PlanningResponse(BaseModel):
    """Resposta do planejamento"""
    workflow_id: str
    topic: str
    keywords: List[str]
    target_audience: List[str]
    strategy: ContentStrategy
    insights: PlanningInsights
    execution_steps: List[Dict[str, Any]]
    next_actions: List[Dict[str, Any]]
    generated_at: datetime = Field(default_factory=datetime.now)