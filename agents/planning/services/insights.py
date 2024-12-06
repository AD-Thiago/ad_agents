# File: agents/planning/services/insights.py

import logging
from typing import Dict, List, Any
from datetime import datetime
from core.domains.adoption_metrics import AdoptionMetricsManager
from core.domains.seasonality import SeasonalityManager
from core.domains.market_segments import MarketSegmentManager, MarketPriority
from ..models import MarketInsight, TechInsight, SeasonalInsight, PlanningInsights

logger = logging.getLogger(__name__)

class InsightService:
    """Serviço para coleta e processamento de insights"""
    
    def __init__(self):
        self.adoption_manager = AdoptionMetricsManager()
        self.seasonality_manager = SeasonalityManager()
        self.market_manager = MarketSegmentManager()

    async def gather_all_insights(self) -> PlanningInsights:
        """Coleta todos os insights necessários"""
        try:
            market_insights = await self.gather_market_insights()
            tech_insights = await self.gather_tech_insights()
            seasonal_insights = await self.gather_seasonal_insights()

            return PlanningInsights(
                market=market_insights,
                tech=tech_insights,
                seasonal=seasonal_insights,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Erro ao coletar insights: {str(e)}")
            raise

    async def gather_market_insights(self) -> List[MarketInsight]:
        """Coleta insights de mercado"""
        insights = []
        try:
            # Coleta insights dos segmentos prioritários
            for segment in self.market_manager.get_segments_by_priority(MarketPriority.CRITICAL):
                metrics = self.market_manager.get_market_metrics(segment.name)
                tech_reqs = self.market_manager.get_tech_requirements(segment.name)
                
                insights.append(MarketInsight(
                    segment=segment.name,
                    industry=segment.industry,
                    priority=segment.priority,
                    metrics=metrics,
                    opportunities=segment.opportunities,
                    key_technologies=tech_reqs.get("technologies", []),
                    target_audience=segment.target_audience
                ))
            
            return insights
        except Exception as e:
            logger.error(f"Erro ao coletar insights de mercado: {str(e)}")
            return []

    async def gather_tech_insights(self) -> List[TechInsight]:
        """Coleta insights tecnológicos"""
        insights = []
        try:
            trending_techs = self.adoption_manager.get_trending_technologies()
            for tech in trending_techs:
                metric = self.adoption_manager.get_metric(tech)
                if metric:
                    recommendations = self.adoption_manager.get_adoption_recommendations(tech)
                    
                    insights.append(TechInsight(
                        technology=tech,
                        adoption_rate=metric.adoption_rate,
                        growth_rate=metric.growth_rate,
                        maturity_level=metric.current_stage.value,
                        market_impact=metric.market_penetration,
                        recommendations=recommendations.get("next_steps", [])
                    ))
            
            return insights
        except Exception as e:
            logger.error(f"Erro ao coletar insights tecnológicos: {str(e)}")
            return []

    async def gather_seasonal_insights(self) -> List[SeasonalInsight]:
        """Coleta insights sazonais"""
        insights = []
        try:
            current_events = self.seasonality_manager.get_current_events()
            upcoming_events = self.seasonality_manager.get_upcoming_events(30)  # próximos 30 dias
            
            for event in current_events + upcoming_events:
                prep_recom = self.seasonality_manager.get_preparation_recommendations(event)
                
                insights.append(SeasonalInsight(
                    event_name=event.name,
                    impact_level=event.impact_level,
                    key_themes=event.key_themes,
                    time_frame="current" if event in current_events else "upcoming",
                    action_items=prep_recom.get("priorities", [])
                ))
            
            return insights
        except Exception as e:
            logger.error(f"Erro ao coletar insights sazonais: {str(e)}")
            return []