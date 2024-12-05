# core\domains\seasonality.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, date
from enum import Enum

class SeasonType(str, Enum):
    """Tipos de sazonalidade"""
    RETAIL = "retail"           # Eventos de varejo
    BUSINESS = "business"       # Ciclos de negócio
    TECH = "tech"              # Eventos tecnológicos
    MARKET = "market"          # Ciclos de mercado
    INDUSTRY = "industry"      # Específico da indústria
    REGULATORY = "regulatory"  # Marcos regulatórios

class EventStatus(str, Enum):
    """Status do evento sazional"""
    UPCOMING = "upcoming"      # Próximo
    ACTIVE = "active"         # Em andamento
    PASSED = "passed"         # Passado
    RECURRING = "recurring"   # Recorrente

class SeasonalEvent(BaseModel):
    """Evento sazional"""
    name: str
    type: SeasonType
    description: str
    start_date: date
    end_date: date
    impact_level: float = Field(ge=0.0, le=1.0)  # 0 a 1
    affected_industries: List[str]
    key_themes: List[str]
    content_priorities: List[str]
    recurrence: str  # annual, quarterly, monthly
    historical_metrics: Dict[str, float]
    preparation_time_days: int

class SeasonalityManager:
    """Gerenciador de sazonalidade"""
    
    def __init__(self):
        self.events = self._initialize_events()
    
    def _initialize_events(self) -> Dict[str, SeasonalEvent]:
        """Inicializa eventos sazonais"""
        return {
            "black_friday": SeasonalEvent(
                name="Black Friday",
                type=SeasonType.RETAIL,
                description="Principal evento de vendas do varejo",
                start_date=date(2024, 11, 29),
                end_date=date(2024, 11, 29),
                impact_level=0.9,
                affected_industries=[
                    "retail",
                    "e-commerce",
                    "fintech",
                    "logistics"
                ],
                key_themes=[
                    "performance",
                    "scalability",
                    "payments",
                    "security"
                ],
                content_priorities=[
                    "system_optimization",
                    "load_testing",
                    "monitoring",
                    "incident_response"
                ],
                recurrence="annual",
                historical_metrics={
                    "2023_transactions": 15000000,
                    "2023_peak_tps": 5000,
                    "2023_conversion": 0.045
                },
                preparation_time_days=90
            ),

            "tax_season": SeasonalEvent(
                name="Tax Season",
                type=SeasonType.REGULATORY,
                description="Período de declaração de impostos",
                start_date=date(2024, 3, 1),
                end_date=date(2024, 4, 30),
                impact_level=0.8,
                affected_industries=[
                    "fintech",
                    "accounting",
                    "banking",
                    "consulting"
                ],
                key_themes=[
                    "tax_compliance",
                    "automation",
                    "data_processing",
                    "security"
                ],
                content_priorities=[
                    "integration_guides",
                    "automation_tools",
                    "security_best_practices",
                    "performance_optimization"
                ],
                recurrence="annual",
                historical_metrics={
                    "2023_declarations": 35000000,
                    "2023_digital_rate": 0.92
                },
                preparation_time_days=60
            ),

            "tech_conference_season": SeasonalEvent(
                name="Tech Conference Season",
                type=SeasonType.TECH,
                description="Principais conferências de tecnologia",
                start_date=date(2024, 5, 1),
                end_date=date(2024, 8, 31),
                impact_level=0.75,
                affected_industries=[
                    "technology",
                    "startup",
                    "enterprise",
                    "education"
                ],
                key_themes=[
                    "innovation",
                    "trends",
                    "networking",
                    "product_launch"
                ],
                content_priorities=[
                    "trend_analysis",
                    "technical_deep_dives",
                    "case_studies",
                    "tutorials"
                ],
                recurrence="annual",
                historical_metrics={
                    "2023_attendees": 500000,
                    "2023_announcements": 250
                },
                preparation_time_days=120
            ),

            "open_banking_phase": SeasonalEvent(
                name="Open Banking Implementation Phase",
                type=SeasonType.REGULATORY,
                description="Fase de implementação do Open Banking",
                start_date=date(2024, 6, 1),
                end_date=date(2024, 8, 31),
                impact_level=0.85,
                affected_industries=[
                    "fintech",
                    "banking",
                    "insurance",
                    "investment"
                ],
                key_themes=[
                    "api_integration",
                    "security",
                    "compliance",
                    "data_sharing"
                ],
                content_priorities=[
                    "technical_standards",
                    "security_guidelines",
                    "integration_tutorials",
                    "compliance_guides"
                ],
                recurrence="quarterly",
                historical_metrics={
                    "2023_apis_launched": 150,
                    "2023_integration_rate": 0.75
                },
                preparation_time_days=90
            )
        }
    
    def get_current_events(self) -> List[SeasonalEvent]:
        """Retorna eventos ativos no momento"""
        today = date.today()
        return [
            event for event in self.events.values()
            if event.start_date <= today <= event.end_date
        ]
    
    def get_upcoming_events(self, days_ahead: int = 90) -> List[SeasonalEvent]:
        """Retorna próximos eventos"""
        from datetime import timedelta
        today = date.today()
        future = today + timedelta(days=days_ahead)
        return [
            event for event in self.events.values()
            if today < event.start_date <= future
        ]
    
    def get_preparation_recommendations(self, event: SeasonalEvent) -> Dict:
        """Gera recomendações de preparação para um evento"""
        days_to_event = (event.start_date - date.today()).days
        
        return {
            "preparation_time_left": days_to_event,
            "status": self._get_preparation_status(days_to_event, event.preparation_time_days),
            "priorities": event.content_priorities,
            "key_themes": event.key_themes,
            "historical_reference": event.historical_metrics
        }
    
    def _get_preparation_status(self, days_left: int, required_days: int) -> str:
        """Determina status de preparação"""
        if days_left <= 0:
            return "CRITICAL"
        elif days_left < required_days * 0.3:
            return "URGENT"
        elif days_left < required_days * 0.7:
            return "ON_TRACK"
        else:
            return "PLANNING"
    
    def get_industry_events(self, industry: str) -> List[SeasonalEvent]:
        """Retorna eventos específicos de uma indústria"""
        return [
            event for event in self.events.values()
            if industry in event.affected_industries
        ]
    
    def get_event_impact(self, event_name: str) -> Dict:
        """Calcula impacto de um evento específico"""
        event = self.events.get(event_name)
        if not event:
            return {}
            
        return {
            "impact_level": event.impact_level,
            "affected_industries": event.affected_industries,
            "preparation_required": event.preparation_time_days,
            "historical_performance": event.historical_metrics
        }

if __name__ == "__main__":
    # Exemplo de uso
    manager = SeasonalityManager()
    
    # Verificar eventos atuais
    current = manager.get_current_events()
    print(f"Eventos ativos: {len(current)}")
    
    # Verificar próximos eventos
    upcoming = manager.get_upcoming_events(30)  # próximos 30 dias
    print(f"Eventos próximos: {len(upcoming)}")
    
    # Para cada evento ativo
    for event in current:
        prep = manager.get_preparation_recommendations(event)
        print(f"\nEvento: {event.name}")
        print(f"Status: {prep['status']}")
        print(f"Prioridades: {prep['priorities']}")