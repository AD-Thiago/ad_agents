# core\domains\adoption_metrics.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, date
from enum import Enum

class AdoptionStage(str, Enum):
    """Estágios de adoção de tecnologia"""
    INNOVATORS = "innovators"         # 2.5% iniciais
    EARLY_ADOPTERS = "early_adopters" # 13.5% seguintes
    EARLY_MAJORITY = "early_majority" # 34% primeiros da maioria
    LATE_MAJORITY = "late_majority"   # 34% últimos da maioria
    LAGGARDS = "laggards"             # 16% finais

class AdoptionMetric(BaseModel):
    """Métricas de adoção de tecnologia"""
    technology: str
    category: str
    current_stage: AdoptionStage
    adoption_rate: float = Field(ge=0.0, le=1.0)
    growth_rate: float  # % de crescimento mensal
    market_penetration: float = Field(ge=0.0, le=1.0)
    enterprise_adoption: float = Field(ge=0.0, le=1.0)
    startup_adoption: float = Field(ge=0.0, le=1.0)
    developer_satisfaction: float = Field(ge=0.0, le=1.0)
    time_to_value: int  # dias
    measured_at: datetime = Field(default_factory=datetime.now)

class ROIMetric(BaseModel):
    """Métricas de ROI"""
    cost_reduction: float
    productivity_gain: float
    time_saved: float  # horas por semana
    quality_improvement: float
    user_satisfaction: float = Field(ge=0.0, le=1.0)
    implementation_cost: float
    maintenance_cost: float
    training_time: int  # horas

class UseCaseMetric(BaseModel):
    """Métricas específicas de casos de uso"""
    use_case: str
    success_rate: float = Field(ge=0.0, le=1.0)
    implementation_time: int  # dias
    roi_metrics: ROIMetric
    adoption_barriers: List[str]
    best_practices: List[str]
    common_pitfalls: List[str]

# Métricas de adoção atuais
CURRENT_ADOPTION_METRICS = {
    "llms": AdoptionMetric(
        technology="Large Language Models",
        category="ai",
        current_stage=AdoptionStage.EARLY_MAJORITY,
        adoption_rate=0.34,
        growth_rate=15.0,  # 15% ao mês
        market_penetration=0.45,
        enterprise_adoption=0.38,
        startup_adoption=0.65,
        developer_satisfaction=0.85,
        time_to_value=14,  # 2 semanas
    ),
    
    "vector_databases": AdoptionMetric(
        technology="Vector Databases",
        category="data",
        current_stage=AdoptionStage.EARLY_ADOPTERS,
        adoption_rate=0.15,
        growth_rate=25.0,
        market_penetration=0.20,
        enterprise_adoption=0.15,
        startup_adoption=0.40,
        developer_satisfaction=0.80,
        time_to_value=7
    ),
    
    "ai_agents": AdoptionMetric(
        technology="AI Agents",
        category="ai",
        current_stage=AdoptionStage.INNOVATORS,
        adoption_rate=0.05,
        growth_rate=45.0,
        market_penetration=0.08,
        enterprise_adoption=0.03,
        startup_adoption=0.12,
        developer_satisfaction=0.75,
        time_to_value=21
    )
}

# Métricas de casos de uso
USE_CASE_METRICS = {
    "customer_service": UseCaseMetric(
        use_case="AI Customer Service",
        success_rate=0.82,
        implementation_time=30,
        roi_metrics=ROIMetric(
            cost_reduction=0.35,  # 35% redução de custos
            productivity_gain=0.45,
            time_saved=15.0,  # 15 horas por semana
            quality_improvement=0.25,
            user_satisfaction=0.78,
            implementation_cost=50000,  # USD
            maintenance_cost=2000,  # USD por mês
            training_time=20  # horas
        ),
        adoption_barriers=[
            "Integration complexity",
            "Data privacy concerns",
            "Staff training",
            "Cost justification"
        ],
        best_practices=[
            "Start with simple use cases",
            "Comprehensive training",
            "Clear escalation paths",
            "Regular performance monitoring"
        ],
        common_pitfalls=[
            "Over-automation",
            "Insufficient testing",
            "Poor integration",
            "Lack of monitoring"
        ]
    ),
    
    "code_assistance": UseCaseMetric(
        use_case="AI Code Assistant",
        success_rate=0.88,
        implementation_time=7,
        roi_metrics=ROIMetric(
            cost_reduction=0.20,
            productivity_gain=0.40,
            time_saved=10.0,
            quality_improvement=0.30,
            user_satisfaction=0.85,
            implementation_cost=15000,
            maintenance_cost=500,
            training_time=8
        ),
        adoption_barriers=[
            "Security concerns",
            "Code quality trust",
            "Learning curve",
            "Integration with existing tools"
        ],
        best_practices=[
            "Code review protocols",
            "Security guidelines",
            "Training programs",
            "Measurement frameworks"
        ],
        common_pitfalls=[
            "Blind trust in suggestions",
            "Inadequate review",
            "Security oversights",
            "Poor prompt practices"
        ]
    )
}

class AdoptionMetricsManager:
    """Gerenciador de métricas de adoção"""
    
    def __init__(self):
        self.metrics = CURRENT_ADOPTION_METRICS
        self.use_cases = USE_CASE_METRICS
    
    def get_metric(self, technology: str) -> Optional[AdoptionMetric]:
        """Retorna métricas para uma tecnologia"""
        return self.metrics.get(technology.lower())
    
    def get_use_case_metrics(self, use_case: str) -> Optional[UseCaseMetric]:
        """Retorna métricas para um caso de uso"""
        return self.use_cases.get(use_case.lower())
    
    def get_trending_technologies(self) -> List[str]:
        """Retorna tecnologias com maior crescimento"""
        return sorted(
            self.metrics.keys(),
            key=lambda x: self.metrics[x].growth_rate,
            reverse=True
        )
    
    def calculate_roi_projection(self, use_case: str, timeframe_months: int) -> Dict:
        """Calcula projeção de ROI para um caso de uso"""
        metrics = self.get_use_case_metrics(use_case)
        if not metrics:
            return {}
            
        roi = metrics.roi_metrics
        monthly_savings = (
            roi.cost_reduction * roi.implementation_cost +
            (roi.productivity_gain * 160 * 50)  # 160h/mês * $50/h
        )
        
        total_cost = (
            roi.implementation_cost +
            (roi.maintenance_cost * timeframe_months)
        )
        
        total_savings = monthly_savings * timeframe_months
        net_roi = (total_savings - total_cost) / total_cost
        
        return {
            "timeframe_months": timeframe_months,
            "total_cost": total_cost,
            "total_savings": total_savings,
            "net_roi": net_roi,
            "break_even_months": roi.implementation_cost / monthly_savings
        }
    
    def get_adoption_recommendations(self, technology: str) -> Dict:
        """Gera recomendações de adoção para uma tecnologia"""
        metric = self.get_metric(technology)
        if not metric:
            return {}
            
        recommendations = {
            "readiness_score": self._calculate_readiness_score(metric),
            "suggested_approach": self._get_suggested_approach(metric),
            "risk_level": self._calculate_risk_level(metric),
            "next_steps": self._get_next_steps(metric)
        }
        
        return recommendations
    
    def _calculate_readiness_score(self, metric: AdoptionMetric) -> float:
        """Calcula score de prontidão para adoção"""
        weights = {
            "market_penetration": 0.3,
            "developer_satisfaction": 0.3,
            "enterprise_adoption": 0.2,
            "time_to_value": 0.2
        }
        
        # Normaliza time_to_value (menor é melhor)
        time_score = 1 - (metric.time_to_value / 90)  # normaliza para 90 dias
        
        score = (
            metric.market_penetration * weights["market_penetration"] +
            metric.developer_satisfaction * weights["developer_satisfaction"] +
            metric.enterprise_adoption * weights["enterprise_adoption"] +
            time_score * weights["time_to_value"]
        )
        
        return min(max(score, 0.0), 1.0)
    
    def _get_suggested_approach(self, metric: AdoptionMetric) -> str:
        """Define abordagem sugerida baseada no estágio de adoção"""
        approaches = {
            AdoptionStage.INNOVATORS: "Experimental pilot with close monitoring",
            AdoptionStage.EARLY_ADOPTERS: "Limited production deployment",
            AdoptionStage.EARLY_MAJORITY: "Full production rollout",
            AdoptionStage.LATE_MAJORITY: "Standardized implementation",
            AdoptionStage.LAGGARDS: "Conservative adoption with proven patterns"
        }
        return approaches[metric.current_stage]
    
    def _calculate_risk_level(self, metric: AdoptionMetric) -> str:
        """Calcula nível de risco baseado nas métricas"""
        risk_score = (
            (1 - metric.market_penetration) * 0.4 +
            (1 - metric.enterprise_adoption) * 0.3 +
            (1 - metric.developer_satisfaction) * 0.3
        )
        
        if risk_score > 0.7:
            return "HIGH"
        elif risk_score > 0.4:
            return "MEDIUM"
        return "LOW"
    
    def _get_next_steps(self, metric: AdoptionMetric) -> List[str]:
        """Define próximos passos baseado no estágio atual"""
        if metric.current_stage == AdoptionStage.INNOVATORS:
            return [
                "Start with proof of concept",
                "Focus on learning and exploration",
                "Build internal expertise",
                "Prepare risk mitigation strategies"
            ]
        elif metric.current_stage == AdoptionStage.EARLY_ADOPTERS:
            return [
                "Plan limited production deployment",
                "Document best practices",
                "Establish monitoring",
                "Create training programs"
            ]
        else:
            return [
                "Scale deployment",
                "Optimize processes",
                "Share knowledge",
                "Measure ROI"
            ]

if __name__ == "__main__":
    # Exemplo de uso
    manager = AdoptionMetricsManager()
    
    # Análise de LLMs
    llm_metrics = manager.get_metric("llms")
    if llm_metrics:
        print(f"LLM Adoption Rate: {llm_metrics.adoption_rate * 100}%")
        print(f"Growth Rate: {llm_metrics.growth_rate}% monthly")
    
    # ROI de caso de uso
    roi_projection = manager.calculate_roi_projection("customer_service", 12)
    print(f"\nCustomer Service ROI (12 months):")
    print(f"Net ROI: {roi_projection['net_roi']:.2f}")
    print(f"Break Even: {roi_projection['break_even_months']:.1f} months")
    
    # Recomendações
    recommendations = manager.get_adoption_recommendations("ai_agents")
    print(f"\nAI Agents Recommendations:")
    print(f"Readiness Score: {recommendations['readiness_score']:.2f}")
    print(f"Risk Level: {recommendations['risk_level']}")
    print(f"Approach: {recommendations['suggested_approach']}")