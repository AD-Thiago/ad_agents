# core\domains\market_segments.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class MarketPriority(str, Enum):
    """Níveis de prioridade para segmentos"""
    CRITICAL = "critical"      # Foco imediato
    HIGH = "high"             # Alta prioridade
    MEDIUM = "medium"         # Prioridade média
    EMERGING = "emerging"     # Mercado emergente
    NICHE = "niche"          # Nicho específico

class Industry(str, Enum):
    """Indústrias principais"""
    FINTECH = "fintech"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    EDUCATION = "education"
    LOGISTICS = "logistics"
    ENERGY = "energy"

class TechMaturity(str, Enum):
    """Nível de maturidade tecnológica"""
    CUTTING_EDGE = "cutting_edge"  # Tecnologia de ponta
    ADVANCED = "advanced"          # Tecnologia avançada
    ESTABLISHED = "established"    # Tecnologia estabelecida
    LEGACY = "legacy"             # Tecnologia legada

class MarketSegment(BaseModel):
    """Definição detalhada de segmento de mercado"""
    name: str
    industry: Industry
    priority: MarketPriority
    market_size: float  # em bilhões USD
    growth_rate: float  # % ao ano
    tech_maturity: TechMaturity
    description: str
    key_technologies: List[str]
    key_players: List[str]
    challenges: List[str]
    opportunities: List[str]
    target_audience: List[str]
    success_metrics: List[str]
    update_frequency: str
    last_updated: datetime = Field(default_factory=datetime.now)

# Definição dos segmentos de mercado
MARKET_SEGMENTS: Dict[str, MarketSegment] = {
    "fintech_banking": MarketSegment(
        name="Digital Banking & Fintech",
        industry=Industry.FINTECH,
        priority=MarketPriority.CRITICAL,
        market_size=475.0,
        growth_rate=23.4,
        tech_maturity=TechMaturity.CUTTING_EDGE,
        description="Serviços financeiros digitais e tecnologias bancárias inovadoras",
        key_technologies=[
            "Blockchain",
            "AI/ML for Risk",
            "Cloud Native",
            "APIs/Open Banking",
            "Microservices"
        ],
        key_players=[
            "Nubank",
            "Stripe",
            "Block (Square)",
            "PicPay",
            "Itaú"
        ],
        challenges=[
            "Regulamentações (PIX, Open Finance)",
            "Segurança e Fraude",
            "Escalabilidade",
            "Integração com Legacy"
        ],
        opportunities=[
            "Banking as a Service",
            "Embedded Finance",
            "AI em Crédito",
            "Pagamentos Instantâneos"
        ],
        target_audience=[
            "Desenvolvedores Fintech",
            "Arquitetos de Sistemas",
            "Product Managers",
            "Tech Leads Bancários"
        ],
        success_metrics=[
            "Uptime do Sistema",
            "Taxa de Conversão",
            "Velocidade de Transação",
            "NPS"
        ],
        update_frequency="daily"
    ),
    
    "healthtech": MarketSegment(
        name="Healthcare Technology",
        industry=Industry.HEALTHCARE,
        priority=MarketPriority.HIGH,
        market_size=280.0,
        growth_rate=18.6,
        tech_maturity=TechMaturity.ADVANCED,
        description="Tecnologias para saúde digital e telemedicina",
        key_technologies=[
            "Telemedicine Platforms",
            "AI Diagnostics",
            "Healthcare APIs",
            "Medical IoT",
            "Cloud Healthcare"
        ],
        key_players=[
            "Hospital Israelita Albert Einstein",
            "Rede D'Or",
            "Hapvida",
            "Dasa"
        ],
        challenges=[
            "Privacidade (LGPD)",
            "Integração de Dados",
            "Regulamentações",
            "Experiência do Paciente"
        ],
        opportunities=[
            "Telemedicina",
            "AI em Diagnóstico",
            "Healthtech SaaS",
            "Medicina Preventiva"
        ],
        target_audience=[
            "Desenvolvedores Healthcare",
            "Gestores Hospitalares",
            "Médicos Tech-Savvy",
            "Healthcare IT"
        ],
        success_metrics=[
            "Precisão Diagnóstica",
            "Tempo de Atendimento",
            "Satisfação do Paciente",
            "Custo por Atendimento"
        ],
        update_frequency="weekly"
    )
}

class MarketSegmentManager:
    """Gerenciador de segmentos de mercado"""
    
    def __init__(self):
        self.segments = MARKET_SEGMENTS
    
    def get_segment(self, segment_id: str) -> Optional[MarketSegment]:
        """Retorna um segmento específico"""
        return self.segments.get(segment_id)
    
    def get_segments_by_priority(self, priority: MarketPriority) -> List[MarketSegment]:
        """Retorna segmentos por nível de prioridade"""
        return [
            segment for segment in self.segments.values()
            if segment.priority == priority
        ]
    
    def get_segments_by_industry(self, industry: Industry) -> List[MarketSegment]:
        """Retorna segmentos por indústria"""
        return [
            segment for segment in self.segments.values()
            if segment.industry == industry
        ]
    
    def get_tech_requirements(self, segment_id: str) -> Dict:
        """Retorna requisitos tecnológicos do segmento"""
        segment = self.get_segment(segment_id)
        if not segment:
            return {}
            
        return {
            "technologies": segment.key_technologies,
            "maturity": segment.tech_maturity,
            "update_frequency": segment.update_frequency
        }
    
    def get_market_metrics(self, segment_id: str) -> Dict:
        """Retorna métricas de mercado do segmento"""
        segment = self.get_segment(segment_id)
        if not segment:
            return {}
            
        return {
            "market_size": segment.market_size,
            "growth_rate": segment.growth_rate,
            "key_players": segment.key_players
        }
    
    def should_update_content(self, segment_id: str) -> bool:
        """Verifica se o conteúdo do segmento precisa ser atualizado"""
        segment = self.get_segment(segment_id)
        if not segment:
            return False
            
        # Calcula tempo desde última atualização
        time_diff = datetime.now() - segment.last_updated
        
        # Define limites baseado na frequência de atualização
        update_limits = {
            "daily": 1,      # 1 dia
            "weekly": 7,     # 7 dias
            "monthly": 30    # 30 dias
        }
        
        days_limit = update_limits.get(segment.update_frequency, 7)
        return time_diff.days >= days_limit