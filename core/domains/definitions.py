# core\domains\definitions.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class DomainType(str, Enum):
    """Tipos de domínios"""
    TECHNICAL = "technical"         # Domínios técnicos
    BUSINESS = "business"           # Domínios de negócio
    INDUSTRY = "industry"           # Domínios de indústria
    EMERGING = "emerging"           # Domínios emergentes
    RESEARCH = "research"           # Domínios de pesquisa

class ContentType(str, Enum):
    """Tipos de conteúdo"""
    TUTORIAL = "tutorial"           # Tutoriais passo-a-passo
    GUIDE = "guide"                # Guias abrangentes
    REFERENCE = "reference"        # Documentação de referência
    CASE_STUDY = "case_study"      # Estudos de caso
    COMPARISON = "comparison"      # Comparativos
    DEEP_DIVE = "deep_dive"        # Análises profundas
    NEWS = "news"                  # Notícias e atualizações

class ExpertiseLevel(str, Enum):
    """Níveis de expertise"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class UpdateFrequency(str, Enum):
    """Frequência de atualização"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class ValidationRule(BaseModel):
    """Regra de validação de conteúdo"""
    name: str
    description: str
    threshold: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    validation_type: str

class DomainDefinition(BaseModel):
    """Definição completa de um domínio"""
    name: str
    type: DomainType
    description: str
    expertise_levels: List[ExpertiseLevel]
    content_types: List[ContentType]
    update_frequency: UpdateFrequency
    primary_keywords: List[str]
    related_keywords: List[str]
    validation_rules: List[ValidationRule]
    required_citations: bool
    requires_code_examples: bool
    requires_diagrams: bool
    last_updated: datetime = Field(default_factory=datetime.now)

# Definições dos domínios principais
DOMAIN_DEFINITIONS = {
    "ai_agents": DomainDefinition(
        name="AI Agents & Automation",
        type=DomainType.TECHNICAL,
        description="Desenvolvimento e implementação de agentes autônomos de IA",
        expertise_levels=[
            ExpertiseLevel.INTERMEDIATE,
            ExpertiseLevel.ADVANCED,
            ExpertiseLevel.EXPERT
        ],
        content_types=[
            ContentType.TUTORIAL,
            ContentType.GUIDE,
            ContentType.DEEP_DIVE,
            ContentType.CASE_STUDY
        ],
        update_frequency=UpdateFrequency.WEEKLY,
        primary_keywords=[
            "ai agents",
            "autonomous systems",
            "agent orchestration",
            "multi-agent systems"
        ],
        related_keywords=[
            "llms",
            "machine learning",
            "neural networks",
            "decision making"
        ],
        validation_rules=[
            ValidationRule(
                name="code_quality",
                description="Qualidade e clareza do código",
                threshold=0.8,
                weight=0.3,
                validation_type="code_analysis"
            ),
            ValidationRule(
                name="technical_accuracy",
                description="Precisão técnica do conteúdo",
                threshold=0.9,
                weight=0.4,
                validation_type="expert_review"
            ),
            ValidationRule(
                name="practical_applicability",
                description="Aplicabilidade prática",
                threshold=0.7,
                weight=0.3,
                validation_type="user_feedback"
            )
        ],
        required_citations=True,
        requires_code_examples=True,
        requires_diagrams=True
    ),

    "mlops": DomainDefinition(
        name="MLOps & Production AI",
        type=DomainType.TECHNICAL,
        description="Práticas e ferramentas para ML em produção",
        expertise_levels=[
            ExpertiseLevel.INTERMEDIATE,
            ExpertiseLevel.ADVANCED
        ],
        content_types=[
            ContentType.GUIDE,
            ContentType.REFERENCE,
            ContentType.CASE_STUDY
        ],
        update_frequency=UpdateFrequency.WEEKLY,
        primary_keywords=[
            "mlops",
            "machine learning",
            "deployment",
            "monitoring"
        ],
        related_keywords=[
            "devops",
            "continuous integration",
            "model registry",
            "feature store"
        ],
        validation_rules=[
            ValidationRule(
                name="deployment_practicality",
                description="Viabilidade de implementação",
                threshold=0.85,
                weight=0.4,
                validation_type="expert_review"
            ),
            ValidationRule(
                name="best_practices",
                description="Aderência às melhores práticas",
                threshold=0.9,
                weight=0.3,
                validation_type="checklist"
            ),
            ValidationRule(
                name="completeness",
                description="Completude da solução",
                threshold=0.8,
                weight=0.3,
                validation_type="coverage_analysis"
            )
        ],
        required_citations=True,
        requires_code_examples=True,
        requires_diagrams=True
    ),

    "enterprise_ai": DomainDefinition(
        name="Enterprise AI Integration",
        type=DomainType.BUSINESS,
        description="Implementação de IA em ambientes empresariais",
        expertise_levels=[
            ExpertiseLevel.INTERMEDIATE,
            ExpertiseLevel.ADVANCED
        ],
        content_types=[
            ContentType.CASE_STUDY,
            ContentType.GUIDE,
            ContentType.COMPARISON
        ],
        update_frequency=UpdateFrequency.MONTHLY,
        primary_keywords=[
            "enterprise ai",
            "business intelligence",
            "digital transformation",
            "ai strategy"
        ],
        related_keywords=[
            "roi",
            "change management",
            "integration",
            "scalability"
        ],
        validation_rules=[
            ValidationRule(
                name="business_impact",
                description="Impacto nos negócios",
                threshold=0.8,
                weight=0.4,
                validation_type="roi_analysis"
            ),
            ValidationRule(
                name="implementation_feasibility",
                description="Viabilidade de implementação",
                threshold=0.85,
                weight=0.3,
                validation_type="expert_review"
            ),
            ValidationRule(
                name="enterprise_readiness",
                description="Prontidão para empresa",
                threshold=0.9,
                weight=0.3,
                validation_type="checklist"
            )
        ],
        required_citations=True,
        requires_code_examples=False,
        requires_diagrams=True
    )
}

class DomainManager:
    """Gerenciador central de domínios"""
    
    def __init__(self):
        self.domains = DOMAIN_DEFINITIONS
    
    def get_domain(self, domain_id: str) -> Optional[DomainDefinition]:
        """Retorna definição de um domínio"""
        return self.domains.get(domain_id.lower())
    
    def get_domains_by_type(self, domain_type: DomainType) -> List[DomainDefinition]:
        """Retorna domínios por tipo"""
        return [
            domain for domain in self.domains.values()
            if domain.type == domain_type
        ]
    
    def get_validation_rules(self, domain_id: str) -> List[ValidationRule]:
        """Retorna regras de validação de um domínio"""
        domain = self.get_domain(domain_id)
        return domain.validation_rules if domain else []
    
    def validate_content_requirements(
        self,
        domain_id: str,
        content: Dict
    ) -> Dict[str, bool]:
        """Valida requisitos de conteúdo para um domínio"""
        domain = self.get_domain(domain_id)
        if not domain:
            return {}
            
        return {
            "has_citations": self._check_citations(content) if domain.required_citations else True,
            "has_code": self._check_code_examples(content) if domain.requires_code_examples else True,
            "has_diagrams": self._check_diagrams(content) if domain.requires_diagrams else True
        }
    
    def _check_citations(self, content: Dict) -> bool:
        """Verifica se o conteúdo tem citações adequadas"""
        # Implementar verificação de citações
        return True
    
    def _check_code_examples(self, content: Dict) -> bool:
        """Verifica se o conteúdo tem exemplos de código"""
        # Implementar verificação de código
        return True
    
    def _check_diagrams(self, content: Dict) -> bool:
        """Verifica se o conteúdo tem diagramas necessários"""
        # Implementar verificação de diagramas
        return True
    
    def get_content_guidelines(self, domain_id: str) -> Dict:
        """Retorna diretrizes de conteúdo para um domínio"""
        domain = self.get_domain(domain_id)
        if not domain:
            return {}
            
        return {
            "expertise_levels": domain.expertise_levels,
            "content_types": domain.content_types,
            "primary_keywords": domain.primary_keywords,
            "update_frequency": domain.update_frequency,
            "requirements": {
                "citations": domain.required_citations,
                "code_examples": domain.requires_code_examples,
                "diagrams": domain.requires_diagrams
            }
        }

if __name__ == "__main__":
    # Exemplo de uso
    manager = DomainManager()
    
    # Verificar domínio de AI Agents
    ai_domain = manager.get_domain("ai_agents")
    if ai_domain:
        print(f"Domain: {ai_domain.name}")
        print(f"Type: {ai_domain.type}")
        print(f"Keywords: {ai_domain.primary_keywords}")
    
    # Obter guidelines
    guidelines = manager.get_content_guidelines("ai_agents")
    print("\nContent Guidelines:")
    print(f"Expertise Levels: {guidelines['expertise_levels']}")
    print(f"Content Types: {guidelines['content_types']}")
    
    # Verificar regras de validação
    rules = manager.get_validation_rules("ai_agents")
    print("\nValidation Rules:")
    for rule in rules:
        print(f"- {rule.name}: {rule.threshold} (weight: {rule.weight})")