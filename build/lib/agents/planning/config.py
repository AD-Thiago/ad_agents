# agents/planning/config.py
from core.config import get_settings
from pydantic import BaseSettings

class PlanningAgentConfig(BaseSettings):
    """
    Configurações para o Planning Agent.
    Estas definições incluem integrações, parâmetros de geração e limites de métricas.
    """

    # OpenAI
    settings = get_settings()
    openai_api_key: str = settings.api.openai_api_key

    # Parâmetros do agente
    temperature: float = 0.7  # Criatividade do modelo
    max_tokens: int = 1500  # Limite de tokens para geração de conteúdo
    planning_interval: int = 120  # Intervalo de planejamento (em segundos)

    # Limiares para métricas e relevância
    min_trend_score: float = 0.5  # Score mínimo para considerar uma tendência
    min_relevance_score: float = 0.6  # Relevância mínima para incluir no plano
    min_confidence_score: float = 0.7  # Confiança mínima para publicar

    # Configurações de cache
    cache_ttl: int = 1800  # Tempo de vida do cache (em segundos)

    # Domínios
    enable_domain_flexibility: bool = True  # Permite criar planos fora de domínios pré-definidos
    default_domain_priority: str = "medium"  # Prioridade padrão para planos fora de domínios

    # Configurações de publicação
    publishing_frequency: str = "daily"  # Frequência de publicação (daily, weekly, monthly)

    class Config:
        env_prefix = "PLANNING_"  # Prefixo para variáveis de ambiente


# Instância global de configuração para facilitar o uso
config = PlanningAgentConfig()
