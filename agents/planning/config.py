# agents/planning/config.py
from pydantic_settings import BaseSettings
from pydantic import Field
from core.config import get_settings

class PlanningAgentConfig(BaseSettings):
    """
    Configurações para o Planning Agent.
    Obtém a OPENAI_API_KEY do get_settings().
    """
    openai_api_key: str = Field(default_factory=lambda: get_settings().OPENAI_API_KEY)

    temperature: float = Field(0.7, env="PLANNING_TEMPERATURE")
    max_tokens: int = Field(1500, env="PLANNING_MAX_TOKENS")
    planning_interval: int = Field(120, env="PLANNING_INTERVAL")

    min_trend_score: float = Field(0.5, env="PLANNING_MIN_TREND_SCORE")
    min_relevance_score: float = Field(0.6, env="PLANNING_MIN_RELEVANCE_SCORE")
    min_confidence_score: float = Field(0.7, env="PLANNING_MIN_CONFIDENCE_SCORE")

    cache_ttl: int = Field(1800, env="PLANNING_CACHE_TTL")

    enable_domain_flexibility: bool = Field(True, env="PLANNING_ENABLE_DOMAIN_FLEXIBILITY")
    default_domain_priority: str = Field("medium", env="PLANNING_DEFAULT_DOMAIN_PRIORITY")

    publishing_frequency: str = Field("daily", env="PLANNING_PUBLISHING_FREQUENCY")

    class Config:
        env_prefix = "PLANNING_"

config = PlanningAgentConfig()