from typing import Dict, Optional
from pydantic import BaseSettings, Field
from functools import lru_cache
import os
from enum import Enum

class Environment(str, Enum):
    """Ambientes de execução"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Níveis de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class APIConfig(BaseSettings):
    """Configurações de APIs externas"""
    # OpenAI
    openai_api_key: str = Field(..., env='OPENAI_API_KEY')
    openai_model: str = Field("gpt-4-1106-preview", env='OPENAI_MODEL')
    openai_temperature: float = Field(0.7, env='OPENAI_TEMPERATURE')
    
    # Pinecone
    pinecone_api_key: str = Field(..., env='PINECONE_API_KEY')
    pinecone_environment: str = Field('gcp-starter', env='PINECONE_ENVIRONMENT')
    pinecone_index_name: str = Field('adagents', env='PINECONE_INDEX_NAME')
    
    # HuggingFace
    hf_api_key: Optional[str] = Field(None, env='HF_API_KEY')
    hf_model: str = Field('sentence-transformers/all-mpnet-base-v2', env='HF_MODEL')

class AgentConfig(BaseSettings):
    """Configurações dos agentes"""
    # Planning Agent
    planning_cache_ttl: int = Field(3600, env='PLANNING_CACHE_TTL')
    planning_max_retries: int = Field(3, env='PLANNING_MAX_RETRIES')
    
    # Search Agent
    search_cache_ttl: int = Field(3600, env='SEARCH_CACHE_TTL')
    search_max_results: int = Field(10, env='SEARCH_MAX_RESULTS')
    search_min_relevance: float = Field(0.7, env='SEARCH_MIN_RELEVANCE')
    
    # Content Agent
    content_max_length: int = Field(4000, env='CONTENT_MAX_LENGTH')
    content_min_length: int = Field(500, env='CONTENT_MIN_LENGTH')
    content_quality_threshold: float = Field(0.8, env='CONTENT_QUALITY_THRESHOLD')
    
    # Review Agent
    review_auto_approve_threshold: float = Field(0.9, env='REVIEW_AUTO_APPROVE_THRESHOLD')
    review_min_quality_score: float = Field(0.7, env='REVIEW_MIN_QUALITY_SCORE')

class DatabaseConfig(BaseSettings):
    """Configurações de banco de dados"""
    db_host: str = Field('localhost', env='DB_HOST')
    db_port: int = Field(5432, env='DB_PORT')
    db_name: str = Field('adagents', env='DB_NAME')
    db_user: str = Field('postgres', env='DB_USER')
    db_password: Optional[str] = Field(None, env='DB_PASSWORD')
    
    @property
    def database_url(self) -> str:
        """Retorna URL de conexão com o banco"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

class CacheConfig(BaseSettings):
    """Configurações de cache"""
    redis_host: str = Field('localhost', env='REDIS_HOST')
    redis_port: int = Field(6379, env='REDIS_PORT')
    redis_db: int = Field(0, env='REDIS_DB')
    redis_password: Optional[str] = Field(None, env='REDIS_PASSWORD')
    default_ttl: int = Field(3600, env='CACHE_DEFAULT_TTL')

class RateLimitConfig(BaseSettings):
    """Configurações de rate limiting"""
    max_requests_per_minute: int = Field(60, env='RATE_LIMIT_RPM')
    max_tokens_per_minute: int = Field(10000, env='RATE_LIMIT_TOKENS')
    max_parallel_requests: int = Field(5, env='RATE_LIMIT_PARALLEL')

class GlobalConfig(BaseSettings):
    """Configurações globais do sistema"""
    environment: Environment = Field(Environment.DEVELOPMENT, env='ENVIRONMENT')
    debug: bool = Field(True, env='DEBUG')
    log_level: LogLevel = Field(LogLevel.INFO, env='LOG_LEVEL')
    
    # Sub-configurações
    api: APIConfig = APIConfig()
    agents: AgentConfig = AgentConfig()
    database: DatabaseConfig = DatabaseConfig()
    cache: CacheConfig = CacheConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    
    # Timeouts e limites
    request_timeout: int = Field(30, env='REQUEST_TIMEOUT')
    max_retries: int = Field(3, env='MAX_RETRIES')
    batch_size: int = Field(100, env='BATCH_SIZE')
    
    # Paths
    base_path: str = Field(os.path.dirname(os.path.dirname(__file__)))
    data_path: str = Field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
    log_path: str = Field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'))
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> GlobalConfig:
    """Retorna instância cacheada das configurações"""
    return GlobalConfig()

def initialize_directories() -> None:
    """Inicializa diretórios necessários"""
    config = get_settings()
    os.makedirs(config.data_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)

def get_environment() -> Environment:
    """Retorna ambiente atual"""
    return get_settings().environment

def is_development() -> bool:
    """Verifica se está em ambiente de desenvolvimento"""
    return get_environment() == Environment.DEVELOPMENT

def is_production() -> bool:
    """Verifica se está em ambiente de produção"""
    return get_environment() == Environment.PRODUCTION

if __name__ == "__main__":
    # Exemplo de uso
    settings = get_settings()
    
    print(f"Environment: {settings.environment}")
    print(f"Debug Mode: {settings.debug}")
    print(f"Log Level: {settings.log_level}")
    
    print("\nAPI Settings:")
    print(f"OpenAI Model: {settings.api.openai_model}")
    print(f"Pinecone Environment: {settings.api.pinecone_environment}")
    
    print("\nAgent Settings:")
    print(f"Search Cache TTL: {settings.agents.search_cache_ttl}")
    print(f"Content Quality Threshold: {settings.agents.content_quality_threshold}")
    
    print("\nRate Limits:")
    print(f"Max RPM: {settings.rate_limit.max_requests_per_minute}")
    print(f"Max Parallel: {settings.rate_limit.max_parallel_requests}")
    
    # Inicializa diretórios
    initialize_directories()
    print(f"\nData Path: {settings.data_path}")
    print(f"Log Path: {settings.log_path}")