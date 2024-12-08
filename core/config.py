# core/config.py
from pydantic_settings import BaseSettings
from pydantic import Field, AnyHttpUrl, field_validator
from urllib.parse import quote_plus

class Settings(BaseSettings):
    """Configurações globais da aplicação"""

    # API OpenAI
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field("gpt-4-1106", env="OPENAI_MODEL")
    OPENAI_TEMPERATURE: float = Field(0.7, env="OPENAI_TEMPERATURE")

    # Pinecone
    PINECONE_API_KEY: str = Field(..., env="PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = Field(..., env="PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: str = Field(..., env="PINECONE_INDEX_NAME")

    # HuggingFace
    HF_API_KEY: str = Field(..., env="HF_API_KEY")
    HF_MODEL: str = Field("sentence-transformers/all-mpnet-base-v2", env="HF_MODEL")

    # Database
    DB_HOST: str = Field("localhost", env="DB_HOST")
    DB_PORT: int = Field(5432, env="DB_PORT")
    DB_NAME: str = Field(..., env="DB_NAME")
    DB_USER: str = Field(..., env="DB_USER")
    DB_PASSWORD: str = Field(..., env="DB_PASSWORD")

    @property
    def DATABASE_URL(self) -> str:
        """Constrói dinamicamente a URL do banco de dados"""
        return (
            f"postgresql://{quote_plus(self.DB_USER)}:{quote_plus(self.DB_PASSWORD)}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # RabbitMQ
    RABBITMQ_HOST: str = Field("localhost", env="RABBITMQ_HOST")
    RABBITMQ_PORT: int = Field(5672, env="RABBITMQ_PORT")
    RABBITMQ_USER: str = Field("guest", env="RABBITMQ_USER")
    RABBITMQ_PASSWORD: str = Field("guest", env="RABBITMQ_PASSWORD")
    RABBITMQ_USE_SSL: bool = Field(False, env="RABBITMQ_USE_SSL")

    # APIs Externas
    DEV_TO_API_KEY: str = Field(..., env="DEV_TO_API_KEY")
    DEV_TO_API_URL: AnyHttpUrl = Field("https://dev.to/api", env="DEV_TO_API_URL")
    HACKER_NEWS_API_URL: AnyHttpUrl = Field("http://hn.algolia.com/api/v1", env="HACKER_NEWS_API_URL")

    # Workflow
    WORKFLOW_TIMEOUT: int = Field(3600, env="WORKFLOW_TIMEOUT")
    MAX_RETRIES: int = Field(3, env="MAX_RETRIES")
    RETRY_DELAY: int = Field(5, env="RETRY_DELAY")

    # Cache
    CACHE_TTL: int = Field(3600, env="CACHE_TTL")
    MAX_CACHE_SIZE: int = Field(10000, env="MAX_CACHE_SIZE")

    # Planning
    PLANNING_MIN_INSIGHTS: int = Field(5, env="PLANNING_MIN_INSIGHTS")
    PLANNING_MAX_TOPICS: int = Field(3, env="PLANNING_MAX_TOPICS")
    PLANNING_UPDATE_INTERVAL: int = Field(3600, env="PLANNING_UPDATE_INTERVAL")

    # Segurança e Logs
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    DEBUG_MODE: bool = Field(False, env="DEBUG_MODE")

    # Validadores para valores positivos
    @field_validator("CACHE_TTL", "WORKFLOW_TIMEOUT", "RETRY_DELAY")
    def validate_positive(cls, value):
        if value <= 0:
            raise ValueError("Os valores devem ser maiores que 0.")
        return value

def get_settings() -> Settings:
    """Obtém as configurações globais da aplicação"""
    return Settings()