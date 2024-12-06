# File: core/config.py

from pydantic import BaseSettings, Field
from typing import Dict, Optional
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente do .env
load_dotenv()

class Settings(BaseSettings):
    """Configurações globais da aplicação"""

    # API
    OPENAI_API_KEY: str = Field(..., env='OPENAI_API_KEY')
    OPENAI_MODEL: str = Field('gpt-4-1106', env='OPENAI_MODEL')
    OPENAI_TEMPERATURE: float = Field(0.7, env='OPENAI_TEMPERATURE')

    # RabbitMQ
    RABBITMQ_HOST: str = Field('localhost', env='RABBITMQ_HOST')
    RABBITMQ_PORT: int = Field(5672, env='RABBITMQ_PORT')
    RABBITMQ_USER: str = Field('guest', env='RABBITMQ_USER')
    RABBITMQ_PASSWORD: str = Field('guest', env='RABBITMQ_PASSWORD')

    # Métricas
    PROMETHEUS_HOST: str = Field('localhost', env='PROMETHEUS_HOST')
    PROMETHEUS_PORT: int = Field(9090, env='PROMETHEUS_PORT')

    # APIs Externas
    HACKER_NEWS_API_URL: str = Field('http://hn.algolia.com/api/v1', env='HACKER_NEWS_API_URL')
    DEV_TO_API_KEY: str = Field('', env='DEV_TO_API_KEY')
    DEV_TO_API_URL: str = Field('https://dev.to/api', env='DEV_TO_API_URL')

    # Workflow
    WORKFLOW_TIMEOUT: int = Field(3600, env='WORKFLOW_TIMEOUT')  # 1 hora
    MAX_RETRIES: int = Field(3, env='MAX_RETRIES')
    RETRY_DELAY: int = Field(5, env='RETRY_DELAY')  # segundos

    # Cache
    CACHE_TTL: int = Field(3600, env='CACHE_TTL')  # 1 hora
    MAX_CACHE_SIZE: int = Field(1000, env='MAX_CACHE_SIZE')

    # Domínios
    DOMAIN_CONFIGS: Dict[str, Dict] = Field({
        'technology': {
            'name': 'Technology',
            'content_guidelines': "Conteúdo técnico, atualizado e com aplicabilidade prática.",
            'priority': 'high'
        },
        'business': {
            'name': 'Business',
            'content_guidelines': "Conteúdo focado em estratégia, finanças e gestão empresarial.",
            'priority': 'medium'
        },
        'lifestyle': {
            'name': 'Lifestyle',
            'content_guidelines': "Conteúdo sobre bem-estar, saúde, hobbies e desenvolvimento pessoal.",
            'priority': 'low'
        }
    }, env='DOMAIN_CONFIGS')

    # Validação
    VALIDATION_THRESHOLDS: Dict[str, float] = Field({
        'content_quality': 0.8,
        'relevance': 0.7,
        'plagiarism': 0.9,
        'technical_accuracy': 0.85
    }, env='VALIDATION_THRESHOLDS')

    # Planning
    PLANNING_MIN_INSIGHTS: int = Field(5, env='PLANNING_MIN_INSIGHTS')
    PLANNING_MAX_TOPICS: int = Field(3, env='PLANNING_MAX_TOPICS')
    PLANNING_UPDATE_INTERVAL: int = Field(3600, env='PLANNING_UPDATE_INTERVAL')  # 1 hora

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

def get_settings() -> Settings:
    """Obtém as configurações globais da aplicação"""
    return Settings()