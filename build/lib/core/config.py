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

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

def get_settings() -> Settings:
    """Obtém as configurações globais da aplicação"""
    return Settings()