# agents\search\config.py
from pydantic import BaseSettings, Field
from typing import Dict, List, Optional

class SearchAgentConfig(BaseSettings):
    """Configurações do Search Agent"""
    
    # Configurações Pinecone
    pinecone_api_key: str = Field(..., env='PINECONE_API_KEY')
    pinecone_environment: str = Field('us-west1-gcp', env='PINECONE_ENV')
    pinecone_index_name: str = Field('ad-agents-index', env='PINECONE_INDEX')
    
    # Configurações HuggingFace
    hf_model_name: str = Field('sentence-transformers/all-mpnet-base-v2', env='HF_MODEL')
    hf_api_key: Optional[str] = Field(None, env='HF_API_KEY')
    
    # Configurações de Cache
    cache_ttl: int = Field(3600, env='CACHE_TTL')  # 1 hora
    max_cache_items: int = Field(1000, env='MAX_CACHE_ITEMS')
    
    # Configurações de Busca
    min_relevance_score: float = Field(0.7, env='MIN_RELEVANCE_SCORE')
    max_results_per_source: int = Field(10, env='MAX_RESULTS_PER_SOURCE')
    default_search_timeout: int = Field(30, env='SEARCH_TIMEOUT')  # segundos
    
    # Configurações de Embeddings
    embedding_model: str = Field('all-mpnet-base-v2', env='EMBEDDING_MODEL')
    embedding_dimension: int = Field(768, env='EMBEDDING_DIM')
    
    # Configurações de Rate Limiting
    rate_limit_searches: int = Field(100, env='RATE_LIMIT_SEARCHES')  # por minuto
    rate_limit_indexing: int = Field(50, env='RATE_LIMIT_INDEXING')  # por minuto
    
    # Fontes de dados confiáveis
    trusted_sources: List[str] = Field(
        default=[
            'github.com',
            'arxiv.org',
            'papers.ssrn.com',
            'scholar.google.com',
            'stackoverflow.com',
            'ieee.org',
            'acm.org'
        ]
    )
    
    # Filtros de conteúdo
    content_filters: Dict[str, List[str]] = Field(
        default={
            'languages': ['en', 'pt-br'],
            'max_age_days': 365,
            'min_words': 100
        }
    )
    
    # Configurações de Retry
    max_retries: int = Field(3, env='MAX_RETRIES')
    retry_delay: int = Field(1, env='RETRY_DELAY')  # segundos
    
    class Config:
        env_prefix = 'SEARCH_'  # Prefixo para variáveis de ambiente
        case_sensitive = False
        
    def get_pinecone_config(self) -> Dict:
        """Retorna configurações formatadas para Pinecone"""
        return {
            "api_key": self.pinecone_api_key,
            "environment": self.pinecone_environment,
            "index_name": self.pinecone_index_name
        }
    
    def get_huggingface_config(self) -> Dict:
        """Retorna configurações formatadas para HuggingFace"""
        return {
            "model_name": self.hf_model_name,
            "api_key": self.hf_api_key,
            "embedding_model": self.embedding_model
        }
    
    def get_cache_config(self) -> Dict:
        """Retorna configurações de cache"""
        return {
            "ttl": self.cache_ttl,
            "max_items": self.max_cache_items
        }
    
    def get_search_config(self) -> Dict:
        """Retorna configurações de busca"""
        return {
            "min_score": self.min_relevance_score,
            "max_results": self.max_results_per_source,
            "timeout": self.default_search_timeout
        }