# agents/search/config.py

from typing import Dict, List, Optional
from pydantic import BaseSettings, Field, HttpUrl

class SearchAgentConfig(BaseSettings):
    """Configurações avançadas para o Search Agent"""
    
    # Configurações de API
    OPENAI_API_KEY: str = Field(..., env='OPENAI_API_KEY')
    
    # Configurações de Embedding
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Vector Store
    VECTOR_STORE_TYPE: str = "faiss"  # faiss ou pinecone
    VECTOR_SIMILARITY_THRESHOLD: float = 0.75
    MAX_RESULTS_PER_QUERY: int = 10
    
    # Cache
    CACHE_TTL: int = 3600  # 1 hora
    MAX_CACHE_ITEMS: int = 10000
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = 60
    MAX_TOKENS_PER_MINUTE: int = 100000
    
    # Content Validation
    MIN_CONFIDENCE_SCORE: float = 0.8
    REQUIRED_SUPPORTING_SOURCES: int = 2
    
    # Notícias e Atualizações
    NEWS_SOURCES: Dict[str, Dict] = {
        "tech_crunch": {
            "name": "TechCrunch",
            "base_url": "https://api.techcrunch.com/v1/",
            "priority": 1,
            "categories": ["technology", "ai", "cloud"]
        },
        "hacker_news": {
            "name": "Hacker News",
            "base_url": "https://hacker-news.firebaseio.com/v0/",
            "priority": 2,
            "categories": ["technology", "programming"]
        },
        "dev_to": {
            "name": "Dev.to",
            "base_url": "https://dev.to/api/",
            "priority": 3,
            "categories": ["development", "programming"]
        }
    }
    
    # Fonte de dados confiáveis
    TRUSTED_DOMAINS: List[str] = [
        "docs.python.org",
        "developer.mozilla.org",
        "kubernetes.io",
        "cloud.google.com",
        "aws.amazon.com",
        "azure.microsoft.com",
        "github.com",
        "stackoverflow.com",
        "arxiv.org",
        "research.google.com",
        "openai.com",
        "pytorch.org",
        "tensorflow.org"
    ]
    
    # Métricas de qualidade
    QUALITY_WEIGHTS: Dict[str, float] = {
        "relevance": 0.4,
        "freshness": 0.2,
        "authority": 0.2,
        "completeness": 0.2
    }
    
    # Parâmetros de processamento
    MAX_CONTENT_LENGTH: int = 100000
    MIN_CONTENT_LENGTH: int = 100
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Configurações de busca
    SEARCH_DEFAULTS: Dict[str, Any] = {
        "min_relevance": 0.6,
        "max_age_days": 365,
        "max_results": 20,
        "include_content": True
    }
    
    # Configurações de análise
    ANALYSIS_OPTIONS: Dict[str, bool] = {
        "extract_code_snippets": True,
        "extract_links": True,
        "analyze_sentiment": False,
        "detect_language": True,
        "generate_summary": True
    }
    
    # Timeouts e tentativas
    REQUEST_TIMEOUT: int = 30  # segundos
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1  # segundos
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PREFIX: str = "search_agent"
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_prefix = "SEARCH_"
        case_sensitive = True

    def get_news_source_config(self, source_id: str) -> Optional[Dict]:
        """Retorna configuração para uma fonte específica"""
        return self.NEWS_SOURCES.get(source_id)
    
    def is_trusted_domain(self, domain: str) -> bool:
        """Verifica se um domínio é confiável"""
        return any(domain.endswith(trusted) for trusted in self.TRUSTED_DOMAINS)
    
    def get_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calcula pontuação de qualidade baseada nos pesos definidos"""
        return sum(
            metrics.get(metric, 0) * weight
            for metric, weight in self.QUALITY_WEIGHTS.items()
        )

# Instância global de configuração
config = SearchAgentConfig()