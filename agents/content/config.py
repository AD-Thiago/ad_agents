# agents\content\config.py
from pydantic import BaseSettings
from typing import Dict, List

class ContentAgentConfig(BaseSettings):
    """Configurações do Content Agent"""
    
    # Parâmetros de geração
    max_tokens: int = 2000
    temperature: float = 0.7
    
    # Templates padrão
    default_templates: Dict[str, Dict] = {
        "blog_post": {
            "sections": ["intro", "main", "conclusion"],
            "tone": "professional",
            "min_words": 1000
        },
        "social_media": {
            "sections": ["header", "body", "cta"],
            "tone": "engaging",
            "min_words": 100
        },
        "technical_article": {
            "sections": ["abstract", "intro", "methods", "results", "conclusion"],
            "tone": "academic",
            "min_words": 2000
        }
    }
    
    # SEO
    min_keyword_density: float = 0.01
    max_keyword_density: float = 0.03
    min_heading_count: int = 3
    
    # Qualidade
    min_readability_score: float = 0.7
    min_seo_score: float = 0.8
    
    # Rate limits
    max_generations_per_minute: int = 10
    
    # Cache
    cache_ttl: int = 3600  # 1 hora
    
    class Config:
        env_prefix = "CONTENT_"