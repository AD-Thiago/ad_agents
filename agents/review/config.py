# agents/review/config.py
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Dict, List

class ReviewAgentConfig(BaseSettings):
   """Configurações do Review Agent"""
   
   # Thresholds de qualidade
   min_quality_score: float = Field(0.7, env='REVIEW_MIN_QUALITY_SCORE')
   min_technical_accuracy: float = Field(0.8, env='REVIEW_MIN_TECHNICAL_ACCURACY')
   min_seo_score: float = Field(0.7, env='REVIEW_MIN_SEO_SCORE')
   min_readability_score: float = Field(0.7, env='REVIEW_MIN_READABILITY_SCORE')
   
   # Auto-aprovação
   auto_approve_threshold: float = Field(0.9, env='REVIEW_AUTO_APPROVE_THRESHOLD')
   require_manual_review: bool = Field(False, env='REVIEW_REQUIRE_MANUAL')
   
   # Cache
   cache_ttl: int = Field(3600, env='REVIEW_CACHE_TTL')
   max_cache_items: int = Field(1000, env='REVIEW_MAX_CACHE_ITEMS')
   
   # Limites
   max_content_length: int = Field(10000, env='REVIEW_MAX_CONTENT_LENGTH')
   max_suggestions: int = Field(10, env='REVIEW_MAX_SUGGESTIONS')
   
   review_metrics: Dict[str, float] = {
       "quality": 0.3,
       "technical": 0.3,
       "seo": 0.2,
       "readability": 0.2
   }
   
   required_checks: List[str] = [
       "grammar",
       "technical_accuracy",
       "seo_optimization",
       "readability",
       "code_quality"
   ]

   class Config:
       env_prefix = 'REVIEW_'
       case_sensitive = False