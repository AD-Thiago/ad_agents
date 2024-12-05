# agents/search/services/news/cache.py

from typing import Dict, Optional, Any
import time
from datetime import datetime
import json
from .models import NewsArticle
from .config import NewsApiConfig

class NewsCache:
    """Gerenciador de cache para requisições de notícias"""
    
    def __init__(self, config: NewsApiConfig):
        self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_times: Dict[str, float] = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Recupera item do cache"""
        if key not in self._cache:
            return None
            
        # Verificar TTL
        if time.time() - self._cache_times[key] > self.config.CACHE_TTL:
            self._remove(key)
            return None
            
        return self._cache[key]
        
    def set(self, key: str, value: Any) -> None:
        """Armazena item no cache"""
        # Limpar cache se necessário
        if len(self._cache) >= self.config.MAX_CACHE_ITEMS:
            self._cleanup_cache()
            
        self._cache[key] = value
        self._cache_times[key] = time.time()
        
    def _remove(self, key: str) -> None:
        """Remove item do cache"""
        if key in self._cache:
            del self._cache[key]
        if key in self._cache_times:
            del self._cache_times[key]
            
    def _cleanup_cache(self) -> None:
        """Limpa itens expirados do cache"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_times.items()
            if current_time - timestamp > self.config.CACHE_TTL
        ]
        
        for key in expired_keys:
            self._remove(key)
            
        # Se ainda precisar de espaço, remove os itens mais antigos
        if len(self._cache) >= self.config.MAX_CACHE_ITEMS:
            sorted_keys = sorted(
                self._cache_times.items(),
                key=lambda x: x[1]
            )
            
            # Remove 20% dos itens mais antigos
            num_to_remove = len(sorted_keys) // 5
            for key, _ in sorted_keys[:num_to_remove]:
                self._remove(key)
                
    def get_cache_key(self, query_params: Dict[str, Any]) -> str:
        """Gera chave de cache para parâmetros de consulta"""
        # Ordenar parâmetros para garantir consistência
        sorted_params = sorted(query_params.items())
        return json.dumps(sorted_params, default=str)
        
    def clear(self) -> None:
        """Limpa todo o cache"""
        self._cache.clear()
        self._cache_times.clear()
        
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        current_time = time.time()
        return {
            "total_items": len(self._cache),
            "expired_items": sum(
                1 for timestamp in self._cache_times.values()
                if current_time - timestamp > self.config.CACHE_TTL
            ),
            "cache_size_percent": (len(self._cache) / self.config.MAX_CACHE_ITEMS) * 100,
            "oldest_item_age": max(
                current_time - min(self._cache_times.values())
                if self._cache_times else 0,
                0
            )
        }