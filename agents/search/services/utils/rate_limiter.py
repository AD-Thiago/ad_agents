# agents/search/services/news/utils/rate_limiter.py

import asyncio
from datetime import datetime, timedelta
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limiter implementando algoritmo de sliding window
    """
    
    def __init__(self, max_calls: int, period: float):
        """
        Args:
            max_calls: Número máximo de chamadas permitidas no período
            period: Período em segundos
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: List[datetime] = []
        self._lock = asyncio.Lock()
        
    async def acquire(self) -> bool:
        """
        Tenta adquirir uma permissão do rate limiter
        
        Returns:
            bool: True se permitido, False se limite excedido
        """
        async with self._lock:
            now = datetime.now()
            
            # Remover chamadas antigas
            window_start = now - timedelta(seconds=self.period)
            self.calls = [call for call in self.calls if call > window_start]
            
            # Verificar se pode fazer nova chamada
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
                
            # Calcular tempo de espera
            if self.calls:
                next_available = self.calls[0] + timedelta(seconds=self.period)
                wait_time = (next_available - now).total_seconds()
                if wait_time > 0:
                    logger.warning(f"Rate limit exceeded. Wait {wait_time:.2f} seconds")
                    return False
                    
            self.calls.append(now)
            return True
    
    async def __aenter__(self):
        """Suporte para uso com 'async with'"""
        while not await self.acquire():
            await asyncio.sleep(1)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup ao sair do contexto"""
        pass

    def reset(self):
        """Reseta o rate limiter"""
        self.calls.clear()