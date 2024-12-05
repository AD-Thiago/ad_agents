# agents/search/services/news/metrics.py

from prometheus_client import Counter, Histogram, Gauge
from prometheus_client.registry import CollectorRegistry
import time

class NewsMetrics:
    """Sistema de métricas para o serviço de notícias"""
    
    def __init__(self):
        # Criar registry separado para evitar colisões
        self.registry = CollectorRegistry()
        
        # Contadores
        self.request_count = Counter(
            "news_integration_requests",
            "Total number of API requests",
            ["source", "status"],
            registry=self.registry
        )
        
        self.article_count = Counter(
            "news_integration_articles",
            "Total number of articles processed",
            ["source"],
            registry=self.registry
        )
        
        # Histogramas
        self.request_latency = Histogram(
            "news_integration_request_duration",
            "Request duration in seconds",
            ["source"],
            registry=self.registry
        )
        
        # Gauges
        self.active_requests = Gauge(
            "news_integration_active_requests",
            "Number of active requests",
            ["source"],
            registry=self.registry
        )

    def track_request(self, source: str):
        """Context manager para rastrear requisições"""
        class RequestTracker:
            def __init__(self, metrics, source):
                self.metrics = metrics
                self.source = source
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.time()
                self.metrics.active_requests.labels(source=self.source).inc()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.metrics.request_latency.labels(source=self.source).observe(duration)
                self.metrics.active_requests.labels(source=self.source).dec()
                
                status = "error" if exc_type else "success"
                self.metrics.request_count.labels(source=self.source, status=status).inc()
                
        return RequestTracker(self, source)

    def record_processed_article(self, source: str):
        """Registra um artigo processado"""
        self.article_count.labels(source=source).inc()

    def get_metrics(self):
        """Retorna métricas atuais"""
        metrics = {
            "requests": {},
            "articles": {},
            "active_requests": {}
        }

        # Obter métricas de requisições para cada fonte
        for source in ["tech_crunch", "hacker_news", "dev.to"]:
            metrics["requests"][source] = {
                "success": self.request_count.labels(source=source, status="success")._value.get(),
                "error": self.request_count.labels(source=source, status="error")._value.get()
            }
            metrics["articles"][source] = self.article_count.labels(source=source)._value.get()
            metrics["active_requests"][source] = self.active_requests.labels(source=source)._value

        return metrics