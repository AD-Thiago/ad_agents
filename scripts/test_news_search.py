# scripts/test_news_search.py

import asyncio
from datetime import datetime, timedelta
import logging
from agents.search.services.news.service import NewsIntegrationService
from agents.search.services.news.models import NewsSearchQuery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_search():
    # Inicializar o serviço
    service = NewsIntegrationService()
    await service.initialize()
    
    try:
        # Criar uma consulta de teste
        query = NewsSearchQuery(
            topic="Python async",
            keywords=["programming", "async"],
            start_date=datetime.now() - timedelta(days=7),
            min_relevance=0.3,
            max_results=5
        )
        
        print(f"\nBuscando artigos sobre: {query.topic}")
        print("=" * 50)
        
        # Realizar a busca
        results = await service.search_news(query)
        
        # Exibir resultados
        print(f"\nEncontrados {len(results)} artigos:\n")
        
        for i, article in enumerate(results, 1):
            print(f"\nArtigo {i}:")
            print(f"Título: {article.title}")
            print(f"Autor: {article.author or 'Não especificado'}")
            print(f"Data: {article.published_date.strftime('%d/%m/%Y %H:%M')}")
            print(f"URL: {article.url}")
            print(f"Tags: {', '.join(article.tags)}")
            if article.metadata:
                print("Metadados:")
                print(f"  Tempo de leitura: {article.metadata.get('reading_time', 'N/A')} minutos")
                print(f"  Comentários: {article.metadata.get('comments_count', 'N/A')}")
                print(f"  Reações: {article.metadata.get('reactions_count', 'N/A')}")
            print("-" * 50)
        
        # Exibir métricas
        metrics = service.metrics.get_metrics()
        print("\nMétricas:")
        for source, data in metrics["requests"].items():
            print(f"\n{source.upper()}:")
            print(f"  Requisições com sucesso: {data['success']}")
            print(f"  Requisições com erro: {data['error']}")
            print(f"  Artigos processados: {metrics['articles'].get(source, 0)}")
            print(f"  Requisições ativas: {metrics['active_requests'].get(source, 0)}")
        
    except Exception as e:
        logger.error(f"Erro durante a busca: {str(e)}")
        raise
    finally:
        # Fechar o serviço
        await service.close()

if __name__ == "__main__":
    try:
        asyncio.run(test_search())
    except KeyboardInterrupt:
        print("\nOperação interrompida pelo usuário")
    except Exception as e:
        logger.error(f"Erro não tratado: {str(e)}")
        raise