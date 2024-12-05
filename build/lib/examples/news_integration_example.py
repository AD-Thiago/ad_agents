# examples/news_integration_example.py

import asyncio
from datetime import datetime, timedelta
from agents.search.services.news.service import NewsIntegrationService
from agents.search.services.news.models import NewsSearchQuery
import json

async def main():
    # Inicializar serviço
    news_service = NewsIntegrationService()
    await news_service.initialize()

    try:
        # Criar consulta de exemplo
        query = NewsSearchQuery(
            topic="AI and Machine Learning",
            keywords=["artificial intelligence", "deep learning", "neural networks"],
            categories=["technology", "ai"],
            start_date=datetime.now() - timedelta(days=7),
            max_results=5,
            min_relevance=0.7,
            language="en"
        )

        print(f"\nBuscando notícias sobre: {query.topic}")
        print("=" * 50)

        # Realizar busca
        articles = await news_service.search_news(query)

        print(f"\nEncontrados {len(articles)} artigos relevantes:\n")

        # Exibir resultados
        for i, article in enumerate(articles, 1):
            print(f"\nArtigo {i}:")
            print(f"Título: {article.title}")
            print(f"Fonte: {article.source}")
            print(f"Data: {article.published_date.strftime('%d/%m/%Y %H:%M')}")
            print(f"Relevância: {article.relevance_score:.2f}")
            print(f"URL: {article.url}")
            print(f"Tempo de leitura: {article.read_time} minutos")
            print("\nResumo:")
            print(article.summary)
            print("-" * 50)

        # Exibir métricas
        metrics = news_service.metrics.get_metrics()
        print("\nMétricas de execução:")
        print(json.dumps(metrics, indent=2))

    finally:
        # Fechar serviço
        await news_service.close()

if __name__ == "__main__":
    asyncio.run(main())