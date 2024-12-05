import asyncio
import logging
from agents.search.agent import EnhancedSearchAgent
from agents.search.services.news.config import NewsApiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Iniciando teste de busca de notícias")

    agent = EnhancedSearchAgent()
    await agent.initialize()

    try:
        print("Buscando artigos sobre: Python async")
        results = await agent.search_recent_developments("Python async")

        print("==================================================")
        for i, result in enumerate(results, start=1):
            print(f"Artigo {i}:")
            print(f"Título: {result.title}")
            print(f"Autor: {result.author}")
            print(f"Data: {result.published_date.strftime('%d/%m/%Y %H:%M')}")
            print(f"URL: {result.url}")
            print(f"Tags: {', '.join(result.tags)}")
            print("Metadados:")
            for key, value in result.metadata.items():
                print(f"  {key.capitalize()}: {value}")
            print("--------------------------------------------------")

        print("Métricas:")
        metrics = agent.metrics.get_metrics()
        for source, source_metrics in metrics["requests"].items():
            print(f"{source.upper()}:")
            print(f"  Requisições com sucesso: {source_metrics['success']}")
            print(f"  Requisições com erro: {source_metrics['error']}")
            print(f"  Artigos processados: {metrics['articles'][source]}")
            print(f"  Requisições ativas: {metrics['active_requests'][source]}")

    except Exception as e:
        logger.error(f"Ocorreu um erro: {str(e)}")
    finally:
        await agent.close()
        logger.info("Teste de busca de notícias concluído")

if __name__ == "__main__":
    asyncio.run(main())