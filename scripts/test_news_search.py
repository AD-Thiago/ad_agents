# scripts/test_news_search.py
import asyncio
import logging
from agents.search.agent import EnhancedSearchAgent
from agents.search.services.news.models import NewsSearchQuery
from datetime import datetime, timedelta, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# scripts/test_news_search.py

# ... (código anterior igual)

# scripts/test_news_search.py

async def main():
    logger.info("Iniciando teste de busca de notícias")
    agent = None
    
    try:
        # Inicializa o agente
        agent = EnhancedSearchAgent()
        await agent.initialize()
        
        # Define parâmetros de teste
        query = NewsSearchQuery(
            topic="Python asyncio development",
            keywords=["python", "asyncio", "async", "await", "coroutines"],
            start_date=datetime.now(timezone.utc) - timedelta(days=90),  # Aumentando período
            end_date=datetime.now(timezone.utc),
            min_relevance=0.05,  # Reduzindo threshold
            max_results=30
        )
        
        # Busca notícias
        logger.info(f"Buscando notícias sobre: {query.topic}")
        results = await agent.search_recent_developments(query.topic)
        
        # Exibe resultados
        total_results = len(results)
        logger.info(f"\nEncontrados {total_results} resultados no total")
        
        # Filtra por relevância com threshold menor
        relevant_results = [(r, r.relevance_score) for r in results if r.relevance_score >= 0.0]  # 20% de relevância mínima
        relevant_results.sort(key=lambda x: x[1], reverse=True)
        
        if relevant_results:
            logger.info(f"Mostrando os {len(relevant_results)} resultados mais relevantes:")
            for i, (result, score) in enumerate(relevant_results[:50], 1):
                logger.info(f"\n{i}. {result.title}")
                logger.info(f"   Fonte: {result.source}")
                logger.info(f"   Data: {result.published_date}")
                logger.info(f"   URL: {result.url}")
                logger.info(f"   Tags: {', '.join(result.tags)}")
                logger.info(f"   Relevância: {score:.2f}")
                if result.summary:
                    logger.info(f"   Resumo: {result.summary[:200]}...")
        else:
            logger.warning("\nNenhum resultado com alta relevância encontrado.")
            # Mostrar alguns resultados com baixa relevância
            logger.info("\nMostrando os 5 resultados com maior pontuação (mesmo que baixa):")
            results_with_scores = [(r, r.relevance_score) for r in results]
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (result, score) in enumerate(results_with_scores[:5], 1):
                logger.info(f"\n{i}. {result.title}")
                logger.info(f"   Fonte: {result.source}")
                logger.info(f"   Tags: {', '.join(result.tags)}")
                logger.info(f"   Relevância: {score:.2f}")
                
    except Exception as e:
        logger.error(f"Erro durante o teste: {str(e)}")
        raise
    finally:
        if agent:
            await agent.close()
        logger.info("Teste finalizado")
        
if __name__ == "__main__":
    asyncio.run(main())