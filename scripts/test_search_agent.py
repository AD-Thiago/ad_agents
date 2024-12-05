import asyncio
import logging
from datetime import datetime
from agents.search.agent import EnhancedSearchAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def test_search_agent():
    """Testa todas as funcionalidades do Search Agent"""
    try:
        # Inicializa o agente
        agent = EnhancedSearchAgent()
        await agent.initialize()
        logging.info("Search Agent inicializado com sucesso")

        # Parâmetros de teste
        test_topic = "Python async programming"
        test_keywords = ["async", "await", "asyncio", "coroutines"]
        test_audience = "intermediate python developers"

        # Testa enriquecimento completo do plano
        logging.info("Testando enriquecimento do plano de conteúdo...")
        enriched_data = await agent.enrich_content_plan(
            topic=test_topic,
            keywords=test_keywords,
            target_audience=test_audience
        )

        # Valida e loga os resultados de cada componente
        logging.info("\n=== Resultados do Teste ===")
        
        # 1. Desenvolvimentos Recentes
        developments = enriched_data["recent_developments"]
        logging.info(f"\nDesenvolvimentos Recentes encontrados: {len(developments)}")
        for dev in developments[:3]:  # Mostra os 3 primeiros resultados
            logging.info(f"- {dev.title} ({dev.source})")

        # 2. Validações Técnicas
        validations = enriched_data["technical_validations"]
        logging.info(f"\nValidações Técnicas realizadas: {len(validations)}")
        for val in validations:
            logging.info(f"- Claim: {val.claim}")
            logging.info(f"  Score: {val.confidence_score}")

        # 3. Análise Competitiva
        comp_analysis = enriched_data["competitive_analysis"]
        logging.info("\nAnálise Competitiva:")
        logging.info(f"- Gaps identificados: {comp_analysis['content_gaps']}")
        logging.info(f"- Ângulos únicos: {comp_analysis['unique_angles']}")

        # 4. Insights SEO
        seo = enriched_data["seo_insights"]
        logging.info("\nInsights SEO:")
        logging.info(f"- Keywords primárias: {seo.primary_keywords}")
        logging.info(f"- Perguntas identificadas: {seo.questions}")

        # 5. Insights de Audiência
        audience = enriched_data["audience_insights"]
        logging.info("\nInsights de Audiência:")
        logging.info(f"- Nível técnico: {audience.technical_level}")
        logging.info(f"- Preferências: {audience.preferences}")
        logging.info(f"- Pontos de dor: {audience.pain_points}")

        # Testa indexação de conteúdo
        test_content = "This is a test content about Python async programming"
        test_metadata = {
            "source": "test",
            "date": datetime.now().isoformat(),
            "author": "test_script"
        }
        
        logging.info("\nTestando indexação de conteúdo...")
        await agent.index_content(test_content, test_metadata)
        
        # Testa busca no vector store
        logging.info("\nTestando busca no vector store...")
        vector_results = await agent._search_vector_store(test_topic)
        logging.info(f"Resultados encontrados no vector store: {len(vector_results)}")

        logging.info("\nTodos os testes completados com sucesso!")

    except Exception as e:
        logging.error(f"Erro durante os testes: {str(e)}")
        raise
    finally:
        # Fecha o agente adequadamente
        await agent.close()
        logging.info("Search Agent finalizado")

if __name__ == "__main__":
    asyncio.run(test_search_agent())