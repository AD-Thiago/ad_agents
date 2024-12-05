# tests\test_search.py
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Adiciona o diretório raiz ao path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from agents.search.agent import SearchAgent, SearchQuery
from agents.search.config import SearchAgentConfig

async def test_search():
    # Inicializa o agente
    agent = SearchAgent()
    
    # Cria uma query de teste
    query = SearchQuery(
        query="Implementação de MLOps em produção",
        context="Buscando informações sobre deploy de modelos ML",
        filters={
            "language": "pt-br",
            "max_age_days": 365
        }
    )
    
    print("\n=== Teste do Search Agent ===")
    print(f"\nQuery: {query.query}")
    print(f"Contexto: {query.context}")
    
    try:
        # Realiza a busca
        results = await agent.search(query)
        
        print(f"\nEncontrados {len(results)} resultados:")
        for i, result in enumerate(results, 1):
            print(f"\n--- Resultado {i} ---")
            print(f"Fonte: {result.source}")
            print(f"Relevância: {result.relevance_score}")
            print(f"Timestamp: {result.timestamp}")
            print(f"Conteúdo: {result.content[:200]}...")
            if result.metadata:
                print(f"Metadata: {result.metadata}")
    
    except Exception as e:
        print(f"\nErro durante a busca: {str(e)}")
        raise

async def test_indexing():
    # Inicializa o agente
    agent = SearchAgent()
    
    # Conteúdo de teste
    test_content = """
    MLOps (Machine Learning Operations) é uma prática que visa automatizar
    e otimizar o ciclo de vida completo de modelos de machine learning em produção.
    """
    
    metadata = {
        "source": "test",
        "author": "AD Team",
        "timestamp": datetime.now().isoformat(),
        "category": "MLOps"
    }
    
    print("\n=== Teste de Indexação ===")
    
    try:
        # Indexa o conteúdo
        await agent.index_content(test_content, metadata)
        print("\nConteúdo indexado com sucesso!")
        
        # Testa a busca do conteúdo indexado
        query = SearchQuery(
            query="MLOps automatização",
            context="Machine Learning Operations"
        )
        
        results = await agent.search(query)
        print(f"\nBusca após indexação: {len(results)} resultados")
        
    except Exception as e:
        print(f"\nErro durante a indexação: {str(e)}")
        raise

if __name__ == "__main__":
    # Executa os testes
    asyncio.run(test_search())
    asyncio.run(test_indexing())