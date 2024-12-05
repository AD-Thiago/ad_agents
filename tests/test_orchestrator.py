# tests/test_orchestrator.py

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Ajusta o path para importações relativas
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from core.orchestrator import Orchestrator, ContentRequest
from agents.search.agent import SearchResult

async def test_content_generation():
    """Testa geração completa de conteúdo"""
    orchestrator = Orchestrator()
    
    # Testa conversão de resultados de busca
    test_search_results = [
        SearchResult(
            content="Test content",
            source="test_source",
            relevance_score=0.8,
            metadata={"key": "value"},
            embedding=None,
            timestamp=datetime.now()
        )
    ]
    
    # Verifica se a conversão está funcionando
    converted_refs = orchestrator._convert_search_results_to_references(test_search_results)
    assert isinstance(converted_refs, list), "Deve retornar uma lista"
    assert len(converted_refs) > 0, "Lista não deve estar vazia"
    assert "title" in converted_refs[0], "Referência deve ter título"
    assert "content" in converted_refs[0], "Referência deve ter conteúdo"
    
    # Teste principal
    request = ContentRequest(
        topic="Implementando AI Agents para Automação de Processos",
        domain="ai_agents",
        content_type="technical_guide", 
        target_audience="desenvolvedores",
        technical_level="advanced",
        keywords=["AI agents", "automação", "processos", "MLOps"],
        references_required=True,
        code_examples=True
    )
    
    print("\n=== Iniciando Teste do Orchestrator ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        result = await orchestrator.process_request(request)
        
        # Validações do resultado
        assert result is not None, "Resultado não pode ser None"
        assert "content" in result, "Resultado deve conter 'content'"
        assert "metadata" in result, "Resultado deve conter 'metadata'"
        assert "suggestions" in result, "Resultado deve conter 'suggestions'"
        assert isinstance(result["content"], str), "Content deve ser string"
        assert len(result["content"]) > 0, "Content não deve estar vazio"
        
        print("\n" + "="*50)
        print("PLANO DE CONTEÚDO:")
        print("="*50)
        print(f"Topic: {result['metadata']['plan']['topic']}")
        print(f"Target Audience: {result['metadata']['plan']['target_audience']}")
        print(f"Priority: {result['metadata']['plan']['priority']}")
        
        print("\n" + "="*50)
        print("MÉTRICAS DE QUALIDADE:")
        print("="*50)
        print(f"Approved: {result['metadata']['approved']}")
        for metric, score in result['metadata']['quality_scores'].items():
            print(f"{metric}: {score:.2f}")
            
        print("\n" + "="*50)
        print("SUGESTÕES DE MELHORIA:")
        print("="*50)
        for suggestion in result['suggestions']:
            print(f"- {suggestion}")
            
        print("\n" + "="*50)
        print("PRÉVIA DO CONTEÚDO:")
        print("="*50)
        preview = result['content'][:500] + "..." if len(result['content']) > 500 else result['content']
        print(preview)
        
        return result
        
    except Exception as e:
        print(f"Erro no teste: {str(e)}")
        raise

async def test_metrics_analysis():
    """Testa análise de métricas"""
    orchestrator = Orchestrator()
    
    request = ContentRequest(
        topic="Guia de Implementação de AI",
        domain="ai_agents",
        content_type="guide",
        target_audience="desenvolvedores",
        technical_level="intermediate",
        keywords=["AI", "implementação", "guia"]
    )
    
    try:
        metrics = await orchestrator.analyze_metrics(request)
        assert metrics is not None, "Métricas não podem ser None"
        assert "metrics" in metrics, "Resultado deve conter 'metrics'"
        assert "timestamp" in metrics, "Resultado deve conter 'timestamp'"
        assert isinstance(metrics["metrics"]["estimated_impact"], float), "Impact deve ser float"
        assert isinstance(metrics["metrics"]["priority"], int), "Priority deve ser int"
        
        print("\n" + "="*50)
        print("ANÁLISE DE MÉTRICAS:")
        print("="*50)
        print(f"Estimated Impact: {metrics['metrics']['estimated_impact']:.2f}")
        print(f"Priority: {metrics['metrics']['priority']}")
        return metrics
        
    except Exception as e:
        print(f"Erro na análise de métricas: {str(e)}")
        raise

async def run_all_tests():
    """Executa todos os testes"""
    print("\n=== Executando Suite de Testes do Orchestrator ===")
    
    try:
        # Teste de geração de conteúdo
        content_result = await test_content_generation()
        print("\n✓ Teste de geração completado")
        
        # Teste de métricas
        metrics_result = await test_metrics_analysis()
        print("\n✓ Teste de métricas completado")
        
        print("\n=== Todos os testes completados com sucesso ===")
        
        return {
            "content_result": content_result,
            "metrics_result": metrics_result
        }
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(run_all_tests())