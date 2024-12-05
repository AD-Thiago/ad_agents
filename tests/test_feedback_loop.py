# tests/test_feedback_loop.py

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Ajusta o path para importações relativas
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from core.feedback_loop import ContentFeedbackLoop, FeedbackMetrics
from agents.content.agent import ContentAgent, ContentTemplate
from agents.review.agent import ReviewAgent
from agents.search.agent import SearchAgent

async def test_feedback_loop():
    """Testa o ciclo de feedback autônomo"""
    
    print("\n=== Iniciando Teste do Feedback Loop ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        # Inicializa agentes
        content_agent = ContentAgent()
        review_agent = ReviewAgent()
        search_agent = SearchAgent()
        
        # Inicializa feedback loop
        feedback_loop = ContentFeedbackLoop(
            content_agent=content_agent,
            review_agent=review_agent,
            search_agent=search_agent,
            max_iterations=3,
            min_quality_threshold=0.8,
            improvement_threshold=0.1
        )
        
        # Cria template de teste
        template = ContentTemplate(
            type="technical_article",
            structure={
                "target_audience": "developers",
                "sections": [
                    "introduction",
                    "concepts",
                    "implementation",
                    "best_practices",
                    "conclusion"
                ]
            },
            tone="technical_but_friendly",
            guidelines=[
                "Include code examples",
                "Focus on practical applications",
                "Explain complex concepts clearly"
            ],
            seo_requirements={
                "min_words": 1500,
                "heading_structure": "h1,h2,h3",
                "keyword_density": 0.02
            }
        )
        
        # Contexto de teste
        context = {
            "domain": "ai_agents",
            "technical_level": "advanced",
            "industry": "technology"
        }
        
        # Gera conteúdo com feedback
        print("\nGerando conteúdo com feedback loop...")
        content, metrics = await feedback_loop.generate_with_feedback(
            topic="Implementing Autonomous AI Agents with Python",
            keywords=[
                "AI agents",
                "automation",
                "machine learning",
                "python",
                "orchestration"
            ],
            template=template,
            context=context
        )
        
        # Validações básicas
        print("\nRealizando validações...")
        assert isinstance(metrics, FeedbackMetrics), "Metrics deve ser instância de FeedbackMetrics"
        assert metrics.iteration_count > 0, "Deve ter pelo menos uma iteração"
        assert metrics.total_duration > 0, "Duração total deve ser maior que zero"
        assert len(metrics.history) > 0, "Deve ter histórico de iterações"
        
        # Imprime resultados
        print("\n=== Resultados do Feedback Loop ===")
        print(f"Número de iterações: {metrics.iteration_count}")
        print(f"Duração total: {metrics.total_duration:.2f} segundos")
        
        print("\n=== Métricas Finais ===")
        print(f"Quality Score: {metrics.quality_score:.2f}")
        print(f"Technical Accuracy: {metrics.technical_accuracy:.2f}")
        print(f"Readability Score: {metrics.readability_score:.2f}")
        print(f"SEO Score: {metrics.seo_score:.2f}")
        
        print("\n=== Histórico de Iterações ===")
        for iteration in metrics.history:
            print(f"\nIteração {iteration.iteration_number}:")
            print("Métricas:")
            for metric, value in iteration.content_metrics.items():
                print(f"- {metric}: {value:.2f}")
            
            print("\nSugestões recebidas:")
            for suggestion in iteration.suggestions:
                print(f"- {suggestion}")
            
            print("\nMelhorias aplicadas:")
            for improvement in iteration.improvements_made:
                print(f"- {improvement}")
        
        print("\n=== Preview do Conteúdo ===")
        print("\nTítulo:")
        print(content.title)
        print("\nMeta Description:")
        print(content.meta_description)
        print("\nPreview do conteúdo:")
        preview = content.content[:500] + "..." if len(content.content) > 500 else content.content
        print(preview)
        
        # Validações adicionais
        assert len(content.content) >= template.seo_requirements["min_words"], \
            "Conteúdo deve atender requisito mínimo de palavras"
        
        assert metrics.quality_score >= feedback_loop.min_quality_threshold or \
               metrics.iteration_count >= feedback_loop.max_iterations, \
            "Deve atingir qualidade mínima ou número máximo de iterações"
        
        print("\n✓ Todas as validações passaram com sucesso!")
        return content, metrics
        
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {str(e)}")
        raise

async def test_quality_improvement():
    """Testa se o feedback loop realmente melhora a qualidade"""
    print("\n=== Testando Melhoria de Qualidade ===")
    
    try:
        # Executa teste principal
        content, metrics = await test_feedback_loop()
        
        # Analisa histórico de qualidade
        if len(metrics.history) > 1:
            initial_quality = metrics.history[0].content_metrics['quality']
            final_quality = metrics.quality_score
            
            print(f"\nQualidade inicial: {initial_quality:.2f}")
            print(f"Qualidade final: {final_quality:.2f}")
            print(f"Melhoria: {(final_quality - initial_quality):.2f}")
            
            assert final_quality > initial_quality, \
                "Qualidade final deve ser maior que a inicial"
            
            print("\n✓ Teste de melhoria de qualidade passou!")
            
    except Exception as e:
        print(f"\n❌ Erro no teste de melhoria: {str(e)}")
        raise

if __name__ == "__main__":
    print("\n=== Iniciando Suite de Testes do Feedback Loop ===")
    
    try:
        # Executa os testes
        asyncio.run(test_feedback_loop())
        asyncio.run(test_quality_improvement())
        
        print("\n=== Todos os testes completados com sucesso! ===")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {str(e)}")
        raise