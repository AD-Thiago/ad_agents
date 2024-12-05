# tests/test_review.py

import asyncio
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from agents.review.agent import ReviewAgent

async def test_review():
   """Testa o Review Agent"""
   
   # Inicializa agente
   agent = ReviewAgent()
   
   # Conteúdo de teste
   test_content = """
   # Implementando AI Agents em Produção
   
   Este guia aborda os principais aspectos da implementação de agentes de IA
   em ambientes produtivos. Vamos explorar as melhores práticas e desafios comuns.
   
   ## Principais Tópicos
   
   1. Arquitetura de Agentes
   2. Integração com LLMs
   3. Monitoramento e Observabilidade
   4. Tratamento de Erros
   """
   
   # Contexto do conteúdo
   context = {
       "domain": "ai_agents",
       "target_audience": "technical_leaders",
       "technical_level": "advanced",
       "content_type": "technical_guide",
       "required_sections": [
           "architecture",
           "implementation",
           "best_practices",
           "monitoring"
       ]
   }
   
   try:
       # Executa revisão
       result = await agent.review_content(
           content=test_content,
           context=context
       )
       
       # Imprime resultados
       print("\n=== Resultado da Revisão ===")
       print(f"Qualidade: {result.quality_score:.2f}")
       print(f"SEO: {result.seo_score:.2f}")
       print(f"Precisão Técnica: {result.technical_accuracy:.2f}")
       print(f"Legibilidade: {result.readability_score:.2f}")
       print(f"Aprovado: {result.approved}")
       
       print("\nSugestões:")
       for suggestion in result.suggestions:
           print(f"- {suggestion}")
       
       print("\nConteúdo Revisado:")
       print(result.content)
       
   except Exception as e:
       print(f"Erro no teste: {str(e)}")
       raise

if __name__ == "__main__":
   asyncio.run(test_review())