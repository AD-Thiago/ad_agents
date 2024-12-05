# tests\test_content.py
import asyncio
import sys
from pathlib import Path

# Ajusta o path para importações relativas
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from agents.content.agent import ContentAgent, ContentTemplate
from agents.content.config import ContentAgentConfig

async def test_content_generation():
    # Carrega configurações
    config = ContentAgentConfig()
    
    # Inicializa agente
    agent = ContentAgent()
    
    # Cria template de teste
    template = ContentTemplate(
        type="blog_post",
        structure={
            "target_audience": "desenvolvedores",
            "sections": ["introdução", "conceitos", "implementação", "conclusão"]
        },
        tone="technical_but_friendly",
        guidelines=[
            "use exemplos de código",
            "explique conceitos complexos de forma simples",
            "inclua referências práticas"
        ],
        seo_requirements={
            "min_words": 1200,
            "heading_structure": "h1,h2,h3",
            "keyword_density": 0.02
        }
    )
    
    # Dados de teste
    topic = "Implementando Machine Learning em Produção"
    keywords = ["MLOps", "machine learning", "deploy", "produção"]
    references = [
        {
            "title": "MLOps Best Practices",
            "content": "MLOps combina Machine Learning com práticas DevOps..."
        }
    ]
    
    try:
        # Gera conteúdo
        content = await agent.generate_content(
            topic=topic,
            keywords=keywords,
            references=references,
            template=template
        )
        
        # Imprime resultados
        print("\n" + "="*50)
        print("TÍTULO:")
        print("="*50)
        print(content.title)
        
        print("\n" + "="*50)
        print("META DESCRIPTION:")
        print("="*50)
        print(content.meta_description)
        
        print("\n" + "="*50)
        print("MÉTRICAS:")
        print("="*50)
        print(f"SEO Score: {content.seo_score}")
        print(f"Readability Score: {content.readability_score}")
        print(f"Tamanho do conteúdo: {len(content.content)} caracteres")
        print(f"Palavras-chave utilizadas: {content.keywords}")
        
        print("\n" + "="*50)
        print("CONTEÚDO COMPLETO:")
        print("="*50)
        print(content.content)
        
    except Exception as e:
        print(f"Erro durante a geração de conteúdo: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_content_generation())