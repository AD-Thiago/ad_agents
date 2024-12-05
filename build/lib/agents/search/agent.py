from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.rabbitmq_utils import RabbitMQUtils
from core.config import get_settings
import json

class SearchResult(BaseModel):
    """Modelo para resultados de busca"""
    content: str
    source: str
    relevance_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class ContentValidation(BaseModel):
    """Modelo para validação de conteúdo"""
    claim: str
    is_valid: bool
    confidence_score: float
    supporting_sources: List[str]
    suggestions: Optional[List[str]]

class AudienceInsight(BaseModel):
    """Modelo para insights sobre audiência"""
    preferences: List[str]
    pain_points: List[str]
    technical_level: str
    common_questions: List[str]
    preferred_formats: List[str]

class SEOInsight(BaseModel):
    """Modelo para insights de SEO"""
    primary_keywords: List[tuple]  # (keyword, volume)
    related_keywords: List[tuple]
    questions: List[str]
    competing_content: List[Dict]
    suggested_structure: Dict[str, Any]

class EnhancedSearchAgent:
    """Agente de busca aprimorado com múltiplas funcionalidades"""

    def __init__(self):
        self.settings = get_settings()
        self.rabbitmq = RabbitMQUtils()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.setup_vector_store()
        self.setup_cache()

    def setup_vector_store(self):
        """Configura armazenamento vetorial"""
        self.vector_store = FAISS.from_texts(
            texts=["inicialização do índice"],
            embedding=self.embeddings,
            metadatas=[{"source": "initialization"}]
        )

    def setup_cache(self):
        """Configura sistema de cache"""
        self.cache = {}
        self.cache_ttl = 3600  # 1 hora

    async def enrich_content_plan(self, topic: str, keywords: List[str], target_audience: str) -> Dict:
        """
        Enriquece o plano de conteúdo com pesquisas e análises
        """
        tasks = [
            self.search_recent_developments(topic),
            self.validate_technical_aspects(topic),
            self.analyze_similar_content(topic, keywords),
            self.gather_seo_insights(keywords),
            self.analyze_audience_preferences(target_audience)
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            "recent_developments": results[0],
            "technical_validations": results[1],
            "competitive_analysis": results[2],
            "seo_insights": results[3],
            "audience_insights": results[4]
        }

    async def search_recent_developments(self, topic: str) -> List[SearchResult]:
        """
        Busca desenvolvimentos recentes sobre o tópico
        """
        # Implementar integração com APIs de notícias/blogs técnicos
        # Por enquanto, retorna exemplo
        return [
            SearchResult(
                content=f"Último desenvolvimento sobre {topic}",
                source="tech_news",
                relevance_score=0.95,
                metadata={"date": datetime.now().isoformat()}
            )
        ]

    async def validate_technical_aspects(self, topic: str) -> List[ContentValidation]:
        """
        Valida aspectos técnicos do tópico
        """
        # Implementar validação contra fontes técnicas confiáveis
        return [
            ContentValidation(
                claim=f"Validação técnica para {topic}",
                is_valid=True,
                confidence_score=0.85,
                supporting_sources=["docs.python.org"],
                suggestions=["Adicionar mais exemplos práticos"]
            )
        ]

    async def analyze_similar_content(self, topic: str, keywords: List[str]) -> Dict:
        """
        Analisa conteúdo similar existente
        """
        results = await self._search_vector_store(topic)
        
        # Análise de gaps e oportunidades
        return {
            "similar_content": results,
            "content_gaps": ["Gap 1", "Gap 2"],
            "unique_angles": ["Ângulo 1", "Ângulo 2"]
        }

    async def gather_seo_insights(self, keywords: List[str]) -> SEOInsight:
        """
        Coleta insights de SEO
        """
        # Implementar integração com APIs de SEO
        return SEOInsight(
            primary_keywords=[("python", 1000)],
            related_keywords=[("python programming", 800)],
            questions=["How to learn Python?"],
            competing_content=[],
            suggested_structure={
                "introduction": ["key_point_1", "key_point_2"],
                "main_sections": ["section_1", "section_2"],
                "conclusion": ["summary", "next_steps"]
            }
        )

    async def analyze_audience_preferences(self, target_audience: str) -> AudienceInsight:
        """
        Analisa preferências da audiência
        """
        return AudienceInsight(
            preferences=["Clear explanations", "Code examples"],
            pain_points=["Complex documentation", "Lack of examples"],
            technical_level="intermediate",
            common_questions=["How to start?", "Best practices?"],
            preferred_formats=["Tutorials", "How-to guides"]
        )

    async def _search_vector_store(self, query: str) -> List[SearchResult]:
        """
        Realiza busca no armazenamento vetorial
        """
        query_embedding = self.embeddings.embed_query(query)
        results = self.vector_store.similarity_search_with_score(query, k=5)
        
        return [
            SearchResult(
                content=result[0].page_content,
                source="vector_store",
                relevance_score=float(result[1]),
                metadata=result[0].metadata
            )
            for result in results
        ]

    async def index_content(self, content: str, metadata: Dict[str, Any]):
        """
        Indexa novo conteúdo no armazenamento vetorial
        """
        chunks = self.text_splitter.split_text(content)
        chunk_metadatas = [metadata for _ in chunks]
        self.vector_store.add_texts(chunks, metadatas=chunk_metadatas)

    def start_consuming(self):
        """
        Inicia consumo de mensagens do RabbitMQ
        """
        def callback(ch, method, properties, body):
            message = json.loads(body)
            print(f"Mensagem recebida: {message}")
            
            # Processar mensagem e enriquecer conteúdo
            enriched_data = asyncio.run(self.enrich_content_plan(
                topic=message.get("topic", ""),
                keywords=message.get("keywords", []),
                target_audience=message.get("target_audience", "")
            ))
            
            # Publicar resultados enriquecidos
            self.rabbitmq.publish_event(
                "search.results",
                json.dumps(enriched_data, default=str)
            )

        self.rabbitmq.consume_event("planning.generated", callback)

if __name__ == "__main__":
    agent = EnhancedSearchAgent()
    print("Search Agent iniciado. Aguardando mensagens...")
    agent.start_consuming()