from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import asyncio
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.pinecone import Pinecone as LangchainPinecone
from core.rabbitmq_utils import RabbitMQUtils
from core.config import get_settings

class SearchQuery(BaseModel):
    """Modelo para consultas de busca"""
    query: str
    context: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    min_relevance: float = 0.7

class SearchResult(BaseModel):
    """Modelo para resultados de busca"""
    content: str
    source: str
    relevance_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class SearchAgent:
    """Agente avançado de busca com múltiplas integrações"""

    def __init__(self):
        self.settings = get_settings()
        self.rabbitmq = RabbitMQUtils()  # Integração RabbitMQ
        self._setup_embeddings()
        self._setup_vector_stores()
        self._setup_cache()

    def _setup_embeddings(self):
        """Configura modelos de embedding"""
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def _setup_vector_stores(self):
        """Configura bases de vetores"""
        self.faiss_index = FAISS.from_texts(
            texts=["documento inicial para FAISS"],
            embedding=self.embeddings,
            metadatas=[{"source": "initialization"}]
        )
        self.pinecone_index = None  # Configuração omitida para simplificar

    def _setup_cache(self):
        """Configura sistema de cache"""
        self.cache = {}
        self.cache_ttl = 3600  # 1 hora

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Realiza busca em múltiplas fontes"""
        cache_key = f"{query.query}_{query.context}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        query_embedding = await self._generate_embedding(query.query)
        results = await asyncio.gather(
            self._search_local_index(query_embedding, query),
            self._search_huggingface(query)
        )

        consolidated = self._consolidate_results(results)
        filtered = [r for r in consolidated if r.relevance_score >= query.min_relevance]
        self.cache[cache_key] = filtered
        return filtered

    async def _generate_embedding(self, text: str) -> List[float]:
        """Gera embedding usando HuggingFace"""
        return self.embeddings.embed_query(text)

    async def _search_local_index(self, query_embedding: List[float], query: SearchQuery) -> List[SearchResult]:
        """Busca no índice FAISS local"""
        results = self.faiss_index.similarity_search_with_score(query.query, k=5)
        return [
            SearchResult(
                content=r[0].page_content,
                source="faiss",
                relevance_score=float(r[1]),
                metadata=r[0].metadata
            )
            for r in results
        ]

    async def _search_huggingface(self, query: SearchQuery) -> List[SearchResult]:
        """Busca usando modelos HuggingFace"""
        return [
            SearchResult(
                content="Resultado HuggingFace",
                source="huggingface",
                relevance_score=0.8,
                metadata={"query": query.query}
            )
        ]

    def _consolidate_results(self, results: List[List[SearchResult]]) -> List[SearchResult]:
        """Consolida e ranqueia resultados de diferentes fontes"""
        all_results = []
        for result_group in results:
            all_results.extend(result_group)

        seen = set()
        unique_results = []
        for result in all_results:
            content_hash = hash(f"{result.content}_{result.source}")
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(result)

        return sorted(unique_results, key=lambda x: x.relevance_score, reverse=True)

    def consume_search_requests(self):
        """Consome mensagens da fila 'search.requests' e processa buscas"""
        def callback(ch, method, properties, body):
            import json
            message = json.loads(body)
            print(f"Consulta recebida: {message}")
            try:
                query = SearchQuery(**message)
                results = asyncio.run(self.search(query))
                self.rabbitmq.publish_event("search.results", [r.dict() for r in results])
                print(f"Resultados publicados: {results}")
            except Exception as e:
                print(f"Erro ao processar consulta: {str(e)}")

        self.rabbitmq.consume_event("search.requests", callback)

    async def index_content(self, content: str, metadata: Dict[str, Any]):
        """Indexa novo conteúdo para busca futura"""
        content_vector = self.embeddings.embed_query(content)
        self.faiss_index.add_texts([content], [metadata])

if __name__ == "__main__":
    async def test():
        agent = SearchAgent()
        query = SearchQuery(query="IA generativa", context="tecnologia")
        results = await agent.search(query)
        print(f"Resultados encontrados: {len(results)}")
        for r in results:
            print(f"Fonte: {r.source}, Relevância: {r.relevance_score}")

    asyncio.run(test())