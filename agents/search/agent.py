# agents/search/agent.py
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import aiohttp
from pydantic import BaseModel, Field
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.rabbitmq_utils import RabbitMQUtils
from core.config import get_settings
from agents.planning.config import config
from .services.news.clients.hackernews import HackerNewsClient
from .services.news.clients.techcrunch import TechCrunchClient
from .services.news.clients.devto import DevToClient
from .services.news.config import NewsApiConfig
from .services.news.metrics import NewsMetrics

logger = logging.getLogger(__name__)

class SearchAgent:
    """Agente de busca com múltiplas funcionalidades"""

    def __init__(self):
        self.config = config
        #self.settings = get_settings()
        self.rabbitmq = RabbitMQUtils()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vector_store = self.setup_vector_store()
        self.cache = {}
        self.cache_ttl = 3600  # Cache com 1 hora de TTL
        self.metrics = NewsMetrics()
        self.news_config = NewsApiConfig()
        self.session = None
        self.hacker_news_client = None
        self.tech_crunch_client = None
        self.dev_to_client = None

    async def initialize(self):
        """Inicializa o agente de pesquisa e configura os clientes"""
        logger.info("Inicializando o agente de pesquisa...")
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Configuração dos clientes com a sessão
        self.hacker_news_client = HackerNewsClient(api_url=self.news_config.HACKER_NEWS_API_URL, session=self.session)
        self.tech_crunch_client = TechCrunchClient(base_url="https://techcrunch.com", session=self.session)
        self.dev_to_client = DevToClient(api_url=self.news_config.DEVTO_API_URL, api_key=self.news_config.DEVTO_API_KEY, session=self.session)

    async def close(self):
        """Fecha conexões do agente de pesquisa"""
        logger.info("Fechando conexões do agente de pesquisa...")
        if self.session:
            await self.session.close()
            self.session = None

    def setup_vector_store(self):
        """Configura o armazenamento vetorial"""
        logger.info("Configurando armazenamento vetorial...")
        return FAISS.from_texts(
            texts=["inicialização do índice"],
            embedding=self.embeddings,
            metadatas=[{"source": "initialization"}]
        )

    async def start_consuming(self):
        """Inicia consumo de mensagens do RabbitMQ"""
        async def process_message(message: Dict[str, Any]):
            logger.info(f"Mensagem recebida: {message}")
            enriched_data = await self.enrich_content_plan(
                topic=message.get("topic", ""),
                keywords=message.get("keywords", []),
                target_audience=message.get("target_audience", "")
            )
            self.rabbitmq.publish_event("search.results", json.dumps(enriched_data, default=str))

        await self.rabbitmq.consume_event("planning.generated", process_message)

    async def enrich_content_plan(self, topic: str, keywords: List[str], target_audience: str) -> Dict:
        """Enriquece o plano de conteúdo"""
        logger.info(f"Enriquecendo plano de conteúdo para o tópico: {topic}")
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

    async def search_recent_developments(self, topic: str) -> List[Dict]:
        """Busca desenvolvimentos recentes sobre o tópico"""
        logger.info(f"Buscando desenvolvimentos recentes sobre: {topic}")
        # Implementação do método aqui
        return []

# Configuração básica de logs e inicialização
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = SearchAgent()
    try:
        asyncio.run(agent.start_consuming())
    except KeyboardInterrupt:
        logger.info("Encerrando o Search Agent.")
        asyncio.run(agent.close())
