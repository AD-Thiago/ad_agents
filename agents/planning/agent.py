# agents/planning/agent.py
import json
import threading
from datetime import datetime
from core.domains.adoption_metrics import AdoptionMetricsManager
from core.domains.seasonality import SeasonalityManager
from core.domains.market_segments import MarketSegmentManager
from core.domains.tech_stacks import TechStackManager
from core.domains.definitions import DomainManager
from core.rabbitmq_utils import RabbitMQUtils
from agents.planning.config import config
from core.config import get_settings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap
from pydantic import AnyHttpUrl
import logging
import asyncio

logger = logging.getLogger(__name__)

class PlanningAgent:
    """Agente de Planejamento Inteligente para Criação de Conteúdos"""

    def __init__(self):
        self.config = config
        self.rabbitmq = RabbitMQUtils()
        self.adoption_manager = AdoptionMetricsManager()
        self.seasonality_manager = SeasonalityManager()
        self.market_manager = MarketSegmentManager()
        self.tech_stack_manager = TechStackManager()
        self.domain_manager = DomainManager()

        self.llm = ChatOpenAI(
            model_name=get_settings().OPENAI_MODEL,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            openai_api_key=self.config.openai_api_key
        )
        self.setup_chain()

        self.stop_event = threading.Event()
        self.cache = {}
        self.cache_ttl = self.config.cache_ttl

    def setup_chain(self):
        self.plan_template = PromptTemplate(
            input_variables=["context", "insights"],
            template=(
                "Você é um planejador de conteúdo avançado. Com base no seguinte contexto:\n"
                "{context}\n"
                "e nas informações a seguir:\n"
                "{insights}\n"
                "Crie um plano de conteúdo no formato JSON com as seguintes chaves:\n"
                "topic (string), keywords (array de strings), target_audience (array de strings), "
                "content_type (string), priority (inteiro), estimated_impact (float).\n"
                "Certifique-se de que o JSON seja válido e bem formatado. Não inclua texto fora do JSON."
            ),
        )
        self.plan_chain = RunnableMap({
            "prompt": self.plan_template,
            "output": self.llm
        })

    async def generate_plan(self):
        """Gera um plano de conteúdo baseado em insights dos domínios"""
        logger.info("Iniciando a geração do plano de conteúdo")
        try:
            insights = await self._get_cached_insights()
            logger.debug("Insights coletados com sucesso")
            plan = await self._generate_plan_with_llm(insights)
            logger.debug("Plano gerado com sucesso")
            await self._publish_plan(plan)
            logger.info("Plano publicado na fila")
        except Exception as e:
            await self._handle_error(e, {"context": "Geração do plano"})
            logger.error("Erro durante a geração do plano", exc_info=True)

    async def _get_cached_insights(self):
        """Obtém insights dos domínios, usando cache se possível"""
        cache_key = self._get_cache_key()
        if cache_key in self.cache:
            if (datetime.now() - self.cache[cache_key]["timestamp"]).seconds < self.cache_ttl:
                logger.debug("Insights recuperados do cache")
                return self.cache[cache_key]["insights"]

        logger.debug("Coletando novos insights")
        adoption_insights = await self._gather_adoption_insights()
        seasonal_insights = await self._gather_seasonal_insights()
        market_insights = await self._gather_market_insights()
        domain_insights = await self._gather_domain_insights()

        insights = {
            "_gather_adoption_insights": adoption_insights,
            "_gather_seasonal_insights": seasonal_insights,
            "_gather_market_insights": market_insights,
            "_gather_domain_insights": domain_insights,
        }
        self.cache[cache_key] = {"insights": insights, "timestamp": datetime.now()}
        return insights

    def _get_cache_key(self):
        """Gera uma chave de cache única"""
        conf_dict = self.config.model_dump()
        items = []
        for k, v in conf_dict.items():
            if isinstance(v, AnyHttpUrl):
                v = str(v)
            val_str = str(v)
            items.append((k, val_str))

        items = sorted(items, key=lambda x: x[0])
        return hash(frozenset(items))

    async def _gather_adoption_insights(self):
        return [self._generate_adoption_insight(metric) for metric in self.adoption_manager.metrics.values()]

    async def _gather_seasonal_insights(self):
        return [self._generate_seasonal_insight(event) for event in self.seasonality_manager.get_current_events()]

    async def _gather_market_insights(self):
        return [self._generate_market_insight(segment) for segment in self.market_manager.segments.values()]

    async def _gather_domain_insights(self):
        context = {
            "domains": [domain.name for domain in self.domain_manager.domains.values()],
            "guidelines": [
                self.domain_manager.get_content_guidelines(domain_id)
                for domain_id in self.domain_manager.domains.keys()
            ],
        }
        return context

    def _generate_adoption_insight(self, metric):
        return {"technology": metric.technology, "adoption_rate": metric.adoption_rate}

    def _generate_seasonal_insight(self, event):
        return {"event": event.name, "impact_level": event.impact_level}

    def _generate_market_insight(self, segment):
        return {"segment": segment.name, "priority": str(segment.priority)}

    async def _generate_plan_with_llm(self, insights):
        """Gera um plano de conteúdo usando o modelo LLM"""
        context = json.dumps(insights["_gather_domain_insights"])
        insights_combined = json.dumps(
            insights["_gather_adoption_insights"] +
            insights["_gather_seasonal_insights"] +
            insights["_gather_market_insights"]
        )
        result = await self.plan_chain.invoke({"context": context, "insights": insights_combined})
        return json.loads(result["output"])

    async def _publish_plan(self, plan):
        """Publica o plano de conteúdo gerado"""
        try:
            await self.rabbitmq.publish_event(
            routing_key="planning.generated",
            message=plan
            )
            logger.info(f"Plano publicado com sucesso")

        except Exception as e:
            await self._handle_error(e, {"context": "Publicação do plano"})
            logger.error("Erro durante a publicação do plano", exc_info=True)
            raise
        
    async def _handle_error(self, error, context=None):
        """Trata erros ocorridos durante a execução do Planning Agent"""
        error_data = {
            "error": str(error),
            "context": context or {}
        }
        await self.rabbitmq.publish_event(
            routing_key="planning.failed",
            message=error_data,
            exchange="workflow.events"
        )
        logger.error(f"Erro no Planning Agent: {error_data}", exc_info=True)

    def start(self):
        """Inicia o agente de planejamento em um intervalo definido"""
        logger.info("Iniciando loop do agente de planejamento.")
        while not self.stop_event.is_set():
            asyncio.run(self.generate_plan())
            self.stop_event.wait(self.config.planning_interval)

    def stop(self):
        """Para a execução do thread"""
        self.stop_event.set()
    
    async def initialize(self):
        # Inicializações que você precisar, ou deixe vazio se não for necessário
        pass

    async def close(self):
        # Limpeza de recursos, se necessário, ou apenas pass
        pass