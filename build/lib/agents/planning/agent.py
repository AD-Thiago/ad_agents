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
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class PlanningAgent:
    """Agente de Planejamento Inteligente para Criação de Conteúdos"""

    def __init__(self):
        # Instâncias principais
        self.rabbitmq = RabbitMQUtils()
        self.adoption_manager = AdoptionMetricsManager()
        self.seasonality_manager = SeasonalityManager()
        self.market_manager = MarketSegmentManager()
        self.tech_stack_manager = TechStackManager()
        self.domain_manager = DomainManager()

        # Configuração do LLM
        self.llm = ChatOpenAI(
            model_name=get_settings().api.openai_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=get_settings().api.openai_api_key,
        )
        self.setup_chain()

        # Evento para controle do loop
        self.stop_event = threading.Event()

    def setup_chain(self):
        """Configura o LangChain para planejamento"""
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
        self.plan_chain = LLMChain(llm=self.llm, prompt=self.plan_template)

    def generate_plan(self):
        """Gera um plano de conteúdo baseado em insights dos domínios"""
        print("[LOG] Gerando plano...")
        insights = []

        # 1. Adicionar insights de frameworks e stacks
        try:
            for framework_id, framework in self.tech_stack_manager.frameworks.items():
                insights.append(self._generate_framework_insight(framework))
        except Exception as e:
            print(f"[ERROR] Falha ao gerar insights de frameworks: {e}")

        # 2. Adicionar eventos sazonais
        try:
            for event in self.seasonality_manager.get_current_events():
                insights.append(self._generate_seasonal_insight(event))
        except Exception as e:
            print(f"[ERROR] Falha ao gerar insights de eventos sazonais: {e}")

        # 3. Adicionar métricas de adoção
        try:
            for metric_id, metric in self.adoption_manager.metrics.items():
                insights.append(self._generate_adoption_insight(metric))
        except Exception as e:
            print(f"[ERROR] Falha ao gerar insights de métricas de adoção: {e}")

        # 4. Adicionar segmentos de mercado
        try:
            for segment_id, segment in self.market_manager.segments.items():
                insights.append(self._generate_market_insight(segment))
        except Exception as e:
            print(f"[ERROR] Falha ao gerar insights de segmentos de mercado: {e}")

        # 5. Contexto de domínios
        try:
            context = {
                "domains": [domain.name for domain in self.domain_manager.domains.values()],
                "guidelines": [
                    self.domain_manager.get_content_guidelines(domain_id)
                    for domain_id in self.domain_manager.domains.keys()
                ],
            }
        except Exception as e:
            print(f"[ERROR] Falha ao gerar contexto de domínios: {e}")
            context = {}

        # Gerar plano com LLM
        try:
            response = self.plan_chain.run({"context": json.dumps(context), "insights": json.dumps(insights)})
            cleaned_response = response.strip("```").strip("json").strip() 
            print(f"[DEBUG] Resposta limpa do LLM: {cleaned_response}")
           #print(f"[DEBUG] Resposta do LLM: {response}")
            plan = json.loads(cleaned_response)
            # Publicar o plano na fila
            self.rabbitmq.publish_event("planning.generated", json.dumps(plan))
            print(f"[LOG] Plano publicado: {plan}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Falha ao decodificar plano gerado: {e}")
        except Exception as e:
            print(f"[ERROR] Falha ao gerar plano com LLM: {e}")

    def _generate_framework_insight(self, framework):
        """Gera insight para frameworks"""
        return {
            "type": "framework",
            "name": framework.name,
            "category": framework.category,
            "features": framework.key_features,
            "use_cases": framework.use_cases,
            "maturity": framework.maturity,
        }

    def _generate_seasonal_insight(self, event):
        """Gera insight para eventos sazonais"""
        return {
            "type": "seasonal_event",
            "name": event.name,
            "impact_level": event.impact_level,
            "key_themes": event.key_themes,
            "affected_industries": event.affected_industries,
        }

    def _generate_adoption_insight(self, metric):
        """Gera insight para métricas de adoção"""
        return {
            "type": "adoption_metric",
            "technology": metric.technology,
            "growth_rate": metric.growth_rate,
            "current_stage": metric.current_stage,
            "market_penetration": metric.market_penetration,
        }

    def _generate_market_insight(self, segment):
        """Gera insight para segmentos de mercado"""
        return {
            "type": "market_segment",
            "name": segment.name,
            "industry": segment.industry,
            "technologies": segment.key_technologies,
            "opportunities": segment.opportunities,
        }

    def start(self):
        """Inicia o agente de planejamento em um intervalo definido"""
        print("[LOG] Iniciando loop do agente de planejamento.")
        while not self.stop_event.is_set():
            self.generate_plan()
            self.stop_event.wait(config.planning_interval)

    def stop(self):
        """Para a execução do thread"""
        self.stop_event.set()


if __name__ == "__main__":
    agent = PlanningAgent()
    print(f"[{datetime.now()}] Planning Agent iniciado.")
    try:
        agent.start()
    except KeyboardInterrupt:
        print("[LOG] Finalizando Planning Agent...")
        agent.stop()