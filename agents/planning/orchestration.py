# File: agents/planning/orchestration.py

import logging
import json
from datetime import datetime
from typing import Optional
from core.constants import QUEUES, EXCHANGES, ROUTING_KEYS
from core.rabbitmq_utils import RabbitMQUtils
from .services.validation import PlanningValidator
from .services.insights import InsightService
from .models import PlanningRequest, PlanningResponse, ContentStrategy
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from core.config import get_settings

logger = logging.getLogger(__name__)

class PlanningOrchestrator:
    """Orquestrador principal do processo de planejamento"""

    def __init__(self):
        self.validation_service = PlanningValidator()
        self.insight_service = InsightService()
        self.rabbitmq = RabbitMQUtils()
        self.llm = ChatOpenAI(
            model_name=get_settings().api.openai_model,
            temperature=0.7,
            max_tokens=2000,
        )
        self.setup_chain()

    def setup_chain(self):
        """Configura o LangChain para geração do plano"""
        self.plan_template = PromptTemplate(
            input_variables=["insights", "requirements"],
            template=(
                "Como um planejador de conteúdo especializado, analise os seguintes insights:\n"
                "{insights}\n\n"
                "E os seguintes requisitos:\n"
                "{requirements}\n\n"
                "Crie um plano de conteúdo estratégico que inclua:\n"
                "1. Tópico principal\n"
                "2. Palavras-chave estratégicas\n"
                "3. Público-alvo específico\n"
                "4. Tipo de conteúdo recomendado\n"
                "5. Estrutura sugerida\n"
                "6. Métricas de sucesso\n"
                "7. Próximos passos\n\n"
                "Forneça o plano em formato JSON válido."
            )
        )
        self.plan_chain = LLMChain(llm=self.llm, prompt=self.plan_template)

    async def initialize(self):
        """Inicializa o orquestrador"""
        logger.info("Inicializando Planning Orchestrator")
        await self.rabbitmq.initialize()
        
        # Configura consumers
        await self.setup_consumers()

    async def setup_consumers(self):
        """Configura os consumers do RabbitMQ"""
        await self.rabbitmq.consume_event(
            queue=QUEUES['planning'],
            callback=self._handle_planning_request,
            exchange=EXCHANGES['workflow'],
            routing_key='workflow.planning.request'
        )

    async def _handle_planning_request(self, ch, method, properties, body):
        """Processa requisições de planejamento"""
        workflow_id = None
        try:
            data = json.loads(body)
            request = PlanningRequest(**data)
            workflow_id = request.workflow_id
            
            logger.info(f"Iniciando planejamento para workflow {workflow_id}")
            
            # Coleta insights
            insights = await self.insight_service.gather_all_insights()
            
            # Gera plano inicial
            plan = await self._generate_plan(insights, request.parameters)
            
            # Valida plano
            validation_result = await self.validation_service.validate_plan(plan)
            
            if validation_result["valid"]:
                # Publica plano validado
                await self._publish_plan(plan)
                logger.info(f"Plano publicado para workflow {workflow_id}")
            else:
                raise ValueError(f"Plano não passou na validação: {validation_result['details']}")
                
        except Exception as e:
            logger.error(f"Erro no planejamento: {str(e)}")
            if workflow_id:
                await self._handle_error(workflow_id, str(e))

    async def _generate_plan(self, insights, parameters) -> PlanningResponse:
        """Gera plano baseado nos insights e parâmetros"""
        try:
            # Prepara dados para o LLM
            plan_input = {
                "insights": json.dumps(insights.dict()),
                "requirements": json.dumps(parameters)
            }
            
            # Gera plano com LLM
            response = await self.plan_chain.arun(plan_input)
            plan_data = json.loads(response)
            
            # Cria resposta estruturada
            return PlanningResponse(
                workflow_id=parameters.get("workflow_id"),
                **plan_data
            )
            
        except Exception as e:
            logger.error(f"Erro ao gerar plano: {str(e)}")
            raise

    async def _publish_plan(self, plan: PlanningResponse):
        """Publica plano gerado"""
        try:
            await self.rabbitmq.publish_event(
                routing_key=ROUTING_KEYS['planning']['complete'],
                message=plan.dict(),
                exchange=EXCHANGES['workflow'],
                headers={'workflow_id': plan.workflow_id}
            )
        except Exception as e:
            logger.error(f"Erro ao publicar plano: {str(e)}")
            raise

    async def _handle_error(self, workflow_id: str, error: str):
        """Manipula erros no processo de planejamento"""
        try:
            await self.rabbitmq.publish_event(
                routing_key=ROUTING_KEYS['planning']['failed'],
                message={
                    'workflow_id': workflow_id,
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                },
                exchange=EXCHANGES['workflow']
            )
        except Exception as e:
            logger.error(f"Erro ao publicar erro: {str(e)}")