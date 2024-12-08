from typing import List, Dict
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI 
from core.rabbitmq_utils import RabbitMQUtils
from core.models import PlanningGenerated, ContentGenerated, ContentImproved
from pydantic import ValidationError
from core.config import get_settings
import json
import threading
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContentAgent")

class ContentAgent:
    """Agente responsável pela geração e melhoria de conteúdo"""

    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model_name=self.settings.api.openai_model,
            temperature=self.settings.api.openai_temperature,
            openai_api_key= self.settings.api.openai_api_key
        )
        self.rabbitmq = RabbitMQUtils()
        self.setup_chains()

    def setup_chains(self):
        """Configura as chains do LangChain"""
        content_template = """
        Crie conteúdo técnico com os seguintes parâmetros:
        Tópico: {topic}
        Palavras-chave: {keywords}
        Público-alvo: {target_audience}
        Tom: {tone}
        Diretrizes de SEO: {seo_guidelines}
        Referências: {references}
        """
        self.content_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["topic", "keywords", "target_audience", "tone", "seo_guidelines", "references"],
                template=content_template
            )
        )

    def _serialize_message(self, message: dict) -> str:
        """Converte mensagem em JSON serializável"""
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Tipo não serializável: {type(obj)}")
        return json.dumps(message, default=default_serializer)

    def consume_plans(self):
        """Consome mensagens da fila 'planning.generated'"""
        def callback(ch, method, properties, body):
           plans = json.loads(body)  # Carregar a lista de planos
           for plan_data in plans:  # Iterar sobre cada plano
              try:
                if isinstance(plan_data.get("target_audience"), list):
                 plan_data["target_audience"] = ", ".join(plan_data["target_audience"])
                plan = PlanningGenerated(**plan_data)  # Criar instância de cada plano
                print(f"Plano recebido: {plan}")

                # Gera conteúdo com base no plano
                content = self.generate_content(plan)

                # Publica na fila 'content.generated'
                self.rabbitmq.publish_event("content.generated", self._serialize_message(content.dict()))
                print(f"Conteúdo gerado publicado: {content}")

              except ValidationError as e:
               print(f"Erro de validação: {e}")
              except Exception as e:
               print(f"Erro ao processar plano: {e}")

        self.rabbitmq.consume_event("planning.generated", callback)

    def consume_improvements(self):
        """Consome solicitações de melhoria da fila 'content.improved'"""
        def callback(ch, method, properties, body):
            try:
                improvement_request = ContentImproved(**json.loads(body))
                logger.info(f"Solicitação de melhoria recebida: {improvement_request}")

                improved_content = self.generate_improved_content(improvement_request)

                self.rabbitmq.publish_event("content.generated", self._serialize_message(improved_content.dict()))
                logger.info(f"Conteúdo melhorado publicado: {improved_content}")

            except ValidationError as e:
                logger.error(f"Erro de validação: {e}")
            except Exception as e:
                logger.error(f"Erro ao processar melhoria: {e}")

        self.rabbitmq.consume_event("content.improved", callback)

    def generate_content(self, plan: PlanningGenerated) -> ContentGenerated:
        """Gera conteúdo baseado no plano"""
        content = self.content_chain.run({
            "topic": plan.topic,
            "keywords": ", ".join(plan.keywords),
            "target_audience": plan.target_audience,
            "tone": "technical",
            "seo_guidelines": "Utilize palavras-chave naturalmente, título otimizado, e meta descrição.",
            "references": "Nenhuma referência disponível."
        })
        return ContentGenerated(
            content=content,
            title=f"Artigo sobre {plan.topic}",
            meta_description=f"Aprenda mais sobre {plan.topic}.",
            keywords=plan.keywords,
            seo_score=0.85,
            readability_score=0.9,
            created_at=datetime.now()
        )

    def generate_improved_content(self, improvement_request: ContentImproved) -> ContentGenerated:
        """Gera conteúdo melhorado"""
        improved_content = f"Versão melhorada: {improvement_request.content}"
        return ContentGenerated(
            content=improved_content,
            title=f"Melhoria: {improvement_request.content_id}",
            meta_description="Texto aprimorado com base no feedback.",
            keywords=[],
            seo_score=0.9,
            readability_score=0.95,
            created_at=datetime.now()
        )

    def start_consumers(self):
        """Inicia os consumidores em threads separadas"""
        threading.Thread(target=self.consume_plans, daemon=True).start()
        threading.Thread(target=self.consume_improvements, daemon=True).start()


if __name__ == "__main__":
    agent = ContentAgent()
    logger.info("Consumidores iniciados. Pressione Ctrl+C para sair.")
    agent.start_consumers()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Finalizando consumidores...")
