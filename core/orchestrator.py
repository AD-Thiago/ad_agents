from typing import Dict, List
from datetime import datetime
import asyncio
from pydantic import BaseModel, Field
from core.rabbitmq_utils import RabbitMQUtils

class ContentRequest(BaseModel):
    """Modelo para solicitação de conteúdo"""
    topic: str
    domain: str
    content_type: str
    target_audience: str
    technical_level: str = "intermediate"
    keywords: List[str] = []
    references_required: bool = True
    code_examples: bool = True

class Orchestrator:
    """
    Orquestrador para funcionalidades específicas.

    Cenários onde o Orchestrator deve ser usado:
    1. Consolidação de Dados:
       - Combina resultados de diferentes agentes para criar respostas finais.
    2. Fluxos Híbridos:
       - Processa fluxos que envolvem lógica adicional ou validações antes de completar uma tarefa.
    3. Monitoramento:
       - Realiza debugging ou logging centralizado para rastrear fluxos.

    Importante:
    - Para fluxos altamente escaláveis ou independentes, use o RabbitMQ para comunicação direta entre agentes.
    """

    def __init__(self):
        self.rabbitmq = RabbitMQUtils()

    def publish_request(self, request: ContentRequest):
        """Publica solicitação inicial na fila `planning.requests`"""
        try:
            self.rabbitmq.publish_event("planning.requests", request.dict())
            print(f"Solicitação publicada na fila `planning.requests`: {request.dict()}")
        except Exception as e:
            print(f"Erro ao publicar solicitação: {str(e)}")

    def consume_and_consolidate(self):
        """
        Consome mensagens de várias filas e consolida os resultados.
        Usado para criar relatórios finais ou responder a solicitações completas.
        """
        def callback(ch, method, properties, body):
            import json
            response = json.loads(body)
            print(f"Resposta recebida: {response}")

            # Exemplo de lógica de consolidação (adapte conforme necessário)
            if "content" in response:
                print(f"Conteúdo recebido: {response['content'][:200]}...")
            elif "plan" in response:
                print(f"Plano estratégico recebido: {response['plan']}")

        # Consumir respostas de filas relevantes
        self.rabbitmq.consume_event("search.results", callback)
        self.rabbitmq.consume_event("content.reviewed", callback)

    async def process_request_with_control(self, request: ContentRequest):
        """
        Executa a solicitação publicando na fila inicial e consolidando respostas.
        Usado para monitorar e consolidar fluxos em tempo real.
        """
        try:
            # Publica a solicitação inicial
            self.publish_request(request)

            # Aguarda e processa respostas
            print("Aguardando respostas das filas...")
            self.consume_and_consolidate()

        except Exception as e:
            print(f"Erro no processamento da solicitação: {str(e)}")

if __name__ == "__main__":
    async def test_orchestrator():
        orchestrator = Orchestrator()

        # Simula uma solicitação de conteúdo
        request = ContentRequest(
            topic="IA Generativa em Produção",
            domain="machine_learning",
            content_type="technical_guide",
            target_audience="desenvolvedores",
            technical_level="advanced",
            keywords=["IA", "Machine Learning", "Produção"]
        )

        # Processa a solicitação
        await orchestrator.process_request_with_control(request)

    asyncio.run(test_orchestrator())
