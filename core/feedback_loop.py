from core.rabbitmq_utils import RabbitMQUtils
import json

class FeedbackLoop:
    def __init__(self, domain: str, max_iterations: int = 5):
        self.rabbitmq = RabbitMQUtils()
        self.domain = domain
        self.max_iterations = max_iterations
        self.iteration = 0

    def publish_content(self):
        """Publica conteúdo gerado na fila."""
        content = {"content_id": 1, "domain": self.domain, "content": "Texto inicial gerado"}
        print(f"Publicando conteúdo: {content}")
        self.rabbitmq.publish_event("content.generated", content)

    def process_reviewed_event(self, ch, method, properties, body):
        """Processa mensagens da fila 'content.reviewed'."""
        message = json.loads(body)
        print(f"Revisão recebida: {message}")

        quality_score = message.get("quality_score", 0)
        if quality_score >= 0.8:
            print(f"Conteúdo aprovado na iteração {self.iteration}")
        else:
            print(f"Qualidade insuficiente ({quality_score}). Melhorando conteúdo...")
            improvement = {
                "content_id": message["content_id"],
                "action": "improve",
                "suggestions": message.get("suggestions", [])
            }
            self.rabbitmq.publish_event("content.improved", improvement)

    def start(self):
        """Inicia o Feedback Loop."""
        print("Iniciando o Feedback Loop...")
        self.publish_content()
        self.rabbitmq.consume_event("content.reviewed", self.process_reviewed_event)
