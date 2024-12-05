from core.rabbitmq_utils import RabbitMQUtils

# Exemplo de publicação na fila 'content.generated'
RabbitMQUtils.publish_event(
    "content.generated",
    {"content_id": 1, "content": "Texto inicial gerado"}
)

