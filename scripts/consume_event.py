from core.rabbitmq_utils import RabbitMQUtils

def process_message(ch, method, properties, body):
    import json
    message = json.loads(body)
    print(f"Mensagem recebida: {message}")

# Consumir mensagens da fila 'content.generated'
RabbitMQUtils.consume_event("content.generated", process_message)
