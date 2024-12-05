from core.rabbitmq_utils import RabbitMQUtils
from core.models import ContentImproved

rabbitmq = RabbitMQUtils()

improvement_request = ContentImproved(
    content_id="12345",
    content="Texto existente que precisa de melhorias.",
    suggestions=["Adicionar mais exemplos práticos.", "Melhorar a clareza na introdução."]
)

rabbitmq.publish_event("content.improved", improvement_request.dict())
print("Solicitação de melhoria publicada na fila 'content.improved'")
