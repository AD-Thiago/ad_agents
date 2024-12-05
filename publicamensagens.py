import json
from datetime import datetime
from core.rabbitmq_utils import RabbitMQUtils
from core.models import PlanningGenerated, ContentImproved


def publish_sample_messages():
    """Publica mensagens de exemplo nas filas configuradas"""
    rabbitmq = RabbitMQUtils()

    # Mensagem de exemplo para a fila 'planning.generated'
    planning_message = PlanningGenerated(
        topic="IA Generativa",
        keywords=["IA", "Machine Learning", "Automação"],
        target_audience="Desenvolvedores",
        content_type="technical_guide",
        priority=5,
        estimated_impact=0.85,
        scheduled_time=datetime.now()
    )

    rabbitmq.publish_event(
        "planning.generated",
        json.dumps(planning_message.dict(), default=str)
    )
    print(f"Mensagem publicada na fila 'planning.generated': {planning_message.dict()}")

    # Mensagem de exemplo para a fila 'content.improved'
    improvement_message = ContentImproved(
        content_id="12345",
        content="Texto existente que precisa de melhorias.",
        suggestions=[
            "Adicionar mais exemplos práticos.",
            "Melhorar a clareza na introdução."
        ]
    )

    rabbitmq.publish_event(
        "content.improved",
        json.dumps(improvement_message.dict(), default=str)
    )
    print(f"Mensagem publicada na fila 'content.improved': {improvement_message.dict()}")


if __name__ == "__main__":
    publish_sample_messages()
