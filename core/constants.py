# File: core/constants.py

from enum import Enum, auto

# Filas do RabbitMQ
QUEUES = {
    'triggers': 'workflows.triggers',
    'planning': 'agents.planning',
    'search': 'agents.search',
    'content': 'agents.content',
    'review': 'agents.review',
    'action': 'agents.action',
    'metrics': 'system.metrics',
    'notifications': 'system.notifications'
}

# Exchanges do RabbitMQ
EXCHANGES = {
    'workflow': 'workflow.events',
    'agents': 'agents.events',
    'system': 'system.events',
    'metrics': 'metrics.events',
    'notifications': 'notifications.events'
}

# Routing Keys para mensagens
ROUTING_KEYS = {
    'workflow': {
        'start': 'workflow.start',
        'complete': 'workflow.complete',
        'failed': 'workflow.failed',
        'update': 'workflow.update'
    },
    'planning': {
        'request': 'planning.request',
        'start': 'planning.start',
        'complete': 'planning.complete',
        'failed': 'planning.failed',
        'update': 'planning.update'
    },
    'search': {
        'request': 'search.request',
        'start': 'search.start',
        'complete': 'search.complete',
        'failed': 'search.failed',
        'update': 'search.update'
    },
    'content': {
        'request': 'content.request',
        'start': 'content.start',
        'complete': 'content.complete',
        'failed': 'content.failed',
        'update': 'content.update'
    },
    'review': {
        'request': 'review.request',
        'start': 'review.start',
        'complete': 'review.complete',
        'failed': 'review.failed',
        'update': 'review.update'
    },
    'action': {
        'request': 'action.request',
        'start': 'action.start',
        'complete': 'action.complete',
        'failed': 'action.failed',
        'update': 'action.update'
    }
}

# Status do Workflow
class WorkflowStatus(str, Enum):
    """Status possíveis de um workflow"""
    PENDING = 'pending'
    PLANNING = 'planning'
    SEARCHING = 'searching'
    GENERATING = 'generating'
    REVIEWING = 'reviewing'
    ACTING = 'acting'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'

# Tipos de Agentes
class AgentType(str, Enum):
    """Tipos de agentes no sistema"""
    PLANNING = 'planning'
    SEARCH = 'search'
    CONTENT = 'content'
    REVIEW = 'review'
    ACTION = 'action'

# Prioridades
class Priority(str, Enum):
    """Níveis de prioridade"""
    CRITICAL = 'critical'
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

# Tipos de Eventos
class EventType(str, Enum):
    """Tipos de eventos do sistema"""
    WORKFLOW = 'workflow'
    AGENT = 'agent'
    SYSTEM = 'system'
    METRIC = 'metric'
    ERROR = 'error'
    NOTIFICATION = 'notification'

# Tipos de Métricas
class MetricType(str, Enum):
    """Tipos de métricas do sistema"""
    PERFORMANCE = 'performance'
    QUALITY = 'quality'
    USAGE = 'usage'
    ERROR = 'error'
    BUSINESS = 'business'

# Headers padrão para mensagens
DEFAULT_HEADERS = {
    'version': '1.0',
    'content_type': 'application/json',
    'encoding': 'utf-8'
}

# Configurações de retry
RETRY_CONFIG = {
    'max_retries': 3,
    'initial_delay': 1,  # segundos
    'max_delay': 30,     # segundos
    'backoff_factor': 2
}

# Timeouts padrão
TIMEOUTS = {
    'workflow': 3600,    # 1 hora
    'planning': 300,     # 5 minutos
    'search': 180,       # 3 minutos
    'content': 600,      # 10 minutos
    'review': 300,       # 5 minutos
    'action': 180        # 3 minutos
}