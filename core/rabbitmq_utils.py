# File: core/rabbitmq_utils.py

import json
import aio_pika
import logging
import asyncio
from typing import Any, Dict, Optional, Callable
from datetime import datetime
from .constants import QUEUES, EXCHANGES, ROUTING_KEYS, DEFAULT_HEADERS, RETRY_CONFIG
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class RabbitMQUtils:
    """Utilitário para comunicação com RabbitMQ"""

    def __init__(self):
        self.connection = None
        self.channel = None
        self.exchanges = {}
        self.queues = {}
        self._setup_from_settings()

    def _setup_from_settings(self):
        """Configura parâmetros do RabbitMQ a partir das settings"""
        self.host = settings.RABBITMQ_HOST
        self.port = settings.RABBITMQ_PORT
        self.user = settings.RABBITMQ_USER
        self.password = settings.RABBITMQ_PASSWORD

    async def initialize(self):
        """Inicializa conexão e configurações"""
        try:
            if not self.connection:
                # Criar conexão
                self.connection = await aio_pika.connect_robust(
                    host=self.host,
                    port=self.port,
                    login=self.user,
                    password=self.password
                )
                logger.info("Conexão RabbitMQ estabelecida")

                # Criar canal
                self.channel = await self.connection.channel()
                await self.channel.set_qos(prefetch_count=1)
                logger.info("Canal RabbitMQ criado")

                # Configurar exchanges e queues
                await self._setup_exchanges()
                await self._setup_queues()

        except Exception as e:
            logger.error(f"Erro ao inicializar RabbitMQ: {str(e)}")
            raise

    async def _setup_exchanges(self):
        """Configura todas as exchanges"""
        for name, exchange_name in EXCHANGES.items():
            try:
                exchange = await self.channel.declare_exchange(
                    name=exchange_name,
                    type=aio_pika.ExchangeType.TOPIC,
                    durable=True
                )
                self.exchanges[name] = exchange
                logger.debug(f"Exchange {name} configurada")
            except Exception as e:
                logger.error(f"Erro ao configurar exchange {name}: {str(e)}")
                raise

    async def _setup_queues(self):
        """Configura todas as queues"""
        for name, queue_name in QUEUES.items():
            try:
                queue = await self.channel.declare_queue(
                    name=queue_name,
                    durable=True
                )
                self.queues[name] = queue
                logger.debug(f"Queue {name} configurada")
            except Exception as e:
                logger.error(f"Erro ao configurar queue {name}: {str(e)}")
                raise

    async def publish_event(
        self,
        routing_key: str,
        message: Dict[str, Any],
        exchange_name: str,
        headers: Optional[Dict] = None
    ):
        """Publica um evento no RabbitMQ"""
        try:
            if not self.channel:
                await self.initialize()

            # Preparar headers
            event_headers = {**DEFAULT_HEADERS, **(headers or {})}
            event_headers['timestamp'] = datetime.now().isoformat()

            # Preparar mensagem
            message_dict = {
                'data': message,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'routing_key': routing_key
                }
            }

            # Criar mensagem AMQP
            amqp_message = aio_pika.Message(
                body=json.dumps(message_dict).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers=event_headers
            )

            # Publicar
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise ValueError(f"Exchange {exchange_name} não encontrada")

            await exchange.publish(
                message=amqp_message,
                routing_key=routing_key
            )

            logger.debug(f"Evento publicado: {routing_key} -> {message}")

        except Exception as e:
            logger.error(f"Erro ao publicar evento: {str(e)}")
            await self._handle_publish_error(e, routing_key, message, exchange_name)

    async def _handle_publish_error(
        self,
        error: Exception,
        routing_key: str,
        message: Dict,
        exchange: str
    ):
        """Manipula erros de publicação"""
        retry_count = 0
        while retry_count < RETRY_CONFIG['max_retries']:
            try:
                await asyncio.sleep(
                    RETRY_CONFIG['initial_delay'] * 
                    (RETRY_CONFIG['backoff_factor'] ** retry_count)
                )
                await self.publish_event(routing_key, message, exchange)
                return
            except Exception as e:
                retry_count += 1
                if retry_count == RETRY_CONFIG['max_retries']:
                    raise

    async def consume_event(
        self,
        queue_name: str,
        callback: Callable,
        exchange_name: Optional[str] = None,
        routing_key: Optional[str] = None
    ):
        """Configura consumer para uma fila"""
        try:
            if not self.channel:
                await self.initialize()

            queue = self.queues.get(queue_name)
            if not queue:
                raise ValueError(f"Queue {queue_name} não encontrada")

            # Binding se necessário
            if exchange_name and routing_key:
                exchange = self.exchanges.get(exchange_name)
                if not exchange:
                    raise ValueError(f"Exchange {exchange_name} não encontrada")
                await queue.bind(exchange, routing_key)

            async def process_message(message: aio_pika.IncomingMessage):
                async with message.process():
                    try:
                        body = json.loads(message.body.decode())
                        await callback(self.channel, message.routing_key, message.headers, body)
                    except Exception as e:
                        logger.error(f"Erro processando mensagem: {str(e)}")
                        await self._handle_consumer_error(message, e)

            # Iniciar consumo
            await queue.consume(process_message)
            logger.info(f"Consumer iniciado para {queue_name}")

        except Exception as e:
            logger.error(f"Erro ao configurar consumer: {str(e)}")
            raise

    async def _handle_consumer_error(self, message: aio_pika.IncomingMessage, error: Exception):
        """Manipula erros no processamento de mensagens"""
        retry_count = message.headers.get('retry_count', 0)
        if retry_count < RETRY_CONFIG['max_retries']:
            # Requeue com delay
            await asyncio.sleep(
                RETRY_CONFIG['initial_delay'] * 
                (RETRY_CONFIG['backoff_factor'] ** retry_count)
            )
            message.headers['retry_count'] = retry_count + 1
            await message.nack(requeue=True)
        else:
            # Dead letter
            await message.reject(requeue=False)

    async def close(self):
        """Fecha conexões"""
        try:
            if self.channel:
                await self.channel.close()
            if self.connection:
                await self.connection.close()
            self.channel = None
            self.connection = None
            logger.info("Conexões RabbitMQ fechadas")
        except Exception as e:
            logger.error(f"Erro ao fechar conexões: {str(e)}")