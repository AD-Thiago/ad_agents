import pika

class RabbitMQUtils:
    """Utilitários para integração com RabbitMQ"""

    def __init__(self, host='localhost', port=5672, username='guest', password='guest', virtual_host='/'):
        self.host = host
        self.port = port
        self.virtual_host = virtual_host
        self.credentials = pika.PlainCredentials(username, password)

    def create_connection(self):
        """Cria uma nova conexão com o RabbitMQ"""
        connection_params = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.virtual_host,
            credentials=self.credentials
        )
        return pika.BlockingConnection(connection_params)

    def publish_event(self, queue, message):
        """Publica um evento em uma fila"""
        connection = self.create_connection()
        channel = connection.channel()
        channel.queue_declare(queue=queue, durable=True)
        channel.basic_publish(
            exchange='',
            routing_key=queue,
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=2  # Faz com que a mensagem seja persistente
            )
        )
        connection.close()

    def consume_event(self, queue, callback):
        """Consome eventos de uma fila"""
        connection = self.create_connection()
        channel = connection.channel()
        channel.queue_declare(queue=queue, durable=True)
        channel.basic_consume(queue=queue, on_message_callback=callback, auto_ack=True)
        try:
            print(f"Consumindo mensagens da fila '{queue}'...")
            channel.start_consuming()
        except KeyboardInterrupt:
            print(f"Parando o consumo da fila '{queue}'...")
        finally:
            connection.close()
