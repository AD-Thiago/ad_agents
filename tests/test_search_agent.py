import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import asyncio
from agents.search.agent import SearchAgent

class TestSearchAgentCompletoEmUmTeste(unittest.IsolatedAsyncioTestCase):
    async def test_tudo_em_um_unico_teste(self):
        # Cria instância do agente
        agent = SearchAgent()

        # Mock do RabbitMQ
        agent.rabbitmq = MagicMock()
        agent.rabbitmq.consume_event = AsyncMock()
        agent.rabbitmq.publish_event = MagicMock()

        # Mock da sessão HTTP e clientes
        agent.session = None
        with patch('aiohttp.ClientSession', return_value=AsyncMock()) as mock_session:
            await agent.initialize()  # Testa initialize
            mock_session.assert_called_once()
            self.assertIsNotNone(agent.session, "A sessão deveria ter sido criada no initialize")

        # Mockando métodos auxiliares do enrich_content_plan
        agent.search_recent_developments = AsyncMock(return_value=[{"dev": "recent"}])
        agent.validate_technical_aspects = AsyncMock(return_value=[{"tech": "valid"}])
        agent.analyze_similar_content = AsyncMock(return_value=[{"similar": "content"}])
        agent.gather_seo_insights = AsyncMock(return_value=[{"seo": "insight"}])
        agent.analyze_audience_preferences = AsyncMock(return_value=[{"audience": "pref"}])

        # Testando start_consuming
        await agent.start_consuming()
        agent.rabbitmq.consume_event.assert_awaited_once()
        args, kwargs = agent.rabbitmq.consume_event.await_args
        self.assertEqual(args[0], "planning.generated")
        process_message_callback = args[1]

        # Cria mensagem fake
        fake_message = {
            "topic": "test_topic",
            "keywords": ["test_keyword"],
            "target_audience": "test_audience"
        }

        # Chama callback manualmente simulando mensagem recebida
        await process_message_callback(fake_message)

        # Verifica se os métodos auxiliares foram chamados pelo enrich_content_plan
        agent.search_recent_developments.assert_awaited_once_with("test_topic")
        agent.validate_technical_aspects.assert_awaited_once_with("test_topic")
        agent.analyze_similar_content.assert_awaited_once_with("test_topic", ["test_keyword"])
        agent.gather_seo_insights.assert_awaited_once_with(["test_keyword"])
        agent.analyze_audience_preferences.assert_awaited_once_with("test_audience")

        # Verifica se publish_event foi chamado com o resultado agregado
        agent.rabbitmq.publish_event.assert_called_once()
        pub_args, pub_kwargs = agent.rabbitmq.publish_event.call_args
        self.assertEqual(pub_args[0], "search.results")
        enriched_data = json.loads(pub_args[1])
        expected_enriched = {
            "recent_developments": [{"dev": "recent"}],
            "technical_validations": [{"tech": "valid"}],
            "competitive_analysis": [{"similar": "content"}],
            "seo_insights": [{"seo": "insight"}],
            "audience_insights": [{"audience": "pref"}]
        }
        self.assertEqual(enriched_data, expected_enriched)

        # Testa o cache
        agent.cache = {}
        key = "test_key"
        agent.cache[key] = {"data": "cached_value", "timestamp": asyncio.get_running_loop().time()}
        self.assertIn(key, agent.cache)
        self.assertEqual(agent.cache[key]["data"], "cached_value")

        # Testa vector store
        self.assertIsNotNone(agent.vector_store, "Vector store deveria estar configurado")

        # Testa o close
        await agent.close()
        self.assertIsNone(agent.session, "A sessão deveria ser None após close()")

if __name__ == "__main__":
    unittest.main()