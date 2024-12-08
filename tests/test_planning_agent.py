# tests/test_planning_agent.py
import unittest
from unittest.mock import AsyncMock, patch
from agents.planning.agent import PlanningAgent

class TestPlanningAgent(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.agent = PlanningAgent()
        self.mock_insights = {
            "_gather_adoption_insights": [{"type": "adoption_metric", "technology": "Tech1"}],
            "_gather_seasonal_insights": [{"type": "seasonal_metric", "season": "Spring"}],
            "_gather_market_insights": [{"type": "market_metric", "market": "Market1"}],
            "_gather_domain_insights": {"domain": "example.com"},
        }

    async def test_get_cached_insights_first_call(self):
        with patch.object(self.agent, '_gather_adoption_insights', return_value=self.mock_insights["_gather_adoption_insights"]), \
             patch.object(self.agent, '_gather_seasonal_insights', return_value=self.mock_insights["_gather_seasonal_insights"]), \
             patch.object(self.agent, '_gather_market_insights', return_value=self.mock_insights["_gather_market_insights"]), \
             patch.object(self.agent, '_gather_domain_insights', return_value=self.mock_insights["_gather_domain_insights"]):
            insights = await self.agent._get_cached_insights()
            self.assertDictEqual(insights, self.mock_insights)

    async def test_generate_plan_with_error(self):
        with patch.object(self.agent, '_get_cached_insights', side_effect=Exception("Test Error")), \
             patch.object(self.agent, '_handle_error', new=AsyncMock()) as mock_handle_error:
            await self.agent.generate_plan()
            mock_handle_error.assert_called_once()
            args, kwargs = mock_handle_error.call_args
            self.assertIsInstance(args[0], Exception)
            self.assertEqual(str(args[0]), "Test Error")
            self.assertEqual(args[1], {"context": "Geração do plano"})

    async def test_publish_plan(self):
        with patch.object(self.agent.rabbitmq, 'publish_event', new=AsyncMock()) as mock_publish:
            await self.agent._publish_plan({"topic": "Test Topic"})
            mock_publish.assert_called_once()
            _, kwargs = mock_publish.call_args
            self.assertEqual(kwargs["routing_key"], "planning.generated")
            self.assertEqual(kwargs["message"], {"topic": "Test Topic"})

    async def test_handle_error(self):
        with patch.object(self.agent.rabbitmq, 'publish_event', new=AsyncMock()) as mock_publish:
            await self.agent._handle_error(Exception("Test Error"), {"context": "Erro de teste"})
            mock_publish.assert_called_once()
            _, kwargs = mock_publish.call_args
            self.assertEqual(kwargs["message"]["error"], "Test Error")
            self.assertEqual(kwargs["message"]["context"], {"context": "Erro de teste"})

    async def test_cache_key_generation(self):
        # Patch o model_dump na classe PlanningAgentConfig
        with patch('agents.planning.config.PlanningAgentConfig.model_dump', return_value={"key1": "value1", "key2": "value2"}):
            cache_key = self.agent._get_cache_key()
            self.assertIsInstance(cache_key, int)