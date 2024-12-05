# tests\test_planning.py
import asyncio
from agents.planning.agent import PlanningAgent
from agents.planning.config import PlanningAgentConfig
from core.config import get_settings

settings = get_settings()

async def test_planning():
    # Carrega configurações
    config = PlanningAgentConfig()
    
    # Inicializa agente
    agent = PlanningAgent()
    
    # Gera plano
    plan = await agent.generate_content_plan()
    
    print("Plano gerado:")
    for item in plan:
        print(f"\nTópico: {item.topic}")
        print(f"Keywords: {item.keywords}")
        print(f"Público: {item.target_audience}")
        print(f"Prioridade: {item.priority}")
        print(f"Impacto estimado: {item.estimated_impact}")

if __name__ == "__main__":
    asyncio.run(test_planning())