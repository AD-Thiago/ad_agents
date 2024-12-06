# File: scripts/start_agent.py

import asyncio
import logging
import argparse
from importlib import import_module

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def start_agent(agent_type: str):
    """Inicia um agente específico"""
    try:
        # Importar dinamicamente o agente
        module = import_module(f'agents.{agent_type}.agent')
        agent_class = getattr(module, f'{agent_type.capitalize()}Agent')
        
        # Instanciar e iniciar agente
        agent = agent_class()
        await agent.initialize()
        
        # Manter rodando
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info(f"Encerrando {agent_type} agent...")
    except Exception as e:
        logger.error(f"Erro fatal: {str(e)}")
        raise
    finally:
        if agent:
            await agent.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', choices=['planning', 'search', 'content', 'review', 'action'])
    args = parser.parse_args()
    
    asyncio.run(start_agent(args.agent))