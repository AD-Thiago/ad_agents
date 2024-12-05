# Documentação do Projeto

Este README foi gerado automaticamente para documentar a estrutura do projeto.

# Índice

- [conftest.py](#conftest.py)
- [main.py](#main.py)
- [publicamensagens copy.py](#publicamensagens-copy.py)
- [publicamensagens.py](#publicamensagens.py)
- [pyproject.toml](#pyproject.toml)
- [pytest.ini](#pytest.ini)
- [requirements.txt](#requirements.txt)
- [setup.py](#setup.py)
- [__init__.py](#__init__.py)
- [.pytest_cache\CACHEDIR.TAG](#.pytest_cache\cachedir.tag)
- [.pytest_cache\v\cache\lastfailed](#.pytest_cache\v\cache\lastfailed)
- [.pytest_cache\v\cache\nodeids](#.pytest_cache\v\cache\nodeids)
- [.pytest_cache\v\cache\stepwise](#.pytest_cache\v\cache\stepwise)
- [ad_agents.egg-info\dependency_links.txt](#ad_agents.egg-info\dependency_links.txt)
- [ad_agents.egg-info\PKG-INFO](#ad_agents.egg-info\pkg-info)
- [ad_agents.egg-info\SOURCES.txt](#ad_agents.egg-info\sources.txt)
- [ad_agents.egg-info\top_level.txt](#ad_agents.egg-info\top_level.txt)
- [agents\__init__.py](#agents\__init__.py)
- [agents\action\__init__.py](#agents\action\__init__.py)
- [agents\content\agent.py](#agents\content\agent.py)
- [agents\content\config.py](#agents\content\config.py)
- [agents\content\__init__.py](#agents\content\__init__.py)
- [agents\planning\agent.py](#agents\planning\agent.py)
- [agents\planning\config.py](#agents\planning\config.py)
- [agents\planning\__init__.py](#agents\planning\__init__.py)
- [agents\review\agent.py](#agents\review\agent.py)
- [agents\review\config.py](#agents\review\config.py)
- [agents\review\__init__.py](#agents\review\__init__.py)
- [agents\search\agent.py](#agents\search\agent.py)
- [agents\search\config.py](#agents\search\config.py)
- [agents\search\__init__.py](#agents\search\__init__.py)
- [agents\search\services\news_integration.py](#agents\search\services\news_integration.py)
- [agents\search\services\__init__.py](#agents\search\services\__init__.py)
- [agents\search\services\news\cache.py](#agents\search\services\news\cache.py)
- [agents\search\services\news\config.py](#agents\search\services\news\config.py)
- [agents\search\services\news\metrics.py](#agents\search\services\news\metrics.py)
- [agents\search\services\news\models.py](#agents\search\services\news\models.py)
- [agents\search\services\news\service.py](#agents\search\services\news\service.py)
- [agents\search\services\news\__init__.py](#agents\search\services\news\__init__.py)
- [agents\search\services\news\clients\devto.py](#agents\search\services\news\clients\devto.py)
- [agents\search\services\news\clients\hackernews.py](#agents\search\services\news\clients\hackernews.py)
- [agents\search\services\news\clients\__init__.py](#agents\search\services\news\clients\__init__.py)
- [agents\search\services\utils\rate_limiter.py](#agents\search\services\utils\rate_limiter.py)
- [agents\search\services\utils\__init__.py](#agents\search\services\utils\__init__.py)
- [api\__init__.py](#api\__init__.py)
- [build\lib\agents\__init__.py](#build\lib\agents\__init__.py)
- [build\lib\agents\action\__init__.py](#build\lib\agents\action\__init__.py)
- [build\lib\agents\content\agent.py](#build\lib\agents\content\agent.py)
- [build\lib\agents\content\config.py](#build\lib\agents\content\config.py)
- [build\lib\agents\content\__init__.py](#build\lib\agents\content\__init__.py)
- [build\lib\agents\planning\agent.py](#build\lib\agents\planning\agent.py)
- [build\lib\agents\planning\config.py](#build\lib\agents\planning\config.py)
- [build\lib\agents\planning\__init__.py](#build\lib\agents\planning\__init__.py)
- [build\lib\agents\review\agent.py](#build\lib\agents\review\agent.py)
- [build\lib\agents\review\config.py](#build\lib\agents\review\config.py)
- [build\lib\agents\review\__init__.py](#build\lib\agents\review\__init__.py)
- [build\lib\agents\search\agent.py](#build\lib\agents\search\agent.py)
- [build\lib\agents\search\config.py](#build\lib\agents\search\config.py)
- [build\lib\agents\search\__init__.py](#build\lib\agents\search\__init__.py)
- [build\lib\agents\search\services\news_integration.py](#build\lib\agents\search\services\news_integration.py)
- [build\lib\agents\search\services\__init__.py](#build\lib\agents\search\services\__init__.py)
- [build\lib\agents\search\services\news\cache.py](#build\lib\agents\search\services\news\cache.py)
- [build\lib\agents\search\services\news\config.py](#build\lib\agents\search\services\news\config.py)
- [build\lib\agents\search\services\news\metrics.py](#build\lib\agents\search\services\news\metrics.py)
- [build\lib\agents\search\services\news\models.py](#build\lib\agents\search\services\news\models.py)
- [build\lib\agents\search\services\news\service.py](#build\lib\agents\search\services\news\service.py)
- [build\lib\agents\search\services\news\__init__.py](#build\lib\agents\search\services\news\__init__.py)
- [build\lib\agents\search\services\news\clients\devto.py](#build\lib\agents\search\services\news\clients\devto.py)
- [build\lib\agents\search\services\news\clients\hackernews.py](#build\lib\agents\search\services\news\clients\hackernews.py)
- [build\lib\agents\search\services\news\clients\__init__.py](#build\lib\agents\search\services\news\clients\__init__.py)
- [build\lib\agents\search\services\utils\rate_limiter.py](#build\lib\agents\search\services\utils\rate_limiter.py)
- [build\lib\agents\search\services\utils\__init__.py](#build\lib\agents\search\services\utils\__init__.py)
- [build\lib\api\__init__.py](#build\lib\api\__init__.py)
- [build\lib\core\config.py](#build\lib\core\config.py)
- [build\lib\core\feedback_loop.py](#build\lib\core\feedback_loop.py)
- [build\lib\core\models.py](#build\lib\core\models.py)
- [build\lib\core\orchestrator.py](#build\lib\core\orchestrator.py)
- [build\lib\core\rabbitmq_utils.py](#build\lib\core\rabbitmq_utils.py)
- [build\lib\core\testa api.py](#build\lib\core\testa-api.py)
- [build\lib\core\__init__.py](#build\lib\core\__init__.py)
- [build\lib\core\domains\adoption_metrics.py](#build\lib\core\domains\adoption_metrics.py)
- [build\lib\core\domains\bigtech_monitor.py](#build\lib\core\domains\bigtech_monitor.py)
- [build\lib\core\domains\definitions.py](#build\lib\core\domains\definitions.py)
- [build\lib\core\domains\market_segments.py](#build\lib\core\domains\market_segments.py)
- [build\lib\core\domains\seasonality.py](#build\lib\core\domains\seasonality.py)
- [build\lib\core\domains\tech_stacks.py](#build\lib\core\domains\tech_stacks.py)
- [build\lib\core\domains\__init__.py](#build\lib\core\domains\__init__.py)
- [build\lib\examples\news_integration_example.py](#build\lib\examples\news_integration_example.py)
- [build\lib\examples\__init__.py](#build\lib\examples\__init__.py)
- [build\lib\infrastructure\__init__.py](#build\lib\infrastructure\__init__.py)
- [build\lib\tests\test_config.py](#build\lib\tests\test_config.py)
- [build\lib\tests\test_content.py](#build\lib\tests\test_content.py)
- [build\lib\tests\test_devto_client.py](#build\lib\tests\test_devto_client.py)
- [build\lib\tests\test_feedback_loop.py](#build\lib\tests\test_feedback_loop.py)
- [build\lib\tests\test_news_integration.py](#build\lib\tests\test_news_integration.py)
- [build\lib\tests\test_orchestrator.py](#build\lib\tests\test_orchestrator.py)
- [build\lib\tests\test_planning.py](#build\lib\tests\test_planning.py)
- [build\lib\tests\test_review.py](#build\lib\tests\test_review.py)
- [build\lib\tests\test_search.py](#build\lib\tests\test_search.py)
- [build\lib\tests\__init__.py](#build\lib\tests\__init__.py)
- [core\config.py](#core\config.py)
- [core\feedback_loop.py](#core\feedback_loop.py)
- [core\models.py](#core\models.py)
- [core\orchestrator.py](#core\orchestrator.py)
- [core\rabbitmq_utils.py](#core\rabbitmq_utils.py)
- [core\testa api.py](#core\testa-api.py)
- [core\__init__.py](#core\__init__.py)
- [core\domains\adoption_metrics.py](#core\domains\adoption_metrics.py)
- [core\domains\bigtech_monitor.py](#core\domains\bigtech_monitor.py)
- [core\domains\definitions.py](#core\domains\definitions.py)
- [core\domains\market_segments.py](#core\domains\market_segments.py)
- [core\domains\seasonality.py](#core\domains\seasonality.py)
- [core\domains\tech_stacks.py](#core\domains\tech_stacks.py)
- [core\domains\__init__.py](#core\domains\__init__.py)
- [examples\news_integration_example.py](#examples\news_integration_example.py)
- [examples\__init__.py](#examples\__init__.py)
- [infrastructure\__init__.py](#infrastructure\__init__.py)
- [infrastructure\docker\Dockerfile](#infrastructure\docker\dockerfile)
- [infrastructure\kubernetes\deployment.yaml](#infrastructure\kubernetes\deployment.yaml)
- [scripts\consume_event.py](#scripts\consume_event.py)
- [scripts\publish_event.py](#scripts\publish_event.py)
- [scripts\test_loop.py](#scripts\test_loop.py)
- [scripts\test_news_search.py](#scripts\test_news_search.py)
- [tests\test_config.py](#tests\test_config.py)
- [tests\test_content.py](#tests\test_content.py)
- [tests\test_devto_client.py](#tests\test_devto_client.py)
- [tests\test_feedback_loop.py](#tests\test_feedback_loop.py)
- [tests\test_news_integration.py](#tests\test_news_integration.py)
- [tests\test_orchestrator.py](#tests\test_orchestrator.py)
- [tests\test_planning.py](#tests\test_planning.py)
- [tests\test_review.py](#tests\test_review.py)
- [tests\test_search.py](#tests\test_search.py)
- [tests\__init__.py](#tests\__init__.py)


## conftest.py

```python
# conftest.py

import pytest
import asyncio

# Define o modo asyncio como auto
def pytest_configure(config):
    config.option.asyncio_mode = "auto"

# Define o escopo do event loop como function
@pytest.fixture(scope="function")
async def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

## main.py

```python
from fastapi import FastAPI 
app = FastAPI() 
@app.get("/") 
def read_root(): 
    return {"Hello": "AD Agents"} 

```

## publicamensagens copy.py

```python
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

```

## publicamensagens.py

```python
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

```

## pyproject.toml

```python
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "ad_agents"
version = "0.1.0"
authors = [
  { name="AD Team" }
]
description = "AD Agents Platform"

[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "session"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v -p no:warnings"
```

## pytest.ini

```python
# pytest.ini
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = session
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v -p no:warnings
```

## requirements.txt

```python
# requirements.txt
# Core
fastapi==0.104.1
uvicorn==0.24.0
pydantic==1.10.13
python-dotenv==1.0.0

# Database
sqlalchemy==2.0.23
python-multipart==0.0.6

# AI/ML Base
langchain==0.0.300
openai==0.27.0

# Monitoring
prometheus-client==0.17.0

# Async
asyncio==3.4.3
aiohttp==3.9.1

# Utils
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.2

pika==1.3.1
```

## setup.py

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="ad_agents",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        'fastapi==0.104.1',
        'langchain==0.0.300',
        'pydantic==1.10.13',
        'python-dotenv==1.0.0',
        'openai==0.27.0',
        'numpy==1.24.0',
        'pandas==2.0.0',
        'scikit-learn==1.3.2',
        'aiohttp==3.9.1'
    ]
)
```

## __init__.py

```python

```

## .pytest_cache\CACHEDIR.TAG

```python
Signature: 8a477f597d28d172789f06886806bc55
# This file is a cache directory tag created by pytest.
# For information about cache directory tags, see:
#	https://bford.info/cachedir/spec.html

```

## .pytest_cache\v\cache\lastfailed

```python
{
  "tests/test_news_integration.py::test_relevance_calculation": true,
  "tests/test_news_integration.py::test_summary_generation": true,
  "tests/test_news_integration.py::test_article_processing": true,
  "tests/test_news_integration.py::test_content_extraction": true,
  "tests/test_news_integration.py::test_filtering_and_sorting": true
}
```

## .pytest_cache\v\cache\nodeids

```python
[
  "tests/test_news_integration.py::test_article_processing",
  "tests/test_news_integration.py::test_cache_functionality",
  "tests/test_news_integration.py::test_content_extraction",
  "tests/test_news_integration.py::test_error_handling",
  "tests/test_news_integration.py::test_filtering_and_sorting",
  "tests/test_news_integration.py::test_relevance_calculation",
  "tests/test_news_integration.py::test_search_news_basic",
  "tests/test_news_integration.py::test_summary_generation"
]
```

## .pytest_cache\v\cache\stepwise

```python
[]
```

## ad_agents.egg-info\dependency_links.txt

```python


```

## ad_agents.egg-info\PKG-INFO

```python
Metadata-Version: 2.1
Name: ad_agents
Version: 0.1.0
Summary: AD Agents Platform
Author: AD Team

```

## ad_agents.egg-info\SOURCES.txt

```python
README.md
pyproject.toml
setup.py
./agents/__init__.py
./agents/action/__init__.py
./agents/content/__init__.py
./agents/content/agent.py
./agents/content/config.py
./agents/planning/__init__.py
./agents/planning/agent.py
./agents/planning/config.py
./agents/review/__init__.py
./agents/review/agent.py
./agents/review/config.py
./agents/search/__init__.py
./agents/search/agent.py
./agents/search/config.py
./agents/search/services/__init__.py
./agents/search/services/news_integration.py
./agents/search/services/news/__init__.py
./agents/search/services/news/cache.py
./agents/search/services/news/config.py
./agents/search/services/news/metrics.py
./agents/search/services/news/models.py
./agents/search/services/news/service.py
./agents/search/services/news/clients/__init__.py
./agents/search/services/news/clients/devto.py
./agents/search/services/news/clients/hackernews.py
./agents/search/services/utils/__init__.py
./agents/search/services/utils/rate_limiter.py
./api/__init__.py
./core/__init__.py
./core/config.py
./core/feedback_loop.py
./core/models.py
./core/orchestrator.py
./core/rabbitmq_utils.py
./core/testa api.py
./core/domains/__init__.py
./core/domains/adoption_metrics.py
./core/domains/bigtech_monitor.py
./core/domains/definitions.py
./core/domains/market_segments.py
./core/domains/seasonality.py
./core/domains/tech_stacks.py
./examples/__init__.py
./examples/news_integration_example.py
./infrastructure/__init__.py
./tests/__init__.py
./tests/test_config.py
./tests/test_content.py
./tests/test_devto_client.py
./tests/test_feedback_loop.py
./tests/test_news_integration.py
./tests/test_orchestrator.py
./tests/test_planning.py
./tests/test_review.py
./tests/test_search.py
ad_agents.egg-info/PKG-INFO
ad_agents.egg-info/SOURCES.txt
ad_agents.egg-info/dependency_links.txt
ad_agents.egg-info/top_level.txt
agents/__init__.py
agents/action/__init__.py
agents/content/__init__.py
agents/content/agent.py
agents/content/config.py
agents/planning/__init__.py
agents/planning/agent.py
agents/planning/config.py
agents/review/__init__.py
agents/review/agent.py
agents/review/config.py
agents/search/__init__.py
agents/search/agent.py
agents/search/config.py
agents/search/services/__init__.py
agents/search/services/news_integration.py
agents/search/services/news/__init__.py
agents/search/services/news/cache.py
agents/search/services/news/config.py
agents/search/services/news/metrics.py
agents/search/services/news/models.py
agents/search/services/news/service.py
agents/search/services/news/clients/__init__.py
agents/search/services/news/clients/devto.py
agents/search/services/news/clients/hackernews.py
agents/search/services/utils/__init__.py
agents/search/services/utils/rate_limiter.py
api/__init__.py
core/__init__.py
core/config.py
core/feedback_loop.py
core/models.py
core/orchestrator.py
core/rabbitmq_utils.py
core/testa api.py
core/domains/__init__.py
core/domains/adoption_metrics.py
core/domains/bigtech_monitor.py
core/domains/definitions.py
core/domains/market_segments.py
core/domains/seasonality.py
core/domains/tech_stacks.py
examples/__init__.py
examples/news_integration_example.py
infrastructure/__init__.py
tests/__init__.py
tests/test_config.py
tests/test_content.py
tests/test_devto_client.py
tests/test_feedback_loop.py
tests/test_news_integration.py
tests/test_orchestrator.py
tests/test_planning.py
tests/test_review.py
tests/test_search.py
```

## ad_agents.egg-info\top_level.txt

```python
agents
api
core
examples
infrastructure
tests

```

## agents\__init__.py

```python
# agents/__init__.py
from agents.search.services.news import NewsIntegrationService, NewsArticle, NewsSearchQuery

__all__ = ['NewsIntegrationService', 'NewsArticle', 'NewsSearchQuery']
```

## agents\action\__init__.py

```python

```

## agents\content\agent.py

```python
from typing import List, Dict
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from core.rabbitmq_utils import RabbitMQUtils
from core.models import PlanningGenerated, ContentGenerated, ContentImproved
from pydantic import ValidationError
from core.config import get_settings
import json
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContentAgent")

class ContentAgent:
    """Agente responsável pela geração e melhoria de conteúdo"""

    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model_name=self.settings.api.openai_model,
            temperature=self.settings.api.openai_temperature,
            openai_api_key=self.settings.api.openai_api_key
        )
        self.rabbitmq = RabbitMQUtils()
        self.setup_chains()

    def setup_chains(self):
        """Configura as chains do LangChain"""
        content_template = """
        Crie conteúdo técnico com os seguintes parâmetros:
        Tópico: {topic}
        Palavras-chave: {keywords}
        Público-alvo: {target_audience}
        Tom: {tone}
        Diretrizes de SEO: {seo_guidelines}
        Referências: {references}
        """
        self.content_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["topic", "keywords", "target_audience", "tone", "seo_guidelines", "references"],
                template=content_template
            )
        )

    def _serialize_message(self, message: dict) -> str:
        """Converte mensagem em JSON serializável"""
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Tipo não serializável: {type(obj)}")
        return json.dumps(message, default=default_serializer)

    def consume_plans(self):
        """Consome mensagens da fila 'planning.generated'"""
        def callback(ch, method, properties, body):
           plans = json.loads(body)  # Carregar a lista de planos
           for plan_data in plans:  # Iterar sobre cada plano
              try:
                if isinstance(plan_data.get("target_audience"), list):
                 plan_data["target_audience"] = ", ".join(plan_data["target_audience"])
                plan = PlanningGenerated(**plan_data)  # Criar instância de cada plano
                print(f"Plano recebido: {plan}")

                # Gera conteúdo com base no plano
                content = self.generate_content(plan)

                # Publica na fila 'content.generated'
                self.rabbitmq.publish_event("content.generated", self._serialize_message(content.dict()))
                print(f"Conteúdo gerado publicado: {content}")

              except ValidationError as e:
               print(f"Erro de validação: {e}")
              except Exception as e:
               print(f"Erro ao processar plano: {e}")

        self.rabbitmq.consume_event("planning.generated", callback)

    def consume_improvements(self):
        """Consome solicitações de melhoria da fila 'content.improved'"""
        def callback(ch, method, properties, body):
            try:
                improvement_request = ContentImproved(**json.loads(body))
                logger.info(f"Solicitação de melhoria recebida: {improvement_request}")

                improved_content = self.generate_improved_content(improvement_request)

                self.rabbitmq.publish_event("content.generated", self._serialize_message(improved_content.dict()))
                logger.info(f"Conteúdo melhorado publicado: {improved_content}")

            except ValidationError as e:
                logger.error(f"Erro de validação: {e}")
            except Exception as e:
                logger.error(f"Erro ao processar melhoria: {e}")

        self.rabbitmq.consume_event("content.improved", callback)

    def generate_content(self, plan: PlanningGenerated) -> ContentGenerated:
        """Gera conteúdo baseado no plano"""
        content = self.content_chain.run({
            "topic": plan.topic,
            "keywords": ", ".join(plan.keywords),
            "target_audience": plan.target_audience,
            "tone": "technical",
            "seo_guidelines": "Utilize palavras-chave naturalmente, título otimizado, e meta descrição.",
            "references": "Nenhuma referência disponível."
        })
        return ContentGenerated(
            content=content,
            title=f"Artigo sobre {plan.topic}",
            meta_description=f"Aprenda mais sobre {plan.topic}.",
            keywords=plan.keywords,
            seo_score=0.85,
            readability_score=0.9,
            created_at=datetime.now()
        )

    def generate_improved_content(self, improvement_request: ContentImproved) -> ContentGenerated:
        """Gera conteúdo melhorado"""
        improved_content = f"Versão melhorada: {improvement_request.content}"
        return ContentGenerated(
            content=improved_content,
            title=f"Melhoria: {improvement_request.content_id}",
            meta_description="Texto aprimorado com base no feedback.",
            keywords=[],
            seo_score=0.9,
            readability_score=0.95,
            created_at=datetime.now()
        )

    def start_consumers(self):
        """Inicia os consumidores em threads separadas"""
        threading.Thread(target=self.consume_plans, daemon=True).start()
        threading.Thread(target=self.consume_improvements, daemon=True).start()


if __name__ == "__main__":
    agent = ContentAgent()
    logger.info("Consumidores iniciados. Pressione Ctrl+C para sair.")
    agent.start_consumers()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Finalizando consumidores...")

```

## agents\content\config.py

```python
# agents\content\config.py
from pydantic import BaseSettings
from typing import Dict, List

class ContentAgentConfig(BaseSettings):
    """Configurações do Content Agent"""
    
    # Parâmetros de geração
    max_tokens: int = 2000
    temperature: float = 0.7
    
    # Templates padrão
    default_templates: Dict[str, Dict] = {
        "blog_post": {
            "sections": ["intro", "main", "conclusion"],
            "tone": "professional",
            "min_words": 1000
        },
        "social_media": {
            "sections": ["header", "body", "cta"],
            "tone": "engaging",
            "min_words": 100
        },
        "technical_article": {
            "sections": ["abstract", "intro", "methods", "results", "conclusion"],
            "tone": "academic",
            "min_words": 2000
        }
    }
    
    # SEO
    min_keyword_density: float = 0.01
    max_keyword_density: float = 0.03
    min_heading_count: int = 3
    
    # Qualidade
    min_readability_score: float = 0.7
    min_seo_score: float = 0.8
    
    # Rate limits
    max_generations_per_minute: int = 10
    
    # Cache
    cache_ttl: int = 3600  # 1 hora
    
    class Config:
        env_prefix = "CONTENT_"
```

## agents\content\__init__.py

```python

```

## agents\planning\agent.py

```python
import json
import threading
from datetime import datetime
from core.domains.adoption_metrics import AdoptionMetricsManager
from core.domains.seasonality import SeasonalityManager
from core.domains.market_segments import MarketSegmentManager
from core.domains.tech_stacks import TechStackManager
from core.domains.definitions import DomainManager
from core.rabbitmq_utils import RabbitMQUtils
from agents.planning.config import config
from core.config import get_settings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class PlanningAgent:
    """Agente de Planejamento Inteligente para Criação de Conteúdos"""

    def __init__(self):
        # Instâncias principais
        self.rabbitmq = RabbitMQUtils()
        self.adoption_manager = AdoptionMetricsManager()
        self.seasonality_manager = SeasonalityManager()
        self.market_manager = MarketSegmentManager()
        self.tech_stack_manager = TechStackManager()
        self.domain_manager = DomainManager()

        # Configuração do LLM
        self.llm = ChatOpenAI(
            model_name=get_settings().api.openai_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=get_settings().api.openai_api_key,
        )
        self.setup_chain()

        # Evento para controle do loop
        self.stop_event = threading.Event()

    def setup_chain(self):
        """Configura o LangChain para planejamento"""
        self.plan_template = PromptTemplate(
            input_variables=["context", "insights"],
            template=(
                "Você é um planejador de conteúdo avançado. Com base no seguinte contexto:\n"
                "{context}\n"
                "e nas informações a seguir:\n"
                "{insights}\n"
                "Crie um plano de conteúdo no formato JSON com as seguintes chaves:\n"
                "topic (string), keywords (array de strings), target_audience (array de strings), "
                "content_type (string), priority (inteiro), estimated_impact (float).\n"
                "Certifique-se de que o JSON seja válido e bem formatado. Não inclua texto fora do JSON."
            ),
        )
        self.plan_chain = LLMChain(llm=self.llm, prompt=self.plan_template)

    def generate_plan(self):
        """Gera um plano de conteúdo baseado em insights dos domínios"""
        print("[LOG] Gerando plano...")
        insights = []

        # 1. Adicionar insights de frameworks e stacks
        try:
            for framework_id, framework in self.tech_stack_manager.frameworks.items():
                insights.append(self._generate_framework_insight(framework))
        except Exception as e:
            print(f"[ERROR] Falha ao gerar insights de frameworks: {e}")

        # 2. Adicionar eventos sazonais
        try:
            for event in self.seasonality_manager.get_current_events():
                insights.append(self._generate_seasonal_insight(event))
        except Exception as e:
            print(f"[ERROR] Falha ao gerar insights de eventos sazonais: {e}")

        # 3. Adicionar métricas de adoção
        try:
            for metric_id, metric in self.adoption_manager.metrics.items():
                insights.append(self._generate_adoption_insight(metric))
        except Exception as e:
            print(f"[ERROR] Falha ao gerar insights de métricas de adoção: {e}")

        # 4. Adicionar segmentos de mercado
        try:
            for segment_id, segment in self.market_manager.segments.items():
                insights.append(self._generate_market_insight(segment))
        except Exception as e:
            print(f"[ERROR] Falha ao gerar insights de segmentos de mercado: {e}")

        # 5. Contexto de domínios
        try:
            context = {
                "domains": [domain.name for domain in self.domain_manager.domains.values()],
                "guidelines": [
                    self.domain_manager.get_content_guidelines(domain_id)
                    for domain_id in self.domain_manager.domains.keys()
                ],
            }
        except Exception as e:
            print(f"[ERROR] Falha ao gerar contexto de domínios: {e}")
            context = {}

        # Gerar plano com LLM
        try:
            response = self.plan_chain.run({"context": json.dumps(context), "insights": json.dumps(insights)})
            cleaned_response = response.strip("```").strip("json").strip() 
            print(f"[DEBUG] Resposta limpa do LLM: {cleaned_response}")
           #print(f"[DEBUG] Resposta do LLM: {response}")
            plan = json.loads(cleaned_response)
            # Publicar o plano na fila
            self.rabbitmq.publish_event("planning.generated", json.dumps(plan))
            print(f"[LOG] Plano publicado: {plan}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Falha ao decodificar plano gerado: {e}")
        except Exception as e:
            print(f"[ERROR] Falha ao gerar plano com LLM: {e}")

    def _generate_framework_insight(self, framework):
        """Gera insight para frameworks"""
        return {
            "type": "framework",
            "name": framework.name,
            "category": framework.category,
            "features": framework.key_features,
            "use_cases": framework.use_cases,
            "maturity": framework.maturity,
        }

    def _generate_seasonal_insight(self, event):
        """Gera insight para eventos sazonais"""
        return {
            "type": "seasonal_event",
            "name": event.name,
            "impact_level": event.impact_level,
            "key_themes": event.key_themes,
            "affected_industries": event.affected_industries,
        }

    def _generate_adoption_insight(self, metric):
        """Gera insight para métricas de adoção"""
        return {
            "type": "adoption_metric",
            "technology": metric.technology,
            "growth_rate": metric.growth_rate,
            "current_stage": metric.current_stage,
            "market_penetration": metric.market_penetration,
        }

    def _generate_market_insight(self, segment):
        """Gera insight para segmentos de mercado"""
        return {
            "type": "market_segment",
            "name": segment.name,
            "industry": segment.industry,
            "technologies": segment.key_technologies,
            "opportunities": segment.opportunities,
        }

    def start(self):
        """Inicia o agente de planejamento em um intervalo definido"""
        print("[LOG] Iniciando loop do agente de planejamento.")
        while not self.stop_event.is_set():
            self.generate_plan()
            self.stop_event.wait(config.planning_interval)

    def stop(self):
        """Para a execução do thread"""
        self.stop_event.set()


if __name__ == "__main__":
    agent = PlanningAgent()
    print(f"[{datetime.now()}] Planning Agent iniciado.")
    try:
        agent.start()
    except KeyboardInterrupt:
        print("[LOG] Finalizando Planning Agent...")
        agent.stop()
```

## agents\planning\config.py

```python
# agents/planning/config.py
from core.config import get_settings
from pydantic import BaseSettings

class PlanningAgentConfig(BaseSettings):
    """
    Configurações para o Planning Agent.
    Estas definições incluem integrações, parâmetros de geração e limites de métricas.
    """

    # OpenAI
    settings = get_settings()
    openai_api_key: str = settings.api.openai_api_key

    # Parâmetros do agente
    temperature: float = 0.7  # Criatividade do modelo
    max_tokens: int = 1500  # Limite de tokens para geração de conteúdo
    planning_interval: int = 120  # Intervalo de planejamento (em segundos)

    # Limiares para métricas e relevância
    min_trend_score: float = 0.5  # Score mínimo para considerar uma tendência
    min_relevance_score: float = 0.6  # Relevância mínima para incluir no plano
    min_confidence_score: float = 0.7  # Confiança mínima para publicar

    # Configurações de cache
    cache_ttl: int = 1800  # Tempo de vida do cache (em segundos)

    # Domínios
    enable_domain_flexibility: bool = True  # Permite criar planos fora de domínios pré-definidos
    default_domain_priority: str = "medium"  # Prioridade padrão para planos fora de domínios

    # Configurações de publicação
    publishing_frequency: str = "daily"  # Frequência de publicação (daily, weekly, monthly)

    class Config:
        env_prefix = "PLANNING_"  # Prefixo para variáveis de ambiente


# Instância global de configuração para facilitar o uso
config = PlanningAgentConfig()

```

## agents\planning\__init__.py

```python

```

## agents\review\agent.py

```python
from typing import Dict, List
from pydantic import BaseModel, Field
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from core.rabbitmq_utils import RabbitMQUtils
from core.config import get_settings
import asyncio

class ReviewResult(BaseModel):
    """Resultado da revisão de conteúdo"""
    content: str
    quality_score: float = Field(ge=0.0, le=1.0)
    seo_score: float = Field(ge=0.0, le=1.0)
    technical_accuracy: float = Field(ge=0.0, le=1.0)
    readability_score: float = Field(ge=0.0, le=1.0)
    suggestions: List[str]
    approved: bool
    review_date: datetime = Field(default_factory=datetime.now)

class ReviewAgent:
    """Agente responsável por revisar e validar conteúdo"""

    def __init__(self):
        settings = get_settings()
        self.rabbitmq = RabbitMQUtils()  # Integração com RabbitMQ
        self.llm = ChatOpenAI(
            model_name="gpt-4-1106-preview",
            temperature=0.3,
            openai_api_key=settings.api.openai_api_key
        )
        self.setup_chains()

    def setup_chains(self):
        """Configura as chains do LangChain"""
        review_template = """
        Como especialista em revisão técnica, avalie o seguinte conteúdo:
        CONTEÚDO:
        {content}
        CONTEXTO:
        {context}
        CRITÉRIOS DE AVALIAÇÃO:
        1. Qualidade geral do conteúdo
        2. Precisão técnica e acurácia
        3. Otimização para SEO
        4. Legibilidade e clareza
        5. Adequação ao público-alvo
        6. Estrutura e formatação
        7. Exemplos e referências
        8. Coerência e consistência
        Por favor, formate sua resposta exatamente assim:
        ---QUALITY_SCORE---
        [score de 0 a 1]
        ---SEO_SCORE---
        [score de 0 a 1]
        ---TECHNICAL_ACCURACY---
        [score de 0 a 1]
        ---READABILITY_SCORE---
        [score de 0 a 1]
        ---APPROVED---
        [true/false]
        ---SUGGESTIONS---
        - [sugestão 1]
        - [sugestão 2]
        ---REVISED_CONTENT---
        [conteúdo revisado e melhorado]
        """
        self.review_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=review_template,
                input_variables=["content", "context"]
            )
        )

    async def review_content(self, content: str, context: Dict) -> ReviewResult:
        """
        Revisa o conteúdo e retorna resultado detalhado.
        Args:
            content: Conteúdo a ser revisado.
            context: Contexto e metadados do conteúdo.
        Returns:
            ReviewResult com scores e sugestões.
        """
        try:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            result = await self.review_chain.arun({"content": content, "context": context_str})
            return self._process_review(result)
        except Exception as e:
            print(f"Erro na revisão: {str(e)}")
            raise

    def consume_generated_content(self):
        """Consome mensagens da fila 'content.generated' e revisa o conteúdo"""
        def callback(ch, method, properties, body):
            import json
            message = json.loads(body)
            print(f"Conteúdo recebido para revisão: {message}")
            try:
                # Revisar conteúdo
                review_result = asyncio.run(
                    self.review_content(content=message["content"], context={"domain": message["domain"]})
                )
                # Publicar resultados da revisão
                self.rabbitmq.publish_event("content.reviewed", review_result.dict())
                print(f"Revisão publicada: {review_result.dict()}")
            except Exception as e:
                print(f"Erro ao processar mensagem: {str(e)}")

        self.rabbitmq.consume_event("content.generated", callback)

    def _process_review(self, raw_review: str) -> ReviewResult:
        """Processa o resultado bruto da revisão"""
        parts = raw_review.split("---")
        review_data = {}
        for i, part in enumerate(parts):
            if "QUALITY_SCORE" in part and i+1 < len(parts):
                review_data["quality_score"] = float(parts[i+1].strip())
            elif "SEO_SCORE" in part and i+1 < len(parts):
                review_data["seo_score"] = float(parts[i+1].strip())
            elif "TECHNICAL_ACCURACY" in part and i+1 < len(parts):
                review_data["technical_accuracy"] = float(parts[i+1].strip())
            elif "READABILITY_SCORE" in part and i+1 < len(parts):
                review_data["readability_score"] = float(parts[i+1].strip())
            elif "APPROVED" in part and i+1 < len(parts):
                review_data["approved"] = "true" in parts[i+1].lower()
            elif "SUGGESTIONS" in part and i+1 < len(parts):
                review_data["suggestions"] = [
                    s.strip() for s in parts[i+1].split("\n") 
                    if s.strip() and s.strip().startswith("-")
                ]
            elif "REVISED_CONTENT" in part and i+1 < len(parts):
                review_data["content"] = parts[i+1].strip()
        return ReviewResult(**review_data)
```

## agents\review\config.py

```python
# agents/review/config.py
from pydantic import BaseSettings, Field
from typing import Dict, List

class ReviewAgentConfig(BaseSettings):
   """Configurações do Review Agent"""
   
   # Thresholds de qualidade
   min_quality_score: float = Field(0.7, env='REVIEW_MIN_QUALITY_SCORE')
   min_technical_accuracy: float = Field(0.8, env='REVIEW_MIN_TECHNICAL_ACCURACY')
   min_seo_score: float = Field(0.7, env='REVIEW_MIN_SEO_SCORE')
   min_readability_score: float = Field(0.7, env='REVIEW_MIN_READABILITY_SCORE')
   
   # Auto-aprovação
   auto_approve_threshold: float = Field(0.9, env='REVIEW_AUTO_APPROVE_THRESHOLD')
   require_manual_review: bool = Field(False, env='REVIEW_REQUIRE_MANUAL')
   
   # Cache
   cache_ttl: int = Field(3600, env='REVIEW_CACHE_TTL')
   max_cache_items: int = Field(1000, env='REVIEW_MAX_CACHE_ITEMS')
   
   # Limites
   max_content_length: int = Field(10000, env='REVIEW_MAX_CONTENT_LENGTH')
   max_suggestions: int = Field(10, env='REVIEW_MAX_SUGGESTIONS')
   
   review_metrics: Dict[str, float] = {
       "quality": 0.3,
       "technical": 0.3,
       "seo": 0.2,
       "readability": 0.2
   }
   
   required_checks: List[str] = [
       "grammar",
       "technical_accuracy",
       "seo_optimization",
       "readability",
       "code_quality"
   ]

   class Config:
       env_prefix = 'REVIEW_'
       case_sensitive = False
```

## agents\review\__init__.py

```python

```

## agents\search\agent.py

```python
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import asyncio
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.rabbitmq_utils import RabbitMQUtils
from core.config import get_settings
import json
import time
from .services.news.metrics import NewsMetrics
import aiohttp
from .services.news.clients.hackernews import HackerNewsClient
from .services.news.clients.devto import DevToClient
from .services.news.config import NewsApiConfig
import logging

logger = logging.getLogger(__name__)

class SearchResult(BaseModel):
    """Modelo para resultados de busca"""
    title: str
    url: str
    author: str
    source: str
    published_date: datetime
    summary: str
    tags: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float

class ContentValidation(BaseModel):
    """Modelo para validação de conteúdo"""
    claim: str
    is_valid: bool
    confidence_score: float
    supporting_sources: List[str]
    suggestions: Optional[List[str]]

class AudienceInsight(BaseModel):
    """Modelo para insights sobre audiência"""
    preferences: List[str]
    pain_points: List[str]
    technical_level: str
    common_questions: List[str]
    preferred_formats: List[str]

class SEOInsight(BaseModel):
    """Modelo para insights de SEO"""
    primary_keywords: List[tuple]  # (keyword, volume)
    related_keywords: List[tuple]
    questions: List[str]
    competing_content: List[Dict]
    suggested_structure: Dict[str, Any]

class EnhancedSearchAgent:
    """Agente de busca aprimorado com múltiplas funcionalidades"""

    def __init__(self):
        self.settings = get_settings()
        self.rabbitmq = RabbitMQUtils()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.setup_vector_store()
        self.setup_cache()
        self.metrics = NewsMetrics()
        self.news_config = NewsApiConfig()
        self.hacker_news_client = HackerNewsClient(self.news_config.HACKER_NEWS_API_URL, self.news_config.HACKER_NEWS_API_KEY)
        self.dev_to_client = DevToClient(self.news_config.DEVTO_API_URL, self.news_config.DEVTO_API_KEY)
        self.session = None

    async def initialize(self):
        """Inicializa o agente de pesquisa"""
        logger.info("Inicializando o agente de pesquisa")
        await self.hacker_news_client.initialize()
        await self.dev_to_client.initialize()
        if not self.session:
            logger.info("Inicializando sessão HTTP para o agente de pesquisa")
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Fecha conexões do agente de pesquisa"""
        logger.info("Fechando conexões do agente de pesquisa")
        await self.hacker_news_client.close()
        await self.dev_to_client.close()
        if self.session:
            await self.session.close()
            self.session = None

    def setup_vector_store(self):
        """Configura armazenamento vetorial"""
        logger.info("Configurando armazenamento vetorial")
        self.vector_store = FAISS.from_texts(
            texts=["inicialização do índice"],
            embedding=self.embeddings,
            metadatas=[{"source": "initialization"}]
        )

    def setup_cache(self):
        """Configura sistema de cache"""
        logger.info("Configurando sistema de cache")
        self.cache = {}
        self.cache_ttl = 3600  # 1 hora

    async def enrich_content_plan(self, topic: str, keywords: List[str], target_audience: str) -> Dict:
        """
        Enriquece o plano de conteúdo com pesquisas e análises
        """
        logger.info(f"Enriquecendo plano de conteúdo para o tópico: {topic}")
        tasks = [
            self.search_recent_developments(topic),
            self.validate_technical_aspects(topic),
            self.analyze_similar_content(topic, keywords),
            self.gather_seo_insights(keywords),
            self.analyze_audience_preferences(target_audience)
        ]

        results = await asyncio.gather(*tasks)

        return {
            "recent_developments": results[0],
            "technical_validations": results[1],
            "competitive_analysis": results[2],
            "seo_insights": results[3],
            "audience_insights": results[4]
        }

    async def search_recent_developments(self, topic: str) -> List[SearchResult]:
        """
        Busca desenvolvimentos recentes sobre o tópico
        """
        logger.info(f"Buscando desenvolvimentos recentes sobre o tópico: {topic}")
        # Integração com a API do Hacker News
        async with self.metrics.track_request("hacker_news"):
            logger.info("Buscando artigos no Hacker News")
            hacker_news_results = await self.hacker_news_client.search_stories(topic)
            logger.debug(f"Resultados do Hacker News: {hacker_news_results}")

        # Integração com a API do Dev.to
        async with self.metrics.track_request("dev_to"):
            logger.info("Buscando artigos no Dev.to")
            dev_to_results = await self.dev_to_client.search_articles(topic)
            logger.debug(f"Resultados do Dev.to: {dev_to_results}")

        # Combinar resultados de todas as fontes
        return hacker_news_results + dev_to_results

    async def validate_technical_aspects(self, topic: str) -> List[ContentValidation]:
        """
        Valida aspectos técnicos do tópico
        """
        logger.info(f"Validando aspectos técnicos do tópico: {topic}")
        # Implementar validação contra fontes técnicas confiáveis
        async with self.metrics.track_request("technical_validation"):
            return [
                ContentValidation(
                    claim=f"Validação técnica para {topic}",
                    is_valid=True,
                    confidence_score=0.85,
                    supporting_sources=["docs.python.org"],
                    suggestions=["Adicionar mais exemplos práticos"]
                )
            ]

    async def analyze_similar_content(self, topic: str, keywords: List[str]) -> Dict:
        """
        Analisa conteúdo similar existente
        """
        logger.info(f"Analisando conteúdo similar para o tópico: {topic}")
        results = await self._search_vector_store(topic)

        # Análise de gaps e oportunidades
        return {
            "similar_content": results,
            "content_gaps": ["Gap 1", "Gap 2"],
            "unique_angles": ["Ângulo 1", "Ângulo 2"]
        }

    async def gather_seo_insights(self, keywords: List[str]) -> SEOInsight:
        """
        Coleta insights de SEO
        """
        logger.info(f"Coletando insights de SEO para as palavras-chave: {keywords}")
        # Implementar integração com APIs de SEO
        async with self.metrics.track_request("seo_insights"):
            return SEOInsight(
                primary_keywords=[("python", 1000)],
                related_keywords=[("python programming", 800)],
                questions=["How to learn Python?"],
                competing_content=[],
                suggested_structure={
                    "introduction": ["key_point_1", "key_point_2"],
                    "main_sections": ["section_1", "section_2"],
                    "conclusion": ["summary", "next_steps"]
                }
            )

    async def analyze_audience_preferences(self, target_audience: str) -> AudienceInsight:
        """
        Analisa preferências da audiência
        """
        logger.info(f"Analisando preferências da audiência: {target_audience}")
        async with self.metrics.track_request("audience_analysis"):
            return AudienceInsight(
                preferences=["Clear explanations", "Code examples"],
                pain_points=["Complex documentation", "Lack of examples"],
                technical_level="intermediate",
                common_questions=["How to start?", "Best practices?"],
                preferred_formats=["Tutorials", "How-to guides"]
            )

    async def _search_vector_store(self, query: str) -> List[SearchResult]:
        """
        Realiza busca no armazenamento vetorial
        """
        logger.info(f"Realizando busca no armazenamento vetorial com o termo: {query}")
        query_embedding = self.embeddings.embed_query(query)
        results = self.vector_store.similarity_search_with_score(query, k=5)

        return [
            SearchResult(
                content=result[0].page_content,
                source="vector_store",
                relevance_score=float(result[1]),
                metadata=result[0].metadata
            )
            for result in results
        ]

    def _calculate_relevance(self, item: Dict, query: str) -> float:
        """Calcula pontuação de relevância para um item"""
        logger.debug(f"Calculando pontuação de relevância para o item: {item}")
        score = 0.0

        # Relevância do título
        if query.lower() in item["title"].lower():
            score += 0.4

        # Relevância do texto/resumo
        if query.lower() in (item.get("story_text") or "").lower() or \
           query.lower() in (item.get("comment_text") or "").lower() or \
           query.lower() in (item.get("description") or "").lower():
            score += 0.3

        # Relevância da pontuação
        if item.get("points", 0) > 100:
            score += 0.2
        elif item.get("points", 0) > 50:
            score += 0.1

        # Relevância do número de comentários
        if item.get("num_comments", 0) > 50:
            score += 0.1

        return min(score, 1.0)

    async def index_content(self, content: str, metadata: Dict[str, Any]):
        """
        Indexa novo conteúdo no armazenamento vetorial
        """
        logger.info("Indexando novo conteúdo no armazenamento vetorial")
        chunks = self.text_splitter.split_text(content)
        chunk_metadatas = [metadata for _ in chunks]
        self.vector_store.add_texts(chunks, metadatas=chunk_metadatas)

    async def start_consuming(self):
        """
        Inicia consumo de mensagens do RabbitMQ
        """
        def callback(ch, method, properties, body):
            message = json.loads(body)
            logger.info(f"Mensagem recebida: {message}")

            # Processar mensagem e enriquecer conteúdo
            enriched_data = asyncio.run(self.enrich_content_plan(
                topic=message.get("topic", ""),
                keywords=message.get("keywords", []),
                target_audience=message.get("target_audience", "")
            ))

            # Publicar resultados enriquecidos
            self.rabbitmq.publish_event(
                "search.results",
                json.dumps(enriched_data, default=str)
            )

        self.rabbitmq.consume_event("planning.generated", callback)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = EnhancedSearchAgent()
    logger.info("Search Agent iniciado. Aguardando mensagens...")
    asyncio.run(agent.start_consuming())
```

## agents\search\config.py

```python
# agents/search/config.py

from typing import Dict, List, Optional
from pydantic import BaseSettings, Field, HttpUrl

class SearchAgentConfig(BaseSettings):
    """Configurações avançadas para o Search Agent"""
    
    # Configurações de API
    OPENAI_API_KEY: str = Field(..., env='OPENAI_API_KEY')
    
    # Configurações de Embedding
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Vector Store
    VECTOR_STORE_TYPE: str = "faiss"  # faiss ou pinecone
    VECTOR_SIMILARITY_THRESHOLD: float = 0.75
    MAX_RESULTS_PER_QUERY: int = 10
    
    # Cache
    CACHE_TTL: int = 3600  # 1 hora
    MAX_CACHE_ITEMS: int = 10000
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = 60
    MAX_TOKENS_PER_MINUTE: int = 100000
    
    # Content Validation
    MIN_CONFIDENCE_SCORE: float = 0.8
    REQUIRED_SUPPORTING_SOURCES: int = 2
    
    # Notícias e Atualizações
    NEWS_SOURCES: Dict[str, Dict] = {
        "tech_crunch": {
            "name": "TechCrunch",
            "base_url": "https://api.techcrunch.com/v1/",
            "priority": 1,
            "categories": ["technology", "ai", "cloud"]
        },
        "hacker_news": {
            "name": "Hacker News",
            "base_url": "http://hn.algolia.com/api/v1",
            "priority": 2,
            "categories": ["technology", "programming"]
        },
        "dev_to": {
            "name": "Dev.to",
            "base_url": "https://dev.to/api/",
            "priority": 3,
            "categories": ["development", "programming"]
        }
    }
    
    # Fonte de dados confiáveis
    TRUSTED_DOMAINS: List[str] = [
        "docs.python.org",
        "developer.mozilla.org",
        "kubernetes.io",
        "cloud.google.com",
        "aws.amazon.com",
        "azure.microsoft.com",
        "github.com",
        "stackoverflow.com",
        "arxiv.org",
        "research.google.com",
        "openai.com",
        "pytorch.org",
        "tensorflow.org"
    ]
    
    # Métricas de qualidade
    QUALITY_WEIGHTS: Dict[str, float] = {
        "relevance": 0.4,
        "freshness": 0.2,
        "authority": 0.2,
        "completeness": 0.2
    }
    
    # Parâmetros de processamento
    MAX_CONTENT_LENGTH: int = 100000
    MIN_CONTENT_LENGTH: int = 100
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Configurações de busca
    SEARCH_DEFAULTS: Dict[str, Any] = {
        "min_relevance": 0.6,
        "max_age_days": 365,
        "max_results": 20,
        "include_content": True
    }
    
    # Configurações de análise
    ANALYSIS_OPTIONS: Dict[str, bool] = {
        "extract_code_snippets": True,
        "extract_links": True,
        "analyze_sentiment": False,
        "detect_language": True,
        "generate_summary": True
    }
    
    # Timeouts e tentativas
    REQUEST_TIMEOUT: int = 30  # segundos
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1  # segundos
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PREFIX: str = "search_agent"
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_prefix = "SEARCH_"
        case_sensitive = True

    def get_news_source_config(self, source_id: str) -> Optional[Dict]:
        """Retorna configuração para uma fonte específica"""
        return self.NEWS_SOURCES.get(source_id)
    
    def is_trusted_domain(self, domain: str) -> bool:
        """Verifica se um domínio é confiável"""
        return any(domain.endswith(trusted) for trusted in self.TRUSTED_DOMAINS)
    
    def get_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calcula pontuação de qualidade baseada nos pesos definidos"""
        return sum(
            metrics.get(metric, 0) * weight
            for metric, weight in self.QUALITY_WEIGHTS.items()
        )

# Instância global de configuração
config = SearchAgentConfig()
```

## agents\search\__init__.py

```python
# agents/search/__init__.py
from agents.search.services.news import NewsIntegrationService, NewsArticle, NewsSearchQuery

__all__ = ['NewsIntegrationService', 'NewsArticle', 'NewsSearchQuery']
```

## agents\search\services\news_integration.py

```python
# agents/search/services/news_integration.py

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import aiohttp
from pydantic import BaseModel
import json

class NewsArticle(BaseModel):
    """Modelo para artigos de notícias"""
    title: str
    url: str
    source: str
    published_date: datetime
    summary: str
    relevance_score: float
    category: str

class NewsIntegrationService:
    """Serviço de integração com APIs de notícias técnicas"""

    def __init__(self):
        # Configurar APIs - em produção, mover para variáveis de ambiente
        self.news_apis = {
            "tech_crunch": {
                "url": "https://api.techcrunch.com/v1/",
                "key": "your_api_key"
            },
            "hacker_news": {
                "url": "https://hacker-news.firebaseio.com/v0/",
                "key": None
            },
            "dev_to": {
                "url": "https://dev.to/api/",
                "key": "your_api_key"
            }
        }
        self.session = None

    async def initialize(self):
        """Inicializa sessão HTTP"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Fecha sessão HTTP"""
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_recent_news(self, topic: str, days: int = 30) -> List[NewsArticle]:
        """Busca notícias recentes sobre um tópico"""
        if not self.session:
            await self.initialize()

        news_items = []
        tasks = [
            self._fetch_tech_crunch(topic, days),
            self._fetch_hacker_news(topic, days),
            self._fetch_dev_to(topic, days)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                news_items.extend(result)

        # Ordenar por relevância e data
        return sorted(
            news_items,
            key=lambda x: (x.relevance_score, x.published_date),
            reverse=True
        )

    async def _fetch_tech_crunch(self, topic: str, days: int) -> List[NewsArticle]:
        """Busca notícias do TechCrunch"""
        try:
            async with self.session.get(
                f"{self.news_apis['tech_crunch']['url']}search",
                params={
                    "q": topic,
                    "from_date": (datetime.now() - timedelta(days=days)).isoformat()
                },
                headers={"Authorization": f"Bearer {self.news_apis['tech_crunch']['key']}"}
            ) as response:
                data = await response.json()
                return [
                    NewsArticle(
                        title=article["title"],
                        url=article["url"],
                        source="TechCrunch",
                        published_date=datetime.fromisoformat(article["published_at"]),
                        summary=article["summary"],
                        relevance_score=self._calculate_relevance(article, topic),
                        category="technology"
                    )
                    for article in data.get("articles", [])
                ]
        except Exception as e:
            print(f"Erro ao buscar do TechCrunch: {str(e)}")
            return []

    async def _fetch_hacker_news(self, topic: str, days: int) -> List[NewsArticle]:
        """Busca notícias do Hacker News"""
        try:
            async with self.session.get(
                f"{self.news_apis['hacker_news']['url']}topstories.json"
            ) as response:
                story_ids = await response.json()
                stories = []
                
                # Limitar a 50 histórias para não sobrecarregar
                for story_id in story_ids[:50]:
                    async with self.session.get(
                        f"{self.news_apis['hacker_news']['url']}item/{story_id}.json"
                    ) as story_response:
                        story = await story_response.json()
                        if topic.lower() in story.get("title", "").lower():
                            stories.append(
                                NewsArticle(
                                    title=story["title"],
                                    url=story.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                                    source="Hacker News",
                                    published_date=datetime.fromtimestamp(story["time"]),
                                    summary=story.get("text", "No summary available"),
                                    relevance_score=self._calculate_relevance(story, topic),
                                    category="technology"
                                )
                            )
                return stories
        except Exception as e:
            print(f"Erro ao buscar do Hacker News: {str(e)}")
            return []

    async def _fetch_dev_to(self, topic: str, days: int) -> List[NewsArticle]:
        """Busca artigos do Dev.to"""
        try:
            async with self.session.get(
                f"{self.news_apis['dev_to']['url']}articles",
                params={"tag": topic, "top": "30"},
                headers={"api-key": self.news_apis['dev_to']['key']}
            ) as response:
                articles = await response.json()
                return [
                    NewsArticle(
                        title=article["title"],
                        url=article["url"],
                        source="Dev.to",
                        published_date=datetime.fromisoformat(article["published_at"]),
                        summary=article["description"],
                        relevance_score=self._calculate_relevance(article, topic),
                        category="technology"
                    )
                    for article in articles
                    if (datetime.now() - datetime.fromisoformat(article["published_at"])).days <= days
                ]
        except Exception as e:
            print(f"Erro ao buscar do Dev.to: {str(e)}")
            return []

    def _calculate_relevance(self, article: Dict, topic: str) -> float:
        """Calcula pontuação de relevância para um artigo"""
        relevance = 0.0
        
        # Relevância do título
        if topic.lower() in article.get("title", "").lower():
            relevance += 0.4
            
        # Relevância do conteúdo/resumo
        if topic.lower() in article.get("summary", "").lower() or \
           topic.lower() in article.get("description", "").lower() or \
           topic.lower() in article.get("text", "").lower():
            relevance += 0.3
            
        # Engajamento (se disponível)
        points = article.get("points", 0)
        comments = article.get("num_comments", 0) or article.get("comment_count", 0)
        
        if points > 100:
            relevance += 0.2
        elif points > 50:
            relevance += 0.1
            
        if comments > 50:
            relevance += 0.1
            
        return min(relevance, 1.0)
```

## agents\search\services\__init__.py

```python
# agents/search/services/__init__.py
from agents.search.services.news import NewsIntegrationService, NewsArticle, NewsSearchQuery

__all__ = ['NewsIntegrationService', 'NewsArticle', 'NewsSearchQuery']
```

## agents\search\services\news\cache.py

```python
# agents/search/services/news/cache.py

from typing import Dict, Optional, Any
import time
from datetime import datetime
import json
from .models import NewsArticle
from .config import NewsApiConfig

class NewsCache:
    """Gerenciador de cache para requisições de notícias"""
    
    def __init__(self, config: NewsApiConfig):
        self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_times: Dict[str, float] = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Recupera item do cache"""
        if key not in self._cache:
            return None
            
        # Verificar TTL
        if time.time() - self._cache_times[key] > self.config.CACHE_TTL:
            self._remove(key)
            return None
            
        return self._cache[key]
        
    def set(self, key: str, value: Any) -> None:
        """Armazena item no cache"""
        # Limpar cache se necessário
        if len(self._cache) >= self.config.MAX_CACHE_ITEMS:
            self._cleanup_cache()
            
        self._cache[key] = value
        self._cache_times[key] = time.time()
        
    def _remove(self, key: str) -> None:
        """Remove item do cache"""
        if key in self._cache:
            del self._cache[key]
        if key in self._cache_times:
            del self._cache_times[key]
            
    def _cleanup_cache(self) -> None:
        """Limpa itens expirados do cache"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_times.items()
            if current_time - timestamp > self.config.CACHE_TTL
        ]
        
        for key in expired_keys:
            self._remove(key)
            
        # Se ainda precisar de espaço, remove os itens mais antigos
        if len(self._cache) >= self.config.MAX_CACHE_ITEMS:
            sorted_keys = sorted(
                self._cache_times.items(),
                key=lambda x: x[1]
            )
            
            # Remove 20% dos itens mais antigos
            num_to_remove = len(sorted_keys) // 5
            for key, _ in sorted_keys[:num_to_remove]:
                self._remove(key)
                
    def get_cache_key(self, query_params: Dict[str, Any]) -> str:
        """Gera chave de cache para parâmetros de consulta"""
        # Ordenar parâmetros para garantir consistência
        sorted_params = sorted(query_params.items())
        return json.dumps(sorted_params, default=str)
        
    def clear(self) -> None:
        """Limpa todo o cache"""
        self._cache.clear()
        self._cache_times.clear()
        
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        current_time = time.time()
        return {
            "total_items": len(self._cache),
            "expired_items": sum(
                1 for timestamp in self._cache_times.values()
                if current_time - timestamp > self.config.CACHE_TTL
            ),
            "cache_size_percent": (len(self._cache) / self.config.MAX_CACHE_ITEMS) * 100,
            "oldest_item_age": max(
                current_time - min(self._cache_times.values())
                if self._cache_times else 0,
                0
            )
        }
```

## agents\search\services\news\config.py

```python
# agents/search/services/news/config.py

from pydantic import BaseSettings, Field
from typing import Dict, List, Optional
from datetime import timedelta
from pathlib import Path
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente do .env
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class NewsApiConfig(BaseSettings):
    """Configurações para integrações com APIs de notícias"""
    
    # Dev.to
    DEVTO_API_URL: str = Field("https://dev.to/api", env='DEVTO_API_URL')
    DEVTO_API_KEY: Optional[str] = Field(None, env='DEVTO_API_KEY')
    DEVTO_MAX_RESULTS: int = Field(100, env='DEVTO_MAX_RESULTS')
    DEVTO_RATE_LIMIT: int = Field(3000, env='DEVTO_RATE_LIMIT')
    
    # Hacker News
    HACKER_NEWS_API_URL: str = Field("http://hn.algolia.com/api/v1", env='HACKER_NEWS_API_URL')
    HACKER_NEWS_API_KEY: Optional[str] = Field(None, env='HACKER_NEWS_API_KEY')
    HACKER_NEWS_MAX_RESULTS: int = Field(100, env='HACKER_NEWS_MAX_RESULTS')
    HACKER_NEWS_RATE_LIMIT: int = Field(3000, env='HACKER_NEWS_RATE_LIMIT')

    # Cache
    CACHE_TTL: int = Field(3600, env='NEWS_CACHE_TTL')
    MAX_CACHE_ITEMS: int = Field(10000, env='NEWS_MAX_CACHE_ITEMS')
    
    # Relevância
    MIN_RELEVANCE_SCORE: float = Field(0.3, env='NEWS_MIN_RELEVANCE')
    RELEVANCE_WEIGHTS: Dict[str, float] = {
        "title_match": 0.4,
        "tag_match": 0.3,
        "content_match": 0.2,
        "engagement": 0.1
    }
    
    # Limites
    DEFAULT_MAX_RESULTS: int = Field(50, env='NEWS_DEFAULT_MAX_RESULTS')
    MAX_SEARCH_PERIOD: timedelta = Field(
        default_factory=lambda: timedelta(days=int(os.getenv('NEWS_MAX_SEARCH_PERIOD', '30'))),
    )
    
    # Configurações de fontes
    ENABLED_SOURCES: List[str] = Field(
        default=["dev.to", "hacker_news"],
        env='NEWS_ENABLED_SOURCES'
    )
    
    # Rate Limiting Global
    RATE_LIMIT_WINDOW: int = Field(60, env='NEWS_RATE_LIMIT_WINDOW')
    MAX_REQUESTS_PER_WINDOW: int = Field(1000, env='NEWS_MAX_REQUESTS_PER_WINDOW')
    
    class Config:
        env_prefix = "NEWS_"
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = 'utf-8'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Configurações de API carregadas")
        logger.debug(f"Configurações: {self.dict()}")
```

## agents\search\services\news\metrics.py

```python
# agents/search/services/news/metrics.py

from prometheus_client import Counter, Histogram, Gauge
from prometheus_client.registry import CollectorRegistry
import time
import asyncio

class NewsMetrics:
    """Sistema de métricas para o serviço de notícias"""

    def __init__(self):
        # Criar registry separado para evitar colisões
        self.registry = CollectorRegistry()

        # Contadores
        self.request_count = Counter(
            "news_integration_requests",
            "Total number of API requests",
            ["source", "status"],
            registry=self.registry
        )

        self.article_count = Counter(
            "news_integration_articles",
            "Total number of articles processed",
            ["source"],
            registry=self.registry
        )

        # Histogramas
        self.request_latency = Histogram(
            "news_integration_request_duration",
            "Request duration in seconds",
            ["source"],
            registry=self.registry
        )

        # Gauges
        self.active_requests = Gauge(
            "news_integration_active_requests",
            "Number of active requests",
            ["source"],
            registry=self.registry
        )

    def track_request(self, source: str):
        """Context manager para rastrear requisições"""
        class RequestTracker:
            def __init__(self, metrics, source):
                self.metrics = metrics
                self.source = source
                self.start_time = None

            async def __aenter__(self):
                self.start_time = time.time()
                self.metrics.active_requests.labels(source=self.source).inc()
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.metrics.request_latency.labels(source=self.source).observe(duration)
                self.metrics.active_requests.labels(source=self.source).dec()

                status = "error" if exc_type else "success"
                self.metrics.request_count.labels(source=self.source, status=status).inc()

        return RequestTracker(self, source)

    def record_processed_article(self, source: str):
        """Registra um artigo processado"""
        self.article_count.labels(source=source).inc()

    def get_metrics(self):
        """Retorna métricas atuais"""
        metrics = {
            "requests": {},
            "articles": {},
            "active_requests": {}
        }

        # Obter métricas de requisições para cada fonte
        for source in ["tech_crunch", "hacker_news", "dev.to"]:
            metrics["requests"][source] = {
                "success": self.request_count.labels(source=source, status="success")._value.get(),
                "error": self.request_count.labels(source=source, status="error")._value.get()
            }
            metrics["articles"][source] = self.article_count.labels(source=source)._value.get()
            metrics["active_requests"][source] = self.active_requests.labels(source=source)._value

        return metrics
```

## agents\search\services\news\models.py

```python
# agents/search/services/news/models.py

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl

class NewsSource(BaseModel):
    """Configuração de fonte de notícias"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=10)
    enabled: bool = True
    categories: List[str] = Field(default_factory=list)
    rate_limit: Optional[int] = None

class NewsSearchQuery(BaseModel):
    """Modelo para consultas de busca de notícias"""
    topic: str
    keywords: List[str] = Field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_relevance: float = 0.3
    max_results: Optional[int] = None

class NewsArticle(BaseModel):
    """Modelo para artigos de notícias"""
    title: str
    url: str
    source: str
    author: Optional[str]
    published_date: datetime
    summary: str
    content: Optional[str]
    tags: List[str] = Field(default_factory=list)
    category: str = "technology"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: Optional[float] = Field(default=0.0)
```

## agents\search\services\news\service.py

```python
# agents/search/services/news/service.py

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
from .models import NewsArticle, NewsSearchQuery
from .config import NewsApiConfig
from .cache import NewsCache
from .metrics import NewsMetrics
from .clients.devto import DevToClient
from .clients.hackernews import HackerNewsClient


logger = logging.getLogger(__name__)

def ensure_timezone(dt: Optional[datetime]) -> Optional[datetime]:
    """Garante que a data tem timezone (UTC se não especificado)"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

class NewsIntegrationService:
    """Serviço principal de integração de notícias"""

    def __init__(self):
        self.config = NewsApiConfig()
        self.cache = NewsCache(self.config)
        self.metrics = NewsMetrics()
        
        # Inicializar clientes
        self.devto_client = DevToClient(api_key=self.config.DEVTO_API_KEY)
        self.hacker_news_client = HackerNewsClient (api_key=self.config.HACKER_NEWS_API_KEY)
        
        self.session = None

    async def initialize(self):
        """Inicializa o serviço"""
        logger.info("Initializing News Integration Service")
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Fecha conexões"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("News Integration Service shut down")

    async def search_news(self, query: NewsSearchQuery) -> List[NewsArticle]:
        """Busca notícias com base nos parâmetros fornecidos"""
        try:
            # Garantir que as datas têm timezone
            query.start_date = ensure_timezone(query.start_date)
            query.end_date = ensure_timezone(query.end_date)

            # Verificar cache
            cache_key = str(query.dict())
            cached_results = self.cache.get(cache_key)
            if cached_results:
                logger.info(f"Cache hit for query: {query.topic}")
                return cached_results

            # Buscar em todas as fontes configuradas
            results = []
            
            # Busca no Dev.to
            async with self.metrics.track_request("dev.to"):
                devto_articles = await self.devto_client.get_articles(
                    search_term=query.topic,
                    tag=query.keywords[0] if query.keywords else None
                )
                for article in devto_articles:
                    results.append(self.devto_client.to_news_article(article))
                    self.metrics.record_processed_article("dev.to")

            # Filtrar e ordenar resultados
            filtered_results = self._filter_and_sort_results(results, query)

            # Armazenar no cache
            self.cache.set(cache_key, filtered_results)

            return filtered_results

        except Exception as e:
            logger.error(f"Error in search_news: {str(e)}")
            return []

    def _filter_and_sort_results(self, articles: List[NewsArticle], query: NewsSearchQuery) -> List[NewsArticle]:
        """Filtra e ordena os resultados"""
        filtered = []
        for article in articles:
            # Garantir que a data do artigo tem timezone
            article.published_date = ensure_timezone(article.published_date)
            
            # Filtrar por data
            if query.start_date and article.published_date < query.start_date:
                continue
            if query.end_date and article.published_date > query.end_date:
                continue

            # Filtrar por relevância
            if self._calculate_relevance(article, query) >= query.min_relevance:
                filtered.append(article)

        # Ordenar por data de publicação (mais recente primeiro)
        filtered.sort(key=lambda x: x.published_date, reverse=True)

        # Limitar número de resultados
        return filtered[:query.max_results] if query.max_results else filtered

    def _calculate_relevance(self, article: NewsArticle, query: NewsSearchQuery) -> float:
        """Calcula pontuação de relevância para um artigo"""
        score = 0.0
        
        # Relevância do título
        if query.topic.lower() in article.title.lower():
            score += 0.4
        
        # Relevância das tags
        if article.tags and any(keyword.lower() in tag.lower() for keyword in query.keywords for tag in article.tags):
            score += 0.3
        
        # Relevância do resumo
        if query.topic.lower() in article.summary.lower():
            score += 0.2
        
        # Bônus por engajamento
        if article.metadata:
            reactions = article.metadata.get("reactions_count", 0)
            comments = article.metadata.get("comments_count", 0)
            if reactions > 50 or comments > 10:
                score += 0.1
        
        return min(score, 1.0)
```

## agents\search\services\news\__init__.py

```python
# agents/search/services/news/__init__.py
from .service import NewsIntegrationService
from .models import NewsArticle, NewsSearchQuery

__all__ = ['NewsIntegrationService', 'NewsArticle', 'NewsSearchQuery']
```

## agents\search\services\news\clients\devto.py

```python
# agents/search/services/news/clients/devto.py

from typing import List, Optional
import aiohttp
from datetime import datetime, timezone
from pydantic import BaseModel, HttpUrl
import logging
from ...utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class DevToArticle(BaseModel):
    """Modelo para artigos do Dev.to"""
    id: int
    title: str
    description: Optional[str]
    url: HttpUrl
    published_at: datetime
    tag_list: List[str]
    user: dict
    reading_time_minutes: Optional[int]
    comments_count: Optional[int]
    public_reactions_count: Optional[int]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DevToClient:
    """Cliente para a API do Dev.to"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://dev.to/api"
        self.api_key = api_key
        # Dev.to tem limite de 3000 requests/hora
        self.rate_limiter = RateLimiter(max_calls=3000, period=3600)
        
    def _ensure_timezone(self, dt: datetime) -> datetime:
        """Garante que a data tem timezone (UTC se não especificado)"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    async def get_articles(
        self,
        search_term: Optional[str] = None,
        tag: Optional[str] = None,
        username: Optional[str] = None,
        page: int = 1,
        per_page: int = 30
    ) -> List[DevToArticle]:
        """Busca artigos no Dev.to"""
        params = {
            "page": page,
            "per_page": min(per_page, 1000)
        }
        
        if tag:
            params["tag"] = tag
        if username:
            params["username"] = username
            
        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    if search_term:
                        url = f"{self.base_url}/articles/search"
                        params["q"] = search_term
                    else:
                        url = f"{self.base_url}/articles"
                        
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status == 429:  # Too Many Requests
                            retry_after = int(response.headers.get("Retry-After", 60))
                            logger.warning(f"Rate limit hit. Retry after {retry_after} seconds")
                            return []
                            
                        response.raise_for_status()
                        data = await response.json()
                        
                        # Processar as datas antes de criar os objetos
                        for article in data:
                            if "published_at" in article:
                                dt = datetime.fromisoformat(article["published_at"].replace("Z", "+00:00"))
                                article["published_at"] = self._ensure_timezone(dt)
                        
                        return [DevToArticle(**article) for article in data]
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching from Dev.to: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return []
    
    def to_news_article(self, article: DevToArticle) -> "NewsArticle":
        """Converte DevToArticle para NewsArticle"""
        from ..models import NewsArticle
        
        return NewsArticle(
            title=article.title,
            url=str(article.url),
            source="dev.to",
            author=article.user.get("name"),
            published_date=self._ensure_timezone(article.published_at),
            summary=article.description or "",
            content=None,
            tags=article.tag_list,
            metadata={
                "reading_time": article.reading_time_minutes,
                "comments_count": article.comments_count,
                "reactions_count": article.public_reactions_count,
                "author_username": article.user.get("username")
            },
            relevance_score=0.0
        )
```

## agents\search\services\news\clients\hackernews.py

```python
# class HackerNewsClient:
  # agents/search/services/news/clients/devto.py

from typing import List, Optional
import aiohttp
from datetime import datetime, timezone
from pydantic import BaseModel, HttpUrl
import logging
from ...utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class HackerNewsClientArticle(BaseModel):
    """Modelo para artigos do Dev.to"""
    id: int
    title: str
    description: Optional[str]
    url: HttpUrl
    published_at: datetime
    tag_list: List[str]
    user: dict
    reading_time_minutes: Optional[int]
    comments_count: Optional[int]
    public_reactions_count: Optional[int]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HackerNewsClient:
    """Cliente para a API do Hacker News"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "http://hn.algolia.com/api/v1"
        self.api_key = api_key
        # Dev.to tem limite de 3000 requests/hora
        self.rate_limiter = RateLimiter(max_calls=3000, period=3600)
        
    def _ensure_timezone(self, dt: datetime) -> datetime:
        """Garante que a data tem timezone (UTC se não especificado)"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    async def get_articles(
        self,
        search_term: Optional[str] = None,
        tag: Optional[str] = None,
        username: Optional[str] = None,
        page: int = 1,
        per_page: int = 30
    ) -> List[HackerNewsClientArticle]:
        """Busca artigos no HackerNews"""
        params = {
            "page": page,
            "per_page": min(per_page, 1000)
        }
        
        if tag:
            params["tag"] = tag
        if username:
            params["username"] = username
            
        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    if search_term:
                        url = f"{self.base_url}/articles/search"
                        params["q"] = search_term
                    else:
                        url = f"{self.base_url}/articles"
                        
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status == 429:  # Too Many Requests
                            retry_after = int(response.headers.get("Retry-After", 60))
                            logger.warning(f"Rate limit hit. Retry after {retry_after} seconds")
                            return []
                            
                        response.raise_for_status()
                        data = await response.json()
                        
                        # Processar as datas antes de criar os objetos
                        for article in data:
                            if "published_at" in article:
                                dt = datetime.fromisoformat(article["published_at"].replace("Z", "+00:00"))
                                article["published_at"] = self._ensure_timezone(dt)
                        
                        return [HackerNewsClientArticle(**article) for article in data]
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching from Dev.to: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return []
    
    def to_news_article(self, article: HackerNewsClientArticle) -> "NewsArticle":
        """Converte DevToArticle para NewsArticle"""
        from ..models import NewsArticle
        
        return NewsArticle(
            title=article.title,
            url=str(article.url),
            source="Hacker News",
            author=article.user.get("name"),
            published_date=self._ensure_timezone(article.published_at),
            summary=article.description or "",
            content=None,
            tags=article.tag_list,
            metadata={
                "reading_time": article.reading_time_minutes,
                "comments_count": article.comments_count,
                "reactions_count": article.public_reactions_count,
                "author_username": article.user.get("username")
            },
            relevance_score=0.0
        )
```

## agents\search\services\news\clients\__init__.py

```python

```

## agents\search\services\utils\rate_limiter.py

```python
# agents/search/services/utils/rate_limiter.py

import asyncio
from datetime import datetime, timedelta
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limiter implementando algoritmo de sliding window
    """
    
    def __init__(self, max_calls: int, period: float):
        """
        Args:
            max_calls: Número máximo de chamadas permitidas no período
            period: Período em segundos
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: List[datetime] = []
        self._lock = asyncio.Lock()
        
    async def acquire(self) -> bool:
        """
        Tenta adquirir uma permissão do rate limiter
        
        Returns:
            bool: True se permitido, False se limite excedido
        """
        async with self._lock:
            now = datetime.now()
            
            # Remover chamadas antigas
            window_start = now - timedelta(seconds=self.period)
            self.calls = [call for call in self.calls if call > window_start]
            
            # Verificar se pode fazer nova chamada
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
                
            # Calcular tempo de espera
            if self.calls:
                next_available = self.calls[0] + timedelta(seconds=self.period)
                wait_time = (next_available - now).total_seconds()
                if wait_time > 0:
                    logger.warning(f"Rate limit exceeded. Wait {wait_time:.2f} seconds")
                    return False
                    
            self.calls.append(now)
            return True
    
    async def __aenter__(self):
        """Suporte para uso com 'async with'"""
        while not await self.acquire():
            await asyncio.sleep(1)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup ao sair do contexto"""
        pass

    def reset(self):
        """Reseta o rate limiter"""
        self.calls.clear()
```

## agents\search\services\utils\__init__.py

```python

```

## api\__init__.py

```python

```

## build\lib\agents\__init__.py

```python
# agents/__init__.py
from agents.search.services.news import NewsIntegrationService, NewsArticle, NewsSearchQuery

__all__ = ['NewsIntegrationService', 'NewsArticle', 'NewsSearchQuery']
```

## build\lib\agents\action\__init__.py

```python

```

## build\lib\agents\content\agent.py

```python
from typing import List, Dict
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from core.rabbitmq_utils import RabbitMQUtils
from core.models import PlanningGenerated, ContentGenerated, ContentImproved
from pydantic import ValidationError
from core.config import get_settings
import json
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContentAgent")

class ContentAgent:
    """Agente responsável pela geração e melhoria de conteúdo"""

    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model_name=self.settings.api.openai_model,
            temperature=self.settings.api.openai_temperature,
            openai_api_key=self.settings.api.openai_api_key
        )
        self.rabbitmq = RabbitMQUtils()
        self.setup_chains()

    def setup_chains(self):
        """Configura as chains do LangChain"""
        content_template = """
        Crie conteúdo técnico com os seguintes parâmetros:
        Tópico: {topic}
        Palavras-chave: {keywords}
        Público-alvo: {target_audience}
        Tom: {tone}
        Diretrizes de SEO: {seo_guidelines}
        Referências: {references}
        """
        self.content_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["topic", "keywords", "target_audience", "tone", "seo_guidelines", "references"],
                template=content_template
            )
        )

    def _serialize_message(self, message: dict) -> str:
        """Converte mensagem em JSON serializável"""
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Tipo não serializável: {type(obj)}")
        return json.dumps(message, default=default_serializer)

    def consume_plans(self):
        """Consome mensagens da fila 'planning.generated'"""
        def callback(ch, method, properties, body):
           plans = json.loads(body)  # Carregar a lista de planos
           for plan_data in plans:  # Iterar sobre cada plano
              try:
                if isinstance(plan_data.get("target_audience"), list):
                 plan_data["target_audience"] = ", ".join(plan_data["target_audience"])
                plan = PlanningGenerated(**plan_data)  # Criar instância de cada plano
                print(f"Plano recebido: {plan}")

                # Gera conteúdo com base no plano
                content = self.generate_content(plan)

                # Publica na fila 'content.generated'
                self.rabbitmq.publish_event("content.generated", self._serialize_message(content.dict()))
                print(f"Conteúdo gerado publicado: {content}")

              except ValidationError as e:
               print(f"Erro de validação: {e}")
              except Exception as e:
               print(f"Erro ao processar plano: {e}")

        self.rabbitmq.consume_event("planning.generated", callback)

    def consume_improvements(self):
        """Consome solicitações de melhoria da fila 'content.improved'"""
        def callback(ch, method, properties, body):
            try:
                improvement_request = ContentImproved(**json.loads(body))
                logger.info(f"Solicitação de melhoria recebida: {improvement_request}")

                improved_content = self.generate_improved_content(improvement_request)

                self.rabbitmq.publish_event("content.generated", self._serialize_message(improved_content.dict()))
                logger.info(f"Conteúdo melhorado publicado: {improved_content}")

            except ValidationError as e:
                logger.error(f"Erro de validação: {e}")
            except Exception as e:
                logger.error(f"Erro ao processar melhoria: {e}")

        self.rabbitmq.consume_event("content.improved", callback)

    def generate_content(self, plan: PlanningGenerated) -> ContentGenerated:
        """Gera conteúdo baseado no plano"""
        content = self.content_chain.run({
            "topic": plan.topic,
            "keywords": ", ".join(plan.keywords),
            "target_audience": plan.target_audience,
            "tone": "technical",
            "seo_guidelines": "Utilize palavras-chave naturalmente, título otimizado, e meta descrição.",
            "references": "Nenhuma referência disponível."
        })
        return ContentGenerated(
            content=content,
            title=f"Artigo sobre {plan.topic}",
            meta_description=f"Aprenda mais sobre {plan.topic}.",
            keywords=plan.keywords,
            seo_score=0.85,
            readability_score=0.9,
            created_at=datetime.now()
        )

    def generate_improved_content(self, improvement_request: ContentImproved) -> ContentGenerated:
        """Gera conteúdo melhorado"""
        improved_content = f"Versão melhorada: {improvement_request.content}"
        return ContentGenerated(
            content=improved_content,
            title=f"Melhoria: {improvement_request.content_id}",
            meta_description="Texto aprimorado com base no feedback.",
            keywords=[],
            seo_score=0.9,
            readability_score=0.95,
            created_at=datetime.now()
        )

    def start_consumers(self):
        """Inicia os consumidores em threads separadas"""
        threading.Thread(target=self.consume_plans, daemon=True).start()
        threading.Thread(target=self.consume_improvements, daemon=True).start()


if __name__ == "__main__":
    agent = ContentAgent()
    logger.info("Consumidores iniciados. Pressione Ctrl+C para sair.")
    agent.start_consumers()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Finalizando consumidores...")

```

## build\lib\agents\content\config.py

```python
# agents\content\config.py
from pydantic import BaseSettings
from typing import Dict, List

class ContentAgentConfig(BaseSettings):
    """Configurações do Content Agent"""
    
    # Parâmetros de geração
    max_tokens: int = 2000
    temperature: float = 0.7
    
    # Templates padrão
    default_templates: Dict[str, Dict] = {
        "blog_post": {
            "sections": ["intro", "main", "conclusion"],
            "tone": "professional",
            "min_words": 1000
        },
        "social_media": {
            "sections": ["header", "body", "cta"],
            "tone": "engaging",
            "min_words": 100
        },
        "technical_article": {
            "sections": ["abstract", "intro", "methods", "results", "conclusion"],
            "tone": "academic",
            "min_words": 2000
        }
    }
    
    # SEO
    min_keyword_density: float = 0.01
    max_keyword_density: float = 0.03
    min_heading_count: int = 3
    
    # Qualidade
    min_readability_score: float = 0.7
    min_seo_score: float = 0.8
    
    # Rate limits
    max_generations_per_minute: int = 10
    
    # Cache
    cache_ttl: int = 3600  # 1 hora
    
    class Config:
        env_prefix = "CONTENT_"
```

## build\lib\agents\content\__init__.py

```python

```

## build\lib\agents\planning\agent.py

```python
import json
import threading
from datetime import datetime
from core.domains.adoption_metrics import AdoptionMetricsManager
from core.domains.seasonality import SeasonalityManager
from core.domains.market_segments import MarketSegmentManager
from core.domains.tech_stacks import TechStackManager
from core.domains.definitions import DomainManager
from core.rabbitmq_utils import RabbitMQUtils
from agents.planning.config import config
from core.config import get_settings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class PlanningAgent:
    """Agente de Planejamento Inteligente para Criação de Conteúdos"""

    def __init__(self):
        # Instâncias principais
        self.rabbitmq = RabbitMQUtils()
        self.adoption_manager = AdoptionMetricsManager()
        self.seasonality_manager = SeasonalityManager()
        self.market_manager = MarketSegmentManager()
        self.tech_stack_manager = TechStackManager()
        self.domain_manager = DomainManager()

        # Configuração do LLM
        self.llm = ChatOpenAI(
            model_name=get_settings().api.openai_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=get_settings().api.openai_api_key,
        )
        self.setup_chain()

        # Evento para controle do loop
        self.stop_event = threading.Event()

    def setup_chain(self):
        """Configura o LangChain para planejamento"""
        self.plan_template = PromptTemplate(
            input_variables=["context", "insights"],
            template=(
                "Você é um planejador de conteúdo avançado. Com base no seguinte contexto:\n"
                "{context}\n"
                "e nas informações a seguir:\n"
                "{insights}\n"
                "Crie um plano de conteúdo no formato JSON com as seguintes chaves:\n"
                "topic (string), keywords (array de strings), target_audience (array de strings), "
                "content_type (string), priority (inteiro), estimated_impact (float).\n"
                "Certifique-se de que o JSON seja válido e bem formatado. Não inclua texto fora do JSON."
            ),
        )
        self.plan_chain = LLMChain(llm=self.llm, prompt=self.plan_template)

    def generate_plan(self):
        """Gera um plano de conteúdo baseado em insights dos domínios"""
        print("[LOG] Gerando plano...")
        insights = []

        # 1. Adicionar insights de frameworks e stacks
        try:
            for framework_id, framework in self.tech_stack_manager.frameworks.items():
                insights.append(self._generate_framework_insight(framework))
        except Exception as e:
            print(f"[ERROR] Falha ao gerar insights de frameworks: {e}")

        # 2. Adicionar eventos sazonais
        try:
            for event in self.seasonality_manager.get_current_events():
                insights.append(self._generate_seasonal_insight(event))
        except Exception as e:
            print(f"[ERROR] Falha ao gerar insights de eventos sazonais: {e}")

        # 3. Adicionar métricas de adoção
        try:
            for metric_id, metric in self.adoption_manager.metrics.items():
                insights.append(self._generate_adoption_insight(metric))
        except Exception as e:
            print(f"[ERROR] Falha ao gerar insights de métricas de adoção: {e}")

        # 4. Adicionar segmentos de mercado
        try:
            for segment_id, segment in self.market_manager.segments.items():
                insights.append(self._generate_market_insight(segment))
        except Exception as e:
            print(f"[ERROR] Falha ao gerar insights de segmentos de mercado: {e}")

        # 5. Contexto de domínios
        try:
            context = {
                "domains": [domain.name for domain in self.domain_manager.domains.values()],
                "guidelines": [
                    self.domain_manager.get_content_guidelines(domain_id)
                    for domain_id in self.domain_manager.domains.keys()
                ],
            }
        except Exception as e:
            print(f"[ERROR] Falha ao gerar contexto de domínios: {e}")
            context = {}

        # Gerar plano com LLM
        try:
            response = self.plan_chain.run({"context": json.dumps(context), "insights": json.dumps(insights)})
            cleaned_response = response.strip("```").strip("json").strip() 
            print(f"[DEBUG] Resposta limpa do LLM: {cleaned_response}")
           #print(f"[DEBUG] Resposta do LLM: {response}")
            plan = json.loads(cleaned_response)
            # Publicar o plano na fila
            self.rabbitmq.publish_event("planning.generated", json.dumps(plan))
            print(f"[LOG] Plano publicado: {plan}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Falha ao decodificar plano gerado: {e}")
        except Exception as e:
            print(f"[ERROR] Falha ao gerar plano com LLM: {e}")

    def _generate_framework_insight(self, framework):
        """Gera insight para frameworks"""
        return {
            "type": "framework",
            "name": framework.name,
            "category": framework.category,
            "features": framework.key_features,
            "use_cases": framework.use_cases,
            "maturity": framework.maturity,
        }

    def _generate_seasonal_insight(self, event):
        """Gera insight para eventos sazonais"""
        return {
            "type": "seasonal_event",
            "name": event.name,
            "impact_level": event.impact_level,
            "key_themes": event.key_themes,
            "affected_industries": event.affected_industries,
        }

    def _generate_adoption_insight(self, metric):
        """Gera insight para métricas de adoção"""
        return {
            "type": "adoption_metric",
            "technology": metric.technology,
            "growth_rate": metric.growth_rate,
            "current_stage": metric.current_stage,
            "market_penetration": metric.market_penetration,
        }

    def _generate_market_insight(self, segment):
        """Gera insight para segmentos de mercado"""
        return {
            "type": "market_segment",
            "name": segment.name,
            "industry": segment.industry,
            "technologies": segment.key_technologies,
            "opportunities": segment.opportunities,
        }

    def start(self):
        """Inicia o agente de planejamento em um intervalo definido"""
        print("[LOG] Iniciando loop do agente de planejamento.")
        while not self.stop_event.is_set():
            self.generate_plan()
            self.stop_event.wait(config.planning_interval)

    def stop(self):
        """Para a execução do thread"""
        self.stop_event.set()


if __name__ == "__main__":
    agent = PlanningAgent()
    print(f"[{datetime.now()}] Planning Agent iniciado.")
    try:
        agent.start()
    except KeyboardInterrupt:
        print("[LOG] Finalizando Planning Agent...")
        agent.stop()
```

## build\lib\agents\planning\config.py

```python
# agents/planning/config.py
from core.config import get_settings
from pydantic import BaseSettings

class PlanningAgentConfig(BaseSettings):
    """
    Configurações para o Planning Agent.
    Estas definições incluem integrações, parâmetros de geração e limites de métricas.
    """

    # OpenAI
    settings = get_settings()
    openai_api_key: str = settings.api.openai_api_key

    # Parâmetros do agente
    temperature: float = 0.7  # Criatividade do modelo
    max_tokens: int = 1500  # Limite de tokens para geração de conteúdo
    planning_interval: int = 120  # Intervalo de planejamento (em segundos)

    # Limiares para métricas e relevância
    min_trend_score: float = 0.5  # Score mínimo para considerar uma tendência
    min_relevance_score: float = 0.6  # Relevância mínima para incluir no plano
    min_confidence_score: float = 0.7  # Confiança mínima para publicar

    # Configurações de cache
    cache_ttl: int = 1800  # Tempo de vida do cache (em segundos)

    # Domínios
    enable_domain_flexibility: bool = True  # Permite criar planos fora de domínios pré-definidos
    default_domain_priority: str = "medium"  # Prioridade padrão para planos fora de domínios

    # Configurações de publicação
    publishing_frequency: str = "daily"  # Frequência de publicação (daily, weekly, monthly)

    class Config:
        env_prefix = "PLANNING_"  # Prefixo para variáveis de ambiente


# Instância global de configuração para facilitar o uso
config = PlanningAgentConfig()

```

## build\lib\agents\planning\__init__.py

```python

```

## build\lib\agents\review\agent.py

```python
from typing import Dict, List
from pydantic import BaseModel, Field
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from core.rabbitmq_utils import RabbitMQUtils
from core.config import get_settings
import asyncio

class ReviewResult(BaseModel):
    """Resultado da revisão de conteúdo"""
    content: str
    quality_score: float = Field(ge=0.0, le=1.0)
    seo_score: float = Field(ge=0.0, le=1.0)
    technical_accuracy: float = Field(ge=0.0, le=1.0)
    readability_score: float = Field(ge=0.0, le=1.0)
    suggestions: List[str]
    approved: bool
    review_date: datetime = Field(default_factory=datetime.now)

class ReviewAgent:
    """Agente responsável por revisar e validar conteúdo"""

    def __init__(self):
        settings = get_settings()
        self.rabbitmq = RabbitMQUtils()  # Integração com RabbitMQ
        self.llm = ChatOpenAI(
            model_name="gpt-4-1106-preview",
            temperature=0.3,
            openai_api_key=settings.api.openai_api_key
        )
        self.setup_chains()

    def setup_chains(self):
        """Configura as chains do LangChain"""
        review_template = """
        Como especialista em revisão técnica, avalie o seguinte conteúdo:
        CONTEÚDO:
        {content}
        CONTEXTO:
        {context}
        CRITÉRIOS DE AVALIAÇÃO:
        1. Qualidade geral do conteúdo
        2. Precisão técnica e acurácia
        3. Otimização para SEO
        4. Legibilidade e clareza
        5. Adequação ao público-alvo
        6. Estrutura e formatação
        7. Exemplos e referências
        8. Coerência e consistência
        Por favor, formate sua resposta exatamente assim:
        ---QUALITY_SCORE---
        [score de 0 a 1]
        ---SEO_SCORE---
        [score de 0 a 1]
        ---TECHNICAL_ACCURACY---
        [score de 0 a 1]
        ---READABILITY_SCORE---
        [score de 0 a 1]
        ---APPROVED---
        [true/false]
        ---SUGGESTIONS---
        - [sugestão 1]
        - [sugestão 2]
        ---REVISED_CONTENT---
        [conteúdo revisado e melhorado]
        """
        self.review_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=review_template,
                input_variables=["content", "context"]
            )
        )

    async def review_content(self, content: str, context: Dict) -> ReviewResult:
        """
        Revisa o conteúdo e retorna resultado detalhado.
        Args:
            content: Conteúdo a ser revisado.
            context: Contexto e metadados do conteúdo.
        Returns:
            ReviewResult com scores e sugestões.
        """
        try:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            result = await self.review_chain.arun({"content": content, "context": context_str})
            return self._process_review(result)
        except Exception as e:
            print(f"Erro na revisão: {str(e)}")
            raise

    def consume_generated_content(self):
        """Consome mensagens da fila 'content.generated' e revisa o conteúdo"""
        def callback(ch, method, properties, body):
            import json
            message = json.loads(body)
            print(f"Conteúdo recebido para revisão: {message}")
            try:
                # Revisar conteúdo
                review_result = asyncio.run(
                    self.review_content(content=message["content"], context={"domain": message["domain"]})
                )
                # Publicar resultados da revisão
                self.rabbitmq.publish_event("content.reviewed", review_result.dict())
                print(f"Revisão publicada: {review_result.dict()}")
            except Exception as e:
                print(f"Erro ao processar mensagem: {str(e)}")

        self.rabbitmq.consume_event("content.generated", callback)

    def _process_review(self, raw_review: str) -> ReviewResult:
        """Processa o resultado bruto da revisão"""
        parts = raw_review.split("---")
        review_data = {}
        for i, part in enumerate(parts):
            if "QUALITY_SCORE" in part and i+1 < len(parts):
                review_data["quality_score"] = float(parts[i+1].strip())
            elif "SEO_SCORE" in part and i+1 < len(parts):
                review_data["seo_score"] = float(parts[i+1].strip())
            elif "TECHNICAL_ACCURACY" in part and i+1 < len(parts):
                review_data["technical_accuracy"] = float(parts[i+1].strip())
            elif "READABILITY_SCORE" in part and i+1 < len(parts):
                review_data["readability_score"] = float(parts[i+1].strip())
            elif "APPROVED" in part and i+1 < len(parts):
                review_data["approved"] = "true" in parts[i+1].lower()
            elif "SUGGESTIONS" in part and i+1 < len(parts):
                review_data["suggestions"] = [
                    s.strip() for s in parts[i+1].split("\n") 
                    if s.strip() and s.strip().startswith("-")
                ]
            elif "REVISED_CONTENT" in part and i+1 < len(parts):
                review_data["content"] = parts[i+1].strip()
        return ReviewResult(**review_data)
```

## build\lib\agents\review\config.py

```python
# agents/review/config.py
from pydantic import BaseSettings, Field
from typing import Dict, List

class ReviewAgentConfig(BaseSettings):
   """Configurações do Review Agent"""
   
   # Thresholds de qualidade
   min_quality_score: float = Field(0.7, env='REVIEW_MIN_QUALITY_SCORE')
   min_technical_accuracy: float = Field(0.8, env='REVIEW_MIN_TECHNICAL_ACCURACY')
   min_seo_score: float = Field(0.7, env='REVIEW_MIN_SEO_SCORE')
   min_readability_score: float = Field(0.7, env='REVIEW_MIN_READABILITY_SCORE')
   
   # Auto-aprovação
   auto_approve_threshold: float = Field(0.9, env='REVIEW_AUTO_APPROVE_THRESHOLD')
   require_manual_review: bool = Field(False, env='REVIEW_REQUIRE_MANUAL')
   
   # Cache
   cache_ttl: int = Field(3600, env='REVIEW_CACHE_TTL')
   max_cache_items: int = Field(1000, env='REVIEW_MAX_CACHE_ITEMS')
   
   # Limites
   max_content_length: int = Field(10000, env='REVIEW_MAX_CONTENT_LENGTH')
   max_suggestions: int = Field(10, env='REVIEW_MAX_SUGGESTIONS')
   
   review_metrics: Dict[str, float] = {
       "quality": 0.3,
       "technical": 0.3,
       "seo": 0.2,
       "readability": 0.2
   }
   
   required_checks: List[str] = [
       "grammar",
       "technical_accuracy",
       "seo_optimization",
       "readability",
       "code_quality"
   ]

   class Config:
       env_prefix = 'REVIEW_'
       case_sensitive = False
```

## build\lib\agents\review\__init__.py

```python

```

## build\lib\agents\search\agent.py

```python
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import asyncio
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.rabbitmq_utils import RabbitMQUtils
from core.config import get_settings
import json
import time
from .services.news.metrics import NewsMetrics
import aiohttp
from .services.news.clients.hackernews import HackerNewsClient
from .services.news.clients.devto import DevToClient
from .services.news.config import NewsApiConfig
import logging

logger = logging.getLogger(__name__)

class SearchResult(BaseModel):
    """Modelo para resultados de busca"""
    title: str
    url: str
    author: str
    source: str
    published_date: datetime
    summary: str
    tags: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float

class ContentValidation(BaseModel):
    """Modelo para validação de conteúdo"""
    claim: str
    is_valid: bool
    confidence_score: float
    supporting_sources: List[str]
    suggestions: Optional[List[str]]

class AudienceInsight(BaseModel):
    """Modelo para insights sobre audiência"""
    preferences: List[str]
    pain_points: List[str]
    technical_level: str
    common_questions: List[str]
    preferred_formats: List[str]

class SEOInsight(BaseModel):
    """Modelo para insights de SEO"""
    primary_keywords: List[tuple]  # (keyword, volume)
    related_keywords: List[tuple]
    questions: List[str]
    competing_content: List[Dict]
    suggested_structure: Dict[str, Any]

class EnhancedSearchAgent:
    """Agente de busca aprimorado com múltiplas funcionalidades"""

    def __init__(self):
        self.settings = get_settings()
        self.rabbitmq = RabbitMQUtils()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.setup_vector_store()
        self.setup_cache()
        self.metrics = NewsMetrics()
        self.news_config = NewsApiConfig()
        self.hacker_news_client = HackerNewsClient(self.news_config.HACKER_NEWS_API_URL, self.news_config.HACKER_NEWS_API_KEY)
        self.dev_to_client = DevToClient(self.news_config.DEVTO_API_URL, self.news_config.DEVTO_API_KEY)
        self.session = None

    async def initialize(self):
        """Inicializa o agente de pesquisa"""
        logger.info("Inicializando o agente de pesquisa")
        await self.hacker_news_client.initialize()
        await self.dev_to_client.initialize()
        if not self.session:
            logger.info("Inicializando sessão HTTP para o agente de pesquisa")
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Fecha conexões do agente de pesquisa"""
        logger.info("Fechando conexões do agente de pesquisa")
        await self.hacker_news_client.close()
        await self.dev_to_client.close()
        if self.session:
            await self.session.close()
            self.session = None

    def setup_vector_store(self):
        """Configura armazenamento vetorial"""
        logger.info("Configurando armazenamento vetorial")
        self.vector_store = FAISS.from_texts(
            texts=["inicialização do índice"],
            embedding=self.embeddings,
            metadatas=[{"source": "initialization"}]
        )

    def setup_cache(self):
        """Configura sistema de cache"""
        logger.info("Configurando sistema de cache")
        self.cache = {}
        self.cache_ttl = 3600  # 1 hora

    async def enrich_content_plan(self, topic: str, keywords: List[str], target_audience: str) -> Dict:
        """
        Enriquece o plano de conteúdo com pesquisas e análises
        """
        logger.info(f"Enriquecendo plano de conteúdo para o tópico: {topic}")
        tasks = [
            self.search_recent_developments(topic),
            self.validate_technical_aspects(topic),
            self.analyze_similar_content(topic, keywords),
            self.gather_seo_insights(keywords),
            self.analyze_audience_preferences(target_audience)
        ]

        results = await asyncio.gather(*tasks)

        return {
            "recent_developments": results[0],
            "technical_validations": results[1],
            "competitive_analysis": results[2],
            "seo_insights": results[3],
            "audience_insights": results[4]
        }

    async def search_recent_developments(self, topic: str) -> List[SearchResult]:
        """
        Busca desenvolvimentos recentes sobre o tópico
        """
        logger.info(f"Buscando desenvolvimentos recentes sobre o tópico: {topic}")
        # Integração com a API do Hacker News
        async with self.metrics.track_request("hacker_news"):
            logger.info("Buscando artigos no Hacker News")
            hacker_news_results = await self.hacker_news_client.search_stories(topic)
            logger.debug(f"Resultados do Hacker News: {hacker_news_results}")

        # Integração com a API do Dev.to
        async with self.metrics.track_request("dev_to"):
            logger.info("Buscando artigos no Dev.to")
            dev_to_results = await self.dev_to_client.search_articles(topic)
            logger.debug(f"Resultados do Dev.to: {dev_to_results}")

        # Combinar resultados de todas as fontes
        return hacker_news_results + dev_to_results

    async def validate_technical_aspects(self, topic: str) -> List[ContentValidation]:
        """
        Valida aspectos técnicos do tópico
        """
        logger.info(f"Validando aspectos técnicos do tópico: {topic}")
        # Implementar validação contra fontes técnicas confiáveis
        async with self.metrics.track_request("technical_validation"):
            return [
                ContentValidation(
                    claim=f"Validação técnica para {topic}",
                    is_valid=True,
                    confidence_score=0.85,
                    supporting_sources=["docs.python.org"],
                    suggestions=["Adicionar mais exemplos práticos"]
                )
            ]

    async def analyze_similar_content(self, topic: str, keywords: List[str]) -> Dict:
        """
        Analisa conteúdo similar existente
        """
        logger.info(f"Analisando conteúdo similar para o tópico: {topic}")
        results = await self._search_vector_store(topic)

        # Análise de gaps e oportunidades
        return {
            "similar_content": results,
            "content_gaps": ["Gap 1", "Gap 2"],
            "unique_angles": ["Ângulo 1", "Ângulo 2"]
        }

    async def gather_seo_insights(self, keywords: List[str]) -> SEOInsight:
        """
        Coleta insights de SEO
        """
        logger.info(f"Coletando insights de SEO para as palavras-chave: {keywords}")
        # Implementar integração com APIs de SEO
        async with self.metrics.track_request("seo_insights"):
            return SEOInsight(
                primary_keywords=[("python", 1000)],
                related_keywords=[("python programming", 800)],
                questions=["How to learn Python?"],
                competing_content=[],
                suggested_structure={
                    "introduction": ["key_point_1", "key_point_2"],
                    "main_sections": ["section_1", "section_2"],
                    "conclusion": ["summary", "next_steps"]
                }
            )

    async def analyze_audience_preferences(self, target_audience: str) -> AudienceInsight:
        """
        Analisa preferências da audiência
        """
        logger.info(f"Analisando preferências da audiência: {target_audience}")
        async with self.metrics.track_request("audience_analysis"):
            return AudienceInsight(
                preferences=["Clear explanations", "Code examples"],
                pain_points=["Complex documentation", "Lack of examples"],
                technical_level="intermediate",
                common_questions=["How to start?", "Best practices?"],
                preferred_formats=["Tutorials", "How-to guides"]
            )

    async def _search_vector_store(self, query: str) -> List[SearchResult]:
        """
        Realiza busca no armazenamento vetorial
        """
        logger.info(f"Realizando busca no armazenamento vetorial com o termo: {query}")
        query_embedding = self.embeddings.embed_query(query)
        results = self.vector_store.similarity_search_with_score(query, k=5)

        return [
            SearchResult(
                content=result[0].page_content,
                source="vector_store",
                relevance_score=float(result[1]),
                metadata=result[0].metadata
            )
            for result in results
        ]

    def _calculate_relevance(self, item: Dict, query: str) -> float:
        """Calcula pontuação de relevância para um item"""
        logger.debug(f"Calculando pontuação de relevância para o item: {item}")
        score = 0.0

        # Relevância do título
        if query.lower() in item["title"].lower():
            score += 0.4

        # Relevância do texto/resumo
        if query.lower() in (item.get("story_text") or "").lower() or \
           query.lower() in (item.get("comment_text") or "").lower() or \
           query.lower() in (item.get("description") or "").lower():
            score += 0.3

        # Relevância da pontuação
        if item.get("points", 0) > 100:
            score += 0.2
        elif item.get("points", 0) > 50:
            score += 0.1

        # Relevância do número de comentários
        if item.get("num_comments", 0) > 50:
            score += 0.1

        return min(score, 1.0)

    async def index_content(self, content: str, metadata: Dict[str, Any]):
        """
        Indexa novo conteúdo no armazenamento vetorial
        """
        logger.info("Indexando novo conteúdo no armazenamento vetorial")
        chunks = self.text_splitter.split_text(content)
        chunk_metadatas = [metadata for _ in chunks]
        self.vector_store.add_texts(chunks, metadatas=chunk_metadatas)

    async def start_consuming(self):
        """
        Inicia consumo de mensagens do RabbitMQ
        """
        def callback(ch, method, properties, body):
            message = json.loads(body)
            logger.info(f"Mensagem recebida: {message}")

            # Processar mensagem e enriquecer conteúdo
            enriched_data = asyncio.run(self.enrich_content_plan(
                topic=message.get("topic", ""),
                keywords=message.get("keywords", []),
                target_audience=message.get("target_audience", "")
            ))

            # Publicar resultados enriquecidos
            self.rabbitmq.publish_event(
                "search.results",
                json.dumps(enriched_data, default=str)
            )

        self.rabbitmq.consume_event("planning.generated", callback)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = EnhancedSearchAgent()
    logger.info("Search Agent iniciado. Aguardando mensagens...")
    asyncio.run(agent.start_consuming())
```

## build\lib\agents\search\config.py

```python
# agents/search/config.py

from typing import Dict, List, Optional
from pydantic import BaseSettings, Field, HttpUrl

class SearchAgentConfig(BaseSettings):
    """Configurações avançadas para o Search Agent"""
    
    # Configurações de API
    OPENAI_API_KEY: str = Field(..., env='OPENAI_API_KEY')
    
    # Configurações de Embedding
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Vector Store
    VECTOR_STORE_TYPE: str = "faiss"  # faiss ou pinecone
    VECTOR_SIMILARITY_THRESHOLD: float = 0.75
    MAX_RESULTS_PER_QUERY: int = 10
    
    # Cache
    CACHE_TTL: int = 3600  # 1 hora
    MAX_CACHE_ITEMS: int = 10000
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = 60
    MAX_TOKENS_PER_MINUTE: int = 100000
    
    # Content Validation
    MIN_CONFIDENCE_SCORE: float = 0.8
    REQUIRED_SUPPORTING_SOURCES: int = 2
    
    # Notícias e Atualizações
    NEWS_SOURCES: Dict[str, Dict] = {
        "tech_crunch": {
            "name": "TechCrunch",
            "base_url": "https://api.techcrunch.com/v1/",
            "priority": 1,
            "categories": ["technology", "ai", "cloud"]
        },
        "hacker_news": {
            "name": "Hacker News",
            "base_url": "http://hn.algolia.com/api/v1",
            "priority": 2,
            "categories": ["technology", "programming"]
        },
        "dev_to": {
            "name": "Dev.to",
            "base_url": "https://dev.to/api/",
            "priority": 3,
            "categories": ["development", "programming"]
        }
    }
    
    # Fonte de dados confiáveis
    TRUSTED_DOMAINS: List[str] = [
        "docs.python.org",
        "developer.mozilla.org",
        "kubernetes.io",
        "cloud.google.com",
        "aws.amazon.com",
        "azure.microsoft.com",
        "github.com",
        "stackoverflow.com",
        "arxiv.org",
        "research.google.com",
        "openai.com",
        "pytorch.org",
        "tensorflow.org"
    ]
    
    # Métricas de qualidade
    QUALITY_WEIGHTS: Dict[str, float] = {
        "relevance": 0.4,
        "freshness": 0.2,
        "authority": 0.2,
        "completeness": 0.2
    }
    
    # Parâmetros de processamento
    MAX_CONTENT_LENGTH: int = 100000
    MIN_CONTENT_LENGTH: int = 100
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Configurações de busca
    SEARCH_DEFAULTS: Dict[str, Any] = {
        "min_relevance": 0.6,
        "max_age_days": 365,
        "max_results": 20,
        "include_content": True
    }
    
    # Configurações de análise
    ANALYSIS_OPTIONS: Dict[str, bool] = {
        "extract_code_snippets": True,
        "extract_links": True,
        "analyze_sentiment": False,
        "detect_language": True,
        "generate_summary": True
    }
    
    # Timeouts e tentativas
    REQUEST_TIMEOUT: int = 30  # segundos
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1  # segundos
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PREFIX: str = "search_agent"
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_prefix = "SEARCH_"
        case_sensitive = True

    def get_news_source_config(self, source_id: str) -> Optional[Dict]:
        """Retorna configuração para uma fonte específica"""
        return self.NEWS_SOURCES.get(source_id)
    
    def is_trusted_domain(self, domain: str) -> bool:
        """Verifica se um domínio é confiável"""
        return any(domain.endswith(trusted) for trusted in self.TRUSTED_DOMAINS)
    
    def get_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calcula pontuação de qualidade baseada nos pesos definidos"""
        return sum(
            metrics.get(metric, 0) * weight
            for metric, weight in self.QUALITY_WEIGHTS.items()
        )

# Instância global de configuração
config = SearchAgentConfig()
```

## build\lib\agents\search\__init__.py

```python
# agents/search/__init__.py
from agents.search.services.news import NewsIntegrationService, NewsArticle, NewsSearchQuery

__all__ = ['NewsIntegrationService', 'NewsArticle', 'NewsSearchQuery']
```

## build\lib\agents\search\services\news_integration.py

```python
# agents/search/services/news_integration.py

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import aiohttp
from pydantic import BaseModel
import json

class NewsArticle(BaseModel):
    """Modelo para artigos de notícias"""
    title: str
    url: str
    source: str
    published_date: datetime
    summary: str
    relevance_score: float
    category: str

class NewsIntegrationService:
    """Serviço de integração com APIs de notícias técnicas"""

    def __init__(self):
        # Configurar APIs - em produção, mover para variáveis de ambiente
        self.news_apis = {
            "tech_crunch": {
                "url": "https://api.techcrunch.com/v1/",
                "key": "your_api_key"
            },
            "hacker_news": {
                "url": "https://hacker-news.firebaseio.com/v0/",
                "key": None
            },
            "dev_to": {
                "url": "https://dev.to/api/",
                "key": "your_api_key"
            }
        }
        self.session = None

    async def initialize(self):
        """Inicializa sessão HTTP"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Fecha sessão HTTP"""
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_recent_news(self, topic: str, days: int = 30) -> List[NewsArticle]:
        """Busca notícias recentes sobre um tópico"""
        if not self.session:
            await self.initialize()

        news_items = []
        tasks = [
            self._fetch_tech_crunch(topic, days),
            self._fetch_hacker_news(topic, days),
            self._fetch_dev_to(topic, days)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                news_items.extend(result)

        # Ordenar por relevância e data
        return sorted(
            news_items,
            key=lambda x: (x.relevance_score, x.published_date),
            reverse=True
        )

    async def _fetch_tech_crunch(self, topic: str, days: int) -> List[NewsArticle]:
        """Busca notícias do TechCrunch"""
        try:
            async with self.session.get(
                f"{self.news_apis['tech_crunch']['url']}search",
                params={
                    "q": topic,
                    "from_date": (datetime.now() - timedelta(days=days)).isoformat()
                },
                headers={"Authorization": f"Bearer {self.news_apis['tech_crunch']['key']}"}
            ) as response:
                data = await response.json()
                return [
                    NewsArticle(
                        title=article["title"],
                        url=article["url"],
                        source="TechCrunch",
                        published_date=datetime.fromisoformat(article["published_at"]),
                        summary=article["summary"],
                        relevance_score=self._calculate_relevance(article, topic),
                        category="technology"
                    )
                    for article in data.get("articles", [])
                ]
        except Exception as e:
            print(f"Erro ao buscar do TechCrunch: {str(e)}")
            return []

    async def _fetch_hacker_news(self, topic: str, days: int) -> List[NewsArticle]:
        """Busca notícias do Hacker News"""
        try:
            async with self.session.get(
                f"{self.news_apis['hacker_news']['url']}topstories.json"
            ) as response:
                story_ids = await response.json()
                stories = []
                
                # Limitar a 50 histórias para não sobrecarregar
                for story_id in story_ids[:50]:
                    async with self.session.get(
                        f"{self.news_apis['hacker_news']['url']}item/{story_id}.json"
                    ) as story_response:
                        story = await story_response.json()
                        if topic.lower() in story.get("title", "").lower():
                            stories.append(
                                NewsArticle(
                                    title=story["title"],
                                    url=story.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                                    source="Hacker News",
                                    published_date=datetime.fromtimestamp(story["time"]),
                                    summary=story.get("text", "No summary available"),
                                    relevance_score=self._calculate_relevance(story, topic),
                                    category="technology"
                                )
                            )
                return stories
        except Exception as e:
            print(f"Erro ao buscar do Hacker News: {str(e)}")
            return []

    async def _fetch_dev_to(self, topic: str, days: int) -> List[NewsArticle]:
        """Busca artigos do Dev.to"""
        try:
            async with self.session.get(
                f"{self.news_apis['dev_to']['url']}articles",
                params={"tag": topic, "top": "30"},
                headers={"api-key": self.news_apis['dev_to']['key']}
            ) as response:
                articles = await response.json()
                return [
                    NewsArticle(
                        title=article["title"],
                        url=article["url"],
                        source="Dev.to",
                        published_date=datetime.fromisoformat(article["published_at"]),
                        summary=article["description"],
                        relevance_score=self._calculate_relevance(article, topic),
                        category="technology"
                    )
                    for article in articles
                    if (datetime.now() - datetime.fromisoformat(article["published_at"])).days <= days
                ]
        except Exception as e:
            print(f"Erro ao buscar do Dev.to: {str(e)}")
            return []

    def _calculate_relevance(self, article: Dict, topic: str) -> float:
        """Calcula pontuação de relevância para um artigo"""
        relevance = 0.0
        
        # Relevância do título
        if topic.lower() in article.get("title", "").lower():
            relevance += 0.4
            
        # Relevância do conteúdo/resumo
        if topic.lower() in article.get("summary", "").lower() or \
           topic.lower() in article.get("description", "").lower() or \
           topic.lower() in article.get("text", "").lower():
            relevance += 0.3
            
        # Engajamento (se disponível)
        points = article.get("points", 0)
        comments = article.get("num_comments", 0) or article.get("comment_count", 0)
        
        if points > 100:
            relevance += 0.2
        elif points > 50:
            relevance += 0.1
            
        if comments > 50:
            relevance += 0.1
            
        return min(relevance, 1.0)
```

## build\lib\agents\search\services\__init__.py

```python
# agents/search/services/__init__.py
from agents.search.services.news import NewsIntegrationService, NewsArticle, NewsSearchQuery

__all__ = ['NewsIntegrationService', 'NewsArticle', 'NewsSearchQuery']
```

## build\lib\agents\search\services\news\cache.py

```python
# agents/search/services/news/cache.py

from typing import Dict, Optional, Any
import time
from datetime import datetime
import json
from .models import NewsArticle
from .config import NewsApiConfig

class NewsCache:
    """Gerenciador de cache para requisições de notícias"""
    
    def __init__(self, config: NewsApiConfig):
        self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_times: Dict[str, float] = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Recupera item do cache"""
        if key not in self._cache:
            return None
            
        # Verificar TTL
        if time.time() - self._cache_times[key] > self.config.CACHE_TTL:
            self._remove(key)
            return None
            
        return self._cache[key]
        
    def set(self, key: str, value: Any) -> None:
        """Armazena item no cache"""
        # Limpar cache se necessário
        if len(self._cache) >= self.config.MAX_CACHE_ITEMS:
            self._cleanup_cache()
            
        self._cache[key] = value
        self._cache_times[key] = time.time()
        
    def _remove(self, key: str) -> None:
        """Remove item do cache"""
        if key in self._cache:
            del self._cache[key]
        if key in self._cache_times:
            del self._cache_times[key]
            
    def _cleanup_cache(self) -> None:
        """Limpa itens expirados do cache"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_times.items()
            if current_time - timestamp > self.config.CACHE_TTL
        ]
        
        for key in expired_keys:
            self._remove(key)
            
        # Se ainda precisar de espaço, remove os itens mais antigos
        if len(self._cache) >= self.config.MAX_CACHE_ITEMS:
            sorted_keys = sorted(
                self._cache_times.items(),
                key=lambda x: x[1]
            )
            
            # Remove 20% dos itens mais antigos
            num_to_remove = len(sorted_keys) // 5
            for key, _ in sorted_keys[:num_to_remove]:
                self._remove(key)
                
    def get_cache_key(self, query_params: Dict[str, Any]) -> str:
        """Gera chave de cache para parâmetros de consulta"""
        # Ordenar parâmetros para garantir consistência
        sorted_params = sorted(query_params.items())
        return json.dumps(sorted_params, default=str)
        
    def clear(self) -> None:
        """Limpa todo o cache"""
        self._cache.clear()
        self._cache_times.clear()
        
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        current_time = time.time()
        return {
            "total_items": len(self._cache),
            "expired_items": sum(
                1 for timestamp in self._cache_times.values()
                if current_time - timestamp > self.config.CACHE_TTL
            ),
            "cache_size_percent": (len(self._cache) / self.config.MAX_CACHE_ITEMS) * 100,
            "oldest_item_age": max(
                current_time - min(self._cache_times.values())
                if self._cache_times else 0,
                0
            )
        }
```

## build\lib\agents\search\services\news\config.py

```python
# agents/search/services/news/config.py

from pydantic import BaseSettings, Field
from typing import Dict, List, Optional
from datetime import timedelta
from pathlib import Path
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente do .env
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class NewsApiConfig(BaseSettings):
    """Configurações para integrações com APIs de notícias"""
    
    # Dev.to
    DEVTO_API_URL: str = Field("https://dev.to/api", env='DEVTO_API_URL')
    DEVTO_API_KEY: Optional[str] = Field(None, env='DEVTO_API_KEY')
    DEVTO_MAX_RESULTS: int = Field(100, env='DEVTO_MAX_RESULTS')
    DEVTO_RATE_LIMIT: int = Field(3000, env='DEVTO_RATE_LIMIT')
    
    # Hacker News
    HACKER_NEWS_API_URL: str = Field("http://hn.algolia.com/api/v1", env='HACKER_NEWS_API_URL')
    HACKER_NEWS_API_KEY: Optional[str] = Field(None, env='HACKER_NEWS_API_KEY')
    HACKER_NEWS_MAX_RESULTS: int = Field(100, env='HACKER_NEWS_MAX_RESULTS')
    HACKER_NEWS_RATE_LIMIT: int = Field(3000, env='HACKER_NEWS_RATE_LIMIT')

    # Cache
    CACHE_TTL: int = Field(3600, env='NEWS_CACHE_TTL')
    MAX_CACHE_ITEMS: int = Field(10000, env='NEWS_MAX_CACHE_ITEMS')
    
    # Relevância
    MIN_RELEVANCE_SCORE: float = Field(0.3, env='NEWS_MIN_RELEVANCE')
    RELEVANCE_WEIGHTS: Dict[str, float] = {
        "title_match": 0.4,
        "tag_match": 0.3,
        "content_match": 0.2,
        "engagement": 0.1
    }
    
    # Limites
    DEFAULT_MAX_RESULTS: int = Field(50, env='NEWS_DEFAULT_MAX_RESULTS')
    MAX_SEARCH_PERIOD: timedelta = Field(
        default_factory=lambda: timedelta(days=int(os.getenv('NEWS_MAX_SEARCH_PERIOD', '30'))),
    )
    
    # Configurações de fontes
    ENABLED_SOURCES: List[str] = Field(
        default=["dev.to", "hacker_news"],
        env='NEWS_ENABLED_SOURCES'
    )
    
    # Rate Limiting Global
    RATE_LIMIT_WINDOW: int = Field(60, env='NEWS_RATE_LIMIT_WINDOW')
    MAX_REQUESTS_PER_WINDOW: int = Field(1000, env='NEWS_MAX_REQUESTS_PER_WINDOW')
    
    class Config:
        env_prefix = "NEWS_"
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = 'utf-8'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Configurações de API carregadas")
        logger.debug(f"Configurações: {self.dict()}")
```

## build\lib\agents\search\services\news\metrics.py

```python
# agents/search/services/news/metrics.py

from prometheus_client import Counter, Histogram, Gauge
from prometheus_client.registry import CollectorRegistry
import time
import asyncio

class NewsMetrics:
    """Sistema de métricas para o serviço de notícias"""

    def __init__(self):
        # Criar registry separado para evitar colisões
        self.registry = CollectorRegistry()

        # Contadores
        self.request_count = Counter(
            "news_integration_requests",
            "Total number of API requests",
            ["source", "status"],
            registry=self.registry
        )

        self.article_count = Counter(
            "news_integration_articles",
            "Total number of articles processed",
            ["source"],
            registry=self.registry
        )

        # Histogramas
        self.request_latency = Histogram(
            "news_integration_request_duration",
            "Request duration in seconds",
            ["source"],
            registry=self.registry
        )

        # Gauges
        self.active_requests = Gauge(
            "news_integration_active_requests",
            "Number of active requests",
            ["source"],
            registry=self.registry
        )

    def track_request(self, source: str):
        """Context manager para rastrear requisições"""
        class RequestTracker:
            def __init__(self, metrics, source):
                self.metrics = metrics
                self.source = source
                self.start_time = None

            async def __aenter__(self):
                self.start_time = time.time()
                self.metrics.active_requests.labels(source=self.source).inc()
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.metrics.request_latency.labels(source=self.source).observe(duration)
                self.metrics.active_requests.labels(source=self.source).dec()

                status = "error" if exc_type else "success"
                self.metrics.request_count.labels(source=self.source, status=status).inc()

        return RequestTracker(self, source)

    def record_processed_article(self, source: str):
        """Registra um artigo processado"""
        self.article_count.labels(source=source).inc()

    def get_metrics(self):
        """Retorna métricas atuais"""
        metrics = {
            "requests": {},
            "articles": {},
            "active_requests": {}
        }

        # Obter métricas de requisições para cada fonte
        for source in ["tech_crunch", "hacker_news", "dev.to"]:
            metrics["requests"][source] = {
                "success": self.request_count.labels(source=source, status="success")._value.get(),
                "error": self.request_count.labels(source=source, status="error")._value.get()
            }
            metrics["articles"][source] = self.article_count.labels(source=source)._value.get()
            metrics["active_requests"][source] = self.active_requests.labels(source=source)._value

        return metrics
```

## build\lib\agents\search\services\news\models.py

```python
# agents/search/services/news/models.py

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl

class NewsSource(BaseModel):
    """Configuração de fonte de notícias"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=10)
    enabled: bool = True
    categories: List[str] = Field(default_factory=list)
    rate_limit: Optional[int] = None

class NewsSearchQuery(BaseModel):
    """Modelo para consultas de busca de notícias"""
    topic: str
    keywords: List[str] = Field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_relevance: float = 0.3
    max_results: Optional[int] = None

class NewsArticle(BaseModel):
    """Modelo para artigos de notícias"""
    title: str
    url: str
    source: str
    author: Optional[str]
    published_date: datetime
    summary: str
    content: Optional[str]
    tags: List[str] = Field(default_factory=list)
    category: str = "technology"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: Optional[float] = Field(default=0.0)
```

## build\lib\agents\search\services\news\service.py

```python
# agents/search/services/news/service.py

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
from .models import NewsArticle, NewsSearchQuery
from .config import NewsApiConfig
from .cache import NewsCache
from .metrics import NewsMetrics
from .clients.devto import DevToClient
from .clients.hackernews import HackerNewsClient


logger = logging.getLogger(__name__)

def ensure_timezone(dt: Optional[datetime]) -> Optional[datetime]:
    """Garante que a data tem timezone (UTC se não especificado)"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

class NewsIntegrationService:
    """Serviço principal de integração de notícias"""

    def __init__(self):
        self.config = NewsApiConfig()
        self.cache = NewsCache(self.config)
        self.metrics = NewsMetrics()
        
        # Inicializar clientes
        self.devto_client = DevToClient(api_key=self.config.DEVTO_API_KEY)
        self.hacker_news_client = HackerNewsClient (api_key=self.config.HACKER_NEWS_API_KEY)
        
        self.session = None

    async def initialize(self):
        """Inicializa o serviço"""
        logger.info("Initializing News Integration Service")
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Fecha conexões"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("News Integration Service shut down")

    async def search_news(self, query: NewsSearchQuery) -> List[NewsArticle]:
        """Busca notícias com base nos parâmetros fornecidos"""
        try:
            # Garantir que as datas têm timezone
            query.start_date = ensure_timezone(query.start_date)
            query.end_date = ensure_timezone(query.end_date)

            # Verificar cache
            cache_key = str(query.dict())
            cached_results = self.cache.get(cache_key)
            if cached_results:
                logger.info(f"Cache hit for query: {query.topic}")
                return cached_results

            # Buscar em todas as fontes configuradas
            results = []
            
            # Busca no Dev.to
            async with self.metrics.track_request("dev.to"):
                devto_articles = await self.devto_client.get_articles(
                    search_term=query.topic,
                    tag=query.keywords[0] if query.keywords else None
                )
                for article in devto_articles:
                    results.append(self.devto_client.to_news_article(article))
                    self.metrics.record_processed_article("dev.to")

            # Filtrar e ordenar resultados
            filtered_results = self._filter_and_sort_results(results, query)

            # Armazenar no cache
            self.cache.set(cache_key, filtered_results)

            return filtered_results

        except Exception as e:
            logger.error(f"Error in search_news: {str(e)}")
            return []

    def _filter_and_sort_results(self, articles: List[NewsArticle], query: NewsSearchQuery) -> List[NewsArticle]:
        """Filtra e ordena os resultados"""
        filtered = []
        for article in articles:
            # Garantir que a data do artigo tem timezone
            article.published_date = ensure_timezone(article.published_date)
            
            # Filtrar por data
            if query.start_date and article.published_date < query.start_date:
                continue
            if query.end_date and article.published_date > query.end_date:
                continue

            # Filtrar por relevância
            if self._calculate_relevance(article, query) >= query.min_relevance:
                filtered.append(article)

        # Ordenar por data de publicação (mais recente primeiro)
        filtered.sort(key=lambda x: x.published_date, reverse=True)

        # Limitar número de resultados
        return filtered[:query.max_results] if query.max_results else filtered

    def _calculate_relevance(self, article: NewsArticle, query: NewsSearchQuery) -> float:
        """Calcula pontuação de relevância para um artigo"""
        score = 0.0
        
        # Relevância do título
        if query.topic.lower() in article.title.lower():
            score += 0.4
        
        # Relevância das tags
        if article.tags and any(keyword.lower() in tag.lower() for keyword in query.keywords for tag in article.tags):
            score += 0.3
        
        # Relevância do resumo
        if query.topic.lower() in article.summary.lower():
            score += 0.2
        
        # Bônus por engajamento
        if article.metadata:
            reactions = article.metadata.get("reactions_count", 0)
            comments = article.metadata.get("comments_count", 0)
            if reactions > 50 or comments > 10:
                score += 0.1
        
        return min(score, 1.0)
```

## build\lib\agents\search\services\news\__init__.py

```python
# agents/search/services/news/__init__.py
from .service import NewsIntegrationService
from .models import NewsArticle, NewsSearchQuery

__all__ = ['NewsIntegrationService', 'NewsArticle', 'NewsSearchQuery']
```

## build\lib\agents\search\services\news\clients\devto.py

```python
# agents/search/services/news/clients/devto.py

from typing import List, Optional
import aiohttp
from datetime import datetime, timezone
from pydantic import BaseModel, HttpUrl
import logging
from ...utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class DevToArticle(BaseModel):
    """Modelo para artigos do Dev.to"""
    id: int
    title: str
    description: Optional[str]
    url: HttpUrl
    published_at: datetime
    tag_list: List[str]
    user: dict
    reading_time_minutes: Optional[int]
    comments_count: Optional[int]
    public_reactions_count: Optional[int]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DevToClient:
    """Cliente para a API do Dev.to"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://dev.to/api"
        self.api_key = api_key
        # Dev.to tem limite de 3000 requests/hora
        self.rate_limiter = RateLimiter(max_calls=3000, period=3600)
        
    def _ensure_timezone(self, dt: datetime) -> datetime:
        """Garante que a data tem timezone (UTC se não especificado)"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    async def get_articles(
        self,
        search_term: Optional[str] = None,
        tag: Optional[str] = None,
        username: Optional[str] = None,
        page: int = 1,
        per_page: int = 30
    ) -> List[DevToArticle]:
        """Busca artigos no Dev.to"""
        params = {
            "page": page,
            "per_page": min(per_page, 1000)
        }
        
        if tag:
            params["tag"] = tag
        if username:
            params["username"] = username
            
        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    if search_term:
                        url = f"{self.base_url}/articles/search"
                        params["q"] = search_term
                    else:
                        url = f"{self.base_url}/articles"
                        
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status == 429:  # Too Many Requests
                            retry_after = int(response.headers.get("Retry-After", 60))
                            logger.warning(f"Rate limit hit. Retry after {retry_after} seconds")
                            return []
                            
                        response.raise_for_status()
                        data = await response.json()
                        
                        # Processar as datas antes de criar os objetos
                        for article in data:
                            if "published_at" in article:
                                dt = datetime.fromisoformat(article["published_at"].replace("Z", "+00:00"))
                                article["published_at"] = self._ensure_timezone(dt)
                        
                        return [DevToArticle(**article) for article in data]
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching from Dev.to: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return []
    
    def to_news_article(self, article: DevToArticle) -> "NewsArticle":
        """Converte DevToArticle para NewsArticle"""
        from ..models import NewsArticle
        
        return NewsArticle(
            title=article.title,
            url=str(article.url),
            source="dev.to",
            author=article.user.get("name"),
            published_date=self._ensure_timezone(article.published_at),
            summary=article.description or "",
            content=None,
            tags=article.tag_list,
            metadata={
                "reading_time": article.reading_time_minutes,
                "comments_count": article.comments_count,
                "reactions_count": article.public_reactions_count,
                "author_username": article.user.get("username")
            },
            relevance_score=0.0
        )
```

## build\lib\agents\search\services\news\clients\hackernews.py

```python
# class HackerNewsClient:
  # agents/search/services/news/clients/devto.py

from typing import List, Optional
import aiohttp
from datetime import datetime, timezone
from pydantic import BaseModel, HttpUrl
import logging
from ...utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class HackerNewsClientArticle(BaseModel):
    """Modelo para artigos do Dev.to"""
    id: int
    title: str
    description: Optional[str]
    url: HttpUrl
    published_at: datetime
    tag_list: List[str]
    user: dict
    reading_time_minutes: Optional[int]
    comments_count: Optional[int]
    public_reactions_count: Optional[int]

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HackerNewsClient:
    """Cliente para a API do Hacker News"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "http://hn.algolia.com/api/v1"
        self.api_key = api_key
        # Dev.to tem limite de 3000 requests/hora
        self.rate_limiter = RateLimiter(max_calls=3000, period=3600)
        
    def _ensure_timezone(self, dt: datetime) -> datetime:
        """Garante que a data tem timezone (UTC se não especificado)"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    async def get_articles(
        self,
        search_term: Optional[str] = None,
        tag: Optional[str] = None,
        username: Optional[str] = None,
        page: int = 1,
        per_page: int = 30
    ) -> List[HackerNewsClientArticle]:
        """Busca artigos no HackerNews"""
        params = {
            "page": page,
            "per_page": min(per_page, 1000)
        }
        
        if tag:
            params["tag"] = tag
        if username:
            params["username"] = username
            
        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        
        try:
            async with self.rate_limiter:
                async with aiohttp.ClientSession() as session:
                    if search_term:
                        url = f"{self.base_url}/articles/search"
                        params["q"] = search_term
                    else:
                        url = f"{self.base_url}/articles"
                        
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status == 429:  # Too Many Requests
                            retry_after = int(response.headers.get("Retry-After", 60))
                            logger.warning(f"Rate limit hit. Retry after {retry_after} seconds")
                            return []
                            
                        response.raise_for_status()
                        data = await response.json()
                        
                        # Processar as datas antes de criar os objetos
                        for article in data:
                            if "published_at" in article:
                                dt = datetime.fromisoformat(article["published_at"].replace("Z", "+00:00"))
                                article["published_at"] = self._ensure_timezone(dt)
                        
                        return [HackerNewsClientArticle(**article) for article in data]
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching from Dev.to: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return []
    
    def to_news_article(self, article: HackerNewsClientArticle) -> "NewsArticle":
        """Converte DevToArticle para NewsArticle"""
        from ..models import NewsArticle
        
        return NewsArticle(
            title=article.title,
            url=str(article.url),
            source="Hacker News",
            author=article.user.get("name"),
            published_date=self._ensure_timezone(article.published_at),
            summary=article.description or "",
            content=None,
            tags=article.tag_list,
            metadata={
                "reading_time": article.reading_time_minutes,
                "comments_count": article.comments_count,
                "reactions_count": article.public_reactions_count,
                "author_username": article.user.get("username")
            },
            relevance_score=0.0
        )
```

## build\lib\agents\search\services\news\clients\__init__.py

```python

```

## build\lib\agents\search\services\utils\rate_limiter.py

```python
# agents/search/services/utils/rate_limiter.py

import asyncio
from datetime import datetime, timedelta
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limiter implementando algoritmo de sliding window
    """
    
    def __init__(self, max_calls: int, period: float):
        """
        Args:
            max_calls: Número máximo de chamadas permitidas no período
            period: Período em segundos
        """
        self.max_calls = max_calls
        self.period = period
        self.calls: List[datetime] = []
        self._lock = asyncio.Lock()
        
    async def acquire(self) -> bool:
        """
        Tenta adquirir uma permissão do rate limiter
        
        Returns:
            bool: True se permitido, False se limite excedido
        """
        async with self._lock:
            now = datetime.now()
            
            # Remover chamadas antigas
            window_start = now - timedelta(seconds=self.period)
            self.calls = [call for call in self.calls if call > window_start]
            
            # Verificar se pode fazer nova chamada
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
                
            # Calcular tempo de espera
            if self.calls:
                next_available = self.calls[0] + timedelta(seconds=self.period)
                wait_time = (next_available - now).total_seconds()
                if wait_time > 0:
                    logger.warning(f"Rate limit exceeded. Wait {wait_time:.2f} seconds")
                    return False
                    
            self.calls.append(now)
            return True
    
    async def __aenter__(self):
        """Suporte para uso com 'async with'"""
        while not await self.acquire():
            await asyncio.sleep(1)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup ao sair do contexto"""
        pass

    def reset(self):
        """Reseta o rate limiter"""
        self.calls.clear()
```

## build\lib\agents\search\services\utils\__init__.py

```python

```

## build\lib\api\__init__.py

```python

```

## build\lib\core\config.py

```python
from typing import Dict, Optional
from pydantic import BaseSettings, Field
from functools import lru_cache
import os
from enum import Enum

class Environment(str, Enum):
    """Ambientes de execução"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Níveis de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class APIConfig(BaseSettings):
    """Configurações de APIs externas"""
    # OpenAI
    openai_api_key: str = Field(..., env='OPENAI_API_KEY')
    openai_model: str = Field("gpt-4-1106-preview", env='OPENAI_MODEL')
    openai_temperature: float = Field(0.7, env='OPENAI_TEMPERATURE')
    
    # Pinecone
    pinecone_api_key: str = Field(..., env='PINECONE_API_KEY')
    pinecone_environment: str = Field('gcp-starter', env='PINECONE_ENVIRONMENT')
    pinecone_index_name: str = Field('adagents', env='PINECONE_INDEX_NAME')
    
    # HuggingFace
    hf_api_key: Optional[str] = Field(None, env='HF_API_KEY')
    hf_model: str = Field('sentence-transformers/all-mpnet-base-v2', env='HF_MODEL')

class AgentConfig(BaseSettings):
    """Configurações dos agentes"""
    # Planning Agent
    planning_cache_ttl: int = Field(3600, env='PLANNING_CACHE_TTL')
    planning_max_retries: int = Field(3, env='PLANNING_MAX_RETRIES')
    
    # Search Agent
    search_cache_ttl: int = Field(3600, env='SEARCH_CACHE_TTL')
    search_max_results: int = Field(10, env='SEARCH_MAX_RESULTS')
    search_min_relevance: float = Field(0.7, env='SEARCH_MIN_RELEVANCE')
    
    # Content Agent
    content_max_length: int = Field(4000, env='CONTENT_MAX_LENGTH')
    content_min_length: int = Field(500, env='CONTENT_MIN_LENGTH')
    content_quality_threshold: float = Field(0.8, env='CONTENT_QUALITY_THRESHOLD')
    
    # Review Agent
    review_auto_approve_threshold: float = Field(0.9, env='REVIEW_AUTO_APPROVE_THRESHOLD')
    review_min_quality_score: float = Field(0.7, env='REVIEW_MIN_QUALITY_SCORE')

class DatabaseConfig(BaseSettings):
    """Configurações de banco de dados"""
    db_host: str = Field('localhost', env='DB_HOST')
    db_port: int = Field(5432, env='DB_PORT')
    db_name: str = Field('adagents', env='DB_NAME')
    db_user: str = Field('postgres', env='DB_USER')
    db_password: Optional[str] = Field(None, env='DB_PASSWORD')
    
    @property
    def database_url(self) -> str:
        """Retorna URL de conexão com o banco"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

class CacheConfig(BaseSettings):
    """Configurações de cache"""
    redis_host: str = Field('localhost', env='REDIS_HOST')
    redis_port: int = Field(6379, env='REDIS_PORT')
    redis_db: int = Field(0, env='REDIS_DB')
    redis_password: Optional[str] = Field(None, env='REDIS_PASSWORD')
    default_ttl: int = Field(3600, env='CACHE_DEFAULT_TTL')

class RateLimitConfig(BaseSettings):
    """Configurações de rate limiting"""
    max_requests_per_minute: int = Field(60, env='RATE_LIMIT_RPM')
    max_tokens_per_minute: int = Field(10000, env='RATE_LIMIT_TOKENS')
    max_parallel_requests: int = Field(5, env='RATE_LIMIT_PARALLEL')

class GlobalConfig(BaseSettings):
    """Configurações globais do sistema"""
    environment: Environment = Field(Environment.DEVELOPMENT, env='ENVIRONMENT')
    debug: bool = Field(True, env='DEBUG')
    log_level: LogLevel = Field(LogLevel.INFO, env='LOG_LEVEL')
    
    # Sub-configurações
    api: APIConfig = APIConfig()
    agents: AgentConfig = AgentConfig()
    database: DatabaseConfig = DatabaseConfig()
    cache: CacheConfig = CacheConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    
    # Timeouts e limites
    request_timeout: int = Field(30, env='REQUEST_TIMEOUT')
    max_retries: int = Field(3, env='MAX_RETRIES')
    batch_size: int = Field(100, env='BATCH_SIZE')
    
    # Paths
    base_path: str = Field(os.path.dirname(os.path.dirname(__file__)))
    data_path: str = Field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
    log_path: str = Field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'))
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> GlobalConfig:
    """Retorna instância cacheada das configurações"""
    return GlobalConfig()

def initialize_directories() -> None:
    """Inicializa diretórios necessários"""
    config = get_settings()
    os.makedirs(config.data_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)

def get_environment() -> Environment:
    """Retorna ambiente atual"""
    return get_settings().environment

def is_development() -> bool:
    """Verifica se está em ambiente de desenvolvimento"""
    return get_environment() == Environment.DEVELOPMENT

def is_production() -> bool:
    """Verifica se está em ambiente de produção"""
    return get_environment() == Environment.PRODUCTION

if __name__ == "__main__":
    # Exemplo de uso
    settings = get_settings()
    
    print(f"Environment: {settings.environment}")
    print(f"Debug Mode: {settings.debug}")
    print(f"Log Level: {settings.log_level}")
    
    print("\nAPI Settings:")
    print(f"OpenAI Model: {settings.api.openai_model}")
    print(f"Pinecone Environment: {settings.api.pinecone_environment}")
    
    print("\nAgent Settings:")
    print(f"Search Cache TTL: {settings.agents.search_cache_ttl}")
    print(f"Content Quality Threshold: {settings.agents.content_quality_threshold}")
    
    print("\nRate Limits:")
    print(f"Max RPM: {settings.rate_limit.max_requests_per_minute}")
    print(f"Max Parallel: {settings.rate_limit.max_parallel_requests}")
    
    # Inicializa diretórios
    initialize_directories()
    print(f"\nData Path: {settings.data_path}")
    print(f"Log Path: {settings.log_path}")
```

## build\lib\core\feedback_loop.py

```python
from core.rabbitmq_utils import RabbitMQUtils
import json

class FeedbackLoop:
    def __init__(self, domain: str, max_iterations: int = 5):
        self.rabbitmq = RabbitMQUtils()
        self.domain = domain
        self.max_iterations = max_iterations
        self.iteration = 0

    def publish_content(self):
        """Publica conteúdo gerado na fila."""
        content = {"content_id": 1, "domain": self.domain, "content": "Texto inicial gerado"}
        print(f"Publicando conteúdo: {content}")
        self.rabbitmq.publish_event("content.generated", content)

    def process_reviewed_event(self, ch, method, properties, body):
        """Processa mensagens da fila 'content.reviewed'."""
        message = json.loads(body)
        print(f"Revisão recebida: {message}")

        quality_score = message.get("quality_score", 0)
        if quality_score >= 0.8:
            print(f"Conteúdo aprovado na iteração {self.iteration}")
        else:
            print(f"Qualidade insuficiente ({quality_score}). Melhorando conteúdo...")
            improvement = {
                "content_id": message["content_id"],
                "action": "improve",
                "suggestions": message.get("suggestions", [])
            }
            self.rabbitmq.publish_event("content.improved", improvement)

    def start(self):
        """Inicia o Feedback Loop."""
        print("Iniciando o Feedback Loop...")
        self.publish_content()
        self.rabbitmq.consume_event("content.reviewed", self.process_reviewed_event)

```

## build\lib\core\models.py

```python
from typing import List, Dict, Union
from pydantic import BaseModel, Field
from datetime import datetime

# Modelo para mensagens consumidas pela fila planning.generated
class PlanningGenerated(BaseModel):
    topic: str
    keywords: List[str]
    target_audience: str
    content_type: str
    priority: int
    estimated_impact: float

# Modelo para mensagens publicadas na fila content.generated
class ContentGenerated(BaseModel):
    content: str
    title: str
    meta_description: str
    keywords: List[str]
    seo_score: float
    readability_score: float
    created_at: datetime = Field(default_factory=datetime.now)

# Modelo para mensagens consumidas pela fila content.improved
class ContentImproved(BaseModel):
    content_id: str
    content: str
    suggestions: List[str]
```

## build\lib\core\orchestrator.py

```python
from typing import Dict, List
from datetime import datetime
import asyncio
from pydantic import BaseModel, Field
from core.rabbitmq_utils import RabbitMQUtils

class ContentRequest(BaseModel):
    """Modelo para solicitação de conteúdo"""
    topic: str
    domain: str
    content_type: str
    target_audience: str
    technical_level: str = "intermediate"
    keywords: List[str] = []
    references_required: bool = True
    code_examples: bool = True

class Orchestrator:
    """
    Orquestrador para funcionalidades específicas.

    Cenários onde o Orchestrator deve ser usado:
    1. Consolidação de Dados:
       - Combina resultados de diferentes agentes para criar respostas finais.
    2. Fluxos Híbridos:
       - Processa fluxos que envolvem lógica adicional ou validações antes de completar uma tarefa.
    3. Monitoramento:
       - Realiza debugging ou logging centralizado para rastrear fluxos.

    Importante:
    - Para fluxos altamente escaláveis ou independentes, use o RabbitMQ para comunicação direta entre agentes.
    """

    def __init__(self):
        self.rabbitmq = RabbitMQUtils()

    def publish_request(self, request: ContentRequest):
        """Publica solicitação inicial na fila `planning.requests`"""
        try:
            self.rabbitmq.publish_event("planning.requests", request.dict())
            print(f"Solicitação publicada na fila `planning.requests`: {request.dict()}")
        except Exception as e:
            print(f"Erro ao publicar solicitação: {str(e)}")

    def consume_and_consolidate(self):
        """
        Consome mensagens de várias filas e consolida os resultados.
        Usado para criar relatórios finais ou responder a solicitações completas.
        """
        def callback(ch, method, properties, body):
            import json
            response = json.loads(body)
            print(f"Resposta recebida: {response}")

            # Exemplo de lógica de consolidação (adapte conforme necessário)
            if "content" in response:
                print(f"Conteúdo recebido: {response['content'][:200]}...")
            elif "plan" in response:
                print(f"Plano estratégico recebido: {response['plan']}")

        # Consumir respostas de filas relevantes
        self.rabbitmq.consume_event("search.results", callback)
        self.rabbitmq.consume_event("content.reviewed", callback)

    async def process_request_with_control(self, request: ContentRequest):
        """
        Executa a solicitação publicando na fila inicial e consolidando respostas.
        Usado para monitorar e consolidar fluxos em tempo real.
        """
        try:
            # Publica a solicitação inicial
            self.publish_request(request)

            # Aguarda e processa respostas
            print("Aguardando respostas das filas...")
            self.consume_and_consolidate()

        except Exception as e:
            print(f"Erro no processamento da solicitação: {str(e)}")

if __name__ == "__main__":
    async def test_orchestrator():
        orchestrator = Orchestrator()

        # Simula uma solicitação de conteúdo
        request = ContentRequest(
            topic="IA Generativa em Produção",
            domain="machine_learning",
            content_type="technical_guide",
            target_audience="desenvolvedores",
            technical_level="advanced",
            keywords=["IA", "Machine Learning", "Produção"]
        )

        # Processa a solicitação
        await orchestrator.process_request_with_control(request)

    asyncio.run(test_orchestrator())

```

## build\lib\core\rabbitmq_utils.py

```python
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

```

## build\lib\core\testa api.py

```python
from core.config import get_settings
from agents.planning.config import config

# Validação da chave da API
settings = get_settings()
if not settings.api.openai_api_key:
    raise ValueError("A chave da API OpenAI (openai_api_key) não foi configurada corretamente.")
print(f"Chave da API OpenAI encontrada: {settings.api.openai_api_key}")



```

## build\lib\core\__init__.py

```python

```

## build\lib\core\domains\adoption_metrics.py

```python
# core\domains\adoption_metrics.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, date
from enum import Enum

class AdoptionStage(str, Enum):
    """Estágios de adoção de tecnologia"""
    INNOVATORS = "innovators"         # 2.5% iniciais
    EARLY_ADOPTERS = "early_adopters" # 13.5% seguintes
    EARLY_MAJORITY = "early_majority" # 34% primeiros da maioria
    LATE_MAJORITY = "late_majority"   # 34% últimos da maioria
    LAGGARDS = "laggards"             # 16% finais

class AdoptionMetric(BaseModel):
    """Métricas de adoção de tecnologia"""
    technology: str
    category: str
    current_stage: AdoptionStage
    adoption_rate: float = Field(ge=0.0, le=1.0)
    growth_rate: float  # % de crescimento mensal
    market_penetration: float = Field(ge=0.0, le=1.0)
    enterprise_adoption: float = Field(ge=0.0, le=1.0)
    startup_adoption: float = Field(ge=0.0, le=1.0)
    developer_satisfaction: float = Field(ge=0.0, le=1.0)
    time_to_value: int  # dias
    measured_at: datetime = Field(default_factory=datetime.now)

class ROIMetric(BaseModel):
    """Métricas de ROI"""
    cost_reduction: float
    productivity_gain: float
    time_saved: float  # horas por semana
    quality_improvement: float
    user_satisfaction: float = Field(ge=0.0, le=1.0)
    implementation_cost: float
    maintenance_cost: float
    training_time: int  # horas

class UseCaseMetric(BaseModel):
    """Métricas específicas de casos de uso"""
    use_case: str
    success_rate: float = Field(ge=0.0, le=1.0)
    implementation_time: int  # dias
    roi_metrics: ROIMetric
    adoption_barriers: List[str]
    best_practices: List[str]
    common_pitfalls: List[str]

# Métricas de adoção atuais
CURRENT_ADOPTION_METRICS = {
    "llms": AdoptionMetric(
        technology="Large Language Models",
        category="ai",
        current_stage=AdoptionStage.EARLY_MAJORITY,
        adoption_rate=0.34,
        growth_rate=15.0,  # 15% ao mês
        market_penetration=0.45,
        enterprise_adoption=0.38,
        startup_adoption=0.65,
        developer_satisfaction=0.85,
        time_to_value=14,  # 2 semanas
    ),
    
    "vector_databases": AdoptionMetric(
        technology="Vector Databases",
        category="data",
        current_stage=AdoptionStage.EARLY_ADOPTERS,
        adoption_rate=0.15,
        growth_rate=25.0,
        market_penetration=0.20,
        enterprise_adoption=0.15,
        startup_adoption=0.40,
        developer_satisfaction=0.80,
        time_to_value=7
    ),
    
    "ai_agents": AdoptionMetric(
        technology="AI Agents",
        category="ai",
        current_stage=AdoptionStage.INNOVATORS,
        adoption_rate=0.05,
        growth_rate=45.0,
        market_penetration=0.08,
        enterprise_adoption=0.03,
        startup_adoption=0.12,
        developer_satisfaction=0.75,
        time_to_value=21
    )
}

# Métricas de casos de uso
USE_CASE_METRICS = {
    "customer_service": UseCaseMetric(
        use_case="AI Customer Service",
        success_rate=0.82,
        implementation_time=30,
        roi_metrics=ROIMetric(
            cost_reduction=0.35,  # 35% redução de custos
            productivity_gain=0.45,
            time_saved=15.0,  # 15 horas por semana
            quality_improvement=0.25,
            user_satisfaction=0.78,
            implementation_cost=50000,  # USD
            maintenance_cost=2000,  # USD por mês
            training_time=20  # horas
        ),
        adoption_barriers=[
            "Integration complexity",
            "Data privacy concerns",
            "Staff training",
            "Cost justification"
        ],
        best_practices=[
            "Start with simple use cases",
            "Comprehensive training",
            "Clear escalation paths",
            "Regular performance monitoring"
        ],
        common_pitfalls=[
            "Over-automation",
            "Insufficient testing",
            "Poor integration",
            "Lack of monitoring"
        ]
    ),
    
    "code_assistance": UseCaseMetric(
        use_case="AI Code Assistant",
        success_rate=0.88,
        implementation_time=7,
        roi_metrics=ROIMetric(
            cost_reduction=0.20,
            productivity_gain=0.40,
            time_saved=10.0,
            quality_improvement=0.30,
            user_satisfaction=0.85,
            implementation_cost=15000,
            maintenance_cost=500,
            training_time=8
        ),
        adoption_barriers=[
            "Security concerns",
            "Code quality trust",
            "Learning curve",
            "Integration with existing tools"
        ],
        best_practices=[
            "Code review protocols",
            "Security guidelines",
            "Training programs",
            "Measurement frameworks"
        ],
        common_pitfalls=[
            "Blind trust in suggestions",
            "Inadequate review",
            "Security oversights",
            "Poor prompt practices"
        ]
    )
}

class AdoptionMetricsManager:
    """Gerenciador de métricas de adoção"""
    
    def __init__(self):
        self.metrics = CURRENT_ADOPTION_METRICS
        self.use_cases = USE_CASE_METRICS
    
    def get_metric(self, technology: str) -> Optional[AdoptionMetric]:
        """Retorna métricas para uma tecnologia"""
        return self.metrics.get(technology.lower())
    
    def get_use_case_metrics(self, use_case: str) -> Optional[UseCaseMetric]:
        """Retorna métricas para um caso de uso"""
        return self.use_cases.get(use_case.lower())
    
    def get_trending_technologies(self) -> List[str]:
        """Retorna tecnologias com maior crescimento"""
        return sorted(
            self.metrics.keys(),
            key=lambda x: self.metrics[x].growth_rate,
            reverse=True
        )
    
    def calculate_roi_projection(self, use_case: str, timeframe_months: int) -> Dict:
        """Calcula projeção de ROI para um caso de uso"""
        metrics = self.get_use_case_metrics(use_case)
        if not metrics:
            return {}
            
        roi = metrics.roi_metrics
        monthly_savings = (
            roi.cost_reduction * roi.implementation_cost +
            (roi.productivity_gain * 160 * 50)  # 160h/mês * $50/h
        )
        
        total_cost = (
            roi.implementation_cost +
            (roi.maintenance_cost * timeframe_months)
        )
        
        total_savings = monthly_savings * timeframe_months
        net_roi = (total_savings - total_cost) / total_cost
        
        return {
            "timeframe_months": timeframe_months,
            "total_cost": total_cost,
            "total_savings": total_savings,
            "net_roi": net_roi,
            "break_even_months": roi.implementation_cost / monthly_savings
        }
    
    def get_adoption_recommendations(self, technology: str) -> Dict:
        """Gera recomendações de adoção para uma tecnologia"""
        metric = self.get_metric(technology)
        if not metric:
            return {}
            
        recommendations = {
            "readiness_score": self._calculate_readiness_score(metric),
            "suggested_approach": self._get_suggested_approach(metric),
            "risk_level": self._calculate_risk_level(metric),
            "next_steps": self._get_next_steps(metric)
        }
        
        return recommendations
    
    def _calculate_readiness_score(self, metric: AdoptionMetric) -> float:
        """Calcula score de prontidão para adoção"""
        weights = {
            "market_penetration": 0.3,
            "developer_satisfaction": 0.3,
            "enterprise_adoption": 0.2,
            "time_to_value": 0.2
        }
        
        # Normaliza time_to_value (menor é melhor)
        time_score = 1 - (metric.time_to_value / 90)  # normaliza para 90 dias
        
        score = (
            metric.market_penetration * weights["market_penetration"] +
            metric.developer_satisfaction * weights["developer_satisfaction"] +
            metric.enterprise_adoption * weights["enterprise_adoption"] +
            time_score * weights["time_to_value"]
        )
        
        return min(max(score, 0.0), 1.0)
    
    def _get_suggested_approach(self, metric: AdoptionMetric) -> str:
        """Define abordagem sugerida baseada no estágio de adoção"""
        approaches = {
            AdoptionStage.INNOVATORS: "Experimental pilot with close monitoring",
            AdoptionStage.EARLY_ADOPTERS: "Limited production deployment",
            AdoptionStage.EARLY_MAJORITY: "Full production rollout",
            AdoptionStage.LATE_MAJORITY: "Standardized implementation",
            AdoptionStage.LAGGARDS: "Conservative adoption with proven patterns"
        }
        return approaches[metric.current_stage]
    
    def _calculate_risk_level(self, metric: AdoptionMetric) -> str:
        """Calcula nível de risco baseado nas métricas"""
        risk_score = (
            (1 - metric.market_penetration) * 0.4 +
            (1 - metric.enterprise_adoption) * 0.3 +
            (1 - metric.developer_satisfaction) * 0.3
        )
        
        if risk_score > 0.7:
            return "HIGH"
        elif risk_score > 0.4:
            return "MEDIUM"
        return "LOW"
    
    def _get_next_steps(self, metric: AdoptionMetric) -> List[str]:
        """Define próximos passos baseado no estágio atual"""
        if metric.current_stage == AdoptionStage.INNOVATORS:
            return [
                "Start with proof of concept",
                "Focus on learning and exploration",
                "Build internal expertise",
                "Prepare risk mitigation strategies"
            ]
        elif metric.current_stage == AdoptionStage.EARLY_ADOPTERS:
            return [
                "Plan limited production deployment",
                "Document best practices",
                "Establish monitoring",
                "Create training programs"
            ]
        else:
            return [
                "Scale deployment",
                "Optimize processes",
                "Share knowledge",
                "Measure ROI"
            ]

if __name__ == "__main__":
    # Exemplo de uso
    manager = AdoptionMetricsManager()
    
    # Análise de LLMs
    llm_metrics = manager.get_metric("llms")
    if llm_metrics:
        print(f"LLM Adoption Rate: {llm_metrics.adoption_rate * 100}%")
        print(f"Growth Rate: {llm_metrics.growth_rate}% monthly")
    
    # ROI de caso de uso
    roi_projection = manager.calculate_roi_projection("customer_service", 12)
    print(f"\nCustomer Service ROI (12 months):")
    print(f"Net ROI: {roi_projection['net_roi']:.2f}")
    print(f"Break Even: {roi_projection['break_even_months']:.1f} months")
    
    # Recomendações
    recommendations = manager.get_adoption_recommendations("ai_agents")
    print(f"\nAI Agents Recommendations:")
    print(f"Readiness Score: {recommendations['readiness_score']:.2f}")
    print(f"Risk Level: {recommendations['risk_level']}")
    print(f"Approach: {recommendations['suggested_approach']}")
```

## build\lib\core\domains\bigtech_monitor.py

```python
# core\domains\bigtech_monitor.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, date
from enum import Enum

class CompanyCategory(str, Enum):
    """Categorias de empresas de tecnologia"""
    AI_SPECIALIST = "ai_specialist"     # Ex: OpenAI, Anthropic
    CLOUD_PROVIDER = "cloud_provider"   # Ex: AWS, GCP
    ENTERPRISE = "enterprise"           # Ex: Microsoft, Oracle
    SOCIAL_MEDIA = "social_media"       # Ex: Meta
    ECOMMERCE = "ecommerce"            # Ex: Amazon, Mercado Livre
    FINTECH = "fintech"                # Ex: Stripe, Square

class TechnologyStatus(str, Enum):
    """Status de uma tecnologia"""
    RESEARCH = "research"          # Em pesquisa
    PREVIEW = "preview"           # Preview/Beta
    GA = "ga"                     # Generally Available
    DEPRECATED = "deprecated"     # Descontinuada

class Innovation(BaseModel):
    """Modelo para inovações tecnológicas"""
    name: str
    company: str
    announcement_date: date
    release_date: Optional[date]
    status: TechnologyStatus
    description: str
    key_features: List[str]
    target_users: List[str]
    competitors: List[str]
    pricing_model: Optional[str]
    documentation_url: Optional[str]
    impact_score: float = Field(ge=0.0, le=1.0)

class Product(BaseModel):
    """Modelo para produtos/serviços"""
    name: str
    category: str
    launch_date: date
    current_version: str
    status: TechnologyStatus
    market_share: float
    key_features: List[str]
    pricing: Dict[str, float]
    enterprise_ready: bool

class BigTechCompany(BaseModel):
    """Modelo para empresas BigTech"""
    name: str
    category: CompanyCategory
    market_cap: float  # em bilhões USD
    key_products: List[Product]
    recent_innovations: List[Innovation]
    strategic_focus: List[str]
    open_source: List[str]
    apis_services: Dict[str, str]
    developer_resources: List[str]
    last_updated: datetime = Field(default_factory=datetime.now)

# Monitoramento das principais BigTechs
BIGTECH_LANDSCAPE = {
    "openai": BigTechCompany(
        name="OpenAI",
        category=CompanyCategory.AI_SPECIALIST,
        market_cap=90.0,  # estimativa
        key_products=[
            Product(
                name="GPT-4",
                category="language_model",
                launch_date=date(2023, 3, 14),
                current_version="GPT-4 Turbo",
                status=TechnologyStatus.GA,
                market_share=0.65,
                key_features=[
                    "128k context window",
                    "Advanced reasoning",
                    "Multimodal capabilities",
                    "JSON mode"
                ],
                pricing={
                    "input": 0.01,
                    "output": 0.03
                },
                enterprise_ready=True
            ),
            Product(
                name="DALL-E 3",
                category="image_generation",
                launch_date=date(2023, 10, 1),
                current_version="3.0",
                status=TechnologyStatus.GA,
                market_share=0.45,
                key_features=[
                    "Photorealistic images",
                    "Text understanding",
                    "Style control"
                ],
                pricing={
                    "standard": 0.02,
                    "hd": 0.04
                },
                enterprise_ready=True
            )
        ],
        recent_innovations=[
            Innovation(
                name="GPT-4 Turbo",
                company="OpenAI",
                announcement_date=date(2023, 11, 6),
                release_date=date(2023, 11, 6),
                status=TechnologyStatus.GA,
                description="Versão melhorada do GPT-4",
                key_features=[
                    "Maior contexto",
                    "Conhecimento mais recente",
                    "Melhor performance"
                ],
                target_users=[
                    "Developers",
                    "Enterprises",
                    "Researchers"
                ],
                competitors=[
                    "Anthropic Claude",
                    "Google PaLM"
                ],
                pricing_model="Pay per token",
                documentation_url="https://platform.openai.com/docs",
                impact_score=0.9
            )
        ],
        strategic_focus=[
            "AGI Development",
            "AI Safety",
            "Enterprise AI"
        ],
        open_source=[
            "Whisper",
            "Gym",
            "CLIP"
        ],
        apis_services={
            "GPT-4": "https://api.openai.com/v1/chat/completions",
            "DALL-E": "https://api.openai.com/v1/images/generations",
            "Embeddings": "https://api.openai.com/v1/embeddings"
        },
        developer_resources=[
            "OpenAI Cookbook",
            "API Documentation",
            "OpenAI Platform"
        ]
    ),

    "anthropic": BigTechCompany(
        name="Anthropic",
        category=CompanyCategory.AI_SPECIALIST,
        market_cap=20.0,  # estimativa
        key_products=[
            Product(
                name="Claude 2.1",
                category="language_model",
                launch_date=date(2023, 11, 21),
                current_version="2.1",
                status=TechnologyStatus.GA,
                market_share=0.15,
                key_features=[
                    "200k context window",
                    "Constitutional AI",
                    "Advanced reasoning",
                    "Tool use"
                ],
                pricing={
                    "input": 0.008,
                    "output": 0.024
                },
                enterprise_ready=True
            )
        ],
        recent_innovations=[],  # Adicionado quando houver
        strategic_focus=[
            "Safe AI Development",
            "Enterprise Solutions",
            "AI Alignment"
        ],
        open_source=[],  # Principalmente closed source
        apis_services={
            "Claude": "https://api.anthropic.com/v1/complete"
        },
        developer_resources=[
            "Claude Documentation",
            "API Guidelines"
        ]
    )
}

class BigTechMonitor:
    """Monitor de BigTechs"""
    
    def __init__(self):
        self.companies = BIGTECH_LANDSCAPE
    
    def get_company(self, company_id: str) -> Optional[BigTechCompany]:
        """Retorna detalhes de uma empresa"""
        return self.companies.get(company_id.lower())
    
    def get_companies_by_category(self, category: CompanyCategory) -> List[BigTechCompany]:
        """Retorna empresas por categoria"""
        return [
            company for company in self.companies.values()
            if company.category == category
        ]
    
    def get_product_details(self, company_id: str, product_name: str) -> Optional[Product]:
        """Retorna detalhes de um produto"""
        company = self.get_company(company_id)
        if not company:
            return None
            
        for product in company.key_products:
            if product.name.lower() == product_name.lower():
                return product
        return None
    
    def get_recent_innovations(self, days: int = 90) -> List[Innovation]:
        """Retorna inovações recentes"""
        recent_innovations = []
        cutoff_date = date.today().replace(day=date.today().day - days)
        
        for company in self.companies.values():
            for innovation in company.recent_innovations:
                if innovation.announcement_date >= cutoff_date:
                    recent_innovations.append(innovation)
        
        return sorted(
            recent_innovations,
            key=lambda x: x.announcement_date,
            reverse=True
        )
    
    def get_competitive_landscape(self, product_category: str) -> Dict:
        """Analisa panorama competitivo de uma categoria"""
        landscape = {
            "products": [],
            "market_share": {},
            "pricing_comparison": {},
            "feature_matrix": {}
        }
        
        for company in self.companies.values():
            for product in company.key_products:
                if product.category == product_category:
                    landscape["products"].append(product.name)
                    landscape["market_share"][product.name] = product.market_share
                    landscape["pricing_comparison"][product.name] = product.pricing
                    landscape["feature_matrix"][product.name] = product.key_features
        
        return landscape
    
    def get_strategic_trends(self) -> Dict[str, List[str]]:
        """Identifica tendências estratégicas"""
        all_focus_areas = []
        for company in self.companies.values():
            all_focus_areas.extend(company.strategic_focus)
        
        # Conta ocorrências
        from collections import Counter
        focus_counter = Counter(all_focus_areas)
        
        # Organiza por relevância
        trends = {
            "high_focus": [],
            "medium_focus": [],
            "emerging": []
        }
        
        for area, count in focus_counter.items():
            if count >= 3:
                trends["high_focus"].append(area)
            elif count == 2:
                trends["medium_focus"].append(area)
            else:
                trends["emerging"].append(area)
                
        return trends

if __name__ == "__main__":
    # Exemplo de uso
    monitor = BigTechMonitor()
    
    # Verificar inovações recentes
    recent = monitor.get_recent_innovations(30)  # últimos 30 dias
    print(f"Inovações recentes: {len(recent)}")
    
    # Análise competitiva de LLMs
    llm_landscape = monitor.get_competitive_landscape("language_model")
    print("\nPanorama LLMs:")
    print(f"Produtos: {llm_landscape['products']}")
    print(f"Market Share: {llm_landscape['market_share']}")
    
    # Tendências estratégicas
    trends = monitor.get_strategic_trends()
    print("\nTendências Principais:", trends["high_focus"])
```

## build\lib\core\domains\definitions.py

```python
# core\domains\definitions.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class DomainType(str, Enum):
    """Tipos de domínios"""
    TECHNICAL = "technical"         # Domínios técnicos
    BUSINESS = "business"           # Domínios de negócio
    INDUSTRY = "industry"           # Domínios de indústria
    EMERGING = "emerging"           # Domínios emergentes
    RESEARCH = "research"           # Domínios de pesquisa

class ContentType(str, Enum):
    """Tipos de conteúdo"""
    TUTORIAL = "tutorial"           # Tutoriais passo-a-passo
    GUIDE = "guide"                # Guias abrangentes
    REFERENCE = "reference"        # Documentação de referência
    CASE_STUDY = "case_study"      # Estudos de caso
    COMPARISON = "comparison"      # Comparativos
    DEEP_DIVE = "deep_dive"        # Análises profundas
    NEWS = "news"                  # Notícias e atualizações

class ExpertiseLevel(str, Enum):
    """Níveis de expertise"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class UpdateFrequency(str, Enum):
    """Frequência de atualização"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class ValidationRule(BaseModel):
    """Regra de validação de conteúdo"""
    name: str
    description: str
    threshold: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    validation_type: str

class DomainDefinition(BaseModel):
    """Definição completa de um domínio"""
    name: str
    type: DomainType
    description: str
    expertise_levels: List[ExpertiseLevel]
    content_types: List[ContentType]
    update_frequency: UpdateFrequency
    primary_keywords: List[str]
    related_keywords: List[str]
    validation_rules: List[ValidationRule]
    required_citations: bool
    requires_code_examples: bool
    requires_diagrams: bool
    last_updated: datetime = Field(default_factory=datetime.now)

# Definições dos domínios principais
DOMAIN_DEFINITIONS = {
    "ai_agents": DomainDefinition(
        name="AI Agents & Automation",
        type=DomainType.TECHNICAL,
        description="Desenvolvimento e implementação de agentes autônomos de IA",
        expertise_levels=[
            ExpertiseLevel.INTERMEDIATE,
            ExpertiseLevel.ADVANCED,
            ExpertiseLevel.EXPERT
        ],
        content_types=[
            ContentType.TUTORIAL,
            ContentType.GUIDE,
            ContentType.DEEP_DIVE,
            ContentType.CASE_STUDY
        ],
        update_frequency=UpdateFrequency.WEEKLY,
        primary_keywords=[
            "ai agents",
            "autonomous systems",
            "agent orchestration",
            "multi-agent systems"
        ],
        related_keywords=[
            "llms",
            "machine learning",
            "neural networks",
            "decision making"
        ],
        validation_rules=[
            ValidationRule(
                name="code_quality",
                description="Qualidade e clareza do código",
                threshold=0.8,
                weight=0.3,
                validation_type="code_analysis"
            ),
            ValidationRule(
                name="technical_accuracy",
                description="Precisão técnica do conteúdo",
                threshold=0.9,
                weight=0.4,
                validation_type="expert_review"
            ),
            ValidationRule(
                name="practical_applicability",
                description="Aplicabilidade prática",
                threshold=0.7,
                weight=0.3,
                validation_type="user_feedback"
            )
        ],
        required_citations=True,
        requires_code_examples=True,
        requires_diagrams=True
    ),

    "mlops": DomainDefinition(
        name="MLOps & Production AI",
        type=DomainType.TECHNICAL,
        description="Práticas e ferramentas para ML em produção",
        expertise_levels=[
            ExpertiseLevel.INTERMEDIATE,
            ExpertiseLevel.ADVANCED
        ],
        content_types=[
            ContentType.GUIDE,
            ContentType.REFERENCE,
            ContentType.CASE_STUDY
        ],
        update_frequency=UpdateFrequency.WEEKLY,
        primary_keywords=[
            "mlops",
            "machine learning",
            "deployment",
            "monitoring"
        ],
        related_keywords=[
            "devops",
            "continuous integration",
            "model registry",
            "feature store"
        ],
        validation_rules=[
            ValidationRule(
                name="deployment_practicality",
                description="Viabilidade de implementação",
                threshold=0.85,
                weight=0.4,
                validation_type="expert_review"
            ),
            ValidationRule(
                name="best_practices",
                description="Aderência às melhores práticas",
                threshold=0.9,
                weight=0.3,
                validation_type="checklist"
            ),
            ValidationRule(
                name="completeness",
                description="Completude da solução",
                threshold=0.8,
                weight=0.3,
                validation_type="coverage_analysis"
            )
        ],
        required_citations=True,
        requires_code_examples=True,
        requires_diagrams=True
    ),

    "enterprise_ai": DomainDefinition(
        name="Enterprise AI Integration",
        type=DomainType.BUSINESS,
        description="Implementação de IA em ambientes empresariais",
        expertise_levels=[
            ExpertiseLevel.INTERMEDIATE,
            ExpertiseLevel.ADVANCED
        ],
        content_types=[
            ContentType.CASE_STUDY,
            ContentType.GUIDE,
            ContentType.COMPARISON
        ],
        update_frequency=UpdateFrequency.MONTHLY,
        primary_keywords=[
            "enterprise ai",
            "business intelligence",
            "digital transformation",
            "ai strategy"
        ],
        related_keywords=[
            "roi",
            "change management",
            "integration",
            "scalability"
        ],
        validation_rules=[
            ValidationRule(
                name="business_impact",
                description="Impacto nos negócios",
                threshold=0.8,
                weight=0.4,
                validation_type="roi_analysis"
            ),
            ValidationRule(
                name="implementation_feasibility",
                description="Viabilidade de implementação",
                threshold=0.85,
                weight=0.3,
                validation_type="expert_review"
            ),
            ValidationRule(
                name="enterprise_readiness",
                description="Prontidão para empresa",
                threshold=0.9,
                weight=0.3,
                validation_type="checklist"
            )
        ],
        required_citations=True,
        requires_code_examples=False,
        requires_diagrams=True
    )
}

class DomainManager:
    """Gerenciador central de domínios"""
    
    def __init__(self):
        self.domains = DOMAIN_DEFINITIONS
    
    def get_domain(self, domain_id: str) -> Optional[DomainDefinition]:
        """Retorna definição de um domínio"""
        return self.domains.get(domain_id.lower())
    
    def get_domains_by_type(self, domain_type: DomainType) -> List[DomainDefinition]:
        """Retorna domínios por tipo"""
        return [
            domain for domain in self.domains.values()
            if domain.type == domain_type
        ]
    
    def get_validation_rules(self, domain_id: str) -> List[ValidationRule]:
        """Retorna regras de validação de um domínio"""
        domain = self.get_domain(domain_id)
        return domain.validation_rules if domain else []
    
    def validate_content_requirements(
        self,
        domain_id: str,
        content: Dict
    ) -> Dict[str, bool]:
        """Valida requisitos de conteúdo para um domínio"""
        domain = self.get_domain(domain_id)
        if not domain:
            return {}
            
        return {
            "has_citations": self._check_citations(content) if domain.required_citations else True,
            "has_code": self._check_code_examples(content) if domain.requires_code_examples else True,
            "has_diagrams": self._check_diagrams(content) if domain.requires_diagrams else True
        }
    
    def _check_citations(self, content: Dict) -> bool:
        """Verifica se o conteúdo tem citações adequadas"""
        # Implementar verificação de citações
        return True
    
    def _check_code_examples(self, content: Dict) -> bool:
        """Verifica se o conteúdo tem exemplos de código"""
        # Implementar verificação de código
        return True
    
    def _check_diagrams(self, content: Dict) -> bool:
        """Verifica se o conteúdo tem diagramas necessários"""
        # Implementar verificação de diagramas
        return True
    
    def get_content_guidelines(self, domain_id: str) -> Dict:
        """Retorna diretrizes de conteúdo para um domínio"""
        domain = self.get_domain(domain_id)
        if not domain:
            return {}
            
        return {
            "expertise_levels": domain.expertise_levels,
            "content_types": domain.content_types,
            "primary_keywords": domain.primary_keywords,
            "update_frequency": domain.update_frequency,
            "requirements": {
                "citations": domain.required_citations,
                "code_examples": domain.requires_code_examples,
                "diagrams": domain.requires_diagrams
            }
        }

if __name__ == "__main__":
    # Exemplo de uso
    manager = DomainManager()
    
    # Verificar domínio de AI Agents
    ai_domain = manager.get_domain("ai_agents")
    if ai_domain:
        print(f"Domain: {ai_domain.name}")
        print(f"Type: {ai_domain.type}")
        print(f"Keywords: {ai_domain.primary_keywords}")
    
    # Obter guidelines
    guidelines = manager.get_content_guidelines("ai_agents")
    print("\nContent Guidelines:")
    print(f"Expertise Levels: {guidelines['expertise_levels']}")
    print(f"Content Types: {guidelines['content_types']}")
    
    # Verificar regras de validação
    rules = manager.get_validation_rules("ai_agents")
    print("\nValidation Rules:")
    for rule in rules:
        print(f"- {rule.name}: {rule.threshold} (weight: {rule.weight})")
```

## build\lib\core\domains\market_segments.py

```python
# core\domains\market_segments.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class MarketPriority(str, Enum):
    """Níveis de prioridade para segmentos"""
    CRITICAL = "critical"      # Foco imediato
    HIGH = "high"             # Alta prioridade
    MEDIUM = "medium"         # Prioridade média
    EMERGING = "emerging"     # Mercado emergente
    NICHE = "niche"          # Nicho específico

class Industry(str, Enum):
    """Indústrias principais"""
    FINTECH = "fintech"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    EDUCATION = "education"
    LOGISTICS = "logistics"
    ENERGY = "energy"

class TechMaturity(str, Enum):
    """Nível de maturidade tecnológica"""
    CUTTING_EDGE = "cutting_edge"  # Tecnologia de ponta
    ADVANCED = "advanced"          # Tecnologia avançada
    ESTABLISHED = "established"    # Tecnologia estabelecida
    LEGACY = "legacy"             # Tecnologia legada

class MarketSegment(BaseModel):
    """Definição detalhada de segmento de mercado"""
    name: str
    industry: Industry
    priority: MarketPriority
    market_size: float  # em bilhões USD
    growth_rate: float  # % ao ano
    tech_maturity: TechMaturity
    description: str
    key_technologies: List[str]
    key_players: List[str]
    challenges: List[str]
    opportunities: List[str]
    target_audience: List[str]
    success_metrics: List[str]
    update_frequency: str
    last_updated: datetime = Field(default_factory=datetime.now)

# Definição dos segmentos de mercado
MARKET_SEGMENTS: Dict[str, MarketSegment] = {
    "fintech_banking": MarketSegment(
        name="Digital Banking & Fintech",
        industry=Industry.FINTECH,
        priority=MarketPriority.CRITICAL,
        market_size=475.0,
        growth_rate=23.4,
        tech_maturity=TechMaturity.CUTTING_EDGE,
        description="Serviços financeiros digitais e tecnologias bancárias inovadoras",
        key_technologies=[
            "Blockchain",
            "AI/ML for Risk",
            "Cloud Native",
            "APIs/Open Banking",
            "Microservices"
        ],
        key_players=[
            "Nubank",
            "Stripe",
            "Block (Square)",
            "PicPay",
            "Itaú"
        ],
        challenges=[
            "Regulamentações (PIX, Open Finance)",
            "Segurança e Fraude",
            "Escalabilidade",
            "Integração com Legacy"
        ],
        opportunities=[
            "Banking as a Service",
            "Embedded Finance",
            "AI em Crédito",
            "Pagamentos Instantâneos"
        ],
        target_audience=[
            "Desenvolvedores Fintech",
            "Arquitetos de Sistemas",
            "Product Managers",
            "Tech Leads Bancários"
        ],
        success_metrics=[
            "Uptime do Sistema",
            "Taxa de Conversão",
            "Velocidade de Transação",
            "NPS"
        ],
        update_frequency="daily"
    ),
    
    "healthtech": MarketSegment(
        name="Healthcare Technology",
        industry=Industry.HEALTHCARE,
        priority=MarketPriority.HIGH,
        market_size=280.0,
        growth_rate=18.6,
        tech_maturity=TechMaturity.ADVANCED,
        description="Tecnologias para saúde digital e telemedicina",
        key_technologies=[
            "Telemedicine Platforms",
            "AI Diagnostics",
            "Healthcare APIs",
            "Medical IoT",
            "Cloud Healthcare"
        ],
        key_players=[
            "Hospital Israelita Albert Einstein",
            "Rede D'Or",
            "Hapvida",
            "Dasa"
        ],
        challenges=[
            "Privacidade (LGPD)",
            "Integração de Dados",
            "Regulamentações",
            "Experiência do Paciente"
        ],
        opportunities=[
            "Telemedicina",
            "AI em Diagnóstico",
            "Healthtech SaaS",
            "Medicina Preventiva"
        ],
        target_audience=[
            "Desenvolvedores Healthcare",
            "Gestores Hospitalares",
            "Médicos Tech-Savvy",
            "Healthcare IT"
        ],
        success_metrics=[
            "Precisão Diagnóstica",
            "Tempo de Atendimento",
            "Satisfação do Paciente",
            "Custo por Atendimento"
        ],
        update_frequency="weekly"
    )
}

class MarketSegmentManager:
    """Gerenciador de segmentos de mercado"""
    
    def __init__(self):
        self.segments = MARKET_SEGMENTS
    
    def get_segment(self, segment_id: str) -> Optional[MarketSegment]:
        """Retorna um segmento específico"""
        return self.segments.get(segment_id)
    
    def get_segments_by_priority(self, priority: MarketPriority) -> List[MarketSegment]:
        """Retorna segmentos por nível de prioridade"""
        return [
            segment for segment in self.segments.values()
            if segment.priority == priority
        ]
    
    def get_segments_by_industry(self, industry: Industry) -> List[MarketSegment]:
        """Retorna segmentos por indústria"""
        return [
            segment for segment in self.segments.values()
            if segment.industry == industry
        ]
    
    def get_tech_requirements(self, segment_id: str) -> Dict:
        """Retorna requisitos tecnológicos do segmento"""
        segment = self.get_segment(segment_id)
        if not segment:
            return {}
            
        return {
            "technologies": segment.key_technologies,
            "maturity": segment.tech_maturity,
            "update_frequency": segment.update_frequency
        }
    
    def get_market_metrics(self, segment_id: str) -> Dict:
        """Retorna métricas de mercado do segmento"""
        segment = self.get_segment(segment_id)
        if not segment:
            return {}
            
        return {
            "market_size": segment.market_size,
            "growth_rate": segment.growth_rate,
            "key_players": segment.key_players
        }
    
    def should_update_content(self, segment_id: str) -> bool:
        """Verifica se o conteúdo do segmento precisa ser atualizado"""
        segment = self.get_segment(segment_id)
        if not segment:
            return False
            
        # Calcula tempo desde última atualização
        time_diff = datetime.now() - segment.last_updated
        
        # Define limites baseado na frequência de atualização
        update_limits = {
            "daily": 1,      # 1 dia
            "weekly": 7,     # 7 dias
            "monthly": 30    # 30 dias
        }
        
        days_limit = update_limits.get(segment.update_frequency, 7)
        return time_diff.days >= days_limit
```

## build\lib\core\domains\seasonality.py

```python
# core\domains\seasonality.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, date
from enum import Enum

class SeasonType(str, Enum):
    """Tipos de sazonalidade"""
    RETAIL = "retail"           # Eventos de varejo
    BUSINESS = "business"       # Ciclos de negócio
    TECH = "tech"              # Eventos tecnológicos
    MARKET = "market"          # Ciclos de mercado
    INDUSTRY = "industry"      # Específico da indústria
    REGULATORY = "regulatory"  # Marcos regulatórios

class EventStatus(str, Enum):
    """Status do evento sazional"""
    UPCOMING = "upcoming"      # Próximo
    ACTIVE = "active"         # Em andamento
    PASSED = "passed"         # Passado
    RECURRING = "recurring"   # Recorrente

class SeasonalEvent(BaseModel):
    """Evento sazional"""
    name: str
    type: SeasonType
    description: str
    start_date: date
    end_date: date
    impact_level: float = Field(ge=0.0, le=1.0)  # 0 a 1
    affected_industries: List[str]
    key_themes: List[str]
    content_priorities: List[str]
    recurrence: str  # annual, quarterly, monthly
    historical_metrics: Dict[str, float]
    preparation_time_days: int

class SeasonalityManager:
    """Gerenciador de sazonalidade"""
    
    def __init__(self):
        self.events = self._initialize_events()
    
    def _initialize_events(self) -> Dict[str, SeasonalEvent]:
        """Inicializa eventos sazonais"""
        return {
            "black_friday": SeasonalEvent(
                name="Black Friday",
                type=SeasonType.RETAIL,
                description="Principal evento de vendas do varejo",
                start_date=date(2024, 11, 29),
                end_date=date(2024, 11, 29),
                impact_level=0.9,
                affected_industries=[
                    "retail",
                    "e-commerce",
                    "fintech",
                    "logistics"
                ],
                key_themes=[
                    "performance",
                    "scalability",
                    "payments",
                    "security"
                ],
                content_priorities=[
                    "system_optimization",
                    "load_testing",
                    "monitoring",
                    "incident_response"
                ],
                recurrence="annual",
                historical_metrics={
                    "2023_transactions": 15000000,
                    "2023_peak_tps": 5000,
                    "2023_conversion": 0.045
                },
                preparation_time_days=90
            ),

            "tax_season": SeasonalEvent(
                name="Tax Season",
                type=SeasonType.REGULATORY,
                description="Período de declaração de impostos",
                start_date=date(2024, 3, 1),
                end_date=date(2024, 4, 30),
                impact_level=0.8,
                affected_industries=[
                    "fintech",
                    "accounting",
                    "banking",
                    "consulting"
                ],
                key_themes=[
                    "tax_compliance",
                    "automation",
                    "data_processing",
                    "security"
                ],
                content_priorities=[
                    "integration_guides",
                    "automation_tools",
                    "security_best_practices",
                    "performance_optimization"
                ],
                recurrence="annual",
                historical_metrics={
                    "2023_declarations": 35000000,
                    "2023_digital_rate": 0.92
                },
                preparation_time_days=60
            ),

            "tech_conference_season": SeasonalEvent(
                name="Tech Conference Season",
                type=SeasonType.TECH,
                description="Principais conferências de tecnologia",
                start_date=date(2024, 5, 1),
                end_date=date(2024, 8, 31),
                impact_level=0.75,
                affected_industries=[
                    "technology",
                    "startup",
                    "enterprise",
                    "education"
                ],
                key_themes=[
                    "innovation",
                    "trends",
                    "networking",
                    "product_launch"
                ],
                content_priorities=[
                    "trend_analysis",
                    "technical_deep_dives",
                    "case_studies",
                    "tutorials"
                ],
                recurrence="annual",
                historical_metrics={
                    "2023_attendees": 500000,
                    "2023_announcements": 250
                },
                preparation_time_days=120
            ),

            "open_banking_phase": SeasonalEvent(
                name="Open Banking Implementation Phase",
                type=SeasonType.REGULATORY,
                description="Fase de implementação do Open Banking",
                start_date=date(2024, 6, 1),
                end_date=date(2024, 8, 31),
                impact_level=0.85,
                affected_industries=[
                    "fintech",
                    "banking",
                    "insurance",
                    "investment"
                ],
                key_themes=[
                    "api_integration",
                    "security",
                    "compliance",
                    "data_sharing"
                ],
                content_priorities=[
                    "technical_standards",
                    "security_guidelines",
                    "integration_tutorials",
                    "compliance_guides"
                ],
                recurrence="quarterly",
                historical_metrics={
                    "2023_apis_launched": 150,
                    "2023_integration_rate": 0.75
                },
                preparation_time_days=90
            )
        }
    
    def get_current_events(self) -> List[SeasonalEvent]:
        """Retorna eventos ativos no momento"""
        today = date.today()
        return [
            event for event in self.events.values()
            if event.start_date <= today <= event.end_date
        ]
    
    def get_upcoming_events(self, days_ahead: int = 90) -> List[SeasonalEvent]:
        """Retorna próximos eventos"""
        from datetime import timedelta
        today = date.today()
        future = today + timedelta(days=days_ahead)
        return [
            event for event in self.events.values()
            if today < event.start_date <= future
        ]
    
    def get_preparation_recommendations(self, event: SeasonalEvent) -> Dict:
        """Gera recomendações de preparação para um evento"""
        days_to_event = (event.start_date - date.today()).days
        
        return {
            "preparation_time_left": days_to_event,
            "status": self._get_preparation_status(days_to_event, event.preparation_time_days),
            "priorities": event.content_priorities,
            "key_themes": event.key_themes,
            "historical_reference": event.historical_metrics
        }
    
    def _get_preparation_status(self, days_left: int, required_days: int) -> str:
        """Determina status de preparação"""
        if days_left <= 0:
            return "CRITICAL"
        elif days_left < required_days * 0.3:
            return "URGENT"
        elif days_left < required_days * 0.7:
            return "ON_TRACK"
        else:
            return "PLANNING"
    
    def get_industry_events(self, industry: str) -> List[SeasonalEvent]:
        """Retorna eventos específicos de uma indústria"""
        return [
            event for event in self.events.values()
            if industry in event.affected_industries
        ]
    
    def get_event_impact(self, event_name: str) -> Dict:
        """Calcula impacto de um evento específico"""
        event = self.events.get(event_name)
        if not event:
            return {}
            
        return {
            "impact_level": event.impact_level,
            "affected_industries": event.affected_industries,
            "preparation_required": event.preparation_time_days,
            "historical_performance": event.historical_metrics
        }

if __name__ == "__main__":
    # Exemplo de uso
    manager = SeasonalityManager()
    
    # Verificar eventos atuais
    current = manager.get_current_events()
    print(f"Eventos ativos: {len(current)}")
    
    # Verificar próximos eventos
    upcoming = manager.get_upcoming_events(30)  # próximos 30 dias
    print(f"Eventos próximos: {len(upcoming)}")
    
    # Para cada evento ativo
    for event in current:
        prep = manager.get_preparation_recommendations(event)
        print(f"\nEvento: {event.name}")
        print(f"Status: {prep['status']}")
        print(f"Prioridades: {prep['priorities']}")
```

## build\lib\core\domains\tech_stacks.py

```python
# core\domains\tech_stacks.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class TechCategory(str, Enum):
    """Categorias de tecnologias"""
    AI_ML = "ai_ml"               # IA e Machine Learning
    DATA_ENGINEERING = "data"     # Engenharia de Dados
    CLOUD = "cloud"               # Cloud Computing
    BACKEND = "backend"           # Backend Development
    FRONTEND = "frontend"         # Frontend Development
    DEVOPS = "devops"            # DevOps & SRE
    SECURITY = "security"         # Segurança
    MOBILE = "mobile"            # Desenvolvimento Mobile

class MaturityLevel(str, Enum):
    """Nível de maturidade da tecnologia"""
    BLEEDING_EDGE = "bleeding_edge"  # Tecnologia muito recente
    EMERGING = "emerging"            # Em emergência
    GROWING = "growing"              # Em crescimento
    STABLE = "stable"                # Estável
    MATURE = "mature"                # Madura
    DECLINING = "declining"          # Em declínio

class Framework(BaseModel):
    """Modelo para frameworks e bibliotecas"""
    name: str
    category: TechCategory
    version: str
    maturity: MaturityLevel
    description: str
    key_features: List[str]
    use_cases: List[str]
    learning_curve: float = Field(ge=0.0, le=1.0)  # 0 a 1
    community_size: int
    github_stars: Optional[int]
    documentation_quality: float = Field(ge=0.0, le=1.0)
    enterprise_ready: bool
    last_release: datetime
    alternatives: List[str]

class TechStack(BaseModel):
    """Stack tecnológico completo"""
    name: str
    description: str
    primary_frameworks: List[Framework]
    supporting_tools: List[str]
    infrastructure: List[str]
    best_practices: List[str]
    pitfalls: List[str]
    learning_resources: List[str]
    success_stories: List[str]

# Definição das principais stacks e frameworks
MODERN_FRAMEWORKS = {
    "ai_agents": Framework(
        name="LangChain",
        category=TechCategory.AI_ML,
        version="0.0.300",
        maturity=MaturityLevel.GROWING,
        description="Framework para desenvolvimento de aplicações com LLMs",
        key_features=[
            "Agents",
            "Chains",
            "Memory",
            "Prompts",
            "Embeddings"
        ],
        use_cases=[
            "Chatbots",
            "Content Generation",
            "Data Analysis",
            "Search Systems"
        ],
        learning_curve=0.7,
        community_size=50000,
        github_stars=65000,
        documentation_quality=0.85,
        enterprise_ready=True,
        last_release=datetime(2023, 11, 15),
        alternatives=[
            "LlamaIndex",
            "Haystack",
            "Semantic Kernel"
        ]
    ),

    "vector_store": Framework(
        name="Pinecone",
        category=TechCategory.AI_ML,
        version="2.0.0",
        maturity=MaturityLevel.STABLE,
        description="Base de dados vetorial para AI e similarity search",
        key_features=[
            "Vector Similarity Search",
            "Real-time Updates",
            "Scalability",
            "Cloud Native"
        ],
        use_cases=[
            "Semantic Search",
            "Recommendation Systems",
            "Duplicate Detection",
            "AI Applications"
        ],
        learning_curve=0.6,
        community_size=30000,
        github_stars=None,  # Closed source
        documentation_quality=0.9,
        enterprise_ready=True,
        last_release=datetime(2023, 10, 1),
        alternatives=[
            "Weaviate",
            "Milvus",
            "Qdrant"
        ]
    ),

    "data_pipeline": Framework(
        name="Apache Airflow",
        category=TechCategory.DATA_ENGINEERING,
        version="2.7.1",
        maturity=MaturityLevel.MATURE,
        description="Plataforma para orquestração de pipelines de dados",
        key_features=[
            "DAG-based Workflows",
            "Rich UI",
            "Extensible",
            "Multiple Executors"
        ],
        use_cases=[
            "ETL Pipelines",
            "ML Pipelines",
            "Data Warehouse Loading",
            "Report Generation"
        ],
        learning_curve=0.8,
        community_size=100000,
        github_stars=31000,
        documentation_quality=0.9,
        enterprise_ready=True,
        last_release=datetime(2023, 9, 15),
        alternatives=[
            "Prefect",
            "Dagster",
            "Luigi"
        ]
    )
}

# Stacks tecnológicos completos
TECH_STACKS = {
    "modern_data_stack": TechStack(
        name="Modern Data Stack",
        description="Stack moderno para engenharia de dados e analytics",
        primary_frameworks=[
            MODERN_FRAMEWORKS["data_pipeline"]
        ],
        supporting_tools=[
            "dbt",
            "Snowflake",
            "Fivetran",
            "Looker"
        ],
        infrastructure=[
            "AWS",
            "Docker",
            "Kubernetes"
        ],
        best_practices=[
            "Data Modeling",
            "Testing",
            "Documentation",
            "Version Control"
        ],
        pitfalls=[
            "Complexity",
            "Cost Management",
            "Team Skills"
        ],
        learning_resources=[
            "dbt Learn",
            "Snowflake University",
            "Airflow Documentation"
        ],
        success_stories=[
            "Magazine Luiza",
            "iFood",
            "Nubank"
        ]
    ),

    "ai_development": TechStack(
        name="AI Development Stack",
        description="Stack completo para desenvolvimento de aplicações AI",
        primary_frameworks=[
            MODERN_FRAMEWORKS["ai_agents"],
            MODERN_FRAMEWORKS["vector_store"]
        ],
        supporting_tools=[
            "OpenAI API",
            "HuggingFace",
            "MLflow"
        ],
        infrastructure=[
            "GPU Instances",
            "Vector Databases",
            "Model Registry"
        ],
        best_practices=[
            "Prompt Engineering",
            "Model Evaluation",
            "Cost Optimization",
            "Safety & Ethics"
        ],
        pitfalls=[
            "Token Limits",
            "Cost Scaling",
            "Hallucinations"
        ],
        learning_resources=[
            "OpenAI Documentation",
            "LangChain Cookbook",
            "Pinecone Tutorials"
        ],
        success_stories=[
            "Globo",
            "Stone",
            "Inter"
        ]
    )
}

class TechStackManager:
    """Gerenciador de stacks tecnológicos"""
    
    def __init__(self):
        self.frameworks = MODERN_FRAMEWORKS
        self.stacks = TECH_STACKS
    
    def get_framework(self, framework_id: str) -> Optional[Framework]:
        """Retorna detalhes de um framework"""
        return self.frameworks.get(framework_id)
    
    def get_stack(self, stack_id: str) -> Optional[TechStack]:
        """Retorna um stack completo"""
        return self.stacks.get(stack_id)
    
    def get_alternatives(self, framework_id: str) -> List[str]:
        """Retorna alternativas para um framework"""
        framework = self.get_framework(framework_id)
        return framework.alternatives if framework else []
    
    def get_frameworks_by_category(self, category: TechCategory) -> List[Framework]:
        """Retorna frameworks por categoria"""
        return [
            framework for framework in self.frameworks.values()
            if framework.category == category
        ]
    
    def get_enterprise_ready_frameworks(self) -> List[Framework]:
        """Retorna frameworks prontos para uso empresarial"""
        return [
            framework for framework in self.frameworks.values()
            if framework.enterprise_ready
        ]
    
    def should_update_stack(self, stack_id: str) -> bool:
        """Verifica se um stack precisa de atualização"""
        stack = self.get_stack(stack_id)
        if not stack:
            return False
        
        # Verifica frameworks desatualizados
        outdated_frameworks = [
            framework for framework in stack.primary_frameworks
            if (datetime.now() - framework.last_release).days > 180
        ]
        
        return len(outdated_frameworks) > 0

if __name__ == "__main__":
    # Exemplo de uso
    manager = TechStackManager()
    
    # Listar frameworks AI/ML
    ai_frameworks = manager.get_frameworks_by_category(TechCategory.AI_ML)
    print(f"Frameworks AI/ML: {len(ai_frameworks)}")
    
    # Verificar frameworks enterprise-ready
    enterprise_frameworks = manager.get_enterprise_ready_frameworks()
    print(f"Frameworks Enterprise: {len(enterprise_frameworks)}")
    
    # Verificar stack AI
    ai_stack = manager.get_stack("ai_development")
    if ai_stack:
        print(f"\nAI Stack:")
        print(f"Frameworks: {len(ai_stack.primary_frameworks)}")
        print(f"Tools: {ai_stack.supporting_tools}")
```

## build\lib\core\domains\__init__.py

```python
# core/domains/__init__.py

from .market_segments import MarketSegmentManager, MarketSegment, MarketPriority, Industry
from .seasonality import SeasonalityManager, SeasonalEvent, SeasonType
from .tech_stacks import TechStackManager, TechStack, Framework, TechCategory, MaturityLevel
from .bigtech_monitor import BigTechMonitor, BigTechCompany, Innovation, CompanyCategory
from .adoption_metrics import AdoptionMetricsManager, AdoptionMetric, ROIMetric, UseCaseMetric
from .definitions import DomainManager, DomainDefinition, DomainType, ContentType, ValidationRule

__all__ = [
   # Market Segments
   'MarketSegmentManager',
   'MarketSegment',
   'MarketPriority',
   'Industry',
   
   # Seasonality
   'SeasonalityManager',
   'SeasonalEvent',
   'SeasonType',
   
   # Tech Stacks
   'TechStackManager',
   'TechStack',
   'Framework', 
   'TechCategory',
   'MaturityLevel',
   
   # BigTech Monitor
   'BigTechMonitor',
   'BigTechCompany',
   'Innovation',
   'CompanyCategory',
   
   # Adoption Metrics
   'AdoptionMetricsManager',
   'AdoptionMetric',
   'ROIMetric',
   'UseCaseMetric',
   
   # Definitions
   'DomainManager',
   'DomainDefinition',
   'DomainType',
   'ContentType',
   'ValidationRule'
]
```

## build\lib\examples\news_integration_example.py

```python
# examples/news_integration_example.py

import asyncio
from datetime import datetime, timedelta
from agents.search.services.news.service import NewsIntegrationService
from agents.search.services.news.models import NewsSearchQuery
import json

async def main():
    # Inicializar serviço
    news_service = NewsIntegrationService()
    await news_service.initialize()

    try:
        # Criar consulta de exemplo
        query = NewsSearchQuery(
            topic="AI and Machine Learning",
            keywords=["artificial intelligence", "deep learning", "neural networks"],
            categories=["technology", "ai"],
            start_date=datetime.now() - timedelta(days=7),
            max_results=5,
            min_relevance=0.7,
            language="en"
        )

        print(f"\nBuscando notícias sobre: {query.topic}")
        print("=" * 50)

        # Realizar busca
        articles = await news_service.search_news(query)

        print(f"\nEncontrados {len(articles)} artigos relevantes:\n")

        # Exibir resultados
        for i, article in enumerate(articles, 1):
            print(f"\nArtigo {i}:")
            print(f"Título: {article.title}")
            print(f"Fonte: {article.source}")
            print(f"Data: {article.published_date.strftime('%d/%m/%Y %H:%M')}")
            print(f"Relevância: {article.relevance_score:.2f}")
            print(f"URL: {article.url}")
            print(f"Tempo de leitura: {article.read_time} minutos")
            print("\nResumo:")
            print(article.summary)
            print("-" * 50)

        # Exibir métricas
        metrics = news_service.metrics.get_metrics()
        print("\nMétricas de execução:")
        print(json.dumps(metrics, indent=2))

    finally:
        # Fechar serviço
        await news_service.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## build\lib\examples\__init__.py

```python

```

## build\lib\infrastructure\__init__.py

```python

```

## build\lib\tests\test_config.py

```python
# test_config.py
from core.config import get_settings

settings = get_settings()
print(f"OpenAI Key: {'*' * len(settings.api.openai_api_key)}")
print(f"Pinecone Key: {'*' * len(settings.api.pinecone_api_key)}")
```

## build\lib\tests\test_content.py

```python
# tests\test_content.py
import asyncio
import sys
from pathlib import Path

# Ajusta o path para importações relativas
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from agents.content.agent import ContentAgent, ContentTemplate
from agents.content.config import ContentAgentConfig

async def test_content_generation():
    # Carrega configurações
    config = ContentAgentConfig()
    
    # Inicializa agente
    agent = ContentAgent()
    
    # Cria template de teste
    template = ContentTemplate(
        type="blog_post",
        structure={
            "target_audience": "desenvolvedores",
            "sections": ["introdução", "conceitos", "implementação", "conclusão"]
        },
        tone="technical_but_friendly",
        guidelines=[
            "use exemplos de código",
            "explique conceitos complexos de forma simples",
            "inclua referências práticas"
        ],
        seo_requirements={
            "min_words": 1200,
            "heading_structure": "h1,h2,h3",
            "keyword_density": 0.02
        }
    )
    
    # Dados de teste
    topic = "Implementando Machine Learning em Produção"
    keywords = ["MLOps", "machine learning", "deploy", "produção"]
    references = [
        {
            "title": "MLOps Best Practices",
            "content": "MLOps combina Machine Learning com práticas DevOps..."
        }
    ]
    
    try:
        # Gera conteúdo
        content = await agent.generate_content(
            topic=topic,
            keywords=keywords,
            references=references,
            template=template
        )
        
        # Imprime resultados
        print("\n" + "="*50)
        print("TÍTULO:")
        print("="*50)
        print(content.title)
        
        print("\n" + "="*50)
        print("META DESCRIPTION:")
        print("="*50)
        print(content.meta_description)
        
        print("\n" + "="*50)
        print("MÉTRICAS:")
        print("="*50)
        print(f"SEO Score: {content.seo_score}")
        print(f"Readability Score: {content.readability_score}")
        print(f"Tamanho do conteúdo: {len(content.content)} caracteres")
        print(f"Palavras-chave utilizadas: {content.keywords}")
        
        print("\n" + "="*50)
        print("CONTEÚDO COMPLETO:")
        print("="*50)
        print(content.content)
        
    except Exception as e:
        print(f"Erro durante a geração de conteúdo: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_content_generation())
```

## build\lib\tests\test_devto_client.py

```python
# tests/test_devto_client.py

import pytest
from datetime import datetime
from unittest.mock import patch, Mock
from agents.search.services.news.clients.devto import DevToClient, DevToArticle

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_response():
    """Mock para resposta da API"""
    return [
        {
            "id": 1,
            "title": "Test Article",
            "description": "Test Description",
            "url": "https://dev.to/test/article",
            "published_at": datetime.now().isoformat(),
            "tag_list": ["python", "testing"],
            "user": {
                "name": "Test User",
                "username": "testuser"
            },
            "reading_time_minutes": 5,
            "comments_count": 10,
            "public_reactions_count": 20
        }
    ]

async def test_get_articles_success(mock_response):
    """Testa busca de artigos com sucesso"""
    client = DevToClient()
    
    # Mock da sessão HTTP
    mock_session = Mock()
    mock_session.get.return_value.__aenter__.return_value.status = 200
    mock_session.get.return_value.__aenter__.return_value.json = Mock(
        return_value=mock_response
    )
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        articles = await client.get_articles(search_term="python")
        
        assert len(articles) == 1
        article = articles[0]
        assert article.title == "Test Article"
        assert article.tag_list == ["python", "testing"]
        
        # Testar conversão para NewsArticle
        news_article = client.to_news_article(article)
        assert news_article.title == article.title
        assert news_article.source == "dev.to"
        assert "reading_time" in news_article.metadata

async def test_get_articles_rate_limit(mock_response):
    """Testa comportamento quando rate limit é atingido"""
    client = DevToClient()
    
    # Mock da sessão HTTP com rate limit
    mock_session = Mock()
    mock_session.get.return_value.__aenter__.return_value.status = 429
    mock_session.get.return_value.__aenter__.return_value.headers = {"Retry-After": "60"}
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        articles = await client.get_articles(search_term="python")
        assert len(articles) == 0

async def test_get_articles_error():
    """Testa tratamento de erro na API"""
    client = DevToClient()
    
    # Mock da sessão HTTP com erro
    mock_session = Mock()
    mock_session.get.side_effect = Exception("API Error")
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        articles = await client.get_articles(search_term="python")
        assert len(articles) == 0

async def test_article_conversion(mock_response):
    """Testa conversão de DevToArticle para NewsArticle"""
    client = DevToClient()
    devto_article = DevToArticle.parse_obj(mock_response[0])
    news_article = client.to_news_article(devto_article)
    
    assert news_article.title == devto_article.title
    assert news_article.url == str(devto_article.url)
    assert news_article.source == "dev.to"
    assert news_article.tags == devto_article.tag_list
    assert news_article.metadata["reading_time"] == devto_article.reading_time_minutes
```

## build\lib\tests\test_feedback_loop.py

```python
# tests/test_feedback_loop.py

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Ajusta o path para importações relativas
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from core.feedback_loop import ContentFeedbackLoop, FeedbackMetrics
from agents.content.agent import ContentAgent, ContentTemplate
from agents.review.agent import ReviewAgent
from agents.search.agent import SearchAgent

async def test_feedback_loop():
    """Testa o ciclo de feedback autônomo"""
    
    print("\n=== Iniciando Teste do Feedback Loop ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        # Inicializa agentes
        content_agent = ContentAgent()
        review_agent = ReviewAgent()
        search_agent = SearchAgent()
        
        # Inicializa feedback loop
        feedback_loop = ContentFeedbackLoop(
            content_agent=content_agent,
            review_agent=review_agent,
            search_agent=search_agent,
            max_iterations=3,
            min_quality_threshold=0.8,
            improvement_threshold=0.1
        )
        
        # Cria template de teste
        template = ContentTemplate(
            type="technical_article",
            structure={
                "target_audience": "developers",
                "sections": [
                    "introduction",
                    "concepts",
                    "implementation",
                    "best_practices",
                    "conclusion"
                ]
            },
            tone="technical_but_friendly",
            guidelines=[
                "Include code examples",
                "Focus on practical applications",
                "Explain complex concepts clearly"
            ],
            seo_requirements={
                "min_words": 1500,
                "heading_structure": "h1,h2,h3",
                "keyword_density": 0.02
            }
        )
        
        # Contexto de teste
        context = {
            "domain": "ai_agents",
            "technical_level": "advanced",
            "industry": "technology"
        }
        
        # Gera conteúdo com feedback
        print("\nGerando conteúdo com feedback loop...")
        content, metrics = await feedback_loop.generate_with_feedback(
            topic="Implementing Autonomous AI Agents with Python",
            keywords=[
                "AI agents",
                "automation",
                "machine learning",
                "python",
                "orchestration"
            ],
            template=template,
            context=context
        )
        
        # Validações básicas
        print("\nRealizando validações...")
        assert isinstance(metrics, FeedbackMetrics), "Metrics deve ser instância de FeedbackMetrics"
        assert metrics.iteration_count > 0, "Deve ter pelo menos uma iteração"
        assert metrics.total_duration > 0, "Duração total deve ser maior que zero"
        assert len(metrics.history) > 0, "Deve ter histórico de iterações"
        
        # Imprime resultados
        print("\n=== Resultados do Feedback Loop ===")
        print(f"Número de iterações: {metrics.iteration_count}")
        print(f"Duração total: {metrics.total_duration:.2f} segundos")
        
        print("\n=== Métricas Finais ===")
        print(f"Quality Score: {metrics.quality_score:.2f}")
        print(f"Technical Accuracy: {metrics.technical_accuracy:.2f}")
        print(f"Readability Score: {metrics.readability_score:.2f}")
        print(f"SEO Score: {metrics.seo_score:.2f}")
        
        print("\n=== Histórico de Iterações ===")
        for iteration in metrics.history:
            print(f"\nIteração {iteration.iteration_number}:")
            print("Métricas:")
            for metric, value in iteration.content_metrics.items():
                print(f"- {metric}: {value:.2f}")
            
            print("\nSugestões recebidas:")
            for suggestion in iteration.suggestions:
                print(f"- {suggestion}")
            
            print("\nMelhorias aplicadas:")
            for improvement in iteration.improvements_made:
                print(f"- {improvement}")
        
        print("\n=== Preview do Conteúdo ===")
        print("\nTítulo:")
        print(content.title)
        print("\nMeta Description:")
        print(content.meta_description)
        print("\nPreview do conteúdo:")
        preview = content.content[:500] + "..." if len(content.content) > 500 else content.content
        print(preview)
        
        # Validações adicionais
        assert len(content.content) >= template.seo_requirements["min_words"], \
            "Conteúdo deve atender requisito mínimo de palavras"
        
        assert metrics.quality_score >= feedback_loop.min_quality_threshold or \
               metrics.iteration_count >= feedback_loop.max_iterations, \
            "Deve atingir qualidade mínima ou número máximo de iterações"
        
        print("\n✓ Todas as validações passaram com sucesso!")
        return content, metrics
        
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {str(e)}")
        raise

async def test_quality_improvement():
    """Testa se o feedback loop realmente melhora a qualidade"""
    print("\n=== Testando Melhoria de Qualidade ===")
    
    try:
        # Executa teste principal
        content, metrics = await test_feedback_loop()
        
        # Analisa histórico de qualidade
        if len(metrics.history) > 1:
            initial_quality = metrics.history[0].content_metrics['quality']
            final_quality = metrics.quality_score
            
            print(f"\nQualidade inicial: {initial_quality:.2f}")
            print(f"Qualidade final: {final_quality:.2f}")
            print(f"Melhoria: {(final_quality - initial_quality):.2f}")
            
            assert final_quality > initial_quality, \
                "Qualidade final deve ser maior que a inicial"
            
            print("\n✓ Teste de melhoria de qualidade passou!")
            
    except Exception as e:
        print(f"\n❌ Erro no teste de melhoria: {str(e)}")
        raise

if __name__ == "__main__":
    print("\n=== Iniciando Suite de Testes do Feedback Loop ===")
    
    try:
        # Executa os testes
        asyncio.run(test_feedback_loop())
        asyncio.run(test_quality_improvement())
        
        print("\n=== Todos os testes completados com sucesso! ===")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {str(e)}")
        raise
```

## build\lib\tests\test_news_integration.py

```python
# tests/test_news_integration.py

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
from agents.search.services.news.service import NewsIntegrationService
from agents.search.services.news.models import NewsSearchQuery
from agents.search.services.news.clients.devto import DevToArticle

pytestmark = pytest.mark.asyncio

@pytest.fixture
async def news_service():
    """Fixture para o serviço de notícias"""
    service = NewsIntegrationService()
    await service.initialize()
    try:
        yield service
    finally:
        await service.close()

@pytest.fixture
def mock_devto_response():
    """Mock para resposta do Dev.to"""
    return [
        {
            "id": 1,
            "title": "Understanding Python Async",
            "description": "A guide to async/await in Python",
            "url": "https://dev.to/test/python-async",
            "published_at": datetime.now().isoformat(),
            "tag_list": ["python", "programming"],
            "user": {
                "name": "Test Author",
                "username": "testuser"
            },
            "reading_time_minutes": 5,
            "comments_count": 10,
            "public_reactions_count": 20
        },
        {
            "id": 2,
            "title": "JavaScript Best Practices",
            "description": "Writing better JavaScript code",
            "url": "https://dev.to/test/javascript",
            "published_at": (datetime.now() - timedelta(days=2)).isoformat(),
            "tag_list": ["javascript", "webdev"],
            "user": {
                "name": "Another Author",
                "username": "jsdev"
            },
            "reading_time_minutes": 8,
            "comments_count": 15,
            "public_reactions_count": 30
        }
    ]

async def test_search_integration(news_service, mock_devto_response):
    """Testa integração completa da busca"""
    # Criar query
    query = NewsSearchQuery(
        topic="python",
        keywords=["programming"],
        start_date=datetime.now() - timedelta(days=7),
        min_relevance=0.3,
        max_results=10
    )
    
    # Mock do cliente Dev.to
    with patch.object(news_service.devto_client, 'get_articles', return_value=[
        DevToArticle.parse_obj(article) for article in mock_devto_response
    ]):
        results = await news_service.search_news(query)
        
        assert len(results) > 0
        assert any("Python" in article.title for article in results)
        assert all(article.source == "dev.to" for article in results)

async def test_cache_integration(news_service, mock_devto_response):
    """Testa integração com cache"""
    query = NewsSearchQuery(topic="python")
    
    # Mock do cliente Dev.to
    with patch.object(news_service.devto_client, 'get_articles', return_value=[
        DevToArticle.parse_obj(mock_devto_response[0])
    ]) as mock_get:
        # Primeira chamada
        results1 = await news_service.search_news(query)
        assert len(results1) > 0
        assert mock_get.called
        
        # Segunda chamada (deve vir do cache)
        mock_get.reset_mock()
        results2 = await news_service.search_news(query)
        assert len(results2) > 0
        assert not mock_get.called
        
        # Verificar se resultados são iguais
        assert results1 == results2

async def test_error_handling(news_service):
    """Testa tratamento de erros na integração"""
    query = NewsSearchQuery(topic="python")
    
    # Simular erro no cliente
    with patch.object(news_service.devto_client, 'get_articles', side_effect=Exception("API Error")):
        results = await news_service.search_news(query)
        assert len(results) == 0

async def test_relevance_filtering(news_service, mock_devto_response):
    """Testa filtro de relevância"""
    query = NewsSearchQuery(
        topic="python",
        min_relevance=0.7  # Relevância alta
    )
    
    # Mock do cliente Dev.to
    with patch.object(news_service.devto_client, 'get_articles', return_value=[
        DevToArticle.parse_obj(article) for article in mock_devto_response
    ]):
        results = await news_service.search_news(query)
        
        # Apenas artigos muito relevantes devem passar
        assert all(
            "Python" in article.title or 
            "python" in article.tags 
            for article in results
        )
```

## build\lib\tests\test_orchestrator.py

```python
# tests/test_orchestrator.py

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Ajusta o path para importações relativas
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from core.orchestrator import Orchestrator, ContentRequest
from agents.search.agent import SearchResult

async def test_content_generation():
    """Testa geração completa de conteúdo"""
    orchestrator = Orchestrator()
    
    # Testa conversão de resultados de busca
    test_search_results = [
        SearchResult(
            content="Test content",
            source="test_source",
            relevance_score=0.8,
            metadata={"key": "value"},
            embedding=None,
            timestamp=datetime.now()
        )
    ]
    
    # Verifica se a conversão está funcionando
    converted_refs = orchestrator._convert_search_results_to_references(test_search_results)
    assert isinstance(converted_refs, list), "Deve retornar uma lista"
    assert len(converted_refs) > 0, "Lista não deve estar vazia"
    assert "title" in converted_refs[0], "Referência deve ter título"
    assert "content" in converted_refs[0], "Referência deve ter conteúdo"
    
    # Teste principal
    request = ContentRequest(
        topic="Implementando AI Agents para Automação de Processos",
        domain="ai_agents",
        content_type="technical_guide", 
        target_audience="desenvolvedores",
        technical_level="advanced",
        keywords=["AI agents", "automação", "processos", "MLOps"],
        references_required=True,
        code_examples=True
    )
    
    print("\n=== Iniciando Teste do Orchestrator ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        result = await orchestrator.process_request(request)
        
        # Validações do resultado
        assert result is not None, "Resultado não pode ser None"
        assert "content" in result, "Resultado deve conter 'content'"
        assert "metadata" in result, "Resultado deve conter 'metadata'"
        assert "suggestions" in result, "Resultado deve conter 'suggestions'"
        assert isinstance(result["content"], str), "Content deve ser string"
        assert len(result["content"]) > 0, "Content não deve estar vazio"
        
        print("\n" + "="*50)
        print("PLANO DE CONTEÚDO:")
        print("="*50)
        print(f"Topic: {result['metadata']['plan']['topic']}")
        print(f"Target Audience: {result['metadata']['plan']['target_audience']}")
        print(f"Priority: {result['metadata']['plan']['priority']}")
        
        print("\n" + "="*50)
        print("MÉTRICAS DE QUALIDADE:")
        print("="*50)
        print(f"Approved: {result['metadata']['approved']}")
        for metric, score in result['metadata']['quality_scores'].items():
            print(f"{metric}: {score:.2f}")
            
        print("\n" + "="*50)
        print("SUGESTÕES DE MELHORIA:")
        print("="*50)
        for suggestion in result['suggestions']:
            print(f"- {suggestion}")
            
        print("\n" + "="*50)
        print("PRÉVIA DO CONTEÚDO:")
        print("="*50)
        preview = result['content'][:500] + "..." if len(result['content']) > 500 else result['content']
        print(preview)
        
        return result
        
    except Exception as e:
        print(f"Erro no teste: {str(e)}")
        raise

async def test_metrics_analysis():
    """Testa análise de métricas"""
    orchestrator = Orchestrator()
    
    request = ContentRequest(
        topic="Guia de Implementação de AI",
        domain="ai_agents",
        content_type="guide",
        target_audience="desenvolvedores",
        technical_level="intermediate",
        keywords=["AI", "implementação", "guia"]
    )
    
    try:
        metrics = await orchestrator.analyze_metrics(request)
        assert metrics is not None, "Métricas não podem ser None"
        assert "metrics" in metrics, "Resultado deve conter 'metrics'"
        assert "timestamp" in metrics, "Resultado deve conter 'timestamp'"
        assert isinstance(metrics["metrics"]["estimated_impact"], float), "Impact deve ser float"
        assert isinstance(metrics["metrics"]["priority"], int), "Priority deve ser int"
        
        print("\n" + "="*50)
        print("ANÁLISE DE MÉTRICAS:")
        print("="*50)
        print(f"Estimated Impact: {metrics['metrics']['estimated_impact']:.2f}")
        print(f"Priority: {metrics['metrics']['priority']}")
        return metrics
        
    except Exception as e:
        print(f"Erro na análise de métricas: {str(e)}")
        raise

async def run_all_tests():
    """Executa todos os testes"""
    print("\n=== Executando Suite de Testes do Orchestrator ===")
    
    try:
        # Teste de geração de conteúdo
        content_result = await test_content_generation()
        print("\n✓ Teste de geração completado")
        
        # Teste de métricas
        metrics_result = await test_metrics_analysis()
        print("\n✓ Teste de métricas completado")
        
        print("\n=== Todos os testes completados com sucesso ===")
        
        return {
            "content_result": content_result,
            "metrics_result": metrics_result
        }
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(run_all_tests())
```

## build\lib\tests\test_planning.py

```python
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
```

## build\lib\tests\test_review.py

```python
# tests/test_review.py

import asyncio
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from agents.review.agent import ReviewAgent

async def test_review():
   """Testa o Review Agent"""
   
   # Inicializa agente
   agent = ReviewAgent()
   
   # Conteúdo de teste
   test_content = """
   # Implementando AI Agents em Produção
   
   Este guia aborda os principais aspectos da implementação de agentes de IA
   em ambientes produtivos. Vamos explorar as melhores práticas e desafios comuns.
   
   ## Principais Tópicos
   
   1. Arquitetura de Agentes
   2. Integração com LLMs
   3. Monitoramento e Observabilidade
   4. Tratamento de Erros
   """
   
   # Contexto do conteúdo
   context = {
       "domain": "ai_agents",
       "target_audience": "technical_leaders",
       "technical_level": "advanced",
       "content_type": "technical_guide",
       "required_sections": [
           "architecture",
           "implementation",
           "best_practices",
           "monitoring"
       ]
   }
   
   try:
       # Executa revisão
       result = await agent.review_content(
           content=test_content,
           context=context
       )
       
       # Imprime resultados
       print("\n=== Resultado da Revisão ===")
       print(f"Qualidade: {result.quality_score:.2f}")
       print(f"SEO: {result.seo_score:.2f}")
       print(f"Precisão Técnica: {result.technical_accuracy:.2f}")
       print(f"Legibilidade: {result.readability_score:.2f}")
       print(f"Aprovado: {result.approved}")
       
       print("\nSugestões:")
       for suggestion in result.suggestions:
           print(f"- {suggestion}")
       
       print("\nConteúdo Revisado:")
       print(result.content)
       
   except Exception as e:
       print(f"Erro no teste: {str(e)}")
       raise

if __name__ == "__main__":
   asyncio.run(test_review())
```

## build\lib\tests\test_search.py

```python
# tests\test_search.py
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Adiciona o diretório raiz ao path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from agents.search.agent import SearchAgent, SearchQuery
from agents.search.config import SearchAgentConfig

async def test_search():
    # Inicializa o agente
    agent = SearchAgent()
    
    # Cria uma query de teste
    query = SearchQuery(
        query="Implementação de MLOps em produção",
        context="Buscando informações sobre deploy de modelos ML",
        filters={
            "language": "pt-br",
            "max_age_days": 365
        }
    )
    
    print("\n=== Teste do Search Agent ===")
    print(f"\nQuery: {query.query}")
    print(f"Contexto: {query.context}")
    
    try:
        # Realiza a busca
        results = await agent.search(query)
        
        print(f"\nEncontrados {len(results)} resultados:")
        for i, result in enumerate(results, 1):
            print(f"\n--- Resultado {i} ---")
            print(f"Fonte: {result.source}")
            print(f"Relevância: {result.relevance_score}")
            print(f"Timestamp: {result.timestamp}")
            print(f"Conteúdo: {result.content[:200]}...")
            if result.metadata:
                print(f"Metadata: {result.metadata}")
    
    except Exception as e:
        print(f"\nErro durante a busca: {str(e)}")
        raise

async def test_indexing():
    # Inicializa o agente
    agent = SearchAgent()
    
    # Conteúdo de teste
    test_content = """
    MLOps (Machine Learning Operations) é uma prática que visa automatizar
    e otimizar o ciclo de vida completo de modelos de machine learning em produção.
    """
    
    metadata = {
        "source": "test",
        "author": "AD Team",
        "timestamp": datetime.now().isoformat(),
        "category": "MLOps"
    }
    
    print("\n=== Teste de Indexação ===")
    
    try:
        # Indexa o conteúdo
        await agent.index_content(test_content, metadata)
        print("\nConteúdo indexado com sucesso!")
        
        # Testa a busca do conteúdo indexado
        query = SearchQuery(
            query="MLOps automatização",
            context="Machine Learning Operations"
        )
        
        results = await agent.search(query)
        print(f"\nBusca após indexação: {len(results)} resultados")
        
    except Exception as e:
        print(f"\nErro durante a indexação: {str(e)}")
        raise

if __name__ == "__main__":
    # Executa os testes
    asyncio.run(test_search())
    asyncio.run(test_indexing())
```

## build\lib\tests\__init__.py

```python

```

## core\config.py

```python
from typing import Dict, Optional
from pydantic import BaseSettings, Field
from functools import lru_cache
import os
from enum import Enum

class Environment(str, Enum):
    """Ambientes de execução"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Níveis de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class APIConfig(BaseSettings):
    """Configurações de APIs externas"""
    # OpenAI
    openai_api_key: str = Field(..., env='OPENAI_API_KEY')
    openai_model: str = Field("gpt-4-1106-preview", env='OPENAI_MODEL')
    openai_temperature: float = Field(0.7, env='OPENAI_TEMPERATURE')
    
    # Pinecone
    pinecone_api_key: str = Field(..., env='PINECONE_API_KEY')
    pinecone_environment: str = Field('gcp-starter', env='PINECONE_ENVIRONMENT')
    pinecone_index_name: str = Field('adagents', env='PINECONE_INDEX_NAME')
    
    # HuggingFace
    hf_api_key: Optional[str] = Field(None, env='HF_API_KEY')
    hf_model: str = Field('sentence-transformers/all-mpnet-base-v2', env='HF_MODEL')

class AgentConfig(BaseSettings):
    """Configurações dos agentes"""
    # Planning Agent
    planning_cache_ttl: int = Field(3600, env='PLANNING_CACHE_TTL')
    planning_max_retries: int = Field(3, env='PLANNING_MAX_RETRIES')
    
    # Search Agent
    search_cache_ttl: int = Field(3600, env='SEARCH_CACHE_TTL')
    search_max_results: int = Field(10, env='SEARCH_MAX_RESULTS')
    search_min_relevance: float = Field(0.7, env='SEARCH_MIN_RELEVANCE')
    
    # Content Agent
    content_max_length: int = Field(4000, env='CONTENT_MAX_LENGTH')
    content_min_length: int = Field(500, env='CONTENT_MIN_LENGTH')
    content_quality_threshold: float = Field(0.8, env='CONTENT_QUALITY_THRESHOLD')
    
    # Review Agent
    review_auto_approve_threshold: float = Field(0.9, env='REVIEW_AUTO_APPROVE_THRESHOLD')
    review_min_quality_score: float = Field(0.7, env='REVIEW_MIN_QUALITY_SCORE')

class DatabaseConfig(BaseSettings):
    """Configurações de banco de dados"""
    db_host: str = Field('localhost', env='DB_HOST')
    db_port: int = Field(5432, env='DB_PORT')
    db_name: str = Field('adagents', env='DB_NAME')
    db_user: str = Field('postgres', env='DB_USER')
    db_password: Optional[str] = Field(None, env='DB_PASSWORD')
    
    @property
    def database_url(self) -> str:
        """Retorna URL de conexão com o banco"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

class CacheConfig(BaseSettings):
    """Configurações de cache"""
    redis_host: str = Field('localhost', env='REDIS_HOST')
    redis_port: int = Field(6379, env='REDIS_PORT')
    redis_db: int = Field(0, env='REDIS_DB')
    redis_password: Optional[str] = Field(None, env='REDIS_PASSWORD')
    default_ttl: int = Field(3600, env='CACHE_DEFAULT_TTL')

class RateLimitConfig(BaseSettings):
    """Configurações de rate limiting"""
    max_requests_per_minute: int = Field(60, env='RATE_LIMIT_RPM')
    max_tokens_per_minute: int = Field(10000, env='RATE_LIMIT_TOKENS')
    max_parallel_requests: int = Field(5, env='RATE_LIMIT_PARALLEL')

class GlobalConfig(BaseSettings):
    """Configurações globais do sistema"""
    environment: Environment = Field(Environment.DEVELOPMENT, env='ENVIRONMENT')
    debug: bool = Field(True, env='DEBUG')
    log_level: LogLevel = Field(LogLevel.INFO, env='LOG_LEVEL')
    
    # Sub-configurações
    api: APIConfig = APIConfig()
    agents: AgentConfig = AgentConfig()
    database: DatabaseConfig = DatabaseConfig()
    cache: CacheConfig = CacheConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    
    # Timeouts e limites
    request_timeout: int = Field(30, env='REQUEST_TIMEOUT')
    max_retries: int = Field(3, env='MAX_RETRIES')
    batch_size: int = Field(100, env='BATCH_SIZE')
    
    # Paths
    base_path: str = Field(os.path.dirname(os.path.dirname(__file__)))
    data_path: str = Field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
    log_path: str = Field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'))
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> GlobalConfig:
    """Retorna instância cacheada das configurações"""
    return GlobalConfig()

def initialize_directories() -> None:
    """Inicializa diretórios necessários"""
    config = get_settings()
    os.makedirs(config.data_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)

def get_environment() -> Environment:
    """Retorna ambiente atual"""
    return get_settings().environment

def is_development() -> bool:
    """Verifica se está em ambiente de desenvolvimento"""
    return get_environment() == Environment.DEVELOPMENT

def is_production() -> bool:
    """Verifica se está em ambiente de produção"""
    return get_environment() == Environment.PRODUCTION

if __name__ == "__main__":
    # Exemplo de uso
    settings = get_settings()
    
    print(f"Environment: {settings.environment}")
    print(f"Debug Mode: {settings.debug}")
    print(f"Log Level: {settings.log_level}")
    
    print("\nAPI Settings:")
    print(f"OpenAI Model: {settings.api.openai_model}")
    print(f"Pinecone Environment: {settings.api.pinecone_environment}")
    
    print("\nAgent Settings:")
    print(f"Search Cache TTL: {settings.agents.search_cache_ttl}")
    print(f"Content Quality Threshold: {settings.agents.content_quality_threshold}")
    
    print("\nRate Limits:")
    print(f"Max RPM: {settings.rate_limit.max_requests_per_minute}")
    print(f"Max Parallel: {settings.rate_limit.max_parallel_requests}")
    
    # Inicializa diretórios
    initialize_directories()
    print(f"\nData Path: {settings.data_path}")
    print(f"Log Path: {settings.log_path}")
```

## core\feedback_loop.py

```python
from core.rabbitmq_utils import RabbitMQUtils
import json

class FeedbackLoop:
    def __init__(self, domain: str, max_iterations: int = 5):
        self.rabbitmq = RabbitMQUtils()
        self.domain = domain
        self.max_iterations = max_iterations
        self.iteration = 0

    def publish_content(self):
        """Publica conteúdo gerado na fila."""
        content = {"content_id": 1, "domain": self.domain, "content": "Texto inicial gerado"}
        print(f"Publicando conteúdo: {content}")
        self.rabbitmq.publish_event("content.generated", content)

    def process_reviewed_event(self, ch, method, properties, body):
        """Processa mensagens da fila 'content.reviewed'."""
        message = json.loads(body)
        print(f"Revisão recebida: {message}")

        quality_score = message.get("quality_score", 0)
        if quality_score >= 0.8:
            print(f"Conteúdo aprovado na iteração {self.iteration}")
        else:
            print(f"Qualidade insuficiente ({quality_score}). Melhorando conteúdo...")
            improvement = {
                "content_id": message["content_id"],
                "action": "improve",
                "suggestions": message.get("suggestions", [])
            }
            self.rabbitmq.publish_event("content.improved", improvement)

    def start(self):
        """Inicia o Feedback Loop."""
        print("Iniciando o Feedback Loop...")
        self.publish_content()
        self.rabbitmq.consume_event("content.reviewed", self.process_reviewed_event)

```

## core\models.py

```python
from typing import List, Dict, Union
from pydantic import BaseModel, Field
from datetime import datetime

# Modelo para mensagens consumidas pela fila planning.generated
class PlanningGenerated(BaseModel):
    topic: str
    keywords: List[str]
    target_audience: str
    content_type: str
    priority: int
    estimated_impact: float

# Modelo para mensagens publicadas na fila content.generated
class ContentGenerated(BaseModel):
    content: str
    title: str
    meta_description: str
    keywords: List[str]
    seo_score: float
    readability_score: float
    created_at: datetime = Field(default_factory=datetime.now)

# Modelo para mensagens consumidas pela fila content.improved
class ContentImproved(BaseModel):
    content_id: str
    content: str
    suggestions: List[str]
```

## core\orchestrator.py

```python
from typing import Dict, List
from datetime import datetime
import asyncio
from pydantic import BaseModel, Field
from core.rabbitmq_utils import RabbitMQUtils

class ContentRequest(BaseModel):
    """Modelo para solicitação de conteúdo"""
    topic: str
    domain: str
    content_type: str
    target_audience: str
    technical_level: str = "intermediate"
    keywords: List[str] = []
    references_required: bool = True
    code_examples: bool = True

class Orchestrator:
    """
    Orquestrador para funcionalidades específicas.

    Cenários onde o Orchestrator deve ser usado:
    1. Consolidação de Dados:
       - Combina resultados de diferentes agentes para criar respostas finais.
    2. Fluxos Híbridos:
       - Processa fluxos que envolvem lógica adicional ou validações antes de completar uma tarefa.
    3. Monitoramento:
       - Realiza debugging ou logging centralizado para rastrear fluxos.

    Importante:
    - Para fluxos altamente escaláveis ou independentes, use o RabbitMQ para comunicação direta entre agentes.
    """

    def __init__(self):
        self.rabbitmq = RabbitMQUtils()

    def publish_request(self, request: ContentRequest):
        """Publica solicitação inicial na fila `planning.requests`"""
        try:
            self.rabbitmq.publish_event("planning.requests", request.dict())
            print(f"Solicitação publicada na fila `planning.requests`: {request.dict()}")
        except Exception as e:
            print(f"Erro ao publicar solicitação: {str(e)}")

    def consume_and_consolidate(self):
        """
        Consome mensagens de várias filas e consolida os resultados.
        Usado para criar relatórios finais ou responder a solicitações completas.
        """
        def callback(ch, method, properties, body):
            import json
            response = json.loads(body)
            print(f"Resposta recebida: {response}")

            # Exemplo de lógica de consolidação (adapte conforme necessário)
            if "content" in response:
                print(f"Conteúdo recebido: {response['content'][:200]}...")
            elif "plan" in response:
                print(f"Plano estratégico recebido: {response['plan']}")

        # Consumir respostas de filas relevantes
        self.rabbitmq.consume_event("search.results", callback)
        self.rabbitmq.consume_event("content.reviewed", callback)

    async def process_request_with_control(self, request: ContentRequest):
        """
        Executa a solicitação publicando na fila inicial e consolidando respostas.
        Usado para monitorar e consolidar fluxos em tempo real.
        """
        try:
            # Publica a solicitação inicial
            self.publish_request(request)

            # Aguarda e processa respostas
            print("Aguardando respostas das filas...")
            self.consume_and_consolidate()

        except Exception as e:
            print(f"Erro no processamento da solicitação: {str(e)}")

if __name__ == "__main__":
    async def test_orchestrator():
        orchestrator = Orchestrator()

        # Simula uma solicitação de conteúdo
        request = ContentRequest(
            topic="IA Generativa em Produção",
            domain="machine_learning",
            content_type="technical_guide",
            target_audience="desenvolvedores",
            technical_level="advanced",
            keywords=["IA", "Machine Learning", "Produção"]
        )

        # Processa a solicitação
        await orchestrator.process_request_with_control(request)

    asyncio.run(test_orchestrator())

```

## core\rabbitmq_utils.py

```python
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

```

## core\testa api.py

```python
from core.config import get_settings
from agents.planning.config import config

# Validação da chave da API
settings = get_settings()
if not settings.api.openai_api_key:
    raise ValueError("A chave da API OpenAI (openai_api_key) não foi configurada corretamente.")
print(f"Chave da API OpenAI encontrada: {settings.api.openai_api_key}")



```

## core\__init__.py

```python

```

## core\domains\adoption_metrics.py

```python
# core\domains\adoption_metrics.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, date
from enum import Enum

class AdoptionStage(str, Enum):
    """Estágios de adoção de tecnologia"""
    INNOVATORS = "innovators"         # 2.5% iniciais
    EARLY_ADOPTERS = "early_adopters" # 13.5% seguintes
    EARLY_MAJORITY = "early_majority" # 34% primeiros da maioria
    LATE_MAJORITY = "late_majority"   # 34% últimos da maioria
    LAGGARDS = "laggards"             # 16% finais

class AdoptionMetric(BaseModel):
    """Métricas de adoção de tecnologia"""
    technology: str
    category: str
    current_stage: AdoptionStage
    adoption_rate: float = Field(ge=0.0, le=1.0)
    growth_rate: float  # % de crescimento mensal
    market_penetration: float = Field(ge=0.0, le=1.0)
    enterprise_adoption: float = Field(ge=0.0, le=1.0)
    startup_adoption: float = Field(ge=0.0, le=1.0)
    developer_satisfaction: float = Field(ge=0.0, le=1.0)
    time_to_value: int  # dias
    measured_at: datetime = Field(default_factory=datetime.now)

class ROIMetric(BaseModel):
    """Métricas de ROI"""
    cost_reduction: float
    productivity_gain: float
    time_saved: float  # horas por semana
    quality_improvement: float
    user_satisfaction: float = Field(ge=0.0, le=1.0)
    implementation_cost: float
    maintenance_cost: float
    training_time: int  # horas

class UseCaseMetric(BaseModel):
    """Métricas específicas de casos de uso"""
    use_case: str
    success_rate: float = Field(ge=0.0, le=1.0)
    implementation_time: int  # dias
    roi_metrics: ROIMetric
    adoption_barriers: List[str]
    best_practices: List[str]
    common_pitfalls: List[str]

# Métricas de adoção atuais
CURRENT_ADOPTION_METRICS = {
    "llms": AdoptionMetric(
        technology="Large Language Models",
        category="ai",
        current_stage=AdoptionStage.EARLY_MAJORITY,
        adoption_rate=0.34,
        growth_rate=15.0,  # 15% ao mês
        market_penetration=0.45,
        enterprise_adoption=0.38,
        startup_adoption=0.65,
        developer_satisfaction=0.85,
        time_to_value=14,  # 2 semanas
    ),
    
    "vector_databases": AdoptionMetric(
        technology="Vector Databases",
        category="data",
        current_stage=AdoptionStage.EARLY_ADOPTERS,
        adoption_rate=0.15,
        growth_rate=25.0,
        market_penetration=0.20,
        enterprise_adoption=0.15,
        startup_adoption=0.40,
        developer_satisfaction=0.80,
        time_to_value=7
    ),
    
    "ai_agents": AdoptionMetric(
        technology="AI Agents",
        category="ai",
        current_stage=AdoptionStage.INNOVATORS,
        adoption_rate=0.05,
        growth_rate=45.0,
        market_penetration=0.08,
        enterprise_adoption=0.03,
        startup_adoption=0.12,
        developer_satisfaction=0.75,
        time_to_value=21
    )
}

# Métricas de casos de uso
USE_CASE_METRICS = {
    "customer_service": UseCaseMetric(
        use_case="AI Customer Service",
        success_rate=0.82,
        implementation_time=30,
        roi_metrics=ROIMetric(
            cost_reduction=0.35,  # 35% redução de custos
            productivity_gain=0.45,
            time_saved=15.0,  # 15 horas por semana
            quality_improvement=0.25,
            user_satisfaction=0.78,
            implementation_cost=50000,  # USD
            maintenance_cost=2000,  # USD por mês
            training_time=20  # horas
        ),
        adoption_barriers=[
            "Integration complexity",
            "Data privacy concerns",
            "Staff training",
            "Cost justification"
        ],
        best_practices=[
            "Start with simple use cases",
            "Comprehensive training",
            "Clear escalation paths",
            "Regular performance monitoring"
        ],
        common_pitfalls=[
            "Over-automation",
            "Insufficient testing",
            "Poor integration",
            "Lack of monitoring"
        ]
    ),
    
    "code_assistance": UseCaseMetric(
        use_case="AI Code Assistant",
        success_rate=0.88,
        implementation_time=7,
        roi_metrics=ROIMetric(
            cost_reduction=0.20,
            productivity_gain=0.40,
            time_saved=10.0,
            quality_improvement=0.30,
            user_satisfaction=0.85,
            implementation_cost=15000,
            maintenance_cost=500,
            training_time=8
        ),
        adoption_barriers=[
            "Security concerns",
            "Code quality trust",
            "Learning curve",
            "Integration with existing tools"
        ],
        best_practices=[
            "Code review protocols",
            "Security guidelines",
            "Training programs",
            "Measurement frameworks"
        ],
        common_pitfalls=[
            "Blind trust in suggestions",
            "Inadequate review",
            "Security oversights",
            "Poor prompt practices"
        ]
    )
}

class AdoptionMetricsManager:
    """Gerenciador de métricas de adoção"""
    
    def __init__(self):
        self.metrics = CURRENT_ADOPTION_METRICS
        self.use_cases = USE_CASE_METRICS
    
    def get_metric(self, technology: str) -> Optional[AdoptionMetric]:
        """Retorna métricas para uma tecnologia"""
        return self.metrics.get(technology.lower())
    
    def get_use_case_metrics(self, use_case: str) -> Optional[UseCaseMetric]:
        """Retorna métricas para um caso de uso"""
        return self.use_cases.get(use_case.lower())
    
    def get_trending_technologies(self) -> List[str]:
        """Retorna tecnologias com maior crescimento"""
        return sorted(
            self.metrics.keys(),
            key=lambda x: self.metrics[x].growth_rate,
            reverse=True
        )
    
    def calculate_roi_projection(self, use_case: str, timeframe_months: int) -> Dict:
        """Calcula projeção de ROI para um caso de uso"""
        metrics = self.get_use_case_metrics(use_case)
        if not metrics:
            return {}
            
        roi = metrics.roi_metrics
        monthly_savings = (
            roi.cost_reduction * roi.implementation_cost +
            (roi.productivity_gain * 160 * 50)  # 160h/mês * $50/h
        )
        
        total_cost = (
            roi.implementation_cost +
            (roi.maintenance_cost * timeframe_months)
        )
        
        total_savings = monthly_savings * timeframe_months
        net_roi = (total_savings - total_cost) / total_cost
        
        return {
            "timeframe_months": timeframe_months,
            "total_cost": total_cost,
            "total_savings": total_savings,
            "net_roi": net_roi,
            "break_even_months": roi.implementation_cost / monthly_savings
        }
    
    def get_adoption_recommendations(self, technology: str) -> Dict:
        """Gera recomendações de adoção para uma tecnologia"""
        metric = self.get_metric(technology)
        if not metric:
            return {}
            
        recommendations = {
            "readiness_score": self._calculate_readiness_score(metric),
            "suggested_approach": self._get_suggested_approach(metric),
            "risk_level": self._calculate_risk_level(metric),
            "next_steps": self._get_next_steps(metric)
        }
        
        return recommendations
    
    def _calculate_readiness_score(self, metric: AdoptionMetric) -> float:
        """Calcula score de prontidão para adoção"""
        weights = {
            "market_penetration": 0.3,
            "developer_satisfaction": 0.3,
            "enterprise_adoption": 0.2,
            "time_to_value": 0.2
        }
        
        # Normaliza time_to_value (menor é melhor)
        time_score = 1 - (metric.time_to_value / 90)  # normaliza para 90 dias
        
        score = (
            metric.market_penetration * weights["market_penetration"] +
            metric.developer_satisfaction * weights["developer_satisfaction"] +
            metric.enterprise_adoption * weights["enterprise_adoption"] +
            time_score * weights["time_to_value"]
        )
        
        return min(max(score, 0.0), 1.0)
    
    def _get_suggested_approach(self, metric: AdoptionMetric) -> str:
        """Define abordagem sugerida baseada no estágio de adoção"""
        approaches = {
            AdoptionStage.INNOVATORS: "Experimental pilot with close monitoring",
            AdoptionStage.EARLY_ADOPTERS: "Limited production deployment",
            AdoptionStage.EARLY_MAJORITY: "Full production rollout",
            AdoptionStage.LATE_MAJORITY: "Standardized implementation",
            AdoptionStage.LAGGARDS: "Conservative adoption with proven patterns"
        }
        return approaches[metric.current_stage]
    
    def _calculate_risk_level(self, metric: AdoptionMetric) -> str:
        """Calcula nível de risco baseado nas métricas"""
        risk_score = (
            (1 - metric.market_penetration) * 0.4 +
            (1 - metric.enterprise_adoption) * 0.3 +
            (1 - metric.developer_satisfaction) * 0.3
        )
        
        if risk_score > 0.7:
            return "HIGH"
        elif risk_score > 0.4:
            return "MEDIUM"
        return "LOW"
    
    def _get_next_steps(self, metric: AdoptionMetric) -> List[str]:
        """Define próximos passos baseado no estágio atual"""
        if metric.current_stage == AdoptionStage.INNOVATORS:
            return [
                "Start with proof of concept",
                "Focus on learning and exploration",
                "Build internal expertise",
                "Prepare risk mitigation strategies"
            ]
        elif metric.current_stage == AdoptionStage.EARLY_ADOPTERS:
            return [
                "Plan limited production deployment",
                "Document best practices",
                "Establish monitoring",
                "Create training programs"
            ]
        else:
            return [
                "Scale deployment",
                "Optimize processes",
                "Share knowledge",
                "Measure ROI"
            ]

if __name__ == "__main__":
    # Exemplo de uso
    manager = AdoptionMetricsManager()
    
    # Análise de LLMs
    llm_metrics = manager.get_metric("llms")
    if llm_metrics:
        print(f"LLM Adoption Rate: {llm_metrics.adoption_rate * 100}%")
        print(f"Growth Rate: {llm_metrics.growth_rate}% monthly")
    
    # ROI de caso de uso
    roi_projection = manager.calculate_roi_projection("customer_service", 12)
    print(f"\nCustomer Service ROI (12 months):")
    print(f"Net ROI: {roi_projection['net_roi']:.2f}")
    print(f"Break Even: {roi_projection['break_even_months']:.1f} months")
    
    # Recomendações
    recommendations = manager.get_adoption_recommendations("ai_agents")
    print(f"\nAI Agents Recommendations:")
    print(f"Readiness Score: {recommendations['readiness_score']:.2f}")
    print(f"Risk Level: {recommendations['risk_level']}")
    print(f"Approach: {recommendations['suggested_approach']}")
```

## core\domains\bigtech_monitor.py

```python
# core\domains\bigtech_monitor.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, date
from enum import Enum

class CompanyCategory(str, Enum):
    """Categorias de empresas de tecnologia"""
    AI_SPECIALIST = "ai_specialist"     # Ex: OpenAI, Anthropic
    CLOUD_PROVIDER = "cloud_provider"   # Ex: AWS, GCP
    ENTERPRISE = "enterprise"           # Ex: Microsoft, Oracle
    SOCIAL_MEDIA = "social_media"       # Ex: Meta
    ECOMMERCE = "ecommerce"            # Ex: Amazon, Mercado Livre
    FINTECH = "fintech"                # Ex: Stripe, Square

class TechnologyStatus(str, Enum):
    """Status de uma tecnologia"""
    RESEARCH = "research"          # Em pesquisa
    PREVIEW = "preview"           # Preview/Beta
    GA = "ga"                     # Generally Available
    DEPRECATED = "deprecated"     # Descontinuada

class Innovation(BaseModel):
    """Modelo para inovações tecnológicas"""
    name: str
    company: str
    announcement_date: date
    release_date: Optional[date]
    status: TechnologyStatus
    description: str
    key_features: List[str]
    target_users: List[str]
    competitors: List[str]
    pricing_model: Optional[str]
    documentation_url: Optional[str]
    impact_score: float = Field(ge=0.0, le=1.0)

class Product(BaseModel):
    """Modelo para produtos/serviços"""
    name: str
    category: str
    launch_date: date
    current_version: str
    status: TechnologyStatus
    market_share: float
    key_features: List[str]
    pricing: Dict[str, float]
    enterprise_ready: bool

class BigTechCompany(BaseModel):
    """Modelo para empresas BigTech"""
    name: str
    category: CompanyCategory
    market_cap: float  # em bilhões USD
    key_products: List[Product]
    recent_innovations: List[Innovation]
    strategic_focus: List[str]
    open_source: List[str]
    apis_services: Dict[str, str]
    developer_resources: List[str]
    last_updated: datetime = Field(default_factory=datetime.now)

# Monitoramento das principais BigTechs
BIGTECH_LANDSCAPE = {
    "openai": BigTechCompany(
        name="OpenAI",
        category=CompanyCategory.AI_SPECIALIST,
        market_cap=90.0,  # estimativa
        key_products=[
            Product(
                name="GPT-4",
                category="language_model",
                launch_date=date(2023, 3, 14),
                current_version="GPT-4 Turbo",
                status=TechnologyStatus.GA,
                market_share=0.65,
                key_features=[
                    "128k context window",
                    "Advanced reasoning",
                    "Multimodal capabilities",
                    "JSON mode"
                ],
                pricing={
                    "input": 0.01,
                    "output": 0.03
                },
                enterprise_ready=True
            ),
            Product(
                name="DALL-E 3",
                category="image_generation",
                launch_date=date(2023, 10, 1),
                current_version="3.0",
                status=TechnologyStatus.GA,
                market_share=0.45,
                key_features=[
                    "Photorealistic images",
                    "Text understanding",
                    "Style control"
                ],
                pricing={
                    "standard": 0.02,
                    "hd": 0.04
                },
                enterprise_ready=True
            )
        ],
        recent_innovations=[
            Innovation(
                name="GPT-4 Turbo",
                company="OpenAI",
                announcement_date=date(2023, 11, 6),
                release_date=date(2023, 11, 6),
                status=TechnologyStatus.GA,
                description="Versão melhorada do GPT-4",
                key_features=[
                    "Maior contexto",
                    "Conhecimento mais recente",
                    "Melhor performance"
                ],
                target_users=[
                    "Developers",
                    "Enterprises",
                    "Researchers"
                ],
                competitors=[
                    "Anthropic Claude",
                    "Google PaLM"
                ],
                pricing_model="Pay per token",
                documentation_url="https://platform.openai.com/docs",
                impact_score=0.9
            )
        ],
        strategic_focus=[
            "AGI Development",
            "AI Safety",
            "Enterprise AI"
        ],
        open_source=[
            "Whisper",
            "Gym",
            "CLIP"
        ],
        apis_services={
            "GPT-4": "https://api.openai.com/v1/chat/completions",
            "DALL-E": "https://api.openai.com/v1/images/generations",
            "Embeddings": "https://api.openai.com/v1/embeddings"
        },
        developer_resources=[
            "OpenAI Cookbook",
            "API Documentation",
            "OpenAI Platform"
        ]
    ),

    "anthropic": BigTechCompany(
        name="Anthropic",
        category=CompanyCategory.AI_SPECIALIST,
        market_cap=20.0,  # estimativa
        key_products=[
            Product(
                name="Claude 2.1",
                category="language_model",
                launch_date=date(2023, 11, 21),
                current_version="2.1",
                status=TechnologyStatus.GA,
                market_share=0.15,
                key_features=[
                    "200k context window",
                    "Constitutional AI",
                    "Advanced reasoning",
                    "Tool use"
                ],
                pricing={
                    "input": 0.008,
                    "output": 0.024
                },
                enterprise_ready=True
            )
        ],
        recent_innovations=[],  # Adicionado quando houver
        strategic_focus=[
            "Safe AI Development",
            "Enterprise Solutions",
            "AI Alignment"
        ],
        open_source=[],  # Principalmente closed source
        apis_services={
            "Claude": "https://api.anthropic.com/v1/complete"
        },
        developer_resources=[
            "Claude Documentation",
            "API Guidelines"
        ]
    )
}

class BigTechMonitor:
    """Monitor de BigTechs"""
    
    def __init__(self):
        self.companies = BIGTECH_LANDSCAPE
    
    def get_company(self, company_id: str) -> Optional[BigTechCompany]:
        """Retorna detalhes de uma empresa"""
        return self.companies.get(company_id.lower())
    
    def get_companies_by_category(self, category: CompanyCategory) -> List[BigTechCompany]:
        """Retorna empresas por categoria"""
        return [
            company for company in self.companies.values()
            if company.category == category
        ]
    
    def get_product_details(self, company_id: str, product_name: str) -> Optional[Product]:
        """Retorna detalhes de um produto"""
        company = self.get_company(company_id)
        if not company:
            return None
            
        for product in company.key_products:
            if product.name.lower() == product_name.lower():
                return product
        return None
    
    def get_recent_innovations(self, days: int = 90) -> List[Innovation]:
        """Retorna inovações recentes"""
        recent_innovations = []
        cutoff_date = date.today().replace(day=date.today().day - days)
        
        for company in self.companies.values():
            for innovation in company.recent_innovations:
                if innovation.announcement_date >= cutoff_date:
                    recent_innovations.append(innovation)
        
        return sorted(
            recent_innovations,
            key=lambda x: x.announcement_date,
            reverse=True
        )
    
    def get_competitive_landscape(self, product_category: str) -> Dict:
        """Analisa panorama competitivo de uma categoria"""
        landscape = {
            "products": [],
            "market_share": {},
            "pricing_comparison": {},
            "feature_matrix": {}
        }
        
        for company in self.companies.values():
            for product in company.key_products:
                if product.category == product_category:
                    landscape["products"].append(product.name)
                    landscape["market_share"][product.name] = product.market_share
                    landscape["pricing_comparison"][product.name] = product.pricing
                    landscape["feature_matrix"][product.name] = product.key_features
        
        return landscape
    
    def get_strategic_trends(self) -> Dict[str, List[str]]:
        """Identifica tendências estratégicas"""
        all_focus_areas = []
        for company in self.companies.values():
            all_focus_areas.extend(company.strategic_focus)
        
        # Conta ocorrências
        from collections import Counter
        focus_counter = Counter(all_focus_areas)
        
        # Organiza por relevância
        trends = {
            "high_focus": [],
            "medium_focus": [],
            "emerging": []
        }
        
        for area, count in focus_counter.items():
            if count >= 3:
                trends["high_focus"].append(area)
            elif count == 2:
                trends["medium_focus"].append(area)
            else:
                trends["emerging"].append(area)
                
        return trends

if __name__ == "__main__":
    # Exemplo de uso
    monitor = BigTechMonitor()
    
    # Verificar inovações recentes
    recent = monitor.get_recent_innovations(30)  # últimos 30 dias
    print(f"Inovações recentes: {len(recent)}")
    
    # Análise competitiva de LLMs
    llm_landscape = monitor.get_competitive_landscape("language_model")
    print("\nPanorama LLMs:")
    print(f"Produtos: {llm_landscape['products']}")
    print(f"Market Share: {llm_landscape['market_share']}")
    
    # Tendências estratégicas
    trends = monitor.get_strategic_trends()
    print("\nTendências Principais:", trends["high_focus"])
```

## core\domains\definitions.py

```python
# core\domains\definitions.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class DomainType(str, Enum):
    """Tipos de domínios"""
    TECHNICAL = "technical"         # Domínios técnicos
    BUSINESS = "business"           # Domínios de negócio
    INDUSTRY = "industry"           # Domínios de indústria
    EMERGING = "emerging"           # Domínios emergentes
    RESEARCH = "research"           # Domínios de pesquisa

class ContentType(str, Enum):
    """Tipos de conteúdo"""
    TUTORIAL = "tutorial"           # Tutoriais passo-a-passo
    GUIDE = "guide"                # Guias abrangentes
    REFERENCE = "reference"        # Documentação de referência
    CASE_STUDY = "case_study"      # Estudos de caso
    COMPARISON = "comparison"      # Comparativos
    DEEP_DIVE = "deep_dive"        # Análises profundas
    NEWS = "news"                  # Notícias e atualizações

class ExpertiseLevel(str, Enum):
    """Níveis de expertise"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class UpdateFrequency(str, Enum):
    """Frequência de atualização"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class ValidationRule(BaseModel):
    """Regra de validação de conteúdo"""
    name: str
    description: str
    threshold: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    validation_type: str

class DomainDefinition(BaseModel):
    """Definição completa de um domínio"""
    name: str
    type: DomainType
    description: str
    expertise_levels: List[ExpertiseLevel]
    content_types: List[ContentType]
    update_frequency: UpdateFrequency
    primary_keywords: List[str]
    related_keywords: List[str]
    validation_rules: List[ValidationRule]
    required_citations: bool
    requires_code_examples: bool
    requires_diagrams: bool
    last_updated: datetime = Field(default_factory=datetime.now)

# Definições dos domínios principais
DOMAIN_DEFINITIONS = {
    "ai_agents": DomainDefinition(
        name="AI Agents & Automation",
        type=DomainType.TECHNICAL,
        description="Desenvolvimento e implementação de agentes autônomos de IA",
        expertise_levels=[
            ExpertiseLevel.INTERMEDIATE,
            ExpertiseLevel.ADVANCED,
            ExpertiseLevel.EXPERT
        ],
        content_types=[
            ContentType.TUTORIAL,
            ContentType.GUIDE,
            ContentType.DEEP_DIVE,
            ContentType.CASE_STUDY
        ],
        update_frequency=UpdateFrequency.WEEKLY,
        primary_keywords=[
            "ai agents",
            "autonomous systems",
            "agent orchestration",
            "multi-agent systems"
        ],
        related_keywords=[
            "llms",
            "machine learning",
            "neural networks",
            "decision making"
        ],
        validation_rules=[
            ValidationRule(
                name="code_quality",
                description="Qualidade e clareza do código",
                threshold=0.8,
                weight=0.3,
                validation_type="code_analysis"
            ),
            ValidationRule(
                name="technical_accuracy",
                description="Precisão técnica do conteúdo",
                threshold=0.9,
                weight=0.4,
                validation_type="expert_review"
            ),
            ValidationRule(
                name="practical_applicability",
                description="Aplicabilidade prática",
                threshold=0.7,
                weight=0.3,
                validation_type="user_feedback"
            )
        ],
        required_citations=True,
        requires_code_examples=True,
        requires_diagrams=True
    ),

    "mlops": DomainDefinition(
        name="MLOps & Production AI",
        type=DomainType.TECHNICAL,
        description="Práticas e ferramentas para ML em produção",
        expertise_levels=[
            ExpertiseLevel.INTERMEDIATE,
            ExpertiseLevel.ADVANCED
        ],
        content_types=[
            ContentType.GUIDE,
            ContentType.REFERENCE,
            ContentType.CASE_STUDY
        ],
        update_frequency=UpdateFrequency.WEEKLY,
        primary_keywords=[
            "mlops",
            "machine learning",
            "deployment",
            "monitoring"
        ],
        related_keywords=[
            "devops",
            "continuous integration",
            "model registry",
            "feature store"
        ],
        validation_rules=[
            ValidationRule(
                name="deployment_practicality",
                description="Viabilidade de implementação",
                threshold=0.85,
                weight=0.4,
                validation_type="expert_review"
            ),
            ValidationRule(
                name="best_practices",
                description="Aderência às melhores práticas",
                threshold=0.9,
                weight=0.3,
                validation_type="checklist"
            ),
            ValidationRule(
                name="completeness",
                description="Completude da solução",
                threshold=0.8,
                weight=0.3,
                validation_type="coverage_analysis"
            )
        ],
        required_citations=True,
        requires_code_examples=True,
        requires_diagrams=True
    ),

    "enterprise_ai": DomainDefinition(
        name="Enterprise AI Integration",
        type=DomainType.BUSINESS,
        description="Implementação de IA em ambientes empresariais",
        expertise_levels=[
            ExpertiseLevel.INTERMEDIATE,
            ExpertiseLevel.ADVANCED
        ],
        content_types=[
            ContentType.CASE_STUDY,
            ContentType.GUIDE,
            ContentType.COMPARISON
        ],
        update_frequency=UpdateFrequency.MONTHLY,
        primary_keywords=[
            "enterprise ai",
            "business intelligence",
            "digital transformation",
            "ai strategy"
        ],
        related_keywords=[
            "roi",
            "change management",
            "integration",
            "scalability"
        ],
        validation_rules=[
            ValidationRule(
                name="business_impact",
                description="Impacto nos negócios",
                threshold=0.8,
                weight=0.4,
                validation_type="roi_analysis"
            ),
            ValidationRule(
                name="implementation_feasibility",
                description="Viabilidade de implementação",
                threshold=0.85,
                weight=0.3,
                validation_type="expert_review"
            ),
            ValidationRule(
                name="enterprise_readiness",
                description="Prontidão para empresa",
                threshold=0.9,
                weight=0.3,
                validation_type="checklist"
            )
        ],
        required_citations=True,
        requires_code_examples=False,
        requires_diagrams=True
    )
}

class DomainManager:
    """Gerenciador central de domínios"""
    
    def __init__(self):
        self.domains = DOMAIN_DEFINITIONS
    
    def get_domain(self, domain_id: str) -> Optional[DomainDefinition]:
        """Retorna definição de um domínio"""
        return self.domains.get(domain_id.lower())
    
    def get_domains_by_type(self, domain_type: DomainType) -> List[DomainDefinition]:
        """Retorna domínios por tipo"""
        return [
            domain for domain in self.domains.values()
            if domain.type == domain_type
        ]
    
    def get_validation_rules(self, domain_id: str) -> List[ValidationRule]:
        """Retorna regras de validação de um domínio"""
        domain = self.get_domain(domain_id)
        return domain.validation_rules if domain else []
    
    def validate_content_requirements(
        self,
        domain_id: str,
        content: Dict
    ) -> Dict[str, bool]:
        """Valida requisitos de conteúdo para um domínio"""
        domain = self.get_domain(domain_id)
        if not domain:
            return {}
            
        return {
            "has_citations": self._check_citations(content) if domain.required_citations else True,
            "has_code": self._check_code_examples(content) if domain.requires_code_examples else True,
            "has_diagrams": self._check_diagrams(content) if domain.requires_diagrams else True
        }
    
    def _check_citations(self, content: Dict) -> bool:
        """Verifica se o conteúdo tem citações adequadas"""
        # Implementar verificação de citações
        return True
    
    def _check_code_examples(self, content: Dict) -> bool:
        """Verifica se o conteúdo tem exemplos de código"""
        # Implementar verificação de código
        return True
    
    def _check_diagrams(self, content: Dict) -> bool:
        """Verifica se o conteúdo tem diagramas necessários"""
        # Implementar verificação de diagramas
        return True
    
    def get_content_guidelines(self, domain_id: str) -> Dict:
        """Retorna diretrizes de conteúdo para um domínio"""
        domain = self.get_domain(domain_id)
        if not domain:
            return {}
            
        return {
            "expertise_levels": domain.expertise_levels,
            "content_types": domain.content_types,
            "primary_keywords": domain.primary_keywords,
            "update_frequency": domain.update_frequency,
            "requirements": {
                "citations": domain.required_citations,
                "code_examples": domain.requires_code_examples,
                "diagrams": domain.requires_diagrams
            }
        }

if __name__ == "__main__":
    # Exemplo de uso
    manager = DomainManager()
    
    # Verificar domínio de AI Agents
    ai_domain = manager.get_domain("ai_agents")
    if ai_domain:
        print(f"Domain: {ai_domain.name}")
        print(f"Type: {ai_domain.type}")
        print(f"Keywords: {ai_domain.primary_keywords}")
    
    # Obter guidelines
    guidelines = manager.get_content_guidelines("ai_agents")
    print("\nContent Guidelines:")
    print(f"Expertise Levels: {guidelines['expertise_levels']}")
    print(f"Content Types: {guidelines['content_types']}")
    
    # Verificar regras de validação
    rules = manager.get_validation_rules("ai_agents")
    print("\nValidation Rules:")
    for rule in rules:
        print(f"- {rule.name}: {rule.threshold} (weight: {rule.weight})")
```

## core\domains\market_segments.py

```python
# core\domains\market_segments.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class MarketPriority(str, Enum):
    """Níveis de prioridade para segmentos"""
    CRITICAL = "critical"      # Foco imediato
    HIGH = "high"             # Alta prioridade
    MEDIUM = "medium"         # Prioridade média
    EMERGING = "emerging"     # Mercado emergente
    NICHE = "niche"          # Nicho específico

class Industry(str, Enum):
    """Indústrias principais"""
    FINTECH = "fintech"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    EDUCATION = "education"
    LOGISTICS = "logistics"
    ENERGY = "energy"

class TechMaturity(str, Enum):
    """Nível de maturidade tecnológica"""
    CUTTING_EDGE = "cutting_edge"  # Tecnologia de ponta
    ADVANCED = "advanced"          # Tecnologia avançada
    ESTABLISHED = "established"    # Tecnologia estabelecida
    LEGACY = "legacy"             # Tecnologia legada

class MarketSegment(BaseModel):
    """Definição detalhada de segmento de mercado"""
    name: str
    industry: Industry
    priority: MarketPriority
    market_size: float  # em bilhões USD
    growth_rate: float  # % ao ano
    tech_maturity: TechMaturity
    description: str
    key_technologies: List[str]
    key_players: List[str]
    challenges: List[str]
    opportunities: List[str]
    target_audience: List[str]
    success_metrics: List[str]
    update_frequency: str
    last_updated: datetime = Field(default_factory=datetime.now)

# Definição dos segmentos de mercado
MARKET_SEGMENTS: Dict[str, MarketSegment] = {
    "fintech_banking": MarketSegment(
        name="Digital Banking & Fintech",
        industry=Industry.FINTECH,
        priority=MarketPriority.CRITICAL,
        market_size=475.0,
        growth_rate=23.4,
        tech_maturity=TechMaturity.CUTTING_EDGE,
        description="Serviços financeiros digitais e tecnologias bancárias inovadoras",
        key_technologies=[
            "Blockchain",
            "AI/ML for Risk",
            "Cloud Native",
            "APIs/Open Banking",
            "Microservices"
        ],
        key_players=[
            "Nubank",
            "Stripe",
            "Block (Square)",
            "PicPay",
            "Itaú"
        ],
        challenges=[
            "Regulamentações (PIX, Open Finance)",
            "Segurança e Fraude",
            "Escalabilidade",
            "Integração com Legacy"
        ],
        opportunities=[
            "Banking as a Service",
            "Embedded Finance",
            "AI em Crédito",
            "Pagamentos Instantâneos"
        ],
        target_audience=[
            "Desenvolvedores Fintech",
            "Arquitetos de Sistemas",
            "Product Managers",
            "Tech Leads Bancários"
        ],
        success_metrics=[
            "Uptime do Sistema",
            "Taxa de Conversão",
            "Velocidade de Transação",
            "NPS"
        ],
        update_frequency="daily"
    ),
    
    "healthtech": MarketSegment(
        name="Healthcare Technology",
        industry=Industry.HEALTHCARE,
        priority=MarketPriority.HIGH,
        market_size=280.0,
        growth_rate=18.6,
        tech_maturity=TechMaturity.ADVANCED,
        description="Tecnologias para saúde digital e telemedicina",
        key_technologies=[
            "Telemedicine Platforms",
            "AI Diagnostics",
            "Healthcare APIs",
            "Medical IoT",
            "Cloud Healthcare"
        ],
        key_players=[
            "Hospital Israelita Albert Einstein",
            "Rede D'Or",
            "Hapvida",
            "Dasa"
        ],
        challenges=[
            "Privacidade (LGPD)",
            "Integração de Dados",
            "Regulamentações",
            "Experiência do Paciente"
        ],
        opportunities=[
            "Telemedicina",
            "AI em Diagnóstico",
            "Healthtech SaaS",
            "Medicina Preventiva"
        ],
        target_audience=[
            "Desenvolvedores Healthcare",
            "Gestores Hospitalares",
            "Médicos Tech-Savvy",
            "Healthcare IT"
        ],
        success_metrics=[
            "Precisão Diagnóstica",
            "Tempo de Atendimento",
            "Satisfação do Paciente",
            "Custo por Atendimento"
        ],
        update_frequency="weekly"
    )
}

class MarketSegmentManager:
    """Gerenciador de segmentos de mercado"""
    
    def __init__(self):
        self.segments = MARKET_SEGMENTS
    
    def get_segment(self, segment_id: str) -> Optional[MarketSegment]:
        """Retorna um segmento específico"""
        return self.segments.get(segment_id)
    
    def get_segments_by_priority(self, priority: MarketPriority) -> List[MarketSegment]:
        """Retorna segmentos por nível de prioridade"""
        return [
            segment for segment in self.segments.values()
            if segment.priority == priority
        ]
    
    def get_segments_by_industry(self, industry: Industry) -> List[MarketSegment]:
        """Retorna segmentos por indústria"""
        return [
            segment for segment in self.segments.values()
            if segment.industry == industry
        ]
    
    def get_tech_requirements(self, segment_id: str) -> Dict:
        """Retorna requisitos tecnológicos do segmento"""
        segment = self.get_segment(segment_id)
        if not segment:
            return {}
            
        return {
            "technologies": segment.key_technologies,
            "maturity": segment.tech_maturity,
            "update_frequency": segment.update_frequency
        }
    
    def get_market_metrics(self, segment_id: str) -> Dict:
        """Retorna métricas de mercado do segmento"""
        segment = self.get_segment(segment_id)
        if not segment:
            return {}
            
        return {
            "market_size": segment.market_size,
            "growth_rate": segment.growth_rate,
            "key_players": segment.key_players
        }
    
    def should_update_content(self, segment_id: str) -> bool:
        """Verifica se o conteúdo do segmento precisa ser atualizado"""
        segment = self.get_segment(segment_id)
        if not segment:
            return False
            
        # Calcula tempo desde última atualização
        time_diff = datetime.now() - segment.last_updated
        
        # Define limites baseado na frequência de atualização
        update_limits = {
            "daily": 1,      # 1 dia
            "weekly": 7,     # 7 dias
            "monthly": 30    # 30 dias
        }
        
        days_limit = update_limits.get(segment.update_frequency, 7)
        return time_diff.days >= days_limit
```

## core\domains\seasonality.py

```python
# core\domains\seasonality.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, date
from enum import Enum

class SeasonType(str, Enum):
    """Tipos de sazonalidade"""
    RETAIL = "retail"           # Eventos de varejo
    BUSINESS = "business"       # Ciclos de negócio
    TECH = "tech"              # Eventos tecnológicos
    MARKET = "market"          # Ciclos de mercado
    INDUSTRY = "industry"      # Específico da indústria
    REGULATORY = "regulatory"  # Marcos regulatórios

class EventStatus(str, Enum):
    """Status do evento sazional"""
    UPCOMING = "upcoming"      # Próximo
    ACTIVE = "active"         # Em andamento
    PASSED = "passed"         # Passado
    RECURRING = "recurring"   # Recorrente

class SeasonalEvent(BaseModel):
    """Evento sazional"""
    name: str
    type: SeasonType
    description: str
    start_date: date
    end_date: date
    impact_level: float = Field(ge=0.0, le=1.0)  # 0 a 1
    affected_industries: List[str]
    key_themes: List[str]
    content_priorities: List[str]
    recurrence: str  # annual, quarterly, monthly
    historical_metrics: Dict[str, float]
    preparation_time_days: int

class SeasonalityManager:
    """Gerenciador de sazonalidade"""
    
    def __init__(self):
        self.events = self._initialize_events()
    
    def _initialize_events(self) -> Dict[str, SeasonalEvent]:
        """Inicializa eventos sazonais"""
        return {
            "black_friday": SeasonalEvent(
                name="Black Friday",
                type=SeasonType.RETAIL,
                description="Principal evento de vendas do varejo",
                start_date=date(2024, 11, 29),
                end_date=date(2024, 11, 29),
                impact_level=0.9,
                affected_industries=[
                    "retail",
                    "e-commerce",
                    "fintech",
                    "logistics"
                ],
                key_themes=[
                    "performance",
                    "scalability",
                    "payments",
                    "security"
                ],
                content_priorities=[
                    "system_optimization",
                    "load_testing",
                    "monitoring",
                    "incident_response"
                ],
                recurrence="annual",
                historical_metrics={
                    "2023_transactions": 15000000,
                    "2023_peak_tps": 5000,
                    "2023_conversion": 0.045
                },
                preparation_time_days=90
            ),

            "tax_season": SeasonalEvent(
                name="Tax Season",
                type=SeasonType.REGULATORY,
                description="Período de declaração de impostos",
                start_date=date(2024, 3, 1),
                end_date=date(2024, 4, 30),
                impact_level=0.8,
                affected_industries=[
                    "fintech",
                    "accounting",
                    "banking",
                    "consulting"
                ],
                key_themes=[
                    "tax_compliance",
                    "automation",
                    "data_processing",
                    "security"
                ],
                content_priorities=[
                    "integration_guides",
                    "automation_tools",
                    "security_best_practices",
                    "performance_optimization"
                ],
                recurrence="annual",
                historical_metrics={
                    "2023_declarations": 35000000,
                    "2023_digital_rate": 0.92
                },
                preparation_time_days=60
            ),

            "tech_conference_season": SeasonalEvent(
                name="Tech Conference Season",
                type=SeasonType.TECH,
                description="Principais conferências de tecnologia",
                start_date=date(2024, 5, 1),
                end_date=date(2024, 8, 31),
                impact_level=0.75,
                affected_industries=[
                    "technology",
                    "startup",
                    "enterprise",
                    "education"
                ],
                key_themes=[
                    "innovation",
                    "trends",
                    "networking",
                    "product_launch"
                ],
                content_priorities=[
                    "trend_analysis",
                    "technical_deep_dives",
                    "case_studies",
                    "tutorials"
                ],
                recurrence="annual",
                historical_metrics={
                    "2023_attendees": 500000,
                    "2023_announcements": 250
                },
                preparation_time_days=120
            ),

            "open_banking_phase": SeasonalEvent(
                name="Open Banking Implementation Phase",
                type=SeasonType.REGULATORY,
                description="Fase de implementação do Open Banking",
                start_date=date(2024, 6, 1),
                end_date=date(2024, 8, 31),
                impact_level=0.85,
                affected_industries=[
                    "fintech",
                    "banking",
                    "insurance",
                    "investment"
                ],
                key_themes=[
                    "api_integration",
                    "security",
                    "compliance",
                    "data_sharing"
                ],
                content_priorities=[
                    "technical_standards",
                    "security_guidelines",
                    "integration_tutorials",
                    "compliance_guides"
                ],
                recurrence="quarterly",
                historical_metrics={
                    "2023_apis_launched": 150,
                    "2023_integration_rate": 0.75
                },
                preparation_time_days=90
            )
        }
    
    def get_current_events(self) -> List[SeasonalEvent]:
        """Retorna eventos ativos no momento"""
        today = date.today()
        return [
            event for event in self.events.values()
            if event.start_date <= today <= event.end_date
        ]
    
    def get_upcoming_events(self, days_ahead: int = 90) -> List[SeasonalEvent]:
        """Retorna próximos eventos"""
        from datetime import timedelta
        today = date.today()
        future = today + timedelta(days=days_ahead)
        return [
            event for event in self.events.values()
            if today < event.start_date <= future
        ]
    
    def get_preparation_recommendations(self, event: SeasonalEvent) -> Dict:
        """Gera recomendações de preparação para um evento"""
        days_to_event = (event.start_date - date.today()).days
        
        return {
            "preparation_time_left": days_to_event,
            "status": self._get_preparation_status(days_to_event, event.preparation_time_days),
            "priorities": event.content_priorities,
            "key_themes": event.key_themes,
            "historical_reference": event.historical_metrics
        }
    
    def _get_preparation_status(self, days_left: int, required_days: int) -> str:
        """Determina status de preparação"""
        if days_left <= 0:
            return "CRITICAL"
        elif days_left < required_days * 0.3:
            return "URGENT"
        elif days_left < required_days * 0.7:
            return "ON_TRACK"
        else:
            return "PLANNING"
    
    def get_industry_events(self, industry: str) -> List[SeasonalEvent]:
        """Retorna eventos específicos de uma indústria"""
        return [
            event for event in self.events.values()
            if industry in event.affected_industries
        ]
    
    def get_event_impact(self, event_name: str) -> Dict:
        """Calcula impacto de um evento específico"""
        event = self.events.get(event_name)
        if not event:
            return {}
            
        return {
            "impact_level": event.impact_level,
            "affected_industries": event.affected_industries,
            "preparation_required": event.preparation_time_days,
            "historical_performance": event.historical_metrics
        }

if __name__ == "__main__":
    # Exemplo de uso
    manager = SeasonalityManager()
    
    # Verificar eventos atuais
    current = manager.get_current_events()
    print(f"Eventos ativos: {len(current)}")
    
    # Verificar próximos eventos
    upcoming = manager.get_upcoming_events(30)  # próximos 30 dias
    print(f"Eventos próximos: {len(upcoming)}")
    
    # Para cada evento ativo
    for event in current:
        prep = manager.get_preparation_recommendations(event)
        print(f"\nEvento: {event.name}")
        print(f"Status: {prep['status']}")
        print(f"Prioridades: {prep['priorities']}")
```

## core\domains\tech_stacks.py

```python
# core\domains\tech_stacks.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class TechCategory(str, Enum):
    """Categorias de tecnologias"""
    AI_ML = "ai_ml"               # IA e Machine Learning
    DATA_ENGINEERING = "data"     # Engenharia de Dados
    CLOUD = "cloud"               # Cloud Computing
    BACKEND = "backend"           # Backend Development
    FRONTEND = "frontend"         # Frontend Development
    DEVOPS = "devops"            # DevOps & SRE
    SECURITY = "security"         # Segurança
    MOBILE = "mobile"            # Desenvolvimento Mobile

class MaturityLevel(str, Enum):
    """Nível de maturidade da tecnologia"""
    BLEEDING_EDGE = "bleeding_edge"  # Tecnologia muito recente
    EMERGING = "emerging"            # Em emergência
    GROWING = "growing"              # Em crescimento
    STABLE = "stable"                # Estável
    MATURE = "mature"                # Madura
    DECLINING = "declining"          # Em declínio

class Framework(BaseModel):
    """Modelo para frameworks e bibliotecas"""
    name: str
    category: TechCategory
    version: str
    maturity: MaturityLevel
    description: str
    key_features: List[str]
    use_cases: List[str]
    learning_curve: float = Field(ge=0.0, le=1.0)  # 0 a 1
    community_size: int
    github_stars: Optional[int]
    documentation_quality: float = Field(ge=0.0, le=1.0)
    enterprise_ready: bool
    last_release: datetime
    alternatives: List[str]

class TechStack(BaseModel):
    """Stack tecnológico completo"""
    name: str
    description: str
    primary_frameworks: List[Framework]
    supporting_tools: List[str]
    infrastructure: List[str]
    best_practices: List[str]
    pitfalls: List[str]
    learning_resources: List[str]
    success_stories: List[str]

# Definição das principais stacks e frameworks
MODERN_FRAMEWORKS = {
    "ai_agents": Framework(
        name="LangChain",
        category=TechCategory.AI_ML,
        version="0.0.300",
        maturity=MaturityLevel.GROWING,
        description="Framework para desenvolvimento de aplicações com LLMs",
        key_features=[
            "Agents",
            "Chains",
            "Memory",
            "Prompts",
            "Embeddings"
        ],
        use_cases=[
            "Chatbots",
            "Content Generation",
            "Data Analysis",
            "Search Systems"
        ],
        learning_curve=0.7,
        community_size=50000,
        github_stars=65000,
        documentation_quality=0.85,
        enterprise_ready=True,
        last_release=datetime(2023, 11, 15),
        alternatives=[
            "LlamaIndex",
            "Haystack",
            "Semantic Kernel"
        ]
    ),

    "vector_store": Framework(
        name="Pinecone",
        category=TechCategory.AI_ML,
        version="2.0.0",
        maturity=MaturityLevel.STABLE,
        description="Base de dados vetorial para AI e similarity search",
        key_features=[
            "Vector Similarity Search",
            "Real-time Updates",
            "Scalability",
            "Cloud Native"
        ],
        use_cases=[
            "Semantic Search",
            "Recommendation Systems",
            "Duplicate Detection",
            "AI Applications"
        ],
        learning_curve=0.6,
        community_size=30000,
        github_stars=None,  # Closed source
        documentation_quality=0.9,
        enterprise_ready=True,
        last_release=datetime(2023, 10, 1),
        alternatives=[
            "Weaviate",
            "Milvus",
            "Qdrant"
        ]
    ),

    "data_pipeline": Framework(
        name="Apache Airflow",
        category=TechCategory.DATA_ENGINEERING,
        version="2.7.1",
        maturity=MaturityLevel.MATURE,
        description="Plataforma para orquestração de pipelines de dados",
        key_features=[
            "DAG-based Workflows",
            "Rich UI",
            "Extensible",
            "Multiple Executors"
        ],
        use_cases=[
            "ETL Pipelines",
            "ML Pipelines",
            "Data Warehouse Loading",
            "Report Generation"
        ],
        learning_curve=0.8,
        community_size=100000,
        github_stars=31000,
        documentation_quality=0.9,
        enterprise_ready=True,
        last_release=datetime(2023, 9, 15),
        alternatives=[
            "Prefect",
            "Dagster",
            "Luigi"
        ]
    )
}

# Stacks tecnológicos completos
TECH_STACKS = {
    "modern_data_stack": TechStack(
        name="Modern Data Stack",
        description="Stack moderno para engenharia de dados e analytics",
        primary_frameworks=[
            MODERN_FRAMEWORKS["data_pipeline"]
        ],
        supporting_tools=[
            "dbt",
            "Snowflake",
            "Fivetran",
            "Looker"
        ],
        infrastructure=[
            "AWS",
            "Docker",
            "Kubernetes"
        ],
        best_practices=[
            "Data Modeling",
            "Testing",
            "Documentation",
            "Version Control"
        ],
        pitfalls=[
            "Complexity",
            "Cost Management",
            "Team Skills"
        ],
        learning_resources=[
            "dbt Learn",
            "Snowflake University",
            "Airflow Documentation"
        ],
        success_stories=[
            "Magazine Luiza",
            "iFood",
            "Nubank"
        ]
    ),

    "ai_development": TechStack(
        name="AI Development Stack",
        description="Stack completo para desenvolvimento de aplicações AI",
        primary_frameworks=[
            MODERN_FRAMEWORKS["ai_agents"],
            MODERN_FRAMEWORKS["vector_store"]
        ],
        supporting_tools=[
            "OpenAI API",
            "HuggingFace",
            "MLflow"
        ],
        infrastructure=[
            "GPU Instances",
            "Vector Databases",
            "Model Registry"
        ],
        best_practices=[
            "Prompt Engineering",
            "Model Evaluation",
            "Cost Optimization",
            "Safety & Ethics"
        ],
        pitfalls=[
            "Token Limits",
            "Cost Scaling",
            "Hallucinations"
        ],
        learning_resources=[
            "OpenAI Documentation",
            "LangChain Cookbook",
            "Pinecone Tutorials"
        ],
        success_stories=[
            "Globo",
            "Stone",
            "Inter"
        ]
    )
}

class TechStackManager:
    """Gerenciador de stacks tecnológicos"""
    
    def __init__(self):
        self.frameworks = MODERN_FRAMEWORKS
        self.stacks = TECH_STACKS
    
    def get_framework(self, framework_id: str) -> Optional[Framework]:
        """Retorna detalhes de um framework"""
        return self.frameworks.get(framework_id)
    
    def get_stack(self, stack_id: str) -> Optional[TechStack]:
        """Retorna um stack completo"""
        return self.stacks.get(stack_id)
    
    def get_alternatives(self, framework_id: str) -> List[str]:
        """Retorna alternativas para um framework"""
        framework = self.get_framework(framework_id)
        return framework.alternatives if framework else []
    
    def get_frameworks_by_category(self, category: TechCategory) -> List[Framework]:
        """Retorna frameworks por categoria"""
        return [
            framework for framework in self.frameworks.values()
            if framework.category == category
        ]
    
    def get_enterprise_ready_frameworks(self) -> List[Framework]:
        """Retorna frameworks prontos para uso empresarial"""
        return [
            framework for framework in self.frameworks.values()
            if framework.enterprise_ready
        ]
    
    def should_update_stack(self, stack_id: str) -> bool:
        """Verifica se um stack precisa de atualização"""
        stack = self.get_stack(stack_id)
        if not stack:
            return False
        
        # Verifica frameworks desatualizados
        outdated_frameworks = [
            framework for framework in stack.primary_frameworks
            if (datetime.now() - framework.last_release).days > 180
        ]
        
        return len(outdated_frameworks) > 0

if __name__ == "__main__":
    # Exemplo de uso
    manager = TechStackManager()
    
    # Listar frameworks AI/ML
    ai_frameworks = manager.get_frameworks_by_category(TechCategory.AI_ML)
    print(f"Frameworks AI/ML: {len(ai_frameworks)}")
    
    # Verificar frameworks enterprise-ready
    enterprise_frameworks = manager.get_enterprise_ready_frameworks()
    print(f"Frameworks Enterprise: {len(enterprise_frameworks)}")
    
    # Verificar stack AI
    ai_stack = manager.get_stack("ai_development")
    if ai_stack:
        print(f"\nAI Stack:")
        print(f"Frameworks: {len(ai_stack.primary_frameworks)}")
        print(f"Tools: {ai_stack.supporting_tools}")
```

## core\domains\__init__.py

```python
# core/domains/__init__.py

from .market_segments import MarketSegmentManager, MarketSegment, MarketPriority, Industry
from .seasonality import SeasonalityManager, SeasonalEvent, SeasonType
from .tech_stacks import TechStackManager, TechStack, Framework, TechCategory, MaturityLevel
from .bigtech_monitor import BigTechMonitor, BigTechCompany, Innovation, CompanyCategory
from .adoption_metrics import AdoptionMetricsManager, AdoptionMetric, ROIMetric, UseCaseMetric
from .definitions import DomainManager, DomainDefinition, DomainType, ContentType, ValidationRule

__all__ = [
   # Market Segments
   'MarketSegmentManager',
   'MarketSegment',
   'MarketPriority',
   'Industry',
   
   # Seasonality
   'SeasonalityManager',
   'SeasonalEvent',
   'SeasonType',
   
   # Tech Stacks
   'TechStackManager',
   'TechStack',
   'Framework', 
   'TechCategory',
   'MaturityLevel',
   
   # BigTech Monitor
   'BigTechMonitor',
   'BigTechCompany',
   'Innovation',
   'CompanyCategory',
   
   # Adoption Metrics
   'AdoptionMetricsManager',
   'AdoptionMetric',
   'ROIMetric',
   'UseCaseMetric',
   
   # Definitions
   'DomainManager',
   'DomainDefinition',
   'DomainType',
   'ContentType',
   'ValidationRule'
]
```

## examples\news_integration_example.py

```python
# examples/news_integration_example.py

import asyncio
from datetime import datetime, timedelta
from agents.search.services.news.service import NewsIntegrationService
from agents.search.services.news.models import NewsSearchQuery
import json

async def main():
    # Inicializar serviço
    news_service = NewsIntegrationService()
    await news_service.initialize()

    try:
        # Criar consulta de exemplo
        query = NewsSearchQuery(
            topic="AI and Machine Learning",
            keywords=["artificial intelligence", "deep learning", "neural networks"],
            categories=["technology", "ai"],
            start_date=datetime.now() - timedelta(days=7),
            max_results=5,
            min_relevance=0.7,
            language="en"
        )

        print(f"\nBuscando notícias sobre: {query.topic}")
        print("=" * 50)

        # Realizar busca
        articles = await news_service.search_news(query)

        print(f"\nEncontrados {len(articles)} artigos relevantes:\n")

        # Exibir resultados
        for i, article in enumerate(articles, 1):
            print(f"\nArtigo {i}:")
            print(f"Título: {article.title}")
            print(f"Fonte: {article.source}")
            print(f"Data: {article.published_date.strftime('%d/%m/%Y %H:%M')}")
            print(f"Relevância: {article.relevance_score:.2f}")
            print(f"URL: {article.url}")
            print(f"Tempo de leitura: {article.read_time} minutos")
            print("\nResumo:")
            print(article.summary)
            print("-" * 50)

        # Exibir métricas
        metrics = news_service.metrics.get_metrics()
        print("\nMétricas de execução:")
        print(json.dumps(metrics, indent=2))

    finally:
        # Fechar serviço
        await news_service.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## examples\__init__.py

```python

```

## infrastructure\__init__.py

```python

```

## infrastructure\docker\Dockerfile

```python
FROM python:3.9-slim 
WORKDIR /app 
COPY requirements.txt . 
RUN pip install -r requirements.txt 
COPY . . 
CMD ["python", "main.py"] 

```

## infrastructure\kubernetes\deployment.yaml

```python
apiVersion: apps/v1 
kind: Deployment 
metadata: 
  name: ad-agents 

```

## scripts\consume_event.py

```python
from core.rabbitmq_utils import RabbitMQUtils

def process_message(ch, method, properties, body):
    import json
    message = json.loads(body)
    print(f"Mensagem recebida: {message}")

# Consumir mensagens da fila 'content.generated'
RabbitMQUtils.consume_event("content.generated", process_message)

```

## scripts\publish_event.py

```python
from core.rabbitmq_utils import RabbitMQUtils

# Exemplo de publicação na fila 'content.generated'
RabbitMQUtils.publish_event(
    "content.generated",
    {"content_id": 1, "content": "Texto inicial gerado"}
)


```

## scripts\test_loop.py

```python

```

## scripts\test_news_search.py

```python
# scripts/test_news_search.py

import asyncio
from datetime import datetime, timedelta
import logging
from agents.search.services.news.service import NewsIntegrationService
from agents.search.services.news.models import NewsSearchQuery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_search():
    # Inicializar o serviço
    service = NewsIntegrationService()
    await service.initialize()
    
    try:
        # Criar uma consulta de teste
        query = NewsSearchQuery(
            topic="Python async",
            keywords=["programming", "async"],
            start_date=datetime.now() - timedelta(days=7),
            min_relevance=0.3,
            max_results=5
        )
        
        print(f"\nBuscando artigos sobre: {query.topic}")
        print("=" * 50)
        
        # Realizar a busca
        results = await service.search_news(query)
        
        # Exibir resultados
        print(f"\nEncontrados {len(results)} artigos:\n")
        
        for i, article in enumerate(results, 1):
            print(f"\nArtigo {i}:")
            print(f"Título: {article.title}")
            print(f"Autor: {article.author or 'Não especificado'}")
            print(f"Data: {article.published_date.strftime('%d/%m/%Y %H:%M')}")
            print(f"URL: {article.url}")
            print(f"Tags: {', '.join(article.tags)}")
            if article.metadata:
                print("Metadados:")
                print(f"  Tempo de leitura: {article.metadata.get('reading_time', 'N/A')} minutos")
                print(f"  Comentários: {article.metadata.get('comments_count', 'N/A')}")
                print(f"  Reações: {article.metadata.get('reactions_count', 'N/A')}")
            print("-" * 50)
        
        # Exibir métricas
        metrics = service.metrics.get_metrics()
        print("\nMétricas:")
        for source, data in metrics["requests"].items():
            print(f"\n{source.upper()}:")
            print(f"  Requisições com sucesso: {data['success']}")
            print(f"  Requisições com erro: {data['error']}")
            print(f"  Artigos processados: {metrics['articles'].get(source, 0)}")
            print(f"  Requisições ativas: {metrics['active_requests'].get(source, 0)}")
        
    except Exception as e:
        logger.error(f"Erro durante a busca: {str(e)}")
        raise
    finally:
        # Fechar o serviço
        await service.close()

if __name__ == "__main__":
    try:
        asyncio.run(test_search())
    except KeyboardInterrupt:
        print("\nOperação interrompida pelo usuário")
    except Exception as e:
        logger.error(f"Erro não tratado: {str(e)}")
        raise
```

## tests\test_config.py

```python
# test_config.py
from core.config import get_settings

settings = get_settings()
print(f"OpenAI Key: {'*' * len(settings.api.openai_api_key)}")
print(f"Pinecone Key: {'*' * len(settings.api.pinecone_api_key)}")
```

## tests\test_content.py

```python
# tests\test_content.py
import asyncio
import sys
from pathlib import Path

# Ajusta o path para importações relativas
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from agents.content.agent import ContentAgent, ContentTemplate
from agents.content.config import ContentAgentConfig

async def test_content_generation():
    # Carrega configurações
    config = ContentAgentConfig()
    
    # Inicializa agente
    agent = ContentAgent()
    
    # Cria template de teste
    template = ContentTemplate(
        type="blog_post",
        structure={
            "target_audience": "desenvolvedores",
            "sections": ["introdução", "conceitos", "implementação", "conclusão"]
        },
        tone="technical_but_friendly",
        guidelines=[
            "use exemplos de código",
            "explique conceitos complexos de forma simples",
            "inclua referências práticas"
        ],
        seo_requirements={
            "min_words": 1200,
            "heading_structure": "h1,h2,h3",
            "keyword_density": 0.02
        }
    )
    
    # Dados de teste
    topic = "Implementando Machine Learning em Produção"
    keywords = ["MLOps", "machine learning", "deploy", "produção"]
    references = [
        {
            "title": "MLOps Best Practices",
            "content": "MLOps combina Machine Learning com práticas DevOps..."
        }
    ]
    
    try:
        # Gera conteúdo
        content = await agent.generate_content(
            topic=topic,
            keywords=keywords,
            references=references,
            template=template
        )
        
        # Imprime resultados
        print("\n" + "="*50)
        print("TÍTULO:")
        print("="*50)
        print(content.title)
        
        print("\n" + "="*50)
        print("META DESCRIPTION:")
        print("="*50)
        print(content.meta_description)
        
        print("\n" + "="*50)
        print("MÉTRICAS:")
        print("="*50)
        print(f"SEO Score: {content.seo_score}")
        print(f"Readability Score: {content.readability_score}")
        print(f"Tamanho do conteúdo: {len(content.content)} caracteres")
        print(f"Palavras-chave utilizadas: {content.keywords}")
        
        print("\n" + "="*50)
        print("CONTEÚDO COMPLETO:")
        print("="*50)
        print(content.content)
        
    except Exception as e:
        print(f"Erro durante a geração de conteúdo: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_content_generation())
```

## tests\test_devto_client.py

```python
# tests/test_devto_client.py

import pytest
from datetime import datetime
from unittest.mock import patch, Mock
from agents.search.services.news.clients.devto import DevToClient, DevToArticle

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_response():
    """Mock para resposta da API"""
    return [
        {
            "id": 1,
            "title": "Test Article",
            "description": "Test Description",
            "url": "https://dev.to/test/article",
            "published_at": datetime.now().isoformat(),
            "tag_list": ["python", "testing"],
            "user": {
                "name": "Test User",
                "username": "testuser"
            },
            "reading_time_minutes": 5,
            "comments_count": 10,
            "public_reactions_count": 20
        }
    ]

async def test_get_articles_success(mock_response):
    """Testa busca de artigos com sucesso"""
    client = DevToClient()
    
    # Mock da sessão HTTP
    mock_session = Mock()
    mock_session.get.return_value.__aenter__.return_value.status = 200
    mock_session.get.return_value.__aenter__.return_value.json = Mock(
        return_value=mock_response
    )
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        articles = await client.get_articles(search_term="python")
        
        assert len(articles) == 1
        article = articles[0]
        assert article.title == "Test Article"
        assert article.tag_list == ["python", "testing"]
        
        # Testar conversão para NewsArticle
        news_article = client.to_news_article(article)
        assert news_article.title == article.title
        assert news_article.source == "dev.to"
        assert "reading_time" in news_article.metadata

async def test_get_articles_rate_limit(mock_response):
    """Testa comportamento quando rate limit é atingido"""
    client = DevToClient()
    
    # Mock da sessão HTTP com rate limit
    mock_session = Mock()
    mock_session.get.return_value.__aenter__.return_value.status = 429
    mock_session.get.return_value.__aenter__.return_value.headers = {"Retry-After": "60"}
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        articles = await client.get_articles(search_term="python")
        assert len(articles) == 0

async def test_get_articles_error():
    """Testa tratamento de erro na API"""
    client = DevToClient()
    
    # Mock da sessão HTTP com erro
    mock_session = Mock()
    mock_session.get.side_effect = Exception("API Error")
    
    with patch('aiohttp.ClientSession', return_value=mock_session):
        articles = await client.get_articles(search_term="python")
        assert len(articles) == 0

async def test_article_conversion(mock_response):
    """Testa conversão de DevToArticle para NewsArticle"""
    client = DevToClient()
    devto_article = DevToArticle.parse_obj(mock_response[0])
    news_article = client.to_news_article(devto_article)
    
    assert news_article.title == devto_article.title
    assert news_article.url == str(devto_article.url)
    assert news_article.source == "dev.to"
    assert news_article.tags == devto_article.tag_list
    assert news_article.metadata["reading_time"] == devto_article.reading_time_minutes
```

## tests\test_feedback_loop.py

```python
# tests/test_feedback_loop.py

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Ajusta o path para importações relativas
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from core.feedback_loop import ContentFeedbackLoop, FeedbackMetrics
from agents.content.agent import ContentAgent, ContentTemplate
from agents.review.agent import ReviewAgent
from agents.search.agent import SearchAgent

async def test_feedback_loop():
    """Testa o ciclo de feedback autônomo"""
    
    print("\n=== Iniciando Teste do Feedback Loop ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        # Inicializa agentes
        content_agent = ContentAgent()
        review_agent = ReviewAgent()
        search_agent = SearchAgent()
        
        # Inicializa feedback loop
        feedback_loop = ContentFeedbackLoop(
            content_agent=content_agent,
            review_agent=review_agent,
            search_agent=search_agent,
            max_iterations=3,
            min_quality_threshold=0.8,
            improvement_threshold=0.1
        )
        
        # Cria template de teste
        template = ContentTemplate(
            type="technical_article",
            structure={
                "target_audience": "developers",
                "sections": [
                    "introduction",
                    "concepts",
                    "implementation",
                    "best_practices",
                    "conclusion"
                ]
            },
            tone="technical_but_friendly",
            guidelines=[
                "Include code examples",
                "Focus on practical applications",
                "Explain complex concepts clearly"
            ],
            seo_requirements={
                "min_words": 1500,
                "heading_structure": "h1,h2,h3",
                "keyword_density": 0.02
            }
        )
        
        # Contexto de teste
        context = {
            "domain": "ai_agents",
            "technical_level": "advanced",
            "industry": "technology"
        }
        
        # Gera conteúdo com feedback
        print("\nGerando conteúdo com feedback loop...")
        content, metrics = await feedback_loop.generate_with_feedback(
            topic="Implementing Autonomous AI Agents with Python",
            keywords=[
                "AI agents",
                "automation",
                "machine learning",
                "python",
                "orchestration"
            ],
            template=template,
            context=context
        )
        
        # Validações básicas
        print("\nRealizando validações...")
        assert isinstance(metrics, FeedbackMetrics), "Metrics deve ser instância de FeedbackMetrics"
        assert metrics.iteration_count > 0, "Deve ter pelo menos uma iteração"
        assert metrics.total_duration > 0, "Duração total deve ser maior que zero"
        assert len(metrics.history) > 0, "Deve ter histórico de iterações"
        
        # Imprime resultados
        print("\n=== Resultados do Feedback Loop ===")
        print(f"Número de iterações: {metrics.iteration_count}")
        print(f"Duração total: {metrics.total_duration:.2f} segundos")
        
        print("\n=== Métricas Finais ===")
        print(f"Quality Score: {metrics.quality_score:.2f}")
        print(f"Technical Accuracy: {metrics.technical_accuracy:.2f}")
        print(f"Readability Score: {metrics.readability_score:.2f}")
        print(f"SEO Score: {metrics.seo_score:.2f}")
        
        print("\n=== Histórico de Iterações ===")
        for iteration in metrics.history:
            print(f"\nIteração {iteration.iteration_number}:")
            print("Métricas:")
            for metric, value in iteration.content_metrics.items():
                print(f"- {metric}: {value:.2f}")
            
            print("\nSugestões recebidas:")
            for suggestion in iteration.suggestions:
                print(f"- {suggestion}")
            
            print("\nMelhorias aplicadas:")
            for improvement in iteration.improvements_made:
                print(f"- {improvement}")
        
        print("\n=== Preview do Conteúdo ===")
        print("\nTítulo:")
        print(content.title)
        print("\nMeta Description:")
        print(content.meta_description)
        print("\nPreview do conteúdo:")
        preview = content.content[:500] + "..." if len(content.content) > 500 else content.content
        print(preview)
        
        # Validações adicionais
        assert len(content.content) >= template.seo_requirements["min_words"], \
            "Conteúdo deve atender requisito mínimo de palavras"
        
        assert metrics.quality_score >= feedback_loop.min_quality_threshold or \
               metrics.iteration_count >= feedback_loop.max_iterations, \
            "Deve atingir qualidade mínima ou número máximo de iterações"
        
        print("\n✓ Todas as validações passaram com sucesso!")
        return content, metrics
        
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {str(e)}")
        raise

async def test_quality_improvement():
    """Testa se o feedback loop realmente melhora a qualidade"""
    print("\n=== Testando Melhoria de Qualidade ===")
    
    try:
        # Executa teste principal
        content, metrics = await test_feedback_loop()
        
        # Analisa histórico de qualidade
        if len(metrics.history) > 1:
            initial_quality = metrics.history[0].content_metrics['quality']
            final_quality = metrics.quality_score
            
            print(f"\nQualidade inicial: {initial_quality:.2f}")
            print(f"Qualidade final: {final_quality:.2f}")
            print(f"Melhoria: {(final_quality - initial_quality):.2f}")
            
            assert final_quality > initial_quality, \
                "Qualidade final deve ser maior que a inicial"
            
            print("\n✓ Teste de melhoria de qualidade passou!")
            
    except Exception as e:
        print(f"\n❌ Erro no teste de melhoria: {str(e)}")
        raise

if __name__ == "__main__":
    print("\n=== Iniciando Suite de Testes do Feedback Loop ===")
    
    try:
        # Executa os testes
        asyncio.run(test_feedback_loop())
        asyncio.run(test_quality_improvement())
        
        print("\n=== Todos os testes completados com sucesso! ===")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {str(e)}")
        raise
```

## tests\test_news_integration.py

```python
# tests/test_news_integration.py

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
from agents.search.services.news.service import NewsIntegrationService
from agents.search.services.news.models import NewsSearchQuery
from agents.search.services.news.clients.devto import DevToArticle

pytestmark = pytest.mark.asyncio

@pytest.fixture
async def news_service():
    """Fixture para o serviço de notícias"""
    service = NewsIntegrationService()
    await service.initialize()
    try:
        yield service
    finally:
        await service.close()

@pytest.fixture
def mock_devto_response():
    """Mock para resposta do Dev.to"""
    return [
        {
            "id": 1,
            "title": "Understanding Python Async",
            "description": "A guide to async/await in Python",
            "url": "https://dev.to/test/python-async",
            "published_at": datetime.now().isoformat(),
            "tag_list": ["python", "programming"],
            "user": {
                "name": "Test Author",
                "username": "testuser"
            },
            "reading_time_minutes": 5,
            "comments_count": 10,
            "public_reactions_count": 20
        },
        {
            "id": 2,
            "title": "JavaScript Best Practices",
            "description": "Writing better JavaScript code",
            "url": "https://dev.to/test/javascript",
            "published_at": (datetime.now() - timedelta(days=2)).isoformat(),
            "tag_list": ["javascript", "webdev"],
            "user": {
                "name": "Another Author",
                "username": "jsdev"
            },
            "reading_time_minutes": 8,
            "comments_count": 15,
            "public_reactions_count": 30
        }
    ]

async def test_search_integration(news_service, mock_devto_response):
    """Testa integração completa da busca"""
    # Criar query
    query = NewsSearchQuery(
        topic="python",
        keywords=["programming"],
        start_date=datetime.now() - timedelta(days=7),
        min_relevance=0.3,
        max_results=10
    )
    
    # Mock do cliente Dev.to
    with patch.object(news_service.devto_client, 'get_articles', return_value=[
        DevToArticle.parse_obj(article) for article in mock_devto_response
    ]):
        results = await news_service.search_news(query)
        
        assert len(results) > 0
        assert any("Python" in article.title for article in results)
        assert all(article.source == "dev.to" for article in results)

async def test_cache_integration(news_service, mock_devto_response):
    """Testa integração com cache"""
    query = NewsSearchQuery(topic="python")
    
    # Mock do cliente Dev.to
    with patch.object(news_service.devto_client, 'get_articles', return_value=[
        DevToArticle.parse_obj(mock_devto_response[0])
    ]) as mock_get:
        # Primeira chamada
        results1 = await news_service.search_news(query)
        assert len(results1) > 0
        assert mock_get.called
        
        # Segunda chamada (deve vir do cache)
        mock_get.reset_mock()
        results2 = await news_service.search_news(query)
        assert len(results2) > 0
        assert not mock_get.called
        
        # Verificar se resultados são iguais
        assert results1 == results2

async def test_error_handling(news_service):
    """Testa tratamento de erros na integração"""
    query = NewsSearchQuery(topic="python")
    
    # Simular erro no cliente
    with patch.object(news_service.devto_client, 'get_articles', side_effect=Exception("API Error")):
        results = await news_service.search_news(query)
        assert len(results) == 0

async def test_relevance_filtering(news_service, mock_devto_response):
    """Testa filtro de relevância"""
    query = NewsSearchQuery(
        topic="python",
        min_relevance=0.7  # Relevância alta
    )
    
    # Mock do cliente Dev.to
    with patch.object(news_service.devto_client, 'get_articles', return_value=[
        DevToArticle.parse_obj(article) for article in mock_devto_response
    ]):
        results = await news_service.search_news(query)
        
        # Apenas artigos muito relevantes devem passar
        assert all(
            "Python" in article.title or 
            "python" in article.tags 
            for article in results
        )
```

## tests\test_orchestrator.py

```python
# tests/test_orchestrator.py

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Ajusta o path para importações relativas
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from core.orchestrator import Orchestrator, ContentRequest
from agents.search.agent import SearchResult

async def test_content_generation():
    """Testa geração completa de conteúdo"""
    orchestrator = Orchestrator()
    
    # Testa conversão de resultados de busca
    test_search_results = [
        SearchResult(
            content="Test content",
            source="test_source",
            relevance_score=0.8,
            metadata={"key": "value"},
            embedding=None,
            timestamp=datetime.now()
        )
    ]
    
    # Verifica se a conversão está funcionando
    converted_refs = orchestrator._convert_search_results_to_references(test_search_results)
    assert isinstance(converted_refs, list), "Deve retornar uma lista"
    assert len(converted_refs) > 0, "Lista não deve estar vazia"
    assert "title" in converted_refs[0], "Referência deve ter título"
    assert "content" in converted_refs[0], "Referência deve ter conteúdo"
    
    # Teste principal
    request = ContentRequest(
        topic="Implementando AI Agents para Automação de Processos",
        domain="ai_agents",
        content_type="technical_guide", 
        target_audience="desenvolvedores",
        technical_level="advanced",
        keywords=["AI agents", "automação", "processos", "MLOps"],
        references_required=True,
        code_examples=True
    )
    
    print("\n=== Iniciando Teste do Orchestrator ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        result = await orchestrator.process_request(request)
        
        # Validações do resultado
        assert result is not None, "Resultado não pode ser None"
        assert "content" in result, "Resultado deve conter 'content'"
        assert "metadata" in result, "Resultado deve conter 'metadata'"
        assert "suggestions" in result, "Resultado deve conter 'suggestions'"
        assert isinstance(result["content"], str), "Content deve ser string"
        assert len(result["content"]) > 0, "Content não deve estar vazio"
        
        print("\n" + "="*50)
        print("PLANO DE CONTEÚDO:")
        print("="*50)
        print(f"Topic: {result['metadata']['plan']['topic']}")
        print(f"Target Audience: {result['metadata']['plan']['target_audience']}")
        print(f"Priority: {result['metadata']['plan']['priority']}")
        
        print("\n" + "="*50)
        print("MÉTRICAS DE QUALIDADE:")
        print("="*50)
        print(f"Approved: {result['metadata']['approved']}")
        for metric, score in result['metadata']['quality_scores'].items():
            print(f"{metric}: {score:.2f}")
            
        print("\n" + "="*50)
        print("SUGESTÕES DE MELHORIA:")
        print("="*50)
        for suggestion in result['suggestions']:
            print(f"- {suggestion}")
            
        print("\n" + "="*50)
        print("PRÉVIA DO CONTEÚDO:")
        print("="*50)
        preview = result['content'][:500] + "..." if len(result['content']) > 500 else result['content']
        print(preview)
        
        return result
        
    except Exception as e:
        print(f"Erro no teste: {str(e)}")
        raise

async def test_metrics_analysis():
    """Testa análise de métricas"""
    orchestrator = Orchestrator()
    
    request = ContentRequest(
        topic="Guia de Implementação de AI",
        domain="ai_agents",
        content_type="guide",
        target_audience="desenvolvedores",
        technical_level="intermediate",
        keywords=["AI", "implementação", "guia"]
    )
    
    try:
        metrics = await orchestrator.analyze_metrics(request)
        assert metrics is not None, "Métricas não podem ser None"
        assert "metrics" in metrics, "Resultado deve conter 'metrics'"
        assert "timestamp" in metrics, "Resultado deve conter 'timestamp'"
        assert isinstance(metrics["metrics"]["estimated_impact"], float), "Impact deve ser float"
        assert isinstance(metrics["metrics"]["priority"], int), "Priority deve ser int"
        
        print("\n" + "="*50)
        print("ANÁLISE DE MÉTRICAS:")
        print("="*50)
        print(f"Estimated Impact: {metrics['metrics']['estimated_impact']:.2f}")
        print(f"Priority: {metrics['metrics']['priority']}")
        return metrics
        
    except Exception as e:
        print(f"Erro na análise de métricas: {str(e)}")
        raise

async def run_all_tests():
    """Executa todos os testes"""
    print("\n=== Executando Suite de Testes do Orchestrator ===")
    
    try:
        # Teste de geração de conteúdo
        content_result = await test_content_generation()
        print("\n✓ Teste de geração completado")
        
        # Teste de métricas
        metrics_result = await test_metrics_analysis()
        print("\n✓ Teste de métricas completado")
        
        print("\n=== Todos os testes completados com sucesso ===")
        
        return {
            "content_result": content_result,
            "metrics_result": metrics_result
        }
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(run_all_tests())
```

## tests\test_planning.py

```python
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
```

## tests\test_review.py

```python
# tests/test_review.py

import asyncio
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from agents.review.agent import ReviewAgent

async def test_review():
   """Testa o Review Agent"""
   
   # Inicializa agente
   agent = ReviewAgent()
   
   # Conteúdo de teste
   test_content = """
   # Implementando AI Agents em Produção
   
   Este guia aborda os principais aspectos da implementação de agentes de IA
   em ambientes produtivos. Vamos explorar as melhores práticas e desafios comuns.
   
   ## Principais Tópicos
   
   1. Arquitetura de Agentes
   2. Integração com LLMs
   3. Monitoramento e Observabilidade
   4. Tratamento de Erros
   """
   
   # Contexto do conteúdo
   context = {
       "domain": "ai_agents",
       "target_audience": "technical_leaders",
       "technical_level": "advanced",
       "content_type": "technical_guide",
       "required_sections": [
           "architecture",
           "implementation",
           "best_practices",
           "monitoring"
       ]
   }
   
   try:
       # Executa revisão
       result = await agent.review_content(
           content=test_content,
           context=context
       )
       
       # Imprime resultados
       print("\n=== Resultado da Revisão ===")
       print(f"Qualidade: {result.quality_score:.2f}")
       print(f"SEO: {result.seo_score:.2f}")
       print(f"Precisão Técnica: {result.technical_accuracy:.2f}")
       print(f"Legibilidade: {result.readability_score:.2f}")
       print(f"Aprovado: {result.approved}")
       
       print("\nSugestões:")
       for suggestion in result.suggestions:
           print(f"- {suggestion}")
       
       print("\nConteúdo Revisado:")
       print(result.content)
       
   except Exception as e:
       print(f"Erro no teste: {str(e)}")
       raise

if __name__ == "__main__":
   asyncio.run(test_review())
```

## tests\test_search.py

```python
# tests\test_search.py
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Adiciona o diretório raiz ao path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from agents.search.agent import SearchAgent, SearchQuery
from agents.search.config import SearchAgentConfig

async def test_search():
    # Inicializa o agente
    agent = SearchAgent()
    
    # Cria uma query de teste
    query = SearchQuery(
        query="Implementação de MLOps em produção",
        context="Buscando informações sobre deploy de modelos ML",
        filters={
            "language": "pt-br",
            "max_age_days": 365
        }
    )
    
    print("\n=== Teste do Search Agent ===")
    print(f"\nQuery: {query.query}")
    print(f"Contexto: {query.context}")
    
    try:
        # Realiza a busca
        results = await agent.search(query)
        
        print(f"\nEncontrados {len(results)} resultados:")
        for i, result in enumerate(results, 1):
            print(f"\n--- Resultado {i} ---")
            print(f"Fonte: {result.source}")
            print(f"Relevância: {result.relevance_score}")
            print(f"Timestamp: {result.timestamp}")
            print(f"Conteúdo: {result.content[:200]}...")
            if result.metadata:
                print(f"Metadata: {result.metadata}")
    
    except Exception as e:
        print(f"\nErro durante a busca: {str(e)}")
        raise

async def test_indexing():
    # Inicializa o agente
    agent = SearchAgent()
    
    # Conteúdo de teste
    test_content = """
    MLOps (Machine Learning Operations) é uma prática que visa automatizar
    e otimizar o ciclo de vida completo de modelos de machine learning em produção.
    """
    
    metadata = {
        "source": "test",
        "author": "AD Team",
        "timestamp": datetime.now().isoformat(),
        "category": "MLOps"
    }
    
    print("\n=== Teste de Indexação ===")
    
    try:
        # Indexa o conteúdo
        await agent.index_content(test_content, metadata)
        print("\nConteúdo indexado com sucesso!")
        
        # Testa a busca do conteúdo indexado
        query = SearchQuery(
            query="MLOps automatização",
            context="Machine Learning Operations"
        )
        
        results = await agent.search(query)
        print(f"\nBusca após indexação: {len(results)} resultados")
        
    except Exception as e:
        print(f"\nErro durante a indexação: {str(e)}")
        raise

if __name__ == "__main__":
    # Executa os testes
    asyncio.run(test_search())
    asyncio.run(test_indexing())
```

## tests\__init__.py

```python

```
