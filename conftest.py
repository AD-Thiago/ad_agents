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