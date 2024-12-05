# test_config.py
from core.config import get_settings

settings = get_settings()
print(f"OpenAI Key: {'*' * len(settings.api.openai_api_key)}")
print(f"Pinecone Key: {'*' * len(settings.api.pinecone_api_key)}")