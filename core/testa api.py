from core.config import get_settings
from agents.planning.config import config

# Validação da chave da API
settings = get_settings()
if not settings.api.openai_api_key:
    raise ValueError("A chave da API OpenAI (openai_api_key) não foi configurada corretamente.")
print(f"Chave da API OpenAI encontrada: {settings.api.openai_api_key}")


