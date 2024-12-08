#import pytest
from core.config import get_settings


def test_settings_load():
    """Teste para verificar se as configurações carregam corretamente."""
    settings = get_settings()

    # Teste geral de configuração
    assert settings.OPENAI_API_KEY is not None, "OPENAI_API_KEY não está carregado!"
    assert settings.DB_HOST == "localhost", "DB_HOST não está configurado corretamente!"
    assert settings.DB_PORT == 5432, "DB_PORT não está configurado corretamente!"
    assert settings.DB_NAME is not None, "DB_NAME não está carregado!"
    assert settings.DATABASE_URL.startswith("postgresql://"), "DATABASE_URL não foi construída corretamente!"

    # Teste de valores numéricos
    assert settings.CACHE_TTL > 0, "CACHE_TTL deve ser maior que 0!"
    assert settings.WORKFLOW_TIMEOUT > 0, "WORKFLOW_TIMEOUT deve ser maior que 0!"

    print("Todas as configurações foram carregadas com sucesso!")


if __name__ == "__main__":
    # Rodar diretamente no terminal
    test_settings_load()
    print("Teste concluído com sucesso!")