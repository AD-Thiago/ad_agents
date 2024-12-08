# test_config.py
from core.config import get_settings

def test_env_loading():
    settings = get_settings()
    assert settings.DB_HOST == "localhost"  # Substitua por algum valor do .env
    assert settings.DB_PORT == 5432         # Confirme se os valores estão corretos
    print("Todas as variáveis foram carregadas corretamente!")

if __name__ == "__main__":
    test_env_loading()
