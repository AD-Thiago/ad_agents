# File: Dockerfile
FROM python:3.11-slim

# Configurar diretório de trabalho
WORKDIR /app

# Copiar arquivos do projeto
COPY requirements.txt .
COPY . .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Comando padrão (será sobrescrito pelo docker-compose)
CMD ["python", "-m", "scripts.start_agent", "planning"]