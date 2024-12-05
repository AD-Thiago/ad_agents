import os
import mimetypes

# Define os arquivos e diretórios a serem excluídos
EXCLUDE = ["venv", "__pycache__", ".git", ".env", "README.md", "generate_readme.py"]

# Caminho para salvar o README
README_PATH = "README.md"


def create_index(sections):
    """Cria uma seção de índice com links"""
    index = "# Índice\n\n"
    for section in sections:
        link = section.lower().replace(" ", "-").replace("/", "-")
        index += f"- [{section}](#{link})\n"
    return index


def get_file_content(file_path):
    """Lê o conteúdo de um arquivo, ignorando arquivos binários"""
    try:
        # Verifica o tipo do arquivo
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and "text" not in mime_type:
            return None  # Ignora arquivos não-textuais

        # Lê arquivos textuais
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except (UnicodeDecodeError, IOError):
        return None  # Ignora arquivos que não podem ser lidos


def list_files(directory):
    """Lista todos os arquivos e diretórios em um projeto"""
    project_structure = {}
    for root, _, files in os.walk(directory):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(full_path, directory)
            
            # Ignora arquivos na lista de exclusão
            if any(exclude in relative_path for exclude in EXCLUDE):
                continue

            content = get_file_content(full_path)
            if content is not None:
                project_structure[relative_path] = content
    return project_structure


def format_file_section(file_path, content):
    """Formata a seção de um arquivo no README"""
    section = f"## {file_path}\n\n"
    section += "```python\n"
    section += content
    section += "\n```\n"
    return section


def generate_readme():
    """Gera o arquivo README.md para documentar o projeto"""
    # Lista os arquivos do projeto
    project_files = list_files(".")

    # Cria o índice com os nomes dos arquivos
    sections = list(project_files.keys())
    index = create_index(sections)

    # Gera o conteúdo para cada arquivo
    content_sections = []
    for file_path, content in project_files.items():
        content_sections.append(format_file_section(file_path, content))

    # Cria o conteúdo completo do README
    readme_content = f"# Documentação do Projeto\n\n"
    readme_content += "Este README foi gerado automaticamente para documentar a estrutura do projeto.\n\n"
    readme_content += index + "\n\n"
    readme_content += "\n".join(content_sections)

    # Salva o README no arquivo
    with open(README_PATH, "w", encoding="utf-8") as readme_file:
        readme_file.write(readme_content)

    print(f"README.md gerado com sucesso em {os.path.abspath(README_PATH)}")


if __name__ == "__main__":
    generate_readme()
