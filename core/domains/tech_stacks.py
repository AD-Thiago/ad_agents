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