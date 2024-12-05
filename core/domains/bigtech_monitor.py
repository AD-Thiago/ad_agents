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