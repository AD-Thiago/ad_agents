from typing import Dict, List
from pydantic import BaseModel, Field 
from datetime import datetime
from langchain_openai import ChatOpenAI
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