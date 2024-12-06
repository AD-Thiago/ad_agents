# File: agents/planning/services/validation.py

import logging
from typing import Dict, Any
from core.domains.definitions import DomainManager
from ..models import PlanningResponse, ContentStrategy

logger = logging.getLogger(__name__)

class PlanningValidator:
    """Serviço de validação de planos de conteúdo"""

    def __init__(self):
        self.domain_manager = DomainManager()

    async def validate_plan(self, plan: PlanningResponse) -> Dict[str, Any]:
        """
        Valida plano completo de conteúdo
        Retorna dict com status e detalhes da validação
        """
        try:
            validations = {
                "domain": await self._validate_domain_requirements(plan),
                "strategy": await self._validate_content_strategy(plan.strategy),
                "insights": await self._validate_insights(plan.insights)
            }

            is_valid = all(v.get("valid", False) for v in validations.values())

            return {
                "valid": is_valid,
                "details": validations,
                "workflow_id": plan.workflow_id
            }
        except Exception as e:
            logger.error(f"Erro na validação do plano: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "workflow_id": plan.workflow_id
            }

    async def _validate_domain_requirements(self, plan: PlanningResponse) -> Dict[str, Any]:
        """Valida requisitos de domínio"""
        try:
            domain = self.domain_manager.get_domain_for_topic(plan.topic)
            if not domain:
                return {"valid": False, "reason": "Domínio não encontrado"}

            guidelines = self.domain_manager.get_content_guidelines(domain.name)
            
            checks = {
                "expertise_level": plan.strategy.target_expertise in guidelines["expertise_levels"],
                "content_type": plan.strategy.content_type in guidelines["content_types"],
                "keywords": any(k in plan.keywords for k in guidelines["primary_keywords"])
            }

            return {
                "valid": all(checks.values()),
                "checks": checks,
                "domain": domain.name
            }
        except Exception as e:
            logger.error(f"Erro na validação de domínio: {str(e)}")
            return {"valid": False, "error": str(e)}

    async def _validate_content_strategy(self, strategy: ContentStrategy) -> Dict[str, Any]:
        """Valida estratégia de conteúdo"""
        try:
            checks = {
                "has_structure": bool(strategy.content_structure),
                "has_metrics": bool(strategy.success_metrics),
                "valid_impact": 0 <= strategy.estimated_impact <= 1,
                "has_topics": bool(strategy.key_topics)
            }

            return {
                "valid": all(checks.values()),
                "checks": checks
            }
        except Exception as e:
            logger.error(f"Erro na validação de estratégia: {str(e)}")
            return {"valid": False, "error": str(e)}

    async def _validate_insights(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Valida insights coletados"""
        try:
            checks = {
                "has_market": bool(insights.market),
                "has_tech": bool(insights.tech),
                "has_seasonal": bool(insights.seasonal),
                "recent_timestamp": (datetime.now() - insights.timestamp).seconds < 3600
            }

            return {
                "valid": all(checks.values()),
                "checks": checks
            }
        except Exception as e:
            logger.error(f"Erro na validação de insights: {str(e)}")
            return {"valid": False, "error": str(e)}