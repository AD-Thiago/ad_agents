# File: core/workflow.py

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
from core.constants import WorkflowStatus, ROUTING_KEYS, EXCHANGES
from core.rabbitmq_utils import RabbitMQUtils

logger = logging.getLogger(__name__)

class WorkflowManager:
    """Gerenciador central de workflows"""

    def __init__(self):
        self.rabbitmq = RabbitMQUtils()
        self.active_workflows: Dict[str, Dict] = {}

    async def initialize(self):
        """Inicializa o gerenciador"""
        await self.rabbitmq.initialize()

    async def create_workflow(self, trigger_data: Dict[str, Any]) -> str:
        """Cria um novo workflow"""
        try:
            workflow_id = str(uuid4())
            
            workflow_data = {
                'id': workflow_id,
                'status': WorkflowStatus.INITIATED,
                'trigger_data': trigger_data,
                'created_at': datetime.now(),
                'steps_completed': [],
                'current_step': None,
                'metadata': {}
            }
            
            self.active_workflows[workflow_id] = workflow_data
            
            # Publicar evento de início
            await self.rabbitmq.publish_event(
                routing_key=ROUTING_KEYS['workflow']['start'],
                message=workflow_data,
                exchange=EXCHANGES['workflow']
            )
            
            logger.info(f"Workflow {workflow_id} criado")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Erro ao criar workflow: {str(e)}")
            raise

    async def update_workflow_status(
        self,
        workflow_id: str,
        status: WorkflowStatus,
        metadata: Optional[Dict] = None
    ):
        """Atualiza status do workflow"""
        try:
            if workflow_id not in self.active_workflows:
                raise ValueError(f"Workflow {workflow_id} não encontrado")
                
            workflow = self.active_workflows[workflow_id]
            old_status = workflow['status']
            
            # Atualizar dados
            workflow['status'] = status
            workflow['last_updated'] = datetime.now()
            if metadata:
                workflow['metadata'].update(metadata)
            
            if status == WorkflowStatus.COMPLETED:
                workflow['completed_at'] = datetime.now()
                
            # Publicar atualização
            await self.rabbitmq.publish_event(
                routing_key=f"workflow.status_update",
                message={
                    'workflow_id': workflow_id,
                    'old_status': old_status,
                    'new_status': status,
                    'metadata': metadata
                },
                exchange=EXCHANGES['status']
            )
            
            logger.info(f"Workflow {workflow_id} atualizado: {status}")
            
        except Exception as e:
            logger.error(f"Erro ao atualizar workflow: {str(e)}")
            raise

    async def complete_workflow_step(
        self,
        workflow_id: str,
        step: str,
        result: Dict[str, Any]
    ):
        """Marca uma etapa do workflow como concluída"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} não encontrado")
                
            # Atualizar workflow
            workflow['steps_completed'].append(step)
            workflow['results'][step] = result
            
            # Determinar próxima etapa
            next_step = self._determine_next_step(workflow, step)
            if next_step:
                await self._trigger_next_step(workflow_id, next_step)
            else:
                await self.complete_workflow(workflow_id)
                
        except Exception as e:
            logger.error(f"Erro ao completar etapa do workflow: {str(e)}")
            raise

    async def handle_workflow_error(
        self,
        workflow_id: str,
        error: str,
        step: Optional[str] = None
    ):
        """Manipula erros no workflow"""
        try:
            await self.update_workflow_status(
                workflow_id,
                WorkflowStatus.FAILED,
                metadata={
                    'error': error,
                    'failed_step': step,
                    'failed_at': datetime.now().isoformat()
                }
            )
            
            # Publicar erro
            await self.rabbitmq.publish_event(
                routing_key=ROUTING_KEYS['workflow']['failed'],
                message={
                    'workflow_id': workflow_id,
                    'error': error,
                    'step': step
                },
                exchange=EXCHANGES['workflow']
            )
            
        except Exception as e:
            logger.error(f"Erro ao manipular erro do workflow: {str(e)}")
            raise

    def _determine_next_step(self, workflow: Dict, current_step: str) -> Optional[str]:
        """Determina próxima etapa do workflow"""
        step_sequence = ['planning', 'search', 'content', 'review', 'action']
        try:
            current_idx = step_sequence.index(current_step)
            if current_idx < len(step_sequence) - 1:
                return step_sequence[current_idx + 1]
        except ValueError:
            pass
        return None

    async def _trigger_next_step(self, workflow_id: str, next_step: str):
        """Dispara próxima etapa do workflow"""
        try:
            workflow = self.active_workflows[workflow_id]
            
            # Atualizar status
            await self.update_workflow_status(
                workflow_id,
                WorkflowStatus(next_step.upper()),
                metadata={'current_step': next_step}
            )
            
            # Publicar evento para próxima etapa
            await self.rabbitmq.publish_event(
                routing_key=ROUTING_KEYS[next_step]['request'],
                message={
                    'workflow_id': workflow_id,
                    'previous_results': workflow.get('results', {}),
                    'trigger_data': workflow['trigger_data']
                },
                exchange=EXCHANGES['workflow']
            )
            
        except Exception as e:
            logger.error(f"Erro ao disparar próxima etapa: {str(e)}")
            raise