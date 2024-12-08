from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from psycopg2 import connect
from psycopg2.extras import RealDictCursor
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
from core.config import get_settings
from agents.search.agent import SearchAgent
from agents.planning.agent import PlanningAgent
#from agents.content.agent import ContentAgent
from agents.review.agent import ReviewAgent

# Configurações
settings = get_settings()
app = FastAPI(
    title="Agents API",
    description="API para interação com agentes inteligentes",
    version="1.0.0"
)

# Modelos de dados para a API
class SearchRequest(BaseModel):
    topic: str
    keywords: List[str]
    target_audience: str
    max_results: Optional[int] = 10

class PlanningRequest(BaseModel):
    topic: str
    domain: str
    technical_level: str = "intermediate"
    content_type: str = "article"
    target_audience: List[str]


# Instâncias dos agentes
search_agent = SearchAgent()
planning_agent = PlanningAgent()
#content_agent = ContentAgent()
#review_agent = ReviewAgent()

# Endpoint existente para teste de DB
@app.get("/db-test")
async def db_test():
    try:
        conn = connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            dbname=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD
        )
        conn.close()
        return {"status": "success", "message": "Database connection is working!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Endpoint existente para search agent
@app.get("/search_agent")
async def search_agent_endpoint():
    try:
        await search_agent.start_consuming()
        return {"status": "success", "message": "Search agent started successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Novos endpoints para teste dos agentes
@app.post("/api/search", tags=["Search"])
async def run_search(request: SearchRequest):
    """Executa busca com o Search Agent"""
    try:
        results = await search_agent.enrich_content_plan(
            topic=request.topic,
            keywords=request.keywords,
            target_audience=request.target_audience
        )
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/plan", tags=["Planning"])
async def generate_plan(request: PlanningRequest):
    """Gera plano com o Planning Agent"""
    try:
        plan = await planning_agent.generate_plan()
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "plan": plan
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#@app.post("/api/content", tags=["Content"])
#async def generate_content(request: ContentRequest):
#    """Gera conteúdo com o Content Agent"""
#    try:
#        content = content_agent.generate_content({
#            "topic": request.topic,
#            "keywords": request.keywords,
#            "target_audience": request.target_audience,
#           "tone": request.tone,
#           "seo_guidelines": request.seo_guidelines or {}
#       })
#       return {
#           "status": "success",
#           "timestamp": datetime.now().isoformat(),
#            "content": content
#        }
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status", tags=["System"])
async def get_system_status():
    """Verifica status do sistema"""
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "agents": {
            "search": "running",
            "planning": "running",
            "content": "running",
            "review": "running"
        }
    }

# Eventos de inicialização e encerramento
@app.on_event("startup")
async def startup_event():
    """Inicializa os agentes quando a API inicia"""
    await search_agent.initialize()
    await planning_agent.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Fecha conexões quando a API é encerrada"""
    await search_agent.close()
    await planning_agent.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)