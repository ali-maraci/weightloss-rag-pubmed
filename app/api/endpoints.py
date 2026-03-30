import json
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.chat_engine import ChatEngine
from app.core.safety import SafetyClassifier

router = APIRouter()
logger = logging.getLogger(__name__)

chat_engine = ChatEngine()
safety_classifier = SafetyClassifier()

class QueryRequest(BaseModel):
    query: str
    session_id: str
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    publication_types: Optional[List[str]] = None
    human_only: Optional[bool] = None

@router.post("/chat")
def stream_chat_response(request: QueryRequest):
    """Core RAG Endpoint returning Server-Sent Events (SSE)."""
    logger.info(f"API Request (Stream) - Session: {request.session_id} | Query: {request.query}")
    try:
        level, disclaimer = safety_classifier.classify_and_disclaim(request.query)

        filters = {k: v for k, v in {
            "year_min": request.year_min,
            "year_max": request.year_max,
            "publication_types": request.publication_types,
            "human_only": request.human_only,
        }.items() if v is not None}

        def generate():
            if disclaimer:
                yield json.dumps({"type": "safety", "level": level.value, "disclaimer": disclaimer}) + "\n\n"
            yield from chat_engine.stream_chat(request.query, session_id=request.session_id, filters=filters or None)

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error processing query.")

@router.get("/health")
def health_check():
    return {"status": "WeightLoss RAG API is ready."}

@router.post("/query")
def structured_query(request: QueryRequest):
    """Non-streaming endpoint returning a StructuredAnswer JSON."""
    logger.info(f"API Request (Structured) - Session: {request.session_id} | Query: {request.query}")
    try:
        level, disclaimer = safety_classifier.classify_and_disclaim(request.query)
        filters = {k: v for k, v in {
            "year_min": request.year_min,
            "year_max": request.year_max,
            "publication_types": request.publication_types,
            "human_only": request.human_only,
        }.items() if v is not None}
        result = chat_engine.qa_chain.structured_query(request.query, filters=filters or None)
        data = result.model_dump()
        data["safety_level"] = level.value
        data["safety_disclaimer"] = disclaimer or None
        return data
    except Exception as e:
        logger.error(f"Error processing structured query: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error processing query.")
