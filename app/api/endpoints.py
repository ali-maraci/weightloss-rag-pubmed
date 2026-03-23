import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.chat_engine import ChatEngine

router = APIRouter()
logger = logging.getLogger(__name__)

chat_engine = ChatEngine()

class QueryRequest(BaseModel):
    query: str
    session_id: str

@router.post("/chat")
def stream_chat_response(request: QueryRequest):
    """Core RAG Endpoint returning Server-Sent Events (SSE)."""
    logger.info(f"API Request (Stream) - Session: {request.session_id} | Query: {request.query}")
    try:
        return StreamingResponse(
            chat_engine.stream_chat(request.query, session_id=request.session_id),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error processing query.")

@router.get("/health")
def health_check():
    return {"status": "WeightLoss RAG API is ready."}
