from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from google.adk.runners import Runner
from .services.session_service import DatabaseSessionService
from google.genai import types
from contextlib import asynccontextmanager
import uuid
import logging

from .agent import root_agent
from .logging_config import configure_logging
from .config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Session service
session_service = DatabaseSessionService()


# Pydantic models
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Starting customer helpdesk API")
    yield
    logger.info("Shutting down customer helpdesk API")


app = FastAPI(title="Customer Helpdesk API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id or "anonymous"

    session = await session_service.get_session(
        app_name=settings.app_name,
        user_id=user_id,
        session_id=session_id,
    )
    if not session:
        session = await session_service.create_session(
            app_name=settings.app_name,
            user_id=user_id,
            session_id=session_id,
        )

    runner = Runner(
        agent=root_agent,
        app_name=settings.app_name,
        session_service=session_service,
    )

    content = types.Content(
        role="user", parts=[types.Part(text=m.content) for m in request.messages]
    )

    try:
        response_text = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            if event.is_final_response():
                for part in event.content.parts:
                    if part.text:
                        response_text += part.text

        return ChatResponse(response=response_text, session_id=session_id)

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}
