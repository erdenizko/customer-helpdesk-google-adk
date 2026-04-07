from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from google.adk.runners import Runner
from .services.session_service import DatabaseSessionService
from .services.validation import validate_final_response
from google.genai import types
from contextlib import asynccontextmanager
import uuid
import logging

from .agent import root_agent
from .logging_config import configure_logging
from .config import get_settings
from .models.errors import ErrorResponse, ErrorCode

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
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins)


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
            state={"user_id": user_id},
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

        try:
            validate_final_response(response_text)
        except Exception as validation_error:
            correlation_id = str(uuid.uuid4())[:8]
            logger.error(
                f"Validation failed [{correlation_id}]: {validation_error}",
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error_code=ErrorCode.VALIDATION_ERROR,
                    message="An error occurred while validating the response. Please try again.",
                    correlation_id=correlation_id,
                ).model_dump(),
            )

        return ChatResponse(response=response_text, session_id=session_id)

    except Exception as e:
        correlation_id = str(uuid.uuid4())[:8]
        logger.error(f"Agent error [{correlation_id}]: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code=ErrorCode.AGENT_ERROR,
                message="An error occurred while processing your request. Please try again.",
                correlation_id=correlation_id,
            ).model_dump(),
        )


@app.get("/health")
async def health():
    return {"status": "healthy"}
