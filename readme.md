# Customer Helpdesk Google ADK

A production-ready 3-agent customer helpdesk system built with Google ADK and FastAPI. Features parallel agent execution, RAG augmentation for technical queries, and NeonDB-backed session persistence.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Endpoint                        │
│                         POST /chat                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ParallelAgent                             │
│         ┌─────────────────┐   ┌─────────────────┐          │
│         │  Classifier     │   │  History Check  │          │
│         │  Agent           │   │  Agent          │          │
│         │  (billing/tech/  │   │  (user tickets  │          │
│         │   general)       │   │   + similar)    │          │
│         └────────┬────────┘   └────────┬────────┘          │
│                  │                      │                   │
│                  └──────────┬───────────┘                   │
│                             ▼                               │
│              session.state: {classifier_intent,             │
│                              history_context}              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Response Enhancer Agent                     │
│                                                              │
│  - Reads classifier_intent from session.state               │
│  - If technical: calls RAG (Qdrant) for KB docs             │
│  - If billing/general: skips RAG                             │
│  - Generates final response                                  │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

- **Parallel Execution**: Classifier and History Check agents run concurrently
- **Conditional RAG**: Only technical queries trigger knowledge base retrieval
- **Session Persistence**: NeonDB-backed DatabaseSessionService (not in-memory)
- **Circuit Breakers**: Tenacity retry decorators for external service calls
- **Structured Logging**: Correlation IDs via structlog

## Tech Stack

| Component | Technology |
|------------|------------|
| Framework | Google ADK + FastAPI |
| LLM | LiteLLM (OpenAI compatible) |
| Database | NeonDB (PostgreSQL) |
| Vector Store | Qdrant |
| Session Storage | NeonDB (custom DatabaseSessionService) |

## Quick Start

### Prerequisites

- Python 3.11+
- NeonDB account (https://neon.tech)
- Qdrant instance (local or cloud)
- OpenAI API key (or compatible)

### Installation

```bash
# Clone and install
git clone <repo-url>
cd customer-helpdesk-google-adk
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host/dbname?sslmode=require

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=helpdesk_kb

# LLM
OPENAI_API_KEY=sk-...
BASIC_MODEL=openai/gpt-4o-mini
COMPLEX_MODEL=openai/gpt-4o

# App
APP_NAME=customer_helpdesk
LOG_LEVEL=INFO
```

### Running

```bash
# Development
uvicorn src.customer_helpdesk.main:app --reload --port 8000

# Production
uvicorn src.customer_helpdesk.main:app --host 0.0.0.0 --port 8000
```

## API Reference

### POST /chat

Send a chat message and receive a response.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need help with my billing"}
  ],
  "session_id": "optional-session-id",
  "user_id": "optional-user-id"
}
```

**Response:**
```json
{
  "response": "I'd be happy to help with your billing question...",
  "session_id": "generated-or-provided-session-id"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{"status": "healthy"}
```

## Project Structure

```
src/customer_helpdesk/
├── __init__.py
├── main.py              # FastAPI app with /chat and /health endpoints
├── agent.py             # ParallelAgent orchestration
├── config.py            # Pydantic settings
├── logging_config.py    # Structured logging with structlog
├── agents/
│   ├── __init__.py
│   ├── classifier/
│   │   ├── __init__.py
│   │   └── agent.py     # Intent classification (billing/technical/general)
│   ├── history_check/
│   │   ├── __init__.py
│   │   └── agent.py     # User history + similar tickets lookup
│   └── response_enhancer/
│       ├── __init__.py
│       └── agent.py     # RAG-augmented response generation
├── services/
│   ├── __init__.py
│   ├── database.py      # NeonDB async service (SQLAlchemy 2.0)
│   ├── models.py       # SQLAlchemy models (User, Ticket, Interaction)
│   ├── session_service.py  # DatabaseSessionService for ADK
│   └── vector_store.py  # Qdrant async service
└── tools/
    ├── __init__.py
    └── helpers.py       # Shared tool utilities

tests/
├── test_agents/         # Unit tests for each agent
├── test_services/       # Integration tests for services
└── test_pipeline/       # End-to-end pipeline tests
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_agents/test_classifier.py -v

# Run with coverage
pytest tests/ --cov=src/customer_helpdesk --cov-report=html
```

## Agent Details

### Classifier Agent
- **Purpose**: Classify user queries into intent categories
- **Output**: `{"intent": "billing" | "technical" | "general"}`
- **Model**: Basic model (fast, cheap)
- **Output Key**: `classifier_intent`

### History Check Agent
- **Purpose**: Look up user's ticket history and similar resolved issues
- **Tools**: `lookup_user_history`, `lookup_similar_issues`
- **Output Key**: `history_context`

### Response Enhancer Agent
- **Purpose**: Generate helpful response using available context
- **RAG**: Only triggered for `technical` intent
- **Model**: Complex model (better reasoning)
- **Output Key**: `final_response`

## Session Management

This implementation uses a custom `DatabaseSessionService` that persists sessions to NeonDB, not the default `InMemorySessionService`. This ensures:

- Session persistence across restarts
- Horizontal scalability
- Audit trail of conversations

## License

MIT