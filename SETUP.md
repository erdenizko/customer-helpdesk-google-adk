# Environment Setup

## Prerequisites

- Python 3.11+
- NeonDB account (https://neon.tech)
- Qdrant instance (local or cloud)
- OpenAI API key (for LiteLLM)

## 1. Clone Repository

git clone <repo-url>
cd customer-helpdesk

## 2. Create Virtual Environment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

## 3. Install Dependencies

pip install -r requirements.txt

## 4. Configure Environment

cp .env.example .env
# Edit .env with your credentials:
# - DATABASE_URL: NeonDB connection string
# - QDRANT_URL: Qdrant server URL
# - OPENAI_API_KEY: Your API key

## 5. Initialize Database

The app auto-creates tables on startup.
Or manually:
python -c "from src.customer_helpdesk.services.database import init_db; import asyncio; asyncio.run(init_db())"

## 6. Run the Application

uvicorn src.customer_helpdesk.main:app --reload --port 8000

## Testing

pytest tests/ -v

## Qdrant Collection Setup

The app auto-creates the collection on first run.
Collection: helpdesk_kb
Vector size: 1536 (OpenAI ada-002)
Distance: COSINE

## Troubleshooting

### Import errors
Make sure you're in the project root and the venv is activated.

### Database connection issues
Check DATABASE_URL format:
postgresql+asyncpg://user:pass@host/dbname?sslmode=require

### Qdrant connection issues
Ensure QDRANT_URL is accessible from your environment.

### LLM API errors
Verify OPENAI_API_KEY is valid and has available quota.
