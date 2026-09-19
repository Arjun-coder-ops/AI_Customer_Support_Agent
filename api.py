import os
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import SupportAgentPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback mock check. If GEMINI_API_KEY is not set, we force MOCK_LLM=true for safety.
mock_env = os.environ.get("MOCK_LLM", "false").lower() == "true"
if not os.environ.get("GEMINI_API_KEY"):
    mock_env = True
    logger.info("GEMINI_API_KEY not found. Forcing MOCK_LLM=true.")

pipeline = SupportAgentPipeline(brand_name="AmazonHelp", use_mock=mock_env)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load training data
    logger.info("Loading training data for pipeline...")
    train_path = "data/processed/train.jsonl"
    zip_path = "data/processed/train.jsonl.zip"
    
    if not os.path.exists(train_path) and os.path.exists(zip_path):
        logger.info(f"Extracting {zip_path}...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # We assume the zip contains data/processed/train.jsonl
            # The exact name inside the zip depends on how it was zipped.
            # Let's extract all just to be safe
            zip_ref.extractall(".")

    if os.path.exists(train_path):
        train_cases = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_cases.append(json.loads(line))
        logger.info(f"Loaded {len(train_cases)} cases. Training and indexing...")
        pipeline.train_and_index(train_cases, train_cases)
        logger.info("Pipeline ready.")
    else:
        logger.warning(f"Training data not found at {train_path}. Operating untrained (mock mode fallback).")
    yield

app = FastAPI(title="AI Customer Support Agent API", lifespan=lifespan)

# Allowed origins for CORS (Vercel frontend)
frontend_url = os.environ.get("FRONTEND_URL", "*")
origins = [frontend_url] if frontend_url != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(req: MessageRequest):
    try:
        result = pipeline.process_message(req.message)
        return result
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "ok", "mock_mode": mock_env, "is_trained": pipeline.is_trained}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
