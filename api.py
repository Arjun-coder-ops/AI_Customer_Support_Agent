import os
import json
import logging
import asyncio
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

def background_train(train_cases, retrieval_cases):
    try:
        pipeline.train_and_index(train_cases, retrieval_cases)
        logger.info("Background training complete. Pipeline ready.")
    except Exception as e:
        logger.error(f"Error during background training: {e}")

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
            zip_ref.extractall(".")

    if os.path.exists(train_path):
        train_cases = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_cases.append(json.loads(line))
        
        # Convert schema for retrieval index
        retrieval_cases = []
        for case in train_cases:
            if "customer_initial_message" in case:
                retrieval_cases.append({
                    "case_id": case.get("conversation_id"),
                    "conversation_id": case.get("conversation_id"),
                    "customer_message": case.get("customer_initial_message", ""),
                    "historical_response": case.get("support_final_response", ""),
                    "turn_count": case.get("turn_count", 1),
                })
                
        logger.info(f"Loaded {len(train_cases)} cases. Starting background training task...")
        # Train in a background thread to prevent Uvicorn from blocking startup and failing Render health checks
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, background_train, train_cases, retrieval_cases)
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
