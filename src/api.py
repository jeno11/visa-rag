"""FastAPI service: POST / ask returns a grounded answer with gov.uk citations."""
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.ask import snapshot_date
from src.bm25 import BM25Retriever
from src.generate import AnswerGenerator
from src.hybrid import HybridRetriever
from src.search import Retriever

app = FastAPI(title="UK Visa RAG Assistant")
logger = logging.getLogger("uvicorn.error")

# Created once when the server starts, then reused for every request
retriever = HybridRetriever([BM25Retriever(), Retriever()])
generator = AnswerGenerator()
SNAPSHOT = snapshot_date()

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)

class Citation(BaseModel):
    n: int
    title_path: str
    url: str

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    not_found: bool
    snapshot_date: str

@app.get("/health")
def health():
    return {"status": "ok", "snapshot_date": SNAPSHOT}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        chunks = retriever.search(req.question, k=5)
        result = generator.answer(req.question, chunks)
    except Exception as e:
        logger.exception("Answering failed for question: %r", req.question)
        raise HTTPException(status_code=502, detail="The answer service failed. Please try again.")
    return AskResponse(**result, snapshot_date=SNAPSHOT)