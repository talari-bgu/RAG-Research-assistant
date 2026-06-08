import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.chain import ask
from src.config import settings
from src.ingest import ingest as run_ingest

app = FastAPI(title="RAG Research Assistant", version="1.0.0")

_UI = Path(__file__).parent.parent / "ui" / "index.html"


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(_UI)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    pdf_dir: Path = settings.pdf_dir
    pdf_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a PDF")
        dest = pdf_dir / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)
    return {"status": "ok", "files": saved}


@app.post("/ingest")
def ingest_documents():
    try:
        run_ingest()
        return {"status": "ok", "message": "Ingestion complete."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        answer = ask(request.question)
        return QueryResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
