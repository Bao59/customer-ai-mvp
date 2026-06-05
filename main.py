from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from pipeline_adapter import get_pipeline


app = FastAPI(
    title="AI Customer Support Assistant MVP",
    description="AI 客服信件分析與回覆草稿系統",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class AnalyzeRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    message = payload.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    start_time = time.perf_counter()

    try:
        workflow = get_pipeline()
        result = workflow.run(message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - start_time

    return {
        "ok": True,
        "processing_time_seconds": round(elapsed, 2),
        "result": result.to_dict(),
    }