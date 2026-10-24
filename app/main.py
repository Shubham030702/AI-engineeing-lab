from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.llm import generate_response

app = FastAPI(title="AI Engineering Lab")


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)


class PromptResponse(BaseModel):
    response: str
    provider: str 


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=PromptResponse)
def generate(request: PromptRequest):
    response = generate_response(request.prompt)

    return PromptResponse(response=response, provider="mock")