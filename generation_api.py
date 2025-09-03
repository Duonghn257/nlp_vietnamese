from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src import VietnamesePoem
from typing import Optional
import uvicorn

app = FastAPI(title="Vietnamese Poem Generator API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
poem_generator = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9


class GenerateResponse(BaseModel):
    generated_text: str


@app.on_event("startup")
async def startup_event():
    global poem_generator
    poem_generator = VietnamesePoem(config_path="config.yaml", device="cpu")


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    generated_text = poem_generator.generate_poem(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
    )

    return GenerateResponse(generated_text=generated_text)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
