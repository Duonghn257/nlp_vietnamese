from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src import VietnamesePoem
from typing import Optional
import uvicorn
import json
import asyncio


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


@app.post("/generate/stream")
async def generate_text_stream(request: GenerateRequest):
    """Streaming endpoint that generates text word by word"""

    async def generate_stream():
        try:
            chunk_count = 0
            for text_chunk in poem_generator.streaming_generate_poem(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            ):
                chunk_count += 1
                print(f"API: Yielding chunk {chunk_count}: '{text_chunk}'")  # Debug

                # Send each chunk as a Server-Sent Event
                data = json.dumps({"chunk": text_chunk, "type": "content"})
                yield f"data: {data}\n\n"

                # Add a small delay to make streaming visible
                # await asyncio.sleep(0.1)

            print(f"API: Completed streaming with {chunk_count} chunks")  # Debug
            # Send completion signal
            done_data = json.dumps({"chunk": "", "type": "done"})
            yield f"data: {done_data}\n\n"

        except Exception as e:
            print(f"API: Error during streaming: {e}")  # Debug
            # Send error signal
            error_data = json.dumps({"chunk": str(e), "type": "error"})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",  # ✅ Sửa ở đây
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",  # ✅ Đảm bảo Content-Type đúng
            "Transfer-Encoding": "chunked",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
