from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
from typing import Dict, Any
from langgraphAgent import streamLanggraph

app = FastAPI()

@app.get("/")
def root():
    return {"status": "LangGraph OpenAI-Compatible API is running"}

@app.post("//chat/completions")
async def chat_completions(request: Request):
    body: Dict[str, Any] = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Get the latest user message
    user_prompt = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            user_prompt = msg["content"]
            break

    if stream:
        # StreamingResponse in OpenAI SSE format
        def event_stream():
            for chunk in streamLanggraph(user_prompt):
                # Format according to OpenAI's streaming spec
                data = {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "delta": {"content": chunk},
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
                

            # Send final done message
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    else:
        # Non-streaming response
        full_reply = "".join(streamLanggraph(user_prompt))
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_reply},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        return response
