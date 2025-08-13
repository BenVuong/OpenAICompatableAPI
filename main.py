import json
import time
import uuid
from typing import Dict, List, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator


from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langgraphAgent import streamLanggraph



app = FastAPI(
    title="OpenAI-Compatible API with LangChain",
    description="A custom FastAPI server to emulate the OpenAI Chat Completions API.",
)


class MessageContent(BaseModel):
    type: str
    text: str = None
    image_url: Dict[str, str] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]

    @field_validator('content',mode="before")
    @classmethod
    def check_content_length(cls, v):
        """
        Validator to ensure that if content is a list, it is not empty.
        This replaces the functionality of conlist(..., min_length=1).
        """
        if isinstance(v, list) and not v:
            raise ValueError("Content list must not be empty.")
        return v

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = 1024
    stream: bool = False
    temperature: float = 0.7

class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "your-langchain-model"
    choices: List[ChatChoice]
    usage: Usage = Field(default_factory=Usage)


class DeltaMessage(BaseModel):
    role: str = "assistant"
    content: str = ""

class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str = None

class ChatCompletionChunkResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "your-langchain-model"
    choices: List[ChatCompletionChunkChoice]

@app.post("//chat/completions")
async def chat_completions_endpoint(request: ChatCompletionRequest):
    """
    Handles OpenAI-compatible chat completion requests, including text and images,
    and now supports streaming responses.
    """
    print(f"Received request for model: {request.model}, stream: {request.stream}")
    
    for message in request.messages:
        if message.role == "system" and message.content != "[Start a new Chat]":
            threadID = message.content
            print(f"Found System message. Thread ID: {threadID}")

    # Extract content from the user message
    # We only process the last message, assuming it's the current user prompt.
    user_message = request.messages[-1]
    
    # Process the messages to create the LangChain input
    langchain_messages = []
    
    # The OpenAI format for multi-modal messages is a list of content blocks.
    # We need to iterate through these and format them for LangChain.
    # The `user_message.content` can be either a string or a list of `MessageContent` objects.
    if isinstance(user_message.content, str):
        # Handle the case where the content is a simple text string
        langchain_messages.append({"type": "text", "text": user_message.content})
    elif isinstance(user_message.content, list):
        for part in user_message.content:
            if part.type == "text":
                print(f"Processing text content: {part.text}")
                langchain_messages.append({"type": "text", "text": part.text})
            elif part.type == "image_url":
                # Extract the URL from the nested dictionary
                image_url = part.image_url.get("url")
                
                if image_url:
                    if image_url.startswith("data:image"):
                        print("Processing base64 image content.")
                        # LangChain's model will handle the decoding from the data URI
                        langchain_messages.append({"type": "image_url", "image_url": {"url": image_url}})
                        print("Image content prepared for LangChain.")
                    else:
                        # For simplicity, this example only handles base64 images.
                        # To support URLs, you would need to add logic to fetch the image.
                        raise HTTPException(
                            status_code=400,
                            detail="Only base64-encoded images are supported in this implementation for now."
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="The 'image_url' object must contain a 'url' key."
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported message content type: {part.type}"
                )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content format: {type(user_message.content)}"
        )
    
    # Create the HumanMessage for LangChain
    langchain_input = [HumanMessage(content=langchain_messages)]
    
    # Check if the user requested a streaming response
    if request.stream:
        def generate_stream_response():
            """
            An async generator to stream the response chunks.
            """
            # Create a unique ID for this conversation
            response_id = f"chatcmpl-{uuid.uuid4()}"
            creation_time = int(time.time())
            
            # First chunk: initial message with role
            chunk_data = ChatCompletionChunkResponse(
                id=response_id,
                created=creation_time,
                model=request.model,
                choices=[ChatCompletionChunkChoice(
                    delta=DeltaMessage(role="assistant"),
                )]
            ).model_dump_json(exclude_unset=True)
            yield f"data: {chunk_data}\n\n"
            
            # Stream the response from the LangChain model
            try:
                for text_chunk in streamLanggraph(langchain_input, threadID):
                    # Each chunk from astream is a string
                    if text_chunk:
                        chunk_data = ChatCompletionChunkResponse(
                            id=response_id,
                            created=creation_time,
                            model=request.model,
                            choices=[ChatCompletionChunkChoice(
                                delta=DeltaMessage(content=text_chunk),
                            )]
                        ).model_dump_json(exclude_unset=True)
                        yield f"data: {chunk_data}\n\n"
            except Exception as e:
                print(f"An error occurred with LangChain streaming: {e}")
                # Send an error chunk
                error_data = json.dumps({"error": str(e)})
                yield f"data: {error_data}\n\n"
            
            # Final chunk: finish reason
            chunk_data = ChatCompletionChunkResponse(
                id=response_id,
                created=creation_time,
                model=request.model,
                choices=[ChatCompletionChunkChoice(
                    delta=DeltaMessage(),
                    finish_reason="stop"
                )]
            ).model_dump_json(exclude_unset=True)
            yield f"data: {chunk_data}\n\n"

            # End of stream signal
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(generate_stream_response(), media_type="text/event-stream")
    
    # Non-streaming fallback for compatibility
    else:
        try:
            print("Invoking LangChain model (non-streaming)...")
            response_text = "".join(streamLanggraph(langchain_input, threadID))
            print("Received response from LangChain.")
        except Exception as e:
            print(f"An error occurred with LangChain: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred with the LangChain model: {e}"
            )

        # Create the OpenAI-compatible response
        completion_message = Message(role="assistant", content=response_text)
        choice = ChatChoice(index=0, message=completion_message)

        return ChatCompletionResponse(
            model=request.model,
            choices=[choice]
        )

if __name__ == "__main__":
    # To run the server, use: python your_script_name.py
    # or uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
