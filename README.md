# OpenAICompatableAPI
A simple fast api server that is open-ai compatable that can be used in SillyTavern

## About
This is a fastapi server that emulates an OpenAI compatable api. It uses the //chat/completions api endpoint.
I built this so that I could integrate my langgraph agent code into a frontend like SillyTavern. With SillyTavern it already has 3D avatars and TTS available.
My Langgraph agent has access to tools that can it can use to assist you. With this you can incorpate your own LLM/Agent code logic.

## Demo Video
A demo video showcasing my langgraph agent code being integrated with SillyTavern frontend
[![YouTube](http://i.ytimg.com/vi/oZcm5u4xZHw/hqdefault.jpg)](https://www.youtube.com/watch?v=oZcm5u4xZHw)

## Use
to run the server run the command `uvicorn main:app --reload --port 5000`

## Integrate into SillyTavern
1. Go to API Connections and set API to Chat Completion
2. Set the Chat Compleition Source to Custom (OpenAI-compatible)
3. Set the Custom Endpoint as http://127.0.0.1:5000/
