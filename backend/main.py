from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app =FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client =OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

@app.post("/chat")
async def chat(request:dict):
    question=request.get("question","")

    response=client.chat.completions.create(
        model="deepseek-chat",
        messages=[
        {"role":"user","content":question}
        ],
        temperature=0.7
    )

    return {"answer":response.choices[0].message.content}

@app.get("/health")
async def health():
    return{"status":"ok"}