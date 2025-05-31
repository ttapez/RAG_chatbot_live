from fastapi import FastAPI, Depends, Header, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import json
import pickle
import db  # your existing module that looks up tenants

from functools import lru_cache
from pathlib import Path

# === LangChain + FAISS (CPU) for RAG ===
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# === Hugging Face + bitsandbytes for Llama 3.1 8B (4-bit) inference ===
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# ------------------------ Setup FastAPI ------------------------
app = FastAPI()

# Serve widget.js (and any other static assets) under /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow CORS so your widget (served from localhost:8000 or another domain) can POST
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # In production, narrow this to your trusted domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],      # must allow X-API-Key
)

# ------------------------ Request / Tenant Logic ------------------------
class QueryRequest(BaseModel):
    question: str

async def get_tenant_id(x_api_key: str = Header(...)):
    """
    Dependency: Extracts 'X-API-Key' header, looks up tenant in db.py.
    If invalid, raises 401.
    """
    tenant = db.get_tenant(x_api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant  # should be a dict with keys "id" and "faq_path"

# ------------------------ FAISS Embeddings & Indexing ------------------------

# Use a small embedding model for FAQ‐vectorization
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@lru_cache(maxsize=32)
def load_index_for(tenant_id: str, faq_json_path: str) -> FAISS:
    """
    Return (and cache) the FAISS index for this tenant.
    If it doesn’t exist, build it from the FAQ JSON on disk.
    """
    tenant_dir = Path(f"data/{tenant_id}")
    index_path = tenant_dir / "faiss.index"
    meta_path = tenant_dir / "meta.pkl"

    # 1) If already built on disk, load and return
    if index_path.exists() and meta_path.exists():
        index = FAISS.load_local(str(index_path), EMBEDDINGS)
        return index

    # 2) Otherwise, build from JSON
    if not Path(faq_json_path).exists():
        raise FileNotFoundError(f"FAQ file not found: {faq_json_path}")

    with open(faq_json_path, "r", encoding="utf-8") as f:
        faq_items = json.load(f)

    docs = [
        Document(
            page_content=f"Q: {item['question']} A: {item['answer']}",
            metadata={"question": item["question"]},
        )
        for item in faq_items
    ]

    index = FAISS.from_documents(docs, EMBEDDINGS)

    # Persist to disk
    tenant_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump({"count": len(docs)}, f)

    return index

# ------------------------ Load Hugging Face Llama 3.1 8B ------------------------

# MODEL_NAME can be changed to any HF‐hosted 8B model (e.g. "meta-llama/Llama-3.1-8B-Instruct")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# 4-bit quantized tokenizer + model; device_map="auto" will place layers on GPU/CPU
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)

model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.eval()

# ------------------------ Prompt Template ------------------------

prompt_template = PromptTemplate.from_template(
    """
SYSTEM:
You are a concise e-commerce FAQ assistant.
• Only answer the customer’s question once.
• Do NOT invent new information.
• If the answer isn’t found in the FAQ context, reply “I’m sorry, I don’t know the answer to that.”
• Stop after the first sentence or paragraph.

FAQ CONTEXT (may be empty):
{context}

CUSTOMER QUESTION:
{question}

ASSISTANT ANSWER:
"""
)

# ------------------------ `/ask` Endpoint ------------------------

@app.post("/ask")
def ask_question(
    payload: QueryRequest,
    tenant = Depends(get_tenant_id)
):
    """
    1) Retrieve tenant (dict with "id" and "faq_path") from `get_tenant_id`.
    2) Load or build FAISS index for that tenant’s FAQ JSON.
    3) Perform similarity search to get top-K relevant docs.
    4) Build a prompt combining the retrieved context + user question.
    5) Run the Hugging Face Llama 3.1 8B model (4-bit) to generate an answer.
    6) Return {"answer": <string>}.
    """
    query = payload.question

    # 1) Load / cache FAISS index
    index = load_index_for(tenant["id"], tenant["faq_path"])

    # 2) Similarity search (K=2)
    relevant_docs = index.similarity_search(query, k=2)
    context = "\n".join(d.page_content for d in relevant_docs) if relevant_docs else ""

    # 3) Build the final prompt
    prompt = prompt_template.format(context=context, question=query)

    # 4) Tokenize & generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.4,
            do_sample=False,
            early_stopping=True,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 5) If no context was found, return fallback
    if not context:
        return {"answer": "I’m sorry, I don’t know the answer to that."}

    return {"answer": answer.strip()}

# ------------------------ Root / Health‐check (Optional) ------------------------
@app.get("/")
def health_check():
    return {"status": "OK"}

