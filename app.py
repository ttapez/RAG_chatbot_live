from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
import os, json, pickle, db

from functools import lru_cache
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# --- Setup FastAPI ---
app = FastAPI()

# --- Serve static files (e.g., widget.js) ---
# Put widget.js in a folder named "static" next to app.py
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Load FAQ Data ---
with open('faq_data.json', 'r') as f:
    faq_list = json.load(f)

# Convert FAQ into list of strings
faq_texts = [f"Q: {item['question']} A: {item['answer']}" for item in faq_list]
faq_docs = [Document(page_content=text) for text in faq_texts]

# --- Embeddings and FAISS Vector Store ---
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_index = FAISS.from_documents(faq_docs, embedding_model)

# --- Load LLaMA Model (local .ggml/.gguf file) ---
llm = Llama(
    model_path= "/home/ubuntu/RAG_chatbot_live/models/llama-3.1-8b-chat.Q4_K_M.gguf",  # adjust path to your model
    use_gpu=True,
    n_gpu_layers=40,
    n_ctx=2048,
    temperature=0.4,
    max_tokens=120,   # keeps replies short
    stop=["\n\n"]     # stops generation at first blank line
)

# --- Prompt Template ---
prompt_template = PromptTemplate.from_template(
"""
SYSTEM:
You are a concise e‑commerce FAQ assistant.  
• Only answer the customer’s question once.  
• Do NOT invent new questions.  
• If the answer is not found in the FAQ context, do not attempt to make an answer up.  
• End your reply after the first sentence or paragraph.

FAQ CONTEXT (may be empty):
{context}

CUSTOMER QUESTION:
{question}

ASSISTANT ANSWER:
"""
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- API-key dependency ----------
async def get_tenant_id(x_api_key: str = Header(...)):
    tenant = db.get_tenant(x_api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant   # dict with id & faq_path


# --- FastAPI Request Schema ---
class QueryRequest(BaseModel):
    question: str

EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@lru_cache(maxsize=32)
def load_index_for(tenant_id: str, faq_json_path: str) -> FAISS:
    """
    Return (and cache) the FAISS index for this tenant.
    If it does not exist yet, build it from the FAQ JSON.
    """
    tenant_dir  = Path(f"data/{tenant_id}")
    index_path  = tenant_dir / "faiss.index"
    meta_path   = tenant_dir / "meta.pkl"

    # 1. Already on disk  ➜  load + return
    if index_path.exists() and meta_path.exists():
        index = FAISS.load_local(str(index_path), EMBEDDINGS)
        return index

    # 2. Build from FAQ JSON
    if not Path(faq_json_path).exists():
        raise FileNotFoundError(f"FAQ file not found: {faq_json_path}")

    with open(faq_json_path, "r", encoding="utf-8") as f:
        faq_items = json.load(f)

    docs = [
        Document(
            page_content=f"Q: {item['question']} A: {item['answer']}",
            metadata={"q": item["question"]},
        )
        for item in faq_items
    ]

    index = FAISS.from_documents(docs, EMBEDDINGS)

    # ensure tenant dir exists & persist
    tenant_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump({"count": len(docs)}, f)

    return index

# --- Endpoint ---
@app.post("/ask")
def ask_question(
    request: QueryRequest,
    tenant = Depends(get_tenant_id)          # <-- injects dict
):
    query = request.question

    # 1) load / cache the correct FAISS index
    index = load_index_for(tenant["id"], tenant["faq_path"])

    # 2) similarity search
    relevant_docs = index.similarity_search(query, k=2)
    context = "\n".join(d.page_content for d in relevant_docs) if relevant_docs else ""

    # 3) generate
    prompt = prompt_template.format(context=context, question=query)
    answer = llm(prompt) if context else "I'm sorry, I don't know the answer to that."
    return {"answer": answer.strip()}
