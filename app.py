from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import json, pickle, db
from functools import lru_cache
from pathlib import Path

# —–– LangChain + FAISS (CPU) for RAG —–
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings          # modern import
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# —–– Transformers (FP16, **no** bitsandbytes) —–
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- tenant header ----------
class QueryRequest(BaseModel):
    question: str

async def get_tenant_id(x_api_key: str = Header(...)):
    tenant = db.get_tenant(x_api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant                    # {"id": "...", "faq_path": "..."}

# ---------- embeddings & index ----------
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@lru_cache(maxsize=32)
def load_index_for(tenant_id: str, faq_json: str) -> FAISS:
    tdir = Path(f"data/{tenant_id}")
    ipath, mpath = tdir / "faiss.index", tdir / "meta.pkl"

    if ipath.exists() and mpath.exists():
        return FAISS.load_local(str(ipath),
                                EMBEDDINGS,
                                allow_dangerous_deserialization=True  # WE TRUST OUR OWN FILE
                                )

    if not Path(faq_json).exists():
        raise FileNotFoundError(f"FAQ file not found: {faq_json}")

    with open(faq_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    docs = [
        Document(
            page_content=f"Q: {it['question']} A: {it['answer']}",
            metadata={"question": it["question"]},
        )
        for it in items
    ]

    index = FAISS.from_documents(docs, EMBEDDINGS)
    tdir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(ipath))
    with open(mpath, "wb") as f:
        pickle.dump({"count": len(docs)}, f)

    return index

# ---------- load Llama-3.1 8B (FP16) ----------
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,  # FP16 weights (≈15 GB VRAM on a T4)
    trust_remote_code=True
)
model.eval()

# ---------- prompt ----------
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

# ---------- /ask ----------
@app.post("/ask")
def ask_question(payload: QueryRequest, tenant=Depends(get_tenant_id)):
    query = payload.question
    index = load_index_for(tenant["id"], tenant["faq_path"])

    docs = index.similarity_search(query, k=2)
    context = "\n".join(d.page_content for d in docs) if docs else ""

    prompt = prompt_template.format(context=context, question=query)

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

    if not context:
        return {"answer": "I’m sorry, I don’t know the answer to that."}
    return {"answer": answer.strip()}

# ---------- health ----------
@app.get("/")
def health():
    return {"status": "OK"}

