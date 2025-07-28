# app.py  — RAG FAQ chatbot (Llama-3.1-8B • 4-bit • FastAPI)
import json, pickle, os, torch
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# —— LangChain for RAG ——
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# —— Transformers / bitsandbytes ——
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ────────────────────────────────────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

async def get_tenant(x_api_key: str = Header(...)):
    import db
    tenant = db.get_tenant(x_api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant   # {"id": "...", "faq_path": "..."}

# ───────────────  Embeddings + FAISS  ───────────────
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
INDEX_ROOT = Path("data")

@lru_cache(maxsize=32)
def load_index_for(tid: str, faq_json: str) -> FAISS:
    tdir  = INDEX_ROOT / tid
    ipath = tdir / "faiss.index"

    if ipath.exists():
        return FAISS.load_local(
            str(ipath),
            EMBEDDINGS,
            allow_dangerous_deserialization=True,
        )

    if not Path(faq_json).exists():
        raise FileNotFoundError(f"FAQ file not found: {faq_json}")

    with open(faq_json, encoding="utf-8") as f:
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
    with open(tdir / "meta.pkl", "wb") as f:
        pickle.dump({"count": len(docs)}, f)

    return index

# ───────────────  Model (4-bit)  ───────────────
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
).eval()

# ───────────────  Prompt  ───────────────
PROMPT_TMPL = PromptTemplate.from_template(
    """SYSTEM:
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

STOP_TOKENS = tokenizer.encode("\nCUSTOMER", add_special_tokens=False)

# ───────────────  Route  ───────────────
@app.post("/ask")
def ask(payload: QueryRequest, tenant=Depends(get_tenant)):
    q      = payload.question.strip()
    index  = load_index_for(tenant["id"], tenant["faq_path"])
    docs   = index.similarity_search(q, k=1)
    ctx    = "\n".join(d.page_content for d in docs) if docs else ""

    prompt = PROMPT_TMPL.format(context=ctx, question=q)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
        )[0]

    # drop prompt tokens
    ans_ids = out_ids[inputs.input_ids.shape[-1]:]

    # stop at first "\nCUSTOMER"
    for i in range(len(ans_ids)):
        if ans_ids[i : i + len(STOP_TOKENS)].tolist() == STOP_TOKENS:
            ans_ids = ans_ids[:i]
            break

    answer = tokenizer.decode(ans_ids, skip_special_tokens=True).strip()
    answer = answer.splitlines()[0].strip() or \
             "I’m sorry, I don’t know the answer to that."

    return {"answer": answer}

@app.get("/")
def health():
    return {"status": "OK"}

