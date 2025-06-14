# app.py  — RAG FAQ chatbot (Llama-3.1-8B, FAISS, FastAPI)
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import json, pickle, os, torch
from functools import lru_cache
from pathlib import Path

# —— LangChain for embedding / retrieval ——
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# —— Transformers Llama-3.1 8B (FP16) ——
from transformers import AutoTokenizer, LlamaForCausalLM

# --------------------------------------------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
# --------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str

async def get_tenant(x_api_key: str = Header(...)):
    import db
    tenant = db.get_tenant(x_api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant               # {"id": "...", "faq_path": "..."}

# --------------------------------------------------------------------
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
INDEX_ROOT = Path("data")

@lru_cache(maxsize=32)
def load_index_for(tenant_id: str, faq_json: str) -> FAISS:
    tdir = INDEX_ROOT / tenant_id
    ipath = tdir / "faiss.index"

    if ipath.exists():
        # safe because the file is created by **our** code
        return FAISS.load_local(
            str(ipath),
            EMBEDDINGS,
            allow_dangerous_deserialization=True,
        )

    # build a new index from the tenant’s JSON
    if not Path(faq_json).exists():
        raise FileNotFoundError(f"FAQ file not found: {faq_json}")

    with open(faq_json, encoding="utf-8") as f:
        faq_items = json.load(f)

    docs = [
        Document(
            page_content=f"Q: {it['question']} A: {it['answer']}",
            metadata={"question": it["question"]},
        )
        for it in faq_items
    ]

    index = FAISS.from_documents(docs, EMBEDDINGS)
    tdir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(ipath))

    # tiny metadata file (count) – not mandatory but handy
    with open(tdir / "meta.pkl", "wb") as f:
        pickle.dump({"count": len(docs)}, f)

    return index

# --------------------------------------------------------------------
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,          # FP16 fits on a T4 GPU
    trust_remote_code=True,
)
model.eval()

# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
@app.post("/ask")
def ask(payload: QueryRequest, tenant=Depends(get_tenant)):
    query     = payload.question.strip()
    index     = load_index_for(tenant["id"], tenant["faq_path"])
    docs      = index.similarity_search(query, k=2)
    context   = "\n".join(d.page_content for d in docs) if docs else ""

    prompt    = PROMPT_TMPL.format(context=context, question=query)

    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.4,
            top_p=0.9,
            do_sample=False,
            early_stopping=True,
        )[0]

    # keep **only** the newly generated tokens (drop the prompt part)
    answer_ids   = output_ids[inputs.input_ids.shape[-1]:]
    answer_text  = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    # collapse to first line / sentence
    answer_text  = answer_text.splitlines()[0].strip()

    if not answer_text:
        answer_text = "I’m sorry, I don’t know the answer to that."

    return {"answer": answer_text}

# simple health-check
@app.get("/")
def health():
    return {"status": "OK"}
