from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json

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
llm = LlamaCpp(
    model_path= r"C:\Users\thoma\OneDrive\Documents\RAG_app\models\llama-2-7b-chat.Q4_K_M.gguf",  # adjust path to your model
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


# --- FastAPI Request Schema ---
class QueryRequest(BaseModel):
    question: str

# --- Endpoint ---
@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.question

    # Retrieve relevant docs
    relevant_docs = faiss_index.similarity_search(query, k=2)
    if not relevant_docs:
        return {"answer": "I'm sorry, I don't know the answer to that."}
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Construct prompt
    prompt = prompt_template.format(context=context, question=query)

    # Generate response
    answer = llm(prompt)
    return {"answer": answer.strip()}
