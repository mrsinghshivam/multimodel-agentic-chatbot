from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List
import os, json, redis, logging
from langchain_core.messages import AIMessage, HumanMessage
from ai_agent1 import get_response_from_ai_agent
from db import SessionLocal, Document, init_db
from io import BytesIO
from pdfminer.high_level import extract_text
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from rag_store import set_vector_store

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Redis
try:
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Connected to Redis successfully.")
except redis.exceptions.ConnectionError as e:
    logger.error("Failed to connect to Redis: %s", str(e))
    raise

# App
app = FastAPI(title="LangGraph AI Agent with Redis Session")
ALLOWED_MODEL_NAMES = ["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "gpt-4o-mini", "deepseek-coder:6.7b", "deepseek-moe:16x7b"]

init_db()

class RequestState(BaseModel):
    session_id: str
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

def serialize_messages(messages):
    return json.dumps([{ "type": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content } for m in messages])

def deserialize_messages(serialized):
    raw = json.loads(serialized)
    return [HumanMessage(content=m["content"]) if m["type"] == "human" else AIMessage(content=m["content"]) for m in raw]

def extract_pdf_text(content: bytes) -> str:
    return extract_text(BytesIO(content))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), use_rag: bool = Form(...)):
    if not use_rag:
        return {"status": "RAG disabled, skipping embedding."}

    session = SessionLocal()
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="ignore") if file.filename.endswith(".txt") else extract_pdf_text(content)
        text = text.replace("\x00", "")

        new_doc = Document(name=file.filename, content=text)
        session.add(new_doc)
        session.commit()
        logger.info(f"Document '{file.filename}' uploaded and saved.")

        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([text])

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store_instance = FAISS.from_documents(docs, embeddings)
        set_vector_store(vector_store_instance)

        logger.info(f"Vector store created with {len(docs)} chunks.")
        return {"status": "uploaded", "chunks": len(docs)}

    except Exception as e:
        session.rollback()
        logger.exception("Error uploading document")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")
    finally:
        session.close()

@app.get("/documents")
def list_documents():
    session = SessionLocal()
    try:
        docs = session.query(Document).all()
        return [{"id": d.id, "name": d.name} for d in docs]
    except Exception as e:
        logger.exception("Error listing documents")
        raise HTTPException(status_code=500, detail="Error fetching documents")
    finally:
        session.close()

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    session = SessionLocal()
    try:
        doc = session.query(Document).get(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        session.delete(doc)
        session.commit()
        logger.info(f"Document {doc_id} deleted.")
        return {"status": "deleted"}
    except Exception as e:
        logger.exception(f"Error deleting document ID {doc_id}")
        raise HTTPException(status_code=500, detail="Error deleting document")
    finally:
        session.close()

@app.post("/chat")
def chat_endpoint(request: RequestState):
    try:
        if request.model_name not in ALLOWED_MODEL_NAMES:
            return {"error": "Invalid model name."}
        if not request.messages:
            return {"error": "No messages provided."}

        redis_key = f"session:{request.session_id}"
        history = deserialize_messages(redis_client.get(redis_key)) if redis_client.exists(redis_key) else []

        for msg in request.messages:
            history.append(HumanMessage(content=msg))

        response = get_response_from_ai_agent(
            llm_id=request.model_name,
            query=history,
            allow_search=request.allow_search,
            system_prompt=request.system_prompt,
            provider=request.model_provider
        )

        if "response" in response:
            history.append(AIMessage(content=response["response"]))
            redis_client.set(redis_key, serialize_messages(history), ex=1800)

        return response

    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail="Error processing chat request")

@app.get("/ping")
def ping():
    try:
        redis_client.ping()
        return {"status": "ok"}
    except redis.exceptions.ConnectionError:
        logger.error("Redis unavailable")
        return {"status": "Redis unavailable"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
