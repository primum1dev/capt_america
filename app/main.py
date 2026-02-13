from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from .auth import create_access_token, get_current_user, hash_password, verify_password
from .database import Base, engine, get_db
from .ingestion import chunk_text, extract_text
from .models import Chunk, Document, User
from .rag import generate_answer, retrieve_top_k
from .schemas import ChatRequest, ChatResponse, LoginRequest, RegisterRequest, TokenResponse, UploadResponse

app = FastAPI(title="Local Docker RAG API", version="0.1.0")


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/auth/register", response_model=TokenResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=payload.email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    token = create_access_token(user.email)
    return TokenResponse(access_token=token)


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user.email)
    return TokenResponse(access_token=token)


@app.post("/documents/upload", response_model=UploadResponse)
def upload_documents(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    total_chunks = 0
    ingested_docs = 0

    for file in files:
        content = file.file.read()
        try:
            text, source_type = extract_text(file.filename, content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        chunks = chunk_text(text)
        document = Document(filename=file.filename, source_type=source_type, owner_id=current_user.id)
        db.add(document)
        db.flush()

        for chunk in chunks:
            db.add(Chunk(document_id=document.id, owner_id=current_user.id, content=chunk))

        total_chunks += len(chunks)
        ingested_docs += 1

    db.commit()
    return UploadResponse(documents_ingested=ingested_docs, chunks_created=total_chunks)


@app.post("/chat/query", response_model=ChatResponse)
def chat_query(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    chunks = db.query(Chunk).filter(Chunk.owner_id == current_user.id).all()
    corpus = [c.content for c in chunks]

    context_chunks = retrieve_top_k(payload.query, corpus, payload.top_k)
    try:
        answer = generate_answer(
            query=payload.query,
            context_chunks=context_chunks,
            provider=payload.provider,
            model=payload.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ChatResponse(answer=answer, context_chunks=context_chunks)
