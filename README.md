# Local Docker RAG Framework (Qwen + DeepSeek)

This repository provides a **local, Dockerized RAG API** with:
- User management (register/login with JWT)
- Document ingestion for **text, PDF, and images (OCR)**
- Retrieval over your private chunks
- LLM answering via **DeepSeek** or **Qwen** API (OpenAI-compatible)

## Stack
- FastAPI
- SQLite (per-container local DB)
- TF-IDF retrieval (scikit-learn)
- OCR with Tesseract

## Quick Start

1. Copy env vars:

```bash
cp .env.example .env
```

2. Set your keys in `.env`:
- `DEEPSEEK_API_KEY`
- `QWEN_API_KEY`
- `JWT_SECRET`

3. Start service:

```bash
docker compose up --build
```

4. Health check:

```bash
curl http://localhost:8000/health
```

## API Flow

### 1) Register

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"StrongPass123"}'
```

### 2) Upload docs (text/pdf/image)

```bash
TOKEN="<your_access_token>"
curl -X POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "files=@/path/to/manual.pdf" \
  -F "files=@/path/to/notes.txt" \
  -F "files=@/path/to/photo.jpg"
```

### 3) Query with provider/model

Use DeepSeek:

```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the warranty period?",
    "provider": "deepseek",
    "model": "deepseek-chat",
    "top_k": 5
  }'
```

Use Qwen:

```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize key installation steps.",
    "provider": "qwen",
    "model": "qwen-plus",
    "top_k": 5
  }'
```

## Notes
- Supported ingestion types: `.txt`, `.md`, `.csv`, `.log`, `.pdf`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`.
- Data is isolated per user.
- For production: swap SQLite for Postgres, add rate limits, rotate JWT secrets, and secure HTTPS.
