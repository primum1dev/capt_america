import os

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_top_k(query: str, corpus: list[str], top_k: int) -> list[str]:
    if not corpus:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus + [query])
    query_vec = matrix[-1]
    doc_vecs = matrix[:-1]

    sims = cosine_similarity(query_vec, doc_vecs).flatten()
    ranked_indices = sims.argsort()[::-1][:top_k]
    return [corpus[i] for i in ranked_indices if sims[i] > 0]


def get_client_and_model(provider: str, model: str):
    if provider == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        return OpenAI(api_key=key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")), model

    if provider == "qwen":
        key = os.getenv("QWEN_API_KEY")
        if not key:
            raise ValueError("QWEN_API_KEY is not set")
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        return OpenAI(api_key=key, base_url=base_url), model

    raise ValueError("Unsupported provider")


def generate_answer(query: str, context_chunks: list[str], provider: str, model: str) -> str:
    client, model_name = get_client_and_model(provider, model)
    context = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."

    prompt = (
        "You are a RAG assistant. Use the context to answer the user's question. "
        "If the context is insufficient, say what is missing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You provide accurate, concise answers grounded in context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content or ""
