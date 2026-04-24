# Copyright (c) 2026 Jernej Hribar

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from langchain_ollama import ChatOllama, OllamaEmbeddings

DATA_PATH = Path("Dataset/qa_rl_dataset.json")
LLM_MODEL = "llama3.2"
EMBEDDINGS_MODEL = "nomic-embed-text"

def load_dataset() -> List[Dict[str, Any]]:
    """Load the QA dataset from the JSON file."""
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def make_docs(rows: List[Dict[str, Any]]) -> List[str]:
    """Convert dataset rows into text documents for embedding."""
    docs = []
    for row in rows:
        docs.append(
            f"Question: {row['question']}\n"
            f"Answer: {row['answer']}\n"
            f"Keywords: {', '.join(row['keywords'])}\n"
            f"Source page: {row['source_page']}"
        )
    return docs


def build_embeddings(rows: List[Dict[str, Any]], model_name: str = EMBEDDINGS_MODEL):
    """Create embeddings for all dataset documents."""
    docs = make_docs(rows)
    emb = OllamaEmbeddings(model=model_name)
    doc_vectors = np.array(emb.embed_documents(docs), dtype=float)
    return emb, docs, doc_vectors


def print_hits(rows: List[Dict[str, Any]], hits: List[Dict[str, Any]]) -> None:
    """Print the top retrieved dataset matches."""
    print("\nTop retrieved items:")
    for rank, hit in enumerate(hits, start=1):
        row = rows[hit["index"]]
        print(
            f"{rank}. score={hit['score']:.4f} | "
            f"Q: {row['question']} | "
            f"A: {row['answer']} | "
            f"page={row['source_page']}"
        )


def grounded_answer(
    llm: ChatOllama,
    rows: List[Dict[str, Any]],
    hits: List[Dict[str, Any]],
    user_question: str
) -> str:
    """Generate an answer using only the retrieved context."""
    context = "\n\n".join(
        [
            f"Item {i+1}\n"
            f"Question: {rows[h['index']]['question']}\n"
            f"Answer: {rows[h['index']]['answer']}\n"
            f"Source page: {rows[h['index']]['source_page']}"
            for i, h in enumerate(hits)
        ]
    )

    prompt = f"""
You are answering only from a small QA dataset extracted from one chapter.
Use only the provided context.
If the answer is not supported by the context, say:
"I don't know from the retrieved dataset."

Context:
{context}

User question:
{user_question}
"""
    return llm.invoke(prompt).content


def semantic_search(query: str, emb: OllamaEmbeddings, doc_vectors: np.ndarray, k: int = 3):
    """Retrieve the most relevant documents for a query."""
    q = np.array(emb.embed_query(query), dtype=float)
    scores = doc_vectors @ q
    top_idx = np.argsort(scores)[::-1][:k]
    return [{"index": int(i), "score": float(scores[i])} for i in top_idx]


def main():
    """Run the interactive QA search and answer flow."""
    rows = load_dataset()
    emb, docs, doc_vectors = build_embeddings(rows)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    print("Ask a question about the dataset.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Enter your question: ").strip()

        if not query:
            print("Please enter a question.\n")
            continue

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        hits = semantic_search(query, emb, doc_vectors, k=3)

        print_hits(rows, hits)
        print("\nAnswer:")
        print(grounded_answer(llm, rows, hits, query))
        print()

if __name__ == "__main__":
    main()