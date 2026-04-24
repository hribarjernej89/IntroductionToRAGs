# Copyright (c) 2026 Jernej Hribar

import json
import re
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Literal

import numpy as np
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, END

DATA_PATH = Path("Dataset/qa_rl_dataset.json")
LLM_MODEL = "llama3.2"
EMBEDINGS_MODEL = "nomic-embed-text"


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


def build_embeddings(rows: List[Dict[str, Any]], model_name: str = EMBEDINGS_MODEL):
    """Create normalized embeddings for all dataset documents."""
    docs = make_docs(rows)
    emb = OllamaEmbeddings(model=model_name)
    doc_vectors = np.array(emb.embed_documents(docs), dtype=float)

    norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    doc_vectors = doc_vectors / np.clip(norms, 1e-12, None)

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


def normalize_text(text: str) -> str:
    """Normalize text for simple keyword matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_score(query: str, row: Dict[str, Any]) -> float:
    """Score a row using exact phrase and token overlap matching."""
    q = normalize_text(query)
    question = normalize_text(row["question"])
    answer = normalize_text(row["answer"])
    keywords = normalize_text(" ".join(row["keywords"]))

    score = 0.0

    if q and q in question:
        score += 5.0
    if q and q in answer:
        score += 3.0
    if q and q in keywords:
        score += 4.0

    query_tokens = set(q.split())
    row_tokens = set((question + " " + answer + " " + keywords).split())
    overlap = len(query_tokens & row_tokens)
    score += overlap * 0.5

    return score


def expand_query(query: str) -> List[str]:
    """Expand the query with useful acronym or full-name variants."""
    variants = [query]
    q_lower = query.lower()

    if "ppo" in q_lower and "proximal policy optimization" not in q_lower:
        variants.append(query + " Proximal Policy Optimization")

    if "proximal policy optimization" in q_lower and "ppo" not in q_lower:
        variants.append(query + " PPO")

    if "rlhf" in q_lower and "reinforcement learning from human feedback" not in q_lower:
        variants.append(query + " Reinforcement Learning from Human Feedback")

    if "reinforcement learning from human feedback" in q_lower and "rlhf" not in q_lower:
        variants.append(query + " RLHF")

    return variants


def hybrid_search(
    query: str,
    rows: List[Dict[str, Any]],
    emb: OllamaEmbeddings,
    doc_vectors: np.ndarray,
    k: int = 5,
):
    """Combine semantic and keyword search for stronger retrieval."""
    all_scores = np.full(len(rows), -1e9, dtype=float)

    for q_text in expand_query(query):
        q_vec = np.array(emb.embed_query(q_text), dtype=float)
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-12)
        semantic_scores = doc_vectors @ q_vec

        lexical_scores = np.array([keyword_score(q_text, row) for row in rows], dtype=float)
        if lexical_scores.max() > 0:
            lexical_scores = lexical_scores / lexical_scores.max()

        combined_scores = 0.7 * semantic_scores + 0.3 * lexical_scores
        all_scores = np.maximum(all_scores, combined_scores)

    top_idx = np.argsort(all_scores)[::-1][:k]
    return [{"index": int(i), "score": float(all_scores[i])} for i in top_idx]


class RAGState(TypedDict):
    question: str
    query: str
    retrieved: list
    needs_retrieval: bool
    needs_rewrite: bool
    rewrite_count: int
    answer: str


def build_graph(
    llm: ChatOllama,
    rows: List[Dict[str, Any]],
    emb: OllamaEmbeddings,
    doc_vectors: np.ndarray,
):
    """Build the agentic RAG graph."""

    def agent_node(state: RAGState):
        """Agent node: decide whether retrieval is needed."""
        prompt = f"""
Decide whether this question needs retrieval from the QA dataset.
Answer only with "yes" or "no".

Question:
{state["question"]}
"""
        decision = llm.invoke(prompt).content.strip().lower()
        needs_retrieval = "yes" in decision
        return {"needs_retrieval": needs_retrieval}

    def should_retrieve(state: RAGState) -> Literal["retrieve", "answer"]:
        """Conditional edge: route to retrieval or direct answer."""
        return "retrieve" if state["needs_retrieval"] else "answer"

    def retrieve_node(state: RAGState):
        """Tool node: retrieve documents for the current query."""
        hits = hybrid_search(state["query"], rows, emb, doc_vectors, k=5)
        return {"retrieved": hits}

    def grade_node(state: RAGState):
        """Check relevance: decide if the retrieved docs are good enough."""
        if not state["retrieved"]:
            return {"needs_rewrite": True}

        top_score = state["retrieved"][0]["score"]
        needs_rewrite = top_score < 0.45
        return {"needs_rewrite": needs_rewrite}

    def route_after_grade(state: RAGState) -> Literal["rewrite", "generate"]:
        """Conditional edge: rewrite weak results or generate an answer."""
        return "rewrite" if state["needs_rewrite"] else "generate"

    def rewrite_node(state: RAGState):
        """Rewrite node: improve the search query but keep technical terms."""
        prompt = f""" Rewrite this question into a short retrieval query.
Preserve exact technical terms.
If useful, include both acronym and full form.
Do not remove important RL terminology.

Question:
{state["question"]}

Current query:
{state["query"]}
"""
        rewritten = llm.invoke(prompt).content.strip().replace("\n", " ")
        return {
            "query": rewritten,
            "rewrite_count": state["rewrite_count"] + 1,
        }

    def generate_node(state: RAGState):
        """Generate node: answer from the retrieved dataset context."""
        if not state["retrieved"]:
            return {"answer": "I don't know from the retrieved dataset."}

        answer = grounded_answer(llm, rows, state["retrieved"], state["question"])
        return {"answer": answer}

    def rewrite_limit_router(state: RAGState) -> Literal["retrieve", "answer"]:
        """Conditional edge: retry retrieval or stop after too many rewrites."""
        return "retrieve" if state["rewrite_count"] < 2 else "answer"

    graph = StateGraph(RAGState)

    graph.add_node("agent", agent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("answer", generate_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_retrieve)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade)
    graph.add_conditional_edges("rewrite", rewrite_limit_router)
    graph.add_edge("generate", END)
    graph.add_edge("answer", END)

    return graph.compile()


def main():
    """Run the interactive agentic RAG application."""
    rows = load_dataset()
    emb, docs, doc_vectors = build_embeddings(rows)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    app = build_graph(llm, rows, emb, doc_vectors)

    print("Agentic RAG ready.")
    print("Ask a question about the dataset.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Enter your question: ").strip()

        if not question:
            print("Please enter a question.\n")
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = app.invoke(
            {
                "question": question,
                "query": question,
                "retrieved": [],
                "needs_retrieval": True,
                "needs_rewrite": False,
                "rewrite_count": 0,
                "answer": "",
            },
            {"recursion_limit": 8},
        )

        print("\nFinal query:", result["query"])

        if result["retrieved"]:
            print_hits(rows, result["retrieved"])
        else:
            print("\nTop retrieved items:")
            print("No documents retrieved.")

        print("\nAnswer:")
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()