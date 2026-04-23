# RAG Demo Bundle

This folder contains a small teaching bundle for demonstrating three local Retrieval-Augmented Generation (RAG) setups with Ollama, LangChain/LangGraph, and a compact QA dataset extracted from Chapter 9 of the attached PDF.

## Files

### `qa_dataset_ch9.json`
A compact question-answer dataset used as the knowledge base for the demos.

Each item contains:
- `id`: unique identifier
- `question`: short QA prompt
- `answer`: expanded answer with a few supporting sentences
- `source_page`: source page in the chapter
- `keywords`: short keyword list used by the hybrid demo

There is also an expanded copy:
- `qa_dataset_ch9_expanded.json`

## Demos

### 1. `demo1_baby_rag.py`
This is the simplest semantic RAG demo.

What it does:
1. Loads the QA dataset
2. Converts each QA item into a retrievable text chunk
3. Creates embeddings with `nomic-embed-text` through Ollama
4. Computes semantic similarity between the user question and dataset items
5. Retrieves the top matching entries
6. Sends the retrieved context to a local LLM (`llama3`) to generate a grounded answer

Why it is useful:
- shows the core RAG pipeline very clearly
- easy to explain in class
- good for demonstrating embeddings and top-k retrieval

Suggested use:
- ask a semantically clear question such as:
  - "How do people use human feedback to improve language models?"

---

### 2. `demo2_agentic_rag.py`
This is a small LangGraph-based agentic RAG demo.

What it does:
1. Retrieves initial results with semantic search
2. Grades whether the retrieval looks strong enough
3. If retrieval is weak, rewrites the query
4. Runs retrieval again
5. Generates the final answer from the retrieved context

Workflow:
`retrieve -> grade -> rewrite -> retrieve -> answer`

Why it is useful:
- shows how agentic RAG differs from naive RAG
- demonstrates a retrieval loop rather than a single fixed pass
- useful for vague or indirect questions

Suggested use:
- ask a broader or less direct question such as:
  - "How is the newer RAG style smarter than the old fixed pipeline?"

---

### 3. `demo3_hybrid_rag.py`
This demo combines semantic retrieval with simple keyword matching.

What it does:
1. Computes semantic similarity using embeddings
2. Computes keyword overlap using the question, answer, and keyword fields
3. Normalizes both scores
4. Fuses them into one hybrid score
5. Retrieves the top matching entries
6. Uses the retrieved context to answer with the local LLM

Why it is useful:
- demonstrates why hybrid search helps with acronyms and exact terms
- useful when semantic retrieval alone may miss precise lexical matches

Suggested use:
- ask an acronym-heavy or exact-term question such as:
  - "What does PPO stand for in RLHF?"

## Requirements

Install dependencies:

```bash
pip install -r requirements_rag_demos.txt
```

Pull local Ollama models:

```bash
ollama pull nomic-embed-text
ollama pull llama3
```

## Run commands

### Demo 1
```bash
python demo1_baby_rag.py
```

### Demo 2
```bash
python demo2_agentic_rag.py
```

### Demo 3
```bash
python demo3_hybrid_rag.py
```

## Teaching notes

A good teaching order is:
1. Start with Demo 1 to explain the basic RAG idea
2. Move to Demo 2 to show why naive retrieval is sometimes not enough
3. Finish with Demo 3 to explain hybrid retrieval and exact-term search

## Notes on the dataset design

The dataset was intentionally designed to:
- keep answers short enough for embedding and retrieval
- distribute topics across the chapter
- include both conceptual and acronym-based questions
- support semantic, agentic, and hybrid retrieval examples with the same corpus

Main covered topics include:
- reinforcement learning
- HHH principles
- RLHF
- reward models
- PPO
- AutoRL
- autonomous AI agents
- planning modules
- Agentic RAG
- action execution
- travel assistant example
- function calling with LLMs
