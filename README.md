# A Practical Introduction to RAG  
## How AI Searches Documents Before Answering

This repository demonstrates three RAG implementations using Ollama for local execution:

1. **Simple RAG** — Basic semantic retrieval and answer
2. **Agentic RAG** — Retrieve → Grade → Rewrite → Retrieve → Answer (LangGraph)  
3. **Hybrid Agentic RAG** — Agentic flow + semantic + keyword retrieval

## Quick Start

### 1. Install Ollama
[https://ollama.com/download](https://ollama.com/download)

### 2. Download Models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Setup Python Environment
```bash
python -m venv rag-env
source rag-env/bin/activate  # Linux/Mac
# or rag-env\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Demo 1: Simple Local RAG
**One-shot retrieval from 14-item JSON dataset**

**Run:** `python demo1_simple_rag.py`  
**Questions to ask:**
- What does RLHF stand for?  
- What is PPO in RLHF?  
- What is Proximal Policy Optimization?

**Expected:** Direct retrieval, grounded answers from dataset

## Demo 2: Agentic RAG with LangGraph
**Self-correcting flow: Retrieve → Grade → Rewrite → Retrieve → Answer**

**Run:** `python demo2_agentic_rag.py`  
**Questions to ask:** Same as Demo 1

**Expected:** Rewrites weak queries, retries retrieval, only answers after good context

## Demo 3: Hybrid Agentic Search
**Agentic flow + semantic similarity + exact phrase matching**

**Run:** `python demo3_hybrid_rag.py`  
**Questions to ask:** Same as Demo 1

**Expected:** Boosts technical terms like "Proximal Policy Optimization" via keyword matching

## Dataset
14 QA pairs on RL, RLHF, PPO, Agentic RAG. See `qa_rl_dataset.json`

## requirements.txt