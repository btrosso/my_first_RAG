# **Building a Retrieval-Augmented Generation (RAG) System From Scratch**

*A complete, step-by-step guide based on implementing a RAG model end-to-end using MS MARCO, FAISS, Sentence Transformers, and a local LLM via Ollama.*

---

# **Table of Contents**

1. [Introduction](#introduction)
2. [What Is RAG?](#what-is-rag)
3. [Conceptual Architecture](#conceptual-architecture)
4. [Project Structure (Recommended)](#project-structure-recommended)
5. [Environment Setup](#environment-setup)
6. [Downloading &amp; Organizing Data/Models](#downloading--organizing-datamodels)
7. [Step 1 — Load &amp; Inspect MS MARCO](#step-1--load--inspect-ms-marco)
8. [Step 2 — Build Your Corpus, Queries, and Qrels](#step-2--build-your-corpus-queries-and-qrels)
9. [Step 3 — Create Embeddings Using Sentence Transformers](#step-3--create-embeddings-using-sentence-transformers)
10. [Step 4 — Build FAISS Vector Index](#step-4--build-faiss-vector-index)
11. [Step 5 — Retrieval Evaluation (Hit@K)](#step-5--retrieval-evaluation-hitatk)
12. [Step 6 — Set Up a Local LLM With Ollama](#step-6--set-up-a-local-llm-with-ollama)
13. [Step 7 — Connect Retrieval to Generation (RAG Pipeline)](#step-7--connect-retrieval-to-generation-rag-pipeline)
14. [Improving RAG &amp; Handling Hallucinations](#improving-rag--handling-hallucinations)
15. [Saving Your Architecture for Reuse](#saving-your-architecture-for-reuse)
16. [Conclusion](#conclusion)

---

# **Introduction** *(Expanded)*

This guide walks you through building a complete **Retrieval-Augmented Generation (RAG)** system **from scratch**, step by step. The goal is to demystify every component involved in RAG, giving you a practical and conceptual foundation strong enough to rebuild the entire system without assistance.

RAG systems combine **information retrieval** with **language generation**, allowing large language models to answer questions using **specific external knowledge** rather than relying solely on what they were trained on. This greatly reduces hallucinations, improves accuracy, and allows applications to incorporate private or dynamic data.

To build a RAG model, you must understand several key concepts:

### **Embeddings**

Embeddings are numerical vector representations of text. Instead of comparing raw strings, we convert sentences or documents into high-dimensional vectors (e.g., 384-dimensional floats). Semantically similar texts end up near one another in this vector space.
For example:

* "Capital of France" → embedding A
* "Paris is the capital city of France" → embedding B
  These embeddings will be close together based on cosine similarity.

### **Vector Search (FAISS)**

Once text is converted into embeddings, we need a fast way to search for the closest vectors to a given query vector. This is where **FAISS** comes in.
FAISS (Facebook AI Similarity Search) is a specialized library optimized for quickly performing nearest-neighbor searches in high-dimensional spaces. It allows us to retrieve the most relevant documents efficiently, even when working with millions of embeddings.

### **Retrieval Evaluation**

A RAG system is only as good as its retriever. Earlier steps cannot fix a poor retriever. To measure retrieval quality, we use metrics such as:

* **Hit@K**: Whether at least one relevant passage appears in the top-K retrieved results
* **MRR@K (Mean Reciprocal Rank)**: How early the first relevant document appears in the ranked list
* **Recall@K**: Fraction of all relevant documents that appear in the top-K results

Good retrieval dramatically improves downstream generation quality.

### **LLM Grounding**

Once relevant passages are retrieved, the generator (LLM) produces an answer **grounded** in the provided context. The better the retrieval quality and the more structured the prompt, the more reliable the answer.

This guide blends:

* Notebook-style experimentation
* Clean engineering practices
* Detailed conceptual explanations

The end goal:

> **If you returned to this project in the future without ChatGPT, you could fully rebuild a working RAG system using only this guide.**

---

# **What Is RAG?** *(Expanded)*

**Retrieval-Augmented Generation (RAG)** is an architecture that enhances language models by combining **information retrieval** with **text generation**. Instead of relying solely on an LLM’s internal knowledge, RAG systems explicitly provide the LLM with relevant external documents at query time.

A RAG system contains three major components:

---

## **1. Retriever**

The retriever’s job is to find the most relevant documents for a given user query. This happens in several steps:

### **Step A — Embedding the Corpus**

Each document (or passage) is converted into an embedding — a numeric vector capturing its semantic meaning. Models like **Sentence Transformers** create embeddings where:

* Similar content has similar embeddings
* Different concepts are far apart

### **Step B — Embedding the Query**

When a user submits a query, we convert that question into its **own embedding**, placing it into the same semantic vector space as the corpus.

### **Step C — Vector Similarity Search**

We compare the query embedding to all document embeddings using a similarity measure (often cosine similarity).
FAISS accelerates this process, making retrieval fast even for large datasets.

### **Why This Matters**

The retriever ensures the LLM sees **relevant, factual evidence**, drastically reducing hallucinations and enabling domain-specific knowledge integration.

---

## **2. Generator (LLM)**

Once relevant documents are retrieved, they are inserted into the LLM’s prompt as **context**.
The generator must:

* Read the retrieved passages
* Answer the question using only those passages
* Avoid using outside knowledge
* Provide accurate, grounded output

Prompts in RAG architectures often include explicit instructions like:

> “Use ONLY the provided passages. If the passages do not contain enough information, say so.”

A strong RAG system demands that the generator remain faithful to evidence rather than guessing.

---

## **3. Evaluator**

Evaluating a RAG system requires assessing two separate components:

### **A. Retrieval Evaluation**

You must measure whether the retriever is finding the right documents. Common metrics:

#### **Hit@K**

Checks if *any* relevant document appears in the top-K retrieved results.

* High Hit@K = strong recall

#### **MRR@K (Mean Reciprocal Rank)**

Measures how early the first relevant result appears.

* Higher = better ranking quality

#### **Recall@K**

How many of the relevant documents were retrieved.

These metrics help diagnose whether generation failures stem from weak retrieval rather than LLM behavior.

---

### **B. Generation Evaluation**

Evaluating the *quality* of LLM-generated answers includes:

* **F1 score** (token overlap with reference answers)
* **Exact Match (EM)** (strict equality)
* **Human judgment** (especially important because RAG answers often vary in wording)
* **Grounding checks** (verifying each answer statement is supported by retrieved passages)

This two-stage evaluation ensures you can improve the right part of the system — retrieval or generation — instead of guessing.

---

## **Why RAG Matters**

RAG is powerful because it allows you to:

* Incorporate new or private knowledge into an LLM
* Update knowledge without retraining
* Reduce hallucinations by grounding responses
* Build domain-specific systems (healthcare, finance, legal, scientific)
* Scale to large document collections

In modern AI applications, RAG is the backbone of real-world LLM systems, from chatbots to search engines to enterprise QA platforms.

---

# **Conceptual Architecture**

```
                  ┌───────────────────┐
                  │      Dataset       │
                  │    (MS MARCO)      │
                  └────────┬──────────┘
                           │
                           ▼
                 ┌─────────────────────┐
                 │  Preprocessing       │
                 │ corpus / queries /   │
                 │ qrels construction    │
                 └────────┬────────────┘
                          │
                          ▼
                 ┌─────────────────────┐
                 │  Embedding Model     │
                 │ (SentenceTransformers)│
                 └────────┬────────────┘
                          │
                          ▼
                ┌───────────────────────┐
                │  FAISS Vector Index    │
                └────────┬──────────────┘
                          │
                 Query →  │  → k documents
                          ▼
                ┌───────────────────────┐
                │  Prompt Builder        │
                │  (context packaging)   │
                └────────┬──────────────┘
                          │
                          ▼
                ┌───────────────────────┐
                │  Local LLM (Ollama)    │
                └────────┬──────────────┘
                          │
                          ▼
                ┌───────────────────────┐
                │      RAG Answer        │
                └────────────────────────┘
```

---

# **Project Structure (Recommended)**

Even though this project was built inside a Jupyter notebook, the following structure is recommended for clean replication and scaling:

```
rag-msmarco/
│
├── data/
│   ├── ms_marco_subset/           # optional: local cached slices
│   └── embeddings/                # saved numpy arrays
│
├── models/
│   └── ollama/                    # downloaded local LLMs (Ollama-managed)
│
├── src/
│   ├── build_corpus.py            # dataset → corpus, queries, qrels
│   ├── build_embeddings.py        # embedding generation
│   ├── build_faiss.py             # vector index construction
│   ├── retrieval.py               # retrieval helpers
│   ├── rag_prompting.py           # prompt templates & context assembly
│   ├── llm_client.py              # Ollama integration
│   └── rag_pipeline.py            # end-to-end answer generation
│
├── notebooks/
│   └── rag_exploration.ipynb      # your current notebook-based workflow
│
└── README.md                      # this document
```

---

# **Environment Setup**

Install required packages:

```bash
pip install datasets sentence-transformers faiss-cpu tqdm requests
```

Ensure Python ≥ 3.10 is recommended.

---

# **Downloading & Organizing Data/Models**

### MS MARCO (via Hugging Face `datasets`):

Downloaded automatically into your Hugging Face cache.
For persistence across runs:

```
data/ms_marco_subset/
```

You can manually write the subset there if desired.

### Sentence Transformer model:

Downloaded automatically to:

```
~/.cache/torch/sentence_transformers/
```

Optional: move it into `models/embeddings/`.

### Ollama models:

Managed automatically by Ollama in:

```
~/.ollama/
```

No manual movement required.

---

# **Step 1 — Load & Inspect MS MARCO**

MS MARCO format:

* `query`: question text
* `answers`: human answers
* `passages.passage_text`: list of candidate passages
* `passages.is_selected`: binary indicator of which passages are relevant

### Load a subset:

```python
from datasets import load_dataset

dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train[:1000]")

print(len(dataset))   # expect: 1000 examples
example = dataset[0]

print(example.keys())
print("QUERY:", example["query"])
print("ANSWERS:", example["answers"])

passages = example["passages"]["passage_text"]
labels = example["passages"]["is_selected"]

print("NUM PASSAGES:", len(passages))
print("IS_SELECTED:", labels)
```

---

# **Step 2 — Build Your Corpus, Queries, and Qrels**

We transform the dataset into RAG-ready structures:

* `corpus`: passage_id → text
* `queries`: query_id → { text, answers }
* `qrels`: query_id → list of relevant passage_ids

### Example code:

```python
from datasets import load_dataset
from collections import defaultdict

dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train[:1000]")

corpus = {}
queries = {}
qrels = defaultdict(list)

next_passage_id = 0

for example in dataset:
    qid = example["query_id"]
    queries[qid] = {
        "text": example["query"],
        "answers": example["answers"]
    }

    passage_texts = example["passages"]["passage_text"]
    labels = example["passages"]["is_selected"]

    for text, label in zip(passage_texts, labels):
        pid = next_passage_id
        corpus[pid] = text
        if label == 1:
            qrels[qid].append(pid)
        next_passage_id += 1

print("Num queries:", len(queries))
print("Num passages:", len(corpus))
```

---

# **Step 3 — Create Embeddings Using Sentence Transformers**

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)

num_passages = len(corpus)
passage_texts = [corpus[pid] for pid in range(num_passages)]

batch_size = 64
all_embeddings = []

for i in tqdm(range(0, num_passages, batch_size)):
    batch = passage_texts[i : i + batch_size]
    batch_emb = embedder.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
    all_embeddings.append(batch_emb)

passage_embeddings = np.vstack(all_embeddings)

print(passage_embeddings.shape)   # (num_passages, embedding_dim)
```

Save embeddings if desired:

```python
np.save("data/embeddings/passage_embeddings.npy", passage_embeddings)
```

---

# **Step 4 — Build FAISS Vector Index**

```python
import faiss
import numpy as np

dim = passage_embeddings.shape[1]  # e.g., 384
index = faiss.IndexFlatIP(dim)     # inner product for cosine similarity
index.add(passage_embeddings)

print("Index size:", index.ntotal)
```

---

# **Step 5 — Retrieval Evaluation (Hit@K)**

This measures how often a relevant passage appears in the top-k retrieved results.

```python
import numpy as np
from tqdm import tqdm

def evaluate_hit_k(queries, qrels, index, embedder, k=5, max_queries=200):
    hits = []
    num_eval = 0

    for qid, qinfo in tqdm(queries.items()):
        rel = qrels.get(qid, [])
        if not rel:
            continue

        q_vec = embedder.encode(
            [qinfo["text"]],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = index.search(q_vec, k)
        retrieved = set(indices[0].tolist())

        hits.append(1 if retrieved & set(rel) else 0)

        num_eval += 1
        if num_eval >= max_queries:
            break

    hit_k = float(np.mean(hits))
    print(f"Evaluated {num_eval} queries.")
    print(f"Hit@{k}: {hit_k:.3f}")
```

---

# **Step 6 — Set Up a Local LLM With Ollama**

### Install Ollama

Download from:
[https://ollama.com](https://ollama.com)

### Pull a model:

```bash
ollama pull llama3.2:3b
```

### Test:

```bash
ollama run llama3.2:3b
```

### Use HTTP API in Python:

```python
import requests, json

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

def call_llm(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
    }

    response = requests.post(OLLAMA_URL, json=payload, stream=True)
    response.raise_for_status()

    chunks = []
    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line.decode())
        if "response" in data:
            chunks.append(data["response"])
        if data.get("done"):
            break

    return "".join(chunks).strip()
```

---

# **Step 7 — Connect Retrieval to Generation (RAG Pipeline)**

### Helper: random query selector

```python
import random

def random_qid_with_labels():
    labeled = [qid for qid, rels in qrels.items() if len(rels) > 0]
    return random.choice(labeled)
```

### Build context block:

```python
def build_context_block(retrieved_docs):
    parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        part = f"[Passage {i}] (pid={doc['pid']}, score={doc['score']:.3f})\n{doc['text']}"
        parts.append(part)
    return "\n\n".join(parts)
```

### Retrieval function:

```python
def retrieve_top_k(query_text, k=5):
    q_vec = embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_vec, k)
    results = []
    for score, pid in zip(scores[0], indices[0]):
        results.append({
            "pid": int(pid),
            "score": float(score),
            "text": corpus[int(pid)],
        })
    return results
```

### Prompt template:

```python
PROMPT_TEMPLATE = """You are a question-answering assistant.

Use ONLY the information in the provided passages.
If the passages lack necessary information, say so plainly.
Do NOT use outside knowledge.

Passages:
{context}

Question:
{question}

Answer in 2-3 sentences.
At the end, list which passages you used: "Sources: [Passage X], ..."
"""
```

### End-to-end RAG pipeline:

```python
def answer_with_rag(qid, k=5):
    qinfo = queries[qid]
    query_text = qinfo["text"]
    refs = qinfo["answers"]

    retrieved = retrieve_top_k(query_text, k=k)
    context = build_context_block(retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, question=query_text)

    answer = call_llm(prompt)

    return {
        "qid": qid,
        "query": query_text,
        "references": refs,
        "retrieved": retrieved,
        "answer": answer,
    }
```

---

# **Improving RAG & Handling Hallucinations**

### Strategies

1. **Prompt Strengthening**

   * Explicit instructions to not use external knowledge
   * Require passage citations
   * Clarify selection rules when multiple values appear in context
2. **Lower LLM Temperature**

   * Encourages more deterministic, grounded answers
3. **Two-Stage Reasoning**

   * Step 1: Extract structured facts
   * Step 2: Answer using only that structured data
4. **Better Retrieval**

   * Use stronger embedding models
   * Increase or decrease `k`
   * Add reranking (bi-encoder → cross-encoder pipeline)
5. **Post-Answer Validation**

   * Check whether generated numbers appear in retrieved passages

---

# **Saving Your Architecture for Reuse**

To replicate or share your setup:

### Save FAISS Index

```python
faiss.write_index(index, "data/index/faiss_index.bin")
```

### Save Embeddings

```python
np.save("data/embeddings/passage_embeddings.npy", passage_embeddings)
```

### Save Corpus/Queries/Qrels

```python
import json

with open("data/corpus.json", "w") as f:
    json.dump(corpus, f)

with open("data/queries.json", "w") as f:
    json.dump(queries, f)

with open("data/qrels.json", "w") as f:
    json.dump(qrels, f)
```

### Export Environment

```bash
pip freeze > requirements.txt
```

---

# **Conclusion**

By following this guide, you have everything required to:

* Load and preprocess a real-world dataset
* Build a retrieval engine using embeddings + FAISS
* Evaluate retrieval quality
* Run a local LLM with Ollama
* Construct prompts that ground generation in retrieved evidence
* Produce complete RAG answers end-to-end

This is a **production-relevant, fully functional RAG setup**, built from first principles.
