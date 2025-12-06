# 🧠 **Baseline RAG Model — Repository Overview**

This repository contains a **from-scratch implementation of a Retrieval-Augmented Generation (RAG) system**, built using:

* **MS MARCO** as the retrieval dataset
* **Sentence Transformers** for embeddings
* **FAISS** for vector search
* **Ollama** for running a local large language model (LLM)
* A single exploratory **Jupyter notebook** implementing the pipeline end-to-end

A full technical guide explaining the concepts and step-by-step implementation is provided in **`RAG_guide.md`**.

---

## 📁 **Project Structure**

```
.
├── data/                  # empty for now; can store embeddings, FAISS indexes, or dataset slices
├── src/
│   └── rag_exploration.ipynb    # the main notebook implementing the end-to-end pipeline
├── RAG_guide.md           # full technical reference and build-from-scratch guide
└── README.md              # (this file)
```

---

## 📘 **What This Repository Contains**

### ✔️ A complete, working RAG implementation inside the notebook:

* Loads a subset of MS MARCO
* Processes corpus, queries, and relevance labels
* Generates embeddings with Sentence Transformers
* Builds a FAISS index for similarity search
* Evaluates retrieval performance (Hit@K)
* Sets up and communicates with a **local LLM via Ollama**
* Generates grounded answers using retrieved documents

### ✔️ A detailed reference guide (`RAG_guide.md`)

This is a standalone document that explains:

* What RAG is
* Core concepts (embeddings, FAISS, evaluation metrics, etc.)
* Execution flow
* Code blocks for each stage
* Instructions to fully recreate the entire pipeline

It serves as both a learning resource and a reproduction guide.

---

## 🚀 **Getting Started**

### 1. Install dependencies

```bash
pip install datasets sentence-transformers faiss-cpu tqdm requests
```

### 2. Install Ollama and pull a model

```bash
brew install ollama
ollama pull llama3.2:3b
```

### 3. Open the notebook

```bash
cd src
jupyter notebook rag_exploration.ipynb
```

Run the notebook cells top to bottom to:

* Build the dataset representation
* Create embeddings
* Construct and query FAISS
* Generate answers using your local LLM

---

## 📂 **Optional: Saving Artifacts**

Although the `data/` directory is empty by default, it is intended to store:

* Saved FAISS indexes
* Embedded vectors (`.npy`)
* Cached MS MARCO subsets
* Any intermediate outputs you want to reuse

This allows for faster experimentation in future runs.

---

## 📄 **Documentation**

All conceptual explanations, definitions, troubleshooting guidance, and code breakdowns are contained in:

### 👉 **`RAG_guide.md`**

If you want to rebuild the entire system from scratch, start there.

---

## 🧩 **Future Improvements**

Potential next steps:

* Convert notebook logic into modular Python scripts in `src/`
* Add reranking (cross-encoder stage)
* Introduce structured extraction → answer reasoning
* Integrate custom datasets
* Containerize the setup for reproducibility

---

## 📬 **Questions or Further Extensions**

If you want to expand the repository structure, add pipelines, or convert this into a production-ready RAG API, the current design already supports modular growth.
