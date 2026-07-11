# MetaCluster

MetaCluster is a semantic 3D asset retrieval framework designed to bridge the gap between intelligent data backends and real-time engines like Unreal Engine 5. By leveraging a FastAPI backend, vector embeddings via CLIP, and a high-performance vector database, it enables seamless, semantic search capabilities for digital assets and 3D environments.

This repository hosts the core backend infrastructure responsible for handling semantic embeddings, managing the vector store, and exposing fast retrieval APIs.

## 📦 Dataset

The rendered 3D asset previews and embeddings used by MetaCluster are hosted on Hugging Face:

🤗 **[MetaCluster Dataset](#)** _(link coming soon)_

## 🚀 Features

- **Semantic Asset Retrieval:** Search and retrieve 3D assets contextually using text or images rather than rigid keyword matching.
- **CLIP Embeddings:** Utilizes OpenAI's CLIP model to generate robust multi-modal embeddings for image and text inputs.
- **Vector Database Storage:** Powered by ChromaDB for fast, scalable vector indexing and querying.
- **FastAPI Core:** A lightweight, high-performance API layer structured for real-time integration with external client applications and engines.

## 📂 Repository Structure

As shown in the workspace (`image_ae167c.png`), the project structure is as follows:

```text
MetaCluster/
├── chromadb/               # ChromaDB persistence storage directory (ignored in version control)
├── vector_db/              # Additional vector index/database artifacts
├── clip_embed.py           # Logic for handling CLIP model inference and embedding generation
├── embedd_images.py        # Script to batch process and insert asset images into the vector store
├── Retriever.py            # Core engine query and semantic search matching algorithms
└── Vector_store.py         # Initialization and management interface for the database collection
```

## 🎬 Workflow in Action

The screenshots below show the semantic retrieval pipeline driving the Unreal Engine 5 client through its three key stages.

### 1. Default State
The shelf is initialized and idle. The top rows hold the rendered previews served from the `/images` endpoint, while the lower slots wait as empty (green) placeholders until a query is made.

![Default state — idle shelf](docs/screenshots/default%20state.png)

### 2. Preview Pooling
When one of the previews is clicked, its caption is sent to the `/search` API. The backend encodes the query with CLIP, runs MMR re-ranking over the vector store, and returns the top related model previews, which are pooled into the side panel.

![Preview pooling — related results returned from /search](docs/screenshots/preview%20pool.png)

### 3. Model Loaded
Selecting a pooled preview loads its corresponding `model.glb` into the scene, rendering the full 3D asset retrieved semantically from the collection.

![Model loaded — retrieved 3D asset rendered in-engine](docs/screenshots/model%20loaded.png)
