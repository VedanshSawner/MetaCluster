from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import open_clip
import chromadb
import numpy as np
import os
from collections import defaultdict
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="MetaCluster Semantic Search API")

BASE_RENDER_DIR = "D:/IIITV ICD/Python workplace/MetaCluster/Rendered_Meshy"
app.mount("/images", StaticFiles(directory=BASE_RENDER_DIR), name="images")

# --- INIT ---
device = "cuda" if torch.cuda.is_available() else "cpu"

client = chromadb.PersistentClient(path="D:/IIITV ICD/Python workplace/MetaCluster/vector_db")
collection = client.get_collection(name="modelnet_embeddings")

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")


# --- REQUEST MODEL ---
class SearchRequest(BaseModel):
    queries: list[str]   # 🔥 multiple queries
    lambda_mult: float = 0.7
    category: str = None


# --- MMR ---
def calculate_mmr(query_embedding, candidate_embeddings, top_k=4, lambda_mult=0.7):
    if len(candidate_embeddings) == 0:
        return []

    sim_to_query = np.dot(candidate_embeddings, query_embedding)
    sim_matrix = np.dot(candidate_embeddings, candidate_embeddings.T)

    selected = []
    remaining = list(range(len(candidate_embeddings)))

    first = int(np.argmax(sim_to_query))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < top_k and remaining:
        best_score = -np.inf
        best_idx = -1

        for idx in remaining:
            relevance = sim_to_query[idx]
            redundancy = np.max(sim_matrix[idx, selected])

            score = (lambda_mult * relevance) - ((1 - lambda_mult) * redundancy)

            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


# --- API ---
@app.post("/search")
def search_models(req: SearchRequest):
    try:
        # 🔥 1. Encode multiple queries
        query_texts = [f"a clean 3D model of a {q}" for q in req.queries]

        text_tokens = tokenizer(query_texts).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)

        query_embeddings = text_features.cpu().numpy()
        query_embeddings = query_embeddings / (
            np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8
        )

        # 🔥 2. Retrieve for each query
        all_embeddings = []
        all_metadatas = []

        fetch_k = 50  # per query

        for q_emb in query_embeddings:
            query_args = {
                "query_embeddings": [q_emb.tolist()],
                "n_results": fetch_k,
                "include": ["metadatas", "embeddings"]
            }

            if req.category:
                query_args["where"] = {"category": req.category}

            results = collection.query(**query_args)

            if results['metadatas'] and len(results['metadatas'][0]) > 0:
                all_embeddings.extend(results['embeddings'][0])
                all_metadatas.extend(results['metadatas'][0])

        if len(all_embeddings) == 0:
            return {"results": []}

        # 🔥 3. Normalize
        candidate_embeddings = np.array(all_embeddings)
        norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        candidate_embeddings = candidate_embeddings / (norms + 1e-8)

        # 🔥 4. Multi-query similarity (MAX across queries)
        sim_scores = np.max(
            np.dot(candidate_embeddings, query_embeddings.T),
            axis=1
        )

        # 🔥 5. Dynamic threshold
        top_sim = np.max(sim_scores)
        valid_indices = [i for i in range(len(sim_scores)) if sim_scores[i] > 0.7 * top_sim]

        if not valid_indices:
            valid_indices = list(range(len(sim_scores)))

        # 🔥 6. GROUP BY MODEL
        model_groups = defaultdict(list)

        for i in valid_indices:
            metadata = all_metadatas[i]

            folder_path = metadata.get('path')
            if not folder_path and metadata.get('model_path'):
                folder_path = os.path.dirname(metadata.get('model_path'))

            if folder_path:
                model_groups[folder_path].append((i, candidate_embeddings[i]))

        # 🔥 7. BEST VIEW PER MODEL
        model_embeddings = []
        model_paths = []

        for folder_path, items in model_groups.items():
            best_sim = -np.inf
            best_embedding = None

            for idx, emb in items:
                sim = np.max(np.dot(emb, query_embeddings.T))  # multi-query sim
                if sim > best_sim:
                    best_sim = sim
                    best_embedding = emb

            if best_embedding is not None:
                model_embeddings.append(best_embedding)
                model_paths.append(folder_path)

        if len(model_embeddings) == 0:
            return {"results": []}

        model_embeddings = np.array(model_embeddings)

        # 🔥 8. Use centroid query for MMR
        query_centroid = np.mean(query_embeddings, axis=0)
        query_centroid /= (np.linalg.norm(query_centroid) + 1e-8)

        # 🔥 9. MMR on models (TOP 4)
        top_k = min(4, len(model_embeddings))

        best_indices = calculate_mmr(
            query_embedding=query_centroid,
            candidate_embeddings=model_embeddings,
            top_k=top_k,
            lambda_mult=req.lambda_mult
        )

        # 🔥 10. Format output
        formatted_results = []

        for idx in best_indices:
            folder_path = model_paths[idx]

            model_path = os.path.join(folder_path, "model.glb").replace('\\', '/')
            relative_path = os.path.relpath(folder_path, BASE_RENDER_DIR).replace('\\', '/')
            image_url = f"http://127.0.0.1:8000/images/{relative_path}/view_0.png"

            formatted_results.append({
                "model_path": model_path,
                "view_file": image_url
            })

        return JSONResponse(
            content={"results": formatted_results},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))