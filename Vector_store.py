import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import trimesh
import pyrender
import torch
import torchvision.transforms as transforms
import open_clip
import chromadb
from chromadb.config import Settings
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

client = chromadb.PersistentClient(
        path="D:/IIITV ICD/Python workplace/MetaCluster/vector_db"
)

collection = client.get_or_create_collection(
    name="modelnet_embeddings",
    metadata={"hnsw:space": "cosine"}
)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

model = model.to(device)
model.eval()

def get_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)

    embedding = embedding.cpu().numpy()[0]

    # Normalize
    embedding = embedding / np.linalg.norm(embedding)

    return embedding

# Path
dataset_path = "D:/IIITV ICD/Python workplace/MetaCluster/Rendered_Meshy" 

print("Scanning directories to count images...")
total_images = 0
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        for model_id in os.listdir(category_path):
            model_folder = os.path.join(category_path, model_id)
            if os.path.isdir(model_folder):
                total_images += sum(1 for f in os.listdir(model_folder) if f.endswith(".png"))

print(f"Found {total_images} images. Starting extraction...")

with tqdm(total=total_images, desc="Generating Embeddings", unit="img") as pbar:
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path): 
            continue
        
        for model_id in os.listdir(category_path):
            model_folder = os.path.join(category_path, model_id)
            if not os.path.isdir(model_folder): 
                continue

            model_path = os.path.join(model_folder, f"{model_id}.glb")
            
            for view_file in os.listdir(model_folder):
                if not view_file.endswith(".png"):
                    continue

                image_path = os.path.join(model_folder, view_file)
                embedding = get_embedding(image_path)
                vector_id = f"{model_id}_{view_file}"

                collection.add(
                    ids=[vector_id],
                    embeddings=[embedding.tolist()],
                    metadatas=[{"category": category, "model_id": model_id, "view_file": view_file, "model_path": model_path}]
                )
                
                pbar.update(1)

print("Database successfully populated!")