import os
import torch
import chromadb
from chromadb.config import Settings
from clip_embed import get_folder_embedding  # your function

# 🔹 Device
device = "cuda" if torch.cuda.is_available() else "cpu"

client = chromadb.PersistentClient(path='D:/IIITV ICD/Python workplace/MetaCluster/chromadb')
print('done')
collection = client.get_or_create_collection(name="modelnet_embeddings")

# 🔹 Parent directory
parent_dir = r'D:\IIITV ICD\Python workplace\MetaCluster\Rendered_Meshy'

# 🔹 Collect all sub-category folders
final_dirs = []

for category in os.listdir(parent_dir):
    category_path = os.path.join(parent_dir, category)

    if not os.path.isdir(category_path):
        continue

    for sub_cat in os.listdir(category_path):
        sub_cat_path = os.path.join(category_path, sub_cat)

        if os.path.isdir(sub_cat_path):
            final_dirs.append(sub_cat_path)

print(f"Total folders found: {len(final_dirs)}")

# 🔹 Batch settings
BATCH_SIZE = 32

batch_ids = []
batch_embeddings = []
batch_metadata = []

done = 0

# 🔹 Process folders
for i, folder in enumerate(final_dirs):

    mean_emb = get_folder_embedding(folder)

    if mean_emb is None:
        continue

    batch_ids.append(f"id_{i}")
    batch_embeddings.append(mean_emb.tolist())
    batch_metadata.append({
        "path": folder
    })

    done += 1

    # 🔹 Print progress
    print(f"Progress: {(done / len(final_dirs)) * 100:.2f}%")

    # 🔹 Insert batch
    if len(batch_ids) == BATCH_SIZE:
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadata
        )

        batch_ids, batch_embeddings, batch_metadata = [], [], []

# 🔹 Insert remaining
if batch_ids:
    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        metadatas=batch_metadata
    )