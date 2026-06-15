import torch
import clip
from PIL import Image
import os
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)


def get_folder_embedding(folder_path):
    embeddings = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            image = preprocess(Image.open(file_path).convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model.encode_image(image)

            emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
            embeddings.append(emb.cpu())

        except Exception as e:
            print(f"Error in {file_path}: {e}")

    if len(embeddings) == 0:
        return None

    embeddings = torch.cat(embeddings, dim=0)

    
    mean_embedding = embeddings.mean(dim=0)

    return mean_embedding.numpy()