import os
import torch
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

from collections import defaultdict
import re

import torch.nn.functional as F
def load_embeds(path: str):
    """
    读取 step1_embedding/libero_razen_embed.py 里保存的 libero_object_embeded.pt
    约定结构:
        {
            "embeds": Tensor [N, D],
            "instructions": List[str]
        }
    """
    data = torch.load(path, map_location="cpu")
    embeds = data["embeds"]              # [N, D]
    instructions = data["instructions"]  # List[str]
    ids=data["episode_ids"]
    embeds_norm = F.normalize(embeds, p=2, dim=1)
    print(f"Loaded embeds: {embeds.shape}, num_instructions={len(instructions)}")
    return embeds_norm, instructions,ids

if __name__ == "__main__":
    embeds_norm, instructions,ids=load_embeds("libero_object_per_instruction_clusters.pt")
    print(ids)