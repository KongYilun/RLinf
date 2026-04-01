import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def load_embeds(path: str):
    data = torch.load(path, map_location="cpu")
    embeds = data["embeds"]              # [N, D]
    instructions = data["instructions"]  # List[str]
    embeds_norm = F.normalize(embeds, p=2, dim=1)
    print(f"Loaded embeds: {embeds.shape}, num_instructions={len(instructions)}")
    return embeds_norm, instructions


def group_by_instruction(instructions):
    groups = defaultdict(list)
    for idx, instr in enumerate(instructions):
        groups[instr].append(idx)
    return groups


def pca_2d_torch(x: torch.Tensor):
    """
    对 x: [N, D] 做 PCA 降到 2 维，返回 [N, 2]
    使用 torch.pca_lowrank，避免对巨大协方差矩阵做 eigh。
    """
    x = x.to(dtype=torch.float32)
    # 中心化
    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean

    N, D = x_centered.shape
    k = min(2, N, D)  # 防止样本数太少的极端情况
    if k < 2:
        # 返回退化结果，至少保证形状正确
        return torch.zeros((N, 2), dtype=torch.float32)

    # U: [N, k], S: [k], V: [D, k]
    U, S, V = torch.pca_lowrank(x_centered, q=k)
    # 投影到前 2 个主成分
    x_2d = x_centered @ V[:, :2]  # [N, 2]
    return x_2d.cpu()


def safe_filename(s: str, max_len: int = 80):
    import re
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")
    if not s:
        s = "instr"
    return s[:max_len]


def main():
    base_dir = os.path.dirname(__file__)
    embeds_path = os.path.join(base_dir, "libero_object_embeded.pt")
    out_dir = os.path.join(base_dir, "per_instruction_pca_2d")
    os.makedirs(out_dir, exist_ok=True)

    embeds, instructions = load_embeds(embeds_path)
    groups = group_by_instruction(instructions)
    print(f"Found {len(groups)} unique instructions/groups.")

    for instr, idxs in groups.items():
        n = len(idxs)
        if n < 2:
            print(f"[SKIP] instruction='{instr[:40]}...' has only {n} samples.")
            continue

        print(f"[Group] instruction='{instr[:60]}...', num_samples={n}")
        idxs_tensor = torch.tensor(idxs, dtype=torch.long)
        sub_embeds = embeds[idxs_tensor]  # [n, D]

        # PCA 到 2 维
        x_2d = pca_2d_torch(sub_embeds).numpy()  # [n, 2]

        # 只在 group 内可视化，不再做聚类，颜色统一
        plt.figure(figsize=(6, 6))
        plt.scatter(x_2d[:, 0], x_2d[:, 1], s=10, alpha=0.8)
        plt.title(instr)
        plt.xlabel("PCA dim 1")
        plt.ylabel("PCA dim 2")

        fname = safe_filename(instr)
        save_path = os.path.join(out_dir, f"pca2d_{fname}_n{n}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  Saved PCA 2D figure to {save_path}")


if __name__ == "__main__":
    main()