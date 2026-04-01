import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_embeds(path: str):
    """加载 torch 保存的 embeds 文件."""
    data = torch.load(path, map_location="cpu")
    embeds = data["embeds"]        # [N, D]
    instructions = data["instructions"]  # List[str]
    print(f"Loaded embeds: {embeds.shape}, num_instructions={len(instructions)}")
    return embeds, instructions


def pca_reduce_torch(embeds: torch.Tensor, out_dim: int = 128, device: str = "cuda"):
    """
    使用 torch.pca_lowrank 在 GPU 上做 PCA 降维.

    embeds: [N, D], 任意 dtype (建议 float32/float16/bfloat16)
    out_dim: 降到的维度
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU")
        device = "cpu"

    # 转为 float32 以保证 pca_lowrank 稳定
    x = embeds.to(device=device, dtype=torch.float32)
    N, D = x.shape
    k = min(out_dim, D)

    print(f"Running PCA on {device}, input shape={x.shape}, target_dim={k}")

    # 去中心化
    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean

    # pca_lowrank 返回 U, S, V；我们只用 V 来构建投影矩阵
    U, S, V = torch.pca_lowrank(x_centered, q=k)  # V: [D, k]

    x_reduced = x_centered @ V[:, :k]  # [N, k]
    print(f"PCA reduced shape: {x_reduced.shape}")

    # 返回到 CPU，方便后续和 sklearn 配合
    return x_reduced.cpu(), V[:, :k].cpu(), mean.cpu()


def tsne_visualize(x: torch.Tensor, instructions, title: str = "t-SNE of Razen Embeddings",
                   max_points: int = 500, save_path: str | None = None):
    """
    对 PCA 后的向量做 t-SNE 降到 2D 并画图.

    x: [N, d]，torch.Tensor or numpy array
    instructions: List[str]，和 x 对应的指令
    max_points: t-SNE 计算上限（太大很慢），超过会随机采样前 max_points 个
    """
    import numpy as np

    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    N = x_np.shape[0]
    if N > max_points:
        print(f"t-SNE using first {max_points}/{N} points for visualization")
        x_np = x_np[:max_points]
        instructions = instructions[:max_points]

    print("Running t-SNE on CPU ...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        init="random",
        verbose=1,
    )
    x_2d = tsne.fit_transform(x_np)  # [n_points, 2]

    plt.figure(figsize=(8, 8))
    plt.scatter(x_2d[:, 0], x_2d[:, 1], s=8, alpha=0.7)

    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"t-SNE figure saved to {save_path}")
    else:
        plt.show()


def main():
    embeds_path = "libero_object_embeded.pt"  # 你当前保存的文件
    target_pca_dim = 128                      # 想要降到的维度，比如 128 或 256
    device = "cuda"                           # 优先用 GPU 做 PCA

    embeds, instructions = load_embeds(embeds_path)

    # 1. PCA 降维（尽量在 GPU）
    embeds_pca, V, mean = pca_reduce_torch(
        embeds,
        out_dim=target_pca_dim,
        device=device,
    )

    # 可选：保存 PCA 结果和投影矩阵，供后续复用
    pca_save_path = f"libero_object_embeded_pca_{target_pca_dim}.pt"
    torch.save(
        {
            "embeds_pca": embeds_pca,  # [N, target_pca_dim]
            "V": V,                    # [D, target_pca_dim] 投影矩阵
            "mean": mean,             # [1, D] 均值
            "instructions": instructions,
        },
        pca_save_path,
    )
    print(f"PCA reduced embeddings saved to {pca_save_path}")

    # 2. 用 t-SNE 把 PCA 后的向量再降到 2D 可视化
    tsne_fig_path = f"libero_object_embeded_tsne_{target_pca_dim}.png"
    tsne_visualize(
        embeds_pca,
        instructions,
        title=f"t-SNE of Razen Embeds (PCA {target_pca_dim}D)",
        max_points=500,  # 可以调大/调小
        save_path=tsne_fig_path,
    )


if __name__ == "__main__":
    main()