import os
import torch
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from collections import defaultdict
import re
from sklearn.metrics import silhouette_score, calinski_harabasz_score
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
    embeds_norm = F.normalize(embeds, p=2, dim=1)
    print(f"Loaded embeds: {embeds.shape}, num_instructions={len(instructions)}")
    return embeds_norm, instructions

def group_by_instruction(instructions):
    """
    按 instruction 文本把样本分组。
    返回: dict[str, List[int]]，键是指令，值是该指令对应的索引列表。
    """
    groups = defaultdict(list)
    for idx, instr in enumerate(instructions):
        groups[instr].append(idx)
    return groups

def cluster_embeddings(
    embeds: torch.Tensor,
    n_clusters: int = 10,
    random_state: int = 42,
    n_iter: int = 50,
    device: str = "cuda",
):
    """
    使用 PyTorch 在 GPU 上做 KMeans 聚类.

    返回:
        labels: [N] 的簇编号 (0 ~ n_clusters-1)
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU")
        device = "cpu"

    torch.manual_seed(random_state)

    # [N, D] -> GPU
    x = embeds.to(device=device, dtype=torch.float32)
    N, D = x.shape
    print(f"Running KMeans (torch) on {device}, n_clusters={n_clusters}, num_points={N}")

    # # 初始化: 随机选择若干样本作为初始中心
    # indices = torch.randperm(N, device=device)[:n_clusters]
    # centroids = x[indices]   # [K, D]
    # 使用 k-means++ 初始化
    centroids = torch.empty((n_clusters, D), device=device, dtype=torch.float32)

    # 1) 第一个中心：从所有点中均匀随机选一个
    first_idx = torch.randint(0, N, (1,), device=device)
    centroids[0] = x[first_idx]

    # 2) 后续中心：按照到最近已选中心的距离平方加权采样
    for c in range(1, n_clusters):
        # 计算每个点到当前已有中心的距离平方 [N, c]
        dists = torch.cdist(x, centroids[:c], p=2)  # [N, c]
        min_dists2, _ = torch.min(dists ** 2, dim=1)  # [N]

        # 避免全零（极端情况）
        probs = min_dists2 / (min_dists2.sum() + 1e-12)

        # 根据概率分布采样下一个中心索引
        next_idx = torch.multinomial(probs, 1)
        centroids[c] = x[next_idx]

    for it in range(n_iter):
        # 计算每个点到各个中心的距离: [N, K]
        # torch.cdist 需要 PyTorch >= 1.1
        dists = torch.cdist(x, centroids, p=2)

        # 取最近的中心
        labels = dists.argmin(dim=1)  # [N]

        # 重新计算每个簇的中心
        new_centroids = []
        for k in range(n_clusters):
            mask = (labels == k)
            if mask.any():
                new_centroids.append(x[mask].mean(dim=0))
            else:
                # 若该簇为空，随机重新初始化一个中心
                rand_idx = torch.randint(0, N, (1,), device=device)
                new_centroids.append(x[rand_idx].squeeze(0))
        new_centroids = torch.stack(new_centroids, dim=0)

        shift = (centroids - new_centroids).pow(2).sum().sqrt().item()
        print(f"iter {it+1}/{n_iter}, centroid shift={shift:.4f}")
        centroids = new_centroids

        if shift < 1e-4:
            print("Converged.")
            break

    # 返回到 CPU / numpy，后续画图方便
    labels = labels.cpu().numpy()
    return labels, centroids.cpu()

def tsne_with_clusters(
    embeds: torch.Tensor,
    labels,
    instructions,
    title: str = "t-SNE of Libero Object Embeddings (clustered)",
    max_points: int = 1000,
    save_path: str | None = None,
):
    """
    对高维向量做 t-SNE 到 2D，并用聚类结果上色。

    embeds: [N, D] Tensor
    labels: [N] cluster id
    instructions: List[str]
    max_points: t-SNE 采样的最大点数（太多会很慢）
    """
    import numpy as np

    x_np = embeds.detach().to(dtype=torch.float32).cpu().numpy()
    labels = torch.as_tensor(labels).cpu().numpy()
    N = x_np.shape[0]

    if N > max_points:
        print(f"t-SNE using first {max_points}/{N} points for visualization")
        x_np = x_np[:max_points]
        labels = labels[:max_points]
        instructions = instructions[:max_points]

    print("Running t-SNE on CPU ...")
    tsne = TSNE(
        n_components=2,
        perplexity=2,
        learning_rate=10,
        # n_iter=1000,
        init="random",
        verbose=1,
    )
    x_2d = tsne.fit_transform(x_np)  # [n_points, 2]

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        x_2d[:, 0],
        x_2d[:, 1],
        c=labels,
        # cmap="tab20",
        s=16,
        alpha=0.8,
    )
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    # 加一个簇编号的 legend（只显示 cluster id）
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.8)
    plt.legend(handles, [f"cluster {i}" for i in range(len(handles))], fontsize=8)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"t-SNE figure with clusters saved to {save_path}")
    else:
        plt.show()

def pca_reduce_torch_local(
    embeds: torch.Tensor,
    out_dim: int = 64,
    device: str = "cuda",
):
    """
    在 GPU 上对嵌入做 PCA 降维，用于每个任务内部的局部 PCA。

    embeds: [N, D]
    out_dim: 降到的维度
    返回:
        embeds_pca: [N, out_dim]
        V: [D, out_dim] 投影矩阵
        mean: [1, D] 均值
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU")
        device = "cpu"

    x = embeds.to(device=device, dtype=torch.float32)
    N, D = x.shape
    k = min(out_dim, D)
    print(f"  [PCA] Running PCA on {device}, input shape={x.shape}, target_dim={k}")

    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean

    # torch.pca_lowrank 比较适合高维
    U, S, V = torch.pca_lowrank(x_centered, q=k)  # V: [D, k]
    x_reduced = x_centered @ V[:, :k]            # [N, k]

    print(f"  [PCA] Reduced shape: {x_reduced.shape}")
    return x_reduced.cpu(), V[:, :k].cpu(), mean.cpu()

def per_instruction_cluster_tsne(
    embeds: torch.Tensor,
    instructions,
    base_out_dir: str,
    global_n_clusters: int = 5,
    global_labels: torch.Tensor | None = None,
    global_cluster_centers: torch.Tensor | None = None,
    raw_embeds: torch.Tensor | None = None,
):
    """
    对每个 instruction 内部单独做 KMeans 聚类和 t-SNE 可视化。

    global_n_clusters: 上限聚类数，实际每个指令会根据样本数自动缩小。
    """
    groups = group_by_instruction(instructions)
    print(f"Found {len(groups)} unique instructions/groups.")

    for instr, idxs in groups.items():
        n = len(idxs)
        if n < 3:
            # 样本太少就跳过，没什么意义
            print(f"[SKIP] instruction='{instr[:40]}...' has only {n} samples.")
            continue

        print(f"\n[Group] instruction='{instr[:60]}...', num_samples={n}")

        # 取该指令对应的子集（归一化后的向量）
        idxs_tensor = torch.tensor(idxs, dtype=torch.long)
        sub_embeds = embeds[idxs_tensor]
        sub_instructions = [instructions[i] for i in idxs]
        # 对应的原始未归一化向量（如果提供的话）
        if raw_embeds is not None:
            sub_raw_embeds = raw_embeds[idxs_tensor]

        # 每组的聚类数不能超过样本数；太小就降到 1
        n_clusters = min(global_n_clusters, max(1, n // 2))
        print(f"  Using n_clusters={n_clusters} for this group.")

        # ===== 1) 先在该任务内部做 PCA 降维 =====
        # 可以调 out_dim，比如 32 / 64 / 128
        sub_pca, _, _ = pca_reduce_torch_local(
            sub_embeds,
            out_dim=min(20,n),
            device="cuda",
        )

        # ===== 2) 在 PCA 空间里做 KMeans（GPU） =====
        labels, centroids = cluster_embeddings(
            sub_pca,                # 注意这里用的是 PCA 后的向量
            n_clusters=n_clusters,
            random_state=42,
            n_iter=50,
            device="cuda",
        )
        # 把该任务内的聚类结果写回全局 labels，保持与 embeds / 数据集同一顺序
        if global_labels is not None:
            labels_tensor = torch.as_tensor(labels, dtype=torch.long)
            # idxs_tensor 是该任务在全局里的索引列表
            global_labels[idxs_tensor] = labels_tensor
        # 保存每个样本对应的聚类中心
        if global_cluster_centers is not None:
            # labels_tensor: [n]
            if 'labels_tensor' not in locals():
                labels_tensor = torch.as_tensor(labels, dtype=torch.long)

            if raw_embeds is not None:
                # 在原始 embedding 空间中按簇重新计算中心
                # sub_raw_embeds: [n, D]
                K = centroids.shape[0]
                centers_raw = []
                for k in range(K):
                    mask = labels_tensor == k
                    if mask.any():
                        centers_raw.append(sub_raw_embeds[mask].mean(dim=0))
                    else:
                        # 该簇在该 instruction 中没有样本，填充为 0 向量
                        centers_raw.append(torch.zeros_like(sub_raw_embeds[0]))
                centers_raw = torch.stack(centers_raw, dim=0)  # [K, D]
                sub_centers = centers_raw[labels_tensor]
            else:
                # 退回到使用归一化空间的聚类中心
                sub_centers = centroids[labels_tensor]  # [n, D]
            global_cluster_centers[idxs_tensor] = sub_centers
        # ===== 3) 质量评估：在 PCA 空间里算 Silhouette & CH =====
        unique_labels = set(labels.tolist())
        if len(unique_labels) > 1:
            sub_np = sub_embeds.detach().to(dtype=torch.float32).cpu().numpy()

            sil = silhouette_score(sub_np, labels, metric="cosine")
            ch = calinski_harabasz_score(sub_np, labels)

            print(f"  Silhouette Score (PCA space): {sil:.4f}")
            print(f"  Calinski-Harabasz Index (PCA space): {ch:.4f}")
        else:
            print("  Only one cluster present, cannot compute silhouette / CH index.")

        # 生成安全的文件名（去掉特殊字符，截断避免过长）
        instr_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", instr).strip("_")
        if not instr_slug:
            instr_slug = "instr"
        instr_slug = instr_slug[:80]

        tsne_path = os.path.join(
            base_out_dir,
            f"tsne_per_instr_{instr_slug}_k{n_clusters}_n{n}.png",
        )

        tsne_with_clusters(
            sub_embeds,
            labels,
            sub_instructions,
            title=f"{instr} (k={n_clusters}, n={n})",
            max_points=n,          # 组内样本不多，全部用上
            save_path=tsne_path,
        )

        print(f"  Saved per-instruction t-SNE to {tsne_path}")

def main():
    # 路径按你当前的目录结构来
    embeds_path = os.path.join(
        os.path.dirname(__file__),
        "libero_object_embeded_id.pt",
    )
    n_clusters = 20  # 聚多少类可以自己改
    max_points = 1000

    embeds, instructions = load_embeds(embeds_path)
    # 再读取一次原始未归一化的 embedding，便于后续使用
    raw_data = torch.load(embeds_path, map_location="cpu")
    raw_embeds = raw_data["embeds"]  # [N, D]
    episode_ids = raw_data["episode_ids"]
    N, D = embeds.shape
    # 初始化全局标签：-1 表示“没有被聚类（或该任务被跳过）”
    all_labels = -torch.ones(N, dtype=torch.long)
    # 初始化全局聚类中心
    all_cluster_centers = torch.zeros(N, D, dtype=embeds.dtype)
    out_dir = os.path.dirname(__file__)
    per_instruction_cluster_tsne(
        embeds,
        instructions,
        base_out_dir=out_dir,
        global_n_clusters=10,   # 每个指令内最多聚 5 类，可以按需要改
        global_labels=all_labels,   # 新增参数
        global_cluster_centers=all_cluster_centers,
        raw_embeds=raw_embeds,
    )
    print(all_labels)
    # 全部任务的聚类结果保存成一个文件，顺序与 embeds / 原始数据集一致
    save_path = os.path.join(
        os.path.dirname(__file__),
        "libero_object_per_instruction_clusters.pt",
    )
    torch.save(
        {
            "labels": all_labels,        # [N]，与 embeds / instructions 顺序一致
            "instructions": instructions,
            "embeds": raw_embeds,        # 原始未归一化的 embedding [N, D]
            "cluster_centers": all_cluster_centers,  # [N, D]，与 embeds / instructions 顺序一致
            "episode_ids": episode_ids, 
        },
        save_path,
    )
    print(f"Global per-instruction cluster labels saved to {save_path}")

    # # 1. 聚类
    # labels = cluster_embeddings(
    #     embeds,
    #     n_clusters=n_clusters,
    #     random_state=42,
    #     n_iter=50,
    #     device="cuda",
    # )

    # # 可选：把聚类结果保存下来，方便后续分析
    # cluster_save_path = os.path.join(
    #     os.path.dirname(__file__),
    #     f"libero_object_cluster_labels_k{n_clusters}.pt",
    # )
    # torch.save(
    #     {
    #         "labels": torch.as_tensor(labels),
    #         "n_clusters": n_clusters,
    #         "instructions": instructions,
    #     },
    #     cluster_save_path,
    # )
    # print(f"Cluster labels saved to {cluster_save_path}")

    # # 2. t-SNE + 按簇可视化
    # tsne_fig_path = os.path.join(
    #     os.path.dirname(__file__),
    #     f"libero_object_tsne_k{n_clusters}.png",
    # )
    # tsne_with_clusters(
    #     embeds,
    #     labels,
    #     instructions,
    #     title=f"t-SNE of Libero Embeds (KMeans k={n_clusters})",
    #     max_points=max_points,
    #     save_path=tsne_fig_path,
    # )


if __name__ == "__main__":
    main()