import torch
import os
def get_orthogonal_projection_matrix(input_dim=3584, target_dim=4096, seed=42):
    """
    生成一个列正交的投影矩阵 P，使得 P.T @ P = I
    """
    torch.manual_seed(seed)
    # 1. 生成高斯随机矩阵
    # 形状为 [target_dim, input_dim] 对应公式 y = Px (若 x 是列向量)
    # 但 PyTorch Linear 通常是 xA^T，这里我们按 y = xP 习惯生成 [input, target] 也可以
    # 这里按标准线性代数习惯：P 是 [4096, 3584]
    random_matrix = torch.randn(target_dim, input_dim)
    
    # 2. QR 分解
    # q 的形状将是 [4096, 3584], r 是 [3584, 3584]
    q, r = torch.linalg.qr(random_matrix, mode='reduced')
    
    # q 就是我们要的 P
    return q

def project_clusters(
    cluster_path="libero_object_per_instruction_clusters.pt",
    output_path="libero_object_per_instruction_clusters_proj_all.pt",
    input_dim=3584,
    target_dim=4096,
    seed=42,
):
    # 1. 读取原始文件
    data = torch.load(cluster_path, map_location="cpu")
    embeds = data["embeds"]
    # 2. 取出聚类中心（根据你的真实结构修改）
    # 情况 A: 原来就是一个 tensor
    if isinstance(data, torch.Tensor):
        centers = data
        data = {"centers": centers}  # 包装成 dict，以便后面加新 key
    # 情况 B: 原来是一个 dict，里面有某个 key 存聚类中心
    elif isinstance(data, dict):
        # 假设 key 叫 "centers"，如果不是，请改成你实际的 key
        centers = data["cluster_centers"]
    else:
        raise TypeError(f"Unsupported type in {cluster_path}: {type(data)}")

    if centers.dim() != 2 or centers.size(1) != input_dim:
        raise ValueError(f"centers shape={centers.shape}, expected [N, {input_dim}]")

    # 3. 生成固定种子的投影矩阵 P
    P = get_orthogonal_projection_matrix(input_dim=input_dim,
                                         target_dim=target_dim,
                                         seed=seed)
    print(P)
    orig_dtype = centers.dtype
    centers_f32 = centers.to(torch.float32)
    projected_centers = centers_f32 @ P.t()  # [N, target_dim], float32
    projected_centers = projected_centers.to(orig_dtype)  # 转回 bfloat16
    orig_dtype = embeds.dtype
    embeds_f32 = embeds.to(torch.float32)
    projected_embeds = embeds_f32 @ P.t()  # [N, target_dim], float32
    projected_embeds = projected_embeds.to(orig_dtype)  # 转回 bfloat16
    # 4. 投影
    # projected_centers = centers @ P.t()  # [N, target_dim]

    # 5. 在原 dict 上添加新 key，而不是新建 dict
    data["cluster_centers"] = projected_centers
    data["projected_embeds"] = projected_embeds
    # data["P"] = P
    # data["seed"] = seed
    # data["input_dim"] = input_dim
    # data["target_dim"] = target_dim

    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    print("saved to", output_path)

if __name__ == "__main__":
    project_clusters()
# # 验证
# P = get_orthogonal_projection_matrix()

# # 验证正交性: P.T @ P 应该等于单位矩阵
# gram_matrix = torch.mm(P.t(), P)
# identity = torch.eye(3584)

# # 检查误差
# error = (gram_matrix - identity).abs().max()
# print(f"Orthonormality Error: {error.item():.2e}")  # 应该非常接近 0 (e.g., 1e-7)

# # 验证模长保留
# x = torch.randn(3584, 1) # 列向量
# y = torch.mm(P, x)
# print(f"Norm diff: {torch.norm(x) - torch.norm(y)}") # 应该接近 0