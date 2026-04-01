import os
from collections import defaultdict

import torch
import tensorflow as tf
import tensorflow_datasets as tfds
import imageio.v2 as imageio
import numpy as np


CLUSTER_PATH = os.path.join(
    os.path.dirname(__file__),
    "libero_object_per_instruction_clusters.pt",
)
DATA_DIR = "/data/dataset"
DATASET_NAME = "libero_object_no_noops"
OUT_DIR = os.path.join(os.path.dirname(__file__), "per_instr_cluster_videos")
os.makedirs(OUT_DIR, exist_ok=True)


def load_clusters(path: str):
    """
    读取 libre_object_per_instruction_clusters.pt
    约定结构:
        {
            "labels": Tensor [N],
            "instructions": List[str]
        }
    """
    data = torch.load(path, map_location="cpu")
    labels = torch.as_tensor(data["labels"]).tolist()
    instructions = list(data["instructions"])
    assert len(labels) == len(instructions), "labels / instructions 长度不一致"
    print(f"Loaded clusters: N={len(labels)}")
    return labels, instructions


def build_selection(labels, instructions, k_per_cluster: int = 2):
    """
    对每个 instruction，在每个聚类簇中挑最多 k_per_cluster 个样本（按 index 顺序）。
    返回:
        selected: dict[(instr, cluster_id)] = List[episode_idx]
    """
    N = len(labels)
    # 先把每个 instruction 下的 indices 收集起来
    instr_to_indices = defaultdict(list)
    for i in range(N):
        instr_to_indices[instructions[i]].append(i)

    selected = dict()

    for instr, idxs in instr_to_indices.items():
        # 该指令下的 labels 子集
        sub_labels = [labels[i] for i in idxs]
        cluster_to_idxs = defaultdict(list)
        for local_i, epi_idx in enumerate(idxs):
            cid = sub_labels[local_i]
            if len(cluster_to_idxs[cid]) < k_per_cluster:
                cluster_to_idxs[cid].append(epi_idx)

        for cid, epi_list in cluster_to_idxs.items():
            selected[(instr, cid)] = epi_list

    return selected


def episode_to_images(episode, target_num_frames: int = 16, use_wrist_image: bool = False):
    """
    从一个 episode 中采样若干帧图像，返回 List[np.ndarray(H,W,3)]
    """
    images = []
    steps = episode["steps"]
    total_steps = len(steps)
    if total_steps <= target_num_frames:
        frame_indices = range(total_steps)
    else:
        frame_indices = np.linspace(0, total_steps - 1, target_num_frames, dtype=int)

    for t, step in enumerate(steps):
        if t in frame_indices:
            obs = step["observation"]
            if use_wrist_image:
                img_np = obs["wrist_image"].numpy()
            else:
                img_np = obs["image"].numpy()
            images.append(img_np.astype("uint8"))
    return images


def save_images_as_mp4(images, out_path: str, fps: int = 8):
    if not images:
        print(f"No images to save for {out_path}")
        return
    imageio.mimsave(out_path, images, fps=fps)
    print(f"Saved video to {out_path}")


def safe_filename(s: str, max_len: int = 80):
    import re
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")
    if not s:
        s = "instr"
    return s[:max_len]


def main():
    # 1. 读取聚类结果
    labels, instructions = load_clusters(CLUSTER_PATH)

    # 2. 按 instruction + cluster 选出每簇的若干 episode index
    k_per_cluster = 2
    selected = build_selection(labels, instructions, k_per_cluster=k_per_cluster)

    # 平铺出所有需要的 episode index，方便快速判断
    target_episode_set = {
        epi_idx for epi_list in selected.values() for epi_idx in epi_list
    }
    print(f"Total target episodes: {len(target_episode_set)}")

    # 3. 加载 TFDS 数据集
    print(f"Loading TFDS dataset '{DATASET_NAME}' from '{DATA_DIR}' ...")
    ds, ds_info = tfds.load(
        DATASET_NAME,
        split="train",
        data_dir=DATA_DIR,
        shuffle_files=False,
        with_info=True,
    )

    # 4. 遍历数据集，根据 episode 索引匹配并导出视频
    # for epi_idx, episode in enumerate(ds):
    for epi_idx, episode in enumerate(ds):
        if epi_idx not in target_episode_set:
            continue

        # 拿到该 episode 的 instruction 字符串，做一个 sanity check（可选）
        # 假设每个 episode 里的所有 step 的 language_instruction 相同，
        # 这里就拿第一步的指令。
        for j,e in enumerate(episode['steps']):
            first_step=e
            break
        # first_step = episode["steps"][0]
        instr = first_step["language_instruction"].numpy().decode("utf-8")

        # 找该 episode 属于哪些 (instr, cluster_id)
        # 理论上只会有一个匹配
        for (instr_key, cid), epi_list in selected.items():
            if epi_idx in epi_list:
                # 再简单 check 一下指令文本是否一致（不一致也继续导出，只是打印 warning）
                if instr != instr_key:
                    print(
                        f"[WARN] Episode {epi_idx} instruction mismatch: "
                        f"from TFDS='{instr}', from clusters='{instr_key}'"
                    )

                images = episode_to_images(
                    episode,
                    target_num_frames=16,
                    use_wrist_image=False,
                )

                instr_slug = safe_filename(instr_key)
                out_name = f"instr_{instr_slug}_cluster{cid}_epi{epi_idx}.mp4"
                out_path = os.path.join(OUT_DIR, out_name)
                save_images_as_mp4(images, out_path, fps=8)

    print("Done.")


if __name__ == "__main__":
    main()