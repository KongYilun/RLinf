import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
import torch
from rzen_embed_inference import RzenEmbed
import torch.nn.functional as F

def load_libero_object_no_noops(
    data_dir: str = "/data/users/kongyilun/code/RLinf/dataset",
    split: str = "train",
):
    """
    加载 TFDS 的 libero_object_no_noops 数据集。
    """
    dataset_name = "libero_object_no_noops"
    print(f"Loading TFDS dataset '{dataset_name}' from data_dir='{data_dir}' ...")
    ds, ds_info = tfds.load(
        dataset_name,
        split=split,
        data_dir=data_dir,
        shuffle_files=False,
        with_info=True,
    )
    # print(ds_info)
    return ds, ds_info


def episode_to_images_and_instruction(
    episode,
    target_num_frames: int = 16,
    use_wrist_image: bool = False,
):
    """
    从一个 episode 中抽取图像序列和语言指令。

    返回:
        images: List[PIL.Image]
        instruction: str
    """
    images = []
    instruction_str = None
    episode_id = None

    steps = episode["steps"]
    total_steps = len(steps)
    print(total_steps)
    if total_steps <= target_num_frames:
        frame_indices = range(total_steps)
    else:
        frame_indices = np.linspace(0, total_steps - 1, target_num_frames, dtype=int)

    for t, step in enumerate(steps):
        # 先从 step 中读取 episode_id
        step_episode_id = int(step["episode_id"].numpy())  # 如有不同字段名，这里改一下

        if episode_id is None:
            episode_id = step_episode_id
        else:
            if step_episode_id != episode_id:
                raise ValueError(
                    f"Inconsistent episode_id in one episode: {episode_id} vs {step_episode_id}"
                )

        if t in frame_indices:
            obs = step["observation"]

            # 选择使用主相机图像或腕部相机图像
            if use_wrist_image:
                img_np = obs["wrist_image"].numpy()
            else:
                img_np = obs["image"].numpy()  # [H, W, 3], uint8

            # numpy -> PIL.Image
            img_pil = Image.fromarray(img_np.astype("uint8"))
            images.append(img_pil)

            # 同一条轨迹里 language_instruction 通常相同，取第一条即可
            if instruction_str is None:
                instruction_str = step["language_instruction"].numpy().decode("utf-8")

    return images, instruction_str, episode_id


def encode_libero_with_razen(
    data_dir: str = "/data/users/kongyilun/code/RLinf/dataset",
    split: str = "train",
    num_episodes: int = 4,
    max_steps: int = 16,
    use_wrist_image: bool = False,
):
    """
    读取 libero_object_no_noops 的若干个 episode，并用 RzenEmbed 编码其中的
    图像序列和语言指令，输出相似度。
    """
    # 1. 加载数据集
    ds, ds_info = load_libero_object_no_noops(data_dir=data_dir, split=split)

    # 2. 初始化 razen 模型（和 step1_embedding/razen_embed.py 一致）
    rzen = RzenEmbed("/data/users/kongyilun/models/RzenEmbed")

    # 我们把每个 episode 的语言指令当成 query，把对应的图像序列当成 candidate
    query_texts = []
    candidate_images = []
    episode_ids = []
    print("\nCollecting episodes ...")
    for epi_idx, episode in enumerate(ds):#.take(num_episodes)
        images, instruction, episode_id = episode_to_images_and_instruction(
            episode,
            target_num_frames=16,
            use_wrist_image=use_wrist_image,
        )

        if not images or instruction is None:
            continue

        query_texts.append(instruction)
        candidate_images.append(images)
        episode_ids.append(episode_id)

        print(f"[Episode {epi_idx}] instruction: {instruction}")
        print(f"  #frames collected: {len(images)}")
        print(f"  episode_id from steps: {episode_id}")

    if not query_texts:
        print("No valid episodes collected.")
        return

    # 3. 使用 razen 编码
    #   - 对文本：用 instruction + queries（这里 queries 就是 episode 指令列表）
    #   - 对图像：用 candidate_instruction + images（这里 images 是每个 episode 的图像序列）
    print("\nEncoding with RzenEmbed ...")


    # 文本指令编码，形状大致为 [N_episodes, D]
    # query_embeds = rzen.get_fused_embeddings(
    #     instruction=query_instruction,
    #     texts=query_texts,
    # )

    # 图像序列编码（每个元素是 List[PIL.Image]），形状大致为 [N_episodes, D]
    embeds = rzen.get_fused_embeddings(
        # text=query_texts,
        images=candidate_images,
    )
    save_path='libero_object_embeded_id.pt'
    torch.save({
        "embeds": embeds.cpu(),          # [N, 3584]
        "instructions": query_texts,     # 对应的语言指令
        "episode_ids": episode_ids,
    }, save_path)
    print(f"Embeddings saved to {save_path}")
    # print(embeds[:,:5])
    # print(embeds.shape)
    # print(type(embeds))
    # cos_sim = F.cosine_similarity(embeds[2].unsqueeze(0), embeds[3].unsqueeze(0), dim=1).item()
    # print(cos_sim)




if __name__ == "__main__":
    # 你可以根据需要修改参数
    encode_libero_with_razen(
        data_dir="/data/users/kongyilun/code/RLinf/dataset",
        split="train",
        num_episodes=4,
        max_steps=512,
        use_wrist_image=False,
    )