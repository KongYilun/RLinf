import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 过滤特定的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message="The given NumPy array is not writable")

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset

from condition_model_sto_new import SiglipConditionModel
from eval_condition_model import evaluate_siglip_condition_model


# 让 Python 能找到 step2 中的 prismatic 代码
ROOT_DIR = Path(__file__).resolve().parents[1]
OPENVLA_COND_DIR = ROOT_DIR / "step2_warmup" / "openvla-oft-conditioned"
sys.path.insert(0, str(OPENVLA_COND_DIR))
from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs  # type: ignore  # noqa: E402
from prismatic.vla.datasets.rlds.dataset import make_single_dataset  # type: ignore  # noqa: E402


class LiberoFirstFrameDataset(Dataset):
    """
    离线遍历 RLDS 数据集，只保留：
      - 每个 episode 的第一帧图像 (timestep == 0)
      - 对应的 language instruction
      - 与 episode_id 对齐的 cluster label 和 residual 监督信号
    """

    def __init__(
        self,
        samples: List[Dict],
    ) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        return {
            "image": sample["image"],  # H x W x 3, uint8 (np.ndarray)
            "instruction": sample["instruction"],  # str
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "residual": torch.from_numpy(sample["residual"]).float(),  # [residual_dim]
        }


def build_rlds_first_frame_samples(
    dataset_root: Path,
    dataset_name: str,
    episode_to_supervision: Dict[str, Tuple[int, np.ndarray]],
) -> List[Dict]:
    """
    使用 prismatic 的 RLDS pipeline 从 TFRecord 中抽取：
      - 每个 episode 的第一帧 primary camera 图像
      - 对应的语言指令
      - 并按照 episode_id 与 cluster 监督对齐

    episode_to_supervision: {episode_id: (label, residual_np)}
    """
    dataset_kwargs = make_oxe_dataset_kwargs(
        dataset_name=dataset_name,
        data_root_dir=dataset_root,
        load_camera_views=("primary",),
        load_depth=False,
        load_proprio=False,
        load_language=True,
    )

    traj_transform_kwargs = dict(
        goal_relabeling_strategy=None,
        goal_relabeling_kwargs={},
        window_size=1,  # 每个样本只包含单步
        future_action_window_size=0,
        subsample_length=None,
        skip_unlabeled=True,
        max_action=None,
        max_proprio=None,
        task_augment_strategy=None,
        task_augment_kwargs={},
    )

    frame_transform_kwargs = dict(
        image_augment_kwargs={},  # 这里不做图像增强
        resize_size=(384, 384),  # SigLIP so400m patch14 默认输入
        depth_resize_size={},
    )
    # import ipdb; ipdb.set_trace()
    dataset, _, _ = make_single_dataset(
        dataset_kwargs=dataset_kwargs,
        train=True,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
    )

    samples: List[Dict] = []
    # import ipdb; ipdb.set_trace()
    for frame in dataset.as_numpy_iterator():
        obs = frame["observation"]
        task = frame["task"]

        # window_size = 1，所以这些维度的第 0 个元素就是当前 step
        timesteps = obs["timestep"]  # [1]
        if int(timesteps[0]) != 0:
            continue

        # episode_id 需要在 rlds 的 trajectory 顶层存在；在 make_dataset_from_rlds 中已经透传到 observation["episode_id"]
        episode_ids = obs.get("episode_id", None)
        if episode_ids is None:
            continue
        episode_id = str(episode_ids[0][0])

        if episode_id not in episode_to_supervision:
            continue

        # primary camera 图像，形状 [1, H, W, 3]，取第 0 帧
        image_primary = obs["image_primary"][0][0]  # H x W x 3, uint8
        lang_arr = task["language_instruction"]
        instruction = str(lang_arr[0].decode("utf-8") if isinstance(lang_arr[0], (bytes, bytearray)) else lang_arr[0])
        # import ipdb; ipdb.set_trace()
        label, residual = episode_to_supervision[episode_id]

        samples.append(
            {
                "episode_id": episode_id,
                "image": image_primary,
                "instruction": instruction,
                "label": int(label),
                "residual": residual,
            }
        )

    return samples


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 加载聚类监督 ---
    clusters_path = ROOT_DIR / "step1_embedding" / "libero_object_per_instruction_clusters_proj_all.pt"
    clusters = torch.load(clusters_path, map_location="cpu")

    labels: torch.Tensor = clusters["labels"]  # [N]
    projected_embeds: torch.Tensor = clusters["projected_embeds"]  # [N, D]
    cluster_centers: torch.Tensor = clusters["cluster_centers"]  # [N, D]，仅用于算 residual 监督
    episode_ids: List = clusters["episode_ids"]  # len N
    # import ipdb; ipdb.set_trace()
    assert projected_embeds.shape == cluster_centers.shape
    # bfloat16 无法直接 .numpy()，先转成 float32 再到 CPU
    residuals = (projected_embeds.float() - cluster_centers.float()).cpu().numpy()  # [N, D]

    residual_dim = projected_embeds.shape[-1]
    num_classes = int(labels.max().item()) + 1

    # 建立 episode_id -> (label, residual) 映射
    episode_to_supervision: Dict[str, Tuple[int, np.ndarray]] = {}
    for idx, eid in enumerate(episode_ids):
        episode_to_supervision[str(eid)] = (int(labels[idx].item()), residuals[idx])

    # --- 2. 从 RLDS 数据集中抽取 (instruction, 第一帧图像) ---
    dataset_root = ROOT_DIR / "dataset"
    dataset_name = "libero_object_no_noops100"
    eval_dataset_name = "libero_object_no_noops"

    siglip_model_path = "/data/users/kongyilun/models/siglip-so400m-patch14-384"
    samples = build_rlds_first_frame_samples(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        episode_to_supervision=episode_to_supervision,
    )

    if not samples:
        raise RuntimeError("未在 RLDS 数据集中找到任何与聚类文件对齐的 episode / timestep == 0 样本。")

    eval_samples = build_rlds_first_frame_samples(
        dataset_root=dataset_root,
        dataset_name=eval_dataset_name,
        episode_to_supervision=episode_to_supervision,
    )
    if not eval_samples:
        raise RuntimeError(
            f"评测集 {eval_dataset_name} 在与 step1 聚类对齐后为空，请检查 dataset 路径与 episode_id。"
        )

    dataset = LiberoFirstFrameDataset(samples=samples)
    eval_dataset = LiberoFirstFrameDataset(samples=eval_samples)

    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # 评测：全量 libero_object_no_noops，顺序固定
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    eval_every_epochs = 20
    print(
        f"num_classes: {num_classes}, residual_dim: {residual_dim} | "
        f"train samples ({dataset_name}): {len(dataset)} | "
        f"eval samples ({eval_dataset_name}): {len(eval_dataset)}"
    )
    # --- 3. 初始化条件编码模型 ---
    center_embed_dim = 512
    model = SiglipConditionModel(
        model_path=siglip_model_path,
        num_classes=num_classes,
        residual_dim=residual_dim,
        center_embed_dim=center_embed_dim,
    )
    model.to(device)

    # ckpt_path = ROOT_DIR / "step3_encoder_training" / "siglip_condition_model_sto_0.pt"
    # ckpt = torch.load(ckpt_path, map_location=device)
    # model.load_state_dict(ckpt["model_state_dict"], strict=True)
    # print(f"Loaded weights from {ckpt_path}")

    finetune_max_steps = 50  # 仅再训练这么多 optimizer step
    print(f"Finetuning for at most {finetune_max_steps} optimizer steps")

    # 分类使用交叉熵，residual 使用 L2/MSE
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()

    initial_lr = 1e-5
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr, weight_decay=1e-2
    )

    num_epochs = 200
    steps_per_epoch = len(dataloader)
    print(f"steps_per_epoch: {steps_per_epoch}")
    total_optimizer_steps = max(1, num_epochs * steps_per_epoch)
    eta_min = max(1e-7, initial_lr * 0.01)
    # 按 optimizer step 线性降到 eta_min，之后保持（避免 Cosine 在 T_max 后回升）
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=eta_min / initial_lr,
        total_iters=total_optimizer_steps,
    )

    # residual L2 正则的权重系数（可以按需调整）
    residual_l2_weight = 1.0
    # 以小概率对残差支路使用非 GT 聚类 id；此类样本不反传 MSE，仅保留 L2(reg)
    sft_residual_mismatch_prob = 0.1

    for epoch in range(num_epochs):
        model.train()
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_residual_l2 = 0.0
        total_samples = 0
        # import ipdb; ipdb.set_trace()

        for batch in dataloader:
            images = batch["image"]  # List[np.ndarray] (default collate keeps object/list-like)
            instructions = batch["instruction"]  # List[str]
            labels_batch = batch["label"].to(device)  # [B]
            residual_targets = batch["residual"].to(device)  # [B, residual_dim]

            optimizer.zero_grad()
            B = labels_batch.size(0)
            k_res = labels_batch.clone()
            if num_classes > 1 and sft_residual_mismatch_prob > 0:
                mis = torch.rand(B, device=device) < sft_residual_mismatch_prob
                if mis.any():
                    rnd = torch.randint(
                        0, num_classes - 1, (int(mis.sum().item()),), device=device
                    )
                    y_sub = labels_batch[mis]
                    k_alt = rnd + (rnd >= y_sub).long()
                    k_res = k_res.clone()
                    k_res[mis] = k_alt

            outputs = model(
                images=images,
                instructions=instructions,
                cluster_ids_for_residual=k_res,
            )
            logits_c = outputs["logits_c"]  # [B, num_classes]
            residual_pred = outputs["residual_embedding"]  # [B, residual_dim]

            cls_loss = cls_criterion(logits_c, labels_batch)
            match_mask = k_res == labels_batch
            reg_mse_per = (residual_pred - residual_targets).pow(2).sum(dim=-1)
            sum_match = match_mask.float().sum()
            reg_loss = torch.where(
                sum_match > 0,
                (reg_mse_per * match_mask.float()).sum() / sum_match ,
                torch.zeros((), device=device, dtype=reg_mse_per.dtype),
            )

            residual_l2 = (residual_pred.pow(2).sum(dim=1).mean())

            loss = cls_loss + reg_loss + residual_l2_weight * residual_l2
            loss.backward()
            optimizer.step()
            scheduler.step()

            bs = labels_batch.size(0)
            total_samples += bs
            total_cls_loss += cls_loss.item() * bs
            total_reg_loss += reg_loss.item() * bs
            total_residual_l2 += residual_l2.item() * bs

        avg_cls = total_cls_loss / total_samples
        avg_reg = total_reg_loss / total_samples
        avg_residual_l2 = total_residual_l2 / total_samples
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{num_epochs} - lr: {lr_now:.2e} - "
            f"cls_loss: {avg_cls:.4f}, reg_loss: {avg_reg:.4f}, residual_l2: {avg_residual_l2:.4f}"
        )

        if (epoch + 1) % eval_every_epochs == 0:
            evaluate_siglip_condition_model(
                model,
                eval_dataloader,
                device,
                log_prefix=f"[Eval epoch {epoch + 1}/{num_epochs}]",
                print_per_task=True,
            )

    # 可选：保存训练好的权重
    out_path = ROOT_DIR / "step3_encoder_training" / "siglip_condition_model_sto_0.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "residual_dim": residual_dim,
            "siglip_model_path": siglip_model_path,
            "center_embed_dim": center_embed_dim,
        },
        out_path,
    )
    print(f"保存训练好的条件编码模型到: {out_path}")


if __name__ == "__main__":
    main()

