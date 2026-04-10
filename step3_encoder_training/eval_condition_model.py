"""
在 RLDS 上评估 SiglipConditionModel（step3 SFT 权重）。

GT：由 episode_id 对齐 step1 ``libero_object_per_instruction_clusters_proj_all.pt`` 中的
label 与 residual（与 train_condition_model 一致）。
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from condition_model_sto_new import SiglipConditionModel,SiglipConditionRLModel

# 复用训练脚本的数据构建（避免重复维护 RLDS 逻辑）
from train_condition_model import ROOT_DIR, LiberoFirstFrameDataset


def set_seed(seed: int) -> None:
    """Fix Python / NumPy / PyTorch RNG and prefer deterministic CUDA kernels where supported."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def _dataloader_worker_init_fn(_worker_id: int) -> None:
    w_seed = torch.initial_seed() % 2**32
    np.random.seed(w_seed)
    random.seed(w_seed)


def _sort_rlds_samples_for_repro(samples: List[dict]) -> List[dict]:
    """
    ``dataset.as_numpy_iterator()`` 依赖 TFDS 分片/预取顺序，跨次运行顺序常不同。
    按 (episode_id, instruction) 稳定排序，并用原始下标打破并列，保证 DataLoader 批顺序可复现。
    """
    indexed = list(enumerate(samples))
    indexed.sort(key=lambda t: (str(t[1]["episode_id"]), t[1]["instruction"], t[0]))
    return [t[1] for t in indexed]


@torch.no_grad()
def evaluate_siglip_condition_model(
    model: SiglipConditionModel,
    loader: DataLoader,
    device: torch.device,
    log_prefix: str = "",
    print_per_task: bool = True,
) -> Dict[str, float]:
    """
    与 standalone 脚本相同的评测：全局分类/残差 MSE + 按 instruction 的分类准确率。
    调用前后会切换 ``model.eval()`` / ``model.train()``（若原为 train 模式则恢复）。
    """
    was_training = model.training
    model.eval()
    try:
        total = 0
        correct_cls = 0
        sum_mse_gt = 0.0
        sum_mse_argmax = 0.0
        sum_mse_argmax_match = 0.0
        n_argmax_match = 0
        per_task_correct: Dict[str, int] = defaultdict(int)
        per_task_total: Dict[str, int] = defaultdict(int)
        per_task_predict: Dict[str, list[int]] = defaultdict(list)

        for batch in loader:
            images = batch["image"]
            instructions = batch["instruction"]
            labels = batch["label"].to(device)
            residual_targets = batch["residual"].to(device)
            bsz = labels.size(0)

            out_gt = model(
                images=images,
                instructions=instructions,
                deterministic_cluster=True,
                # cluster_ids_for_residual=None
            )
            out_am = model(images=images, instructions=instructions, deterministic_cluster=False)
            # cluster_ids_for_residual=None
            
            logits = out_gt["logits_c"]
            pred_cls = logits.argmax(dim=-1)
            import ipdb; ipdb.set_trace()
            match_vec = pred_cls == labels
            correct_cls += match_vec.sum().item()

            for i in range(bsz):
                inst = instructions[i]
                per_task_total[inst] += 1
                per_task_predict[inst].append(pred_cls[i].item())
                if bool(match_vec[i].item()):
                    per_task_correct[inst] += 1

            mse_gt = F.mse_loss(out_gt["residual_embedding"], residual_targets, reduction="none").mean(
                dim=-1
            )
            mse_am = F.mse_loss(out_am["residual_embedding"], residual_targets, reduction="none").mean(
                dim=-1
            )

            sum_mse_gt += mse_gt.sum().item()
            sum_mse_argmax += mse_am.sum().item()

            match = pred_cls == labels
            if match.any():
                sum_mse_argmax_match += mse_am[match].sum().item()
                n_argmax_match += int(match.sum().item())

            total += bsz

        acc = correct_cls / max(total, 1)
        mse_gt_mean = sum_mse_gt / max(total, 1)
        mse_argmax_mean = sum_mse_argmax / max(total, 1)
        mse_argmax_when_correct = sum_mse_argmax_match / max(n_argmax_match, 1)

        pre = f"{log_prefix} " if log_prefix else ""
        print(f"\n{pre}--- Metrics (global) ---")
        print(f"{pre}classification_accuracy (argmax vs GT label): {acc:.4f}")
        print(f"{pre}residual_mse (GT cluster id in residual branch): {mse_gt_mean:.6f}")
        print(
            f"{pre}residual_mse (argmax cluster in residual branch vs same GT residual target): "
            f"{mse_argmax_mean:.6f}"
        )
        print(
            f"{pre}residual_mse (argmax branch, only samples where argmax==GT): "
            f"{mse_argmax_when_correct:.6f} (n={n_argmax_match}/{total})"
        )

        if print_per_task:
            print(
                f"\n{pre}--- Per-task classification accuracy (by instruction), "
                f"{len(per_task_total)} tasks ---"
            )
            for inst in sorted(per_task_total.keys()):
                n_t = per_task_total[inst]
                c_t = per_task_correct[inst]
                acc_t = c_t / max(n_t, 1)
                print(f"{pre}n={n_t:4d}  acc={acc_t:.4f}  ({c_t}/{n_t})  |  {inst}")
                print(f"{pre}predict: {per_task_predict[inst]}")

        return {
            "accuracy": float(acc),
            "mse_residual_gt_path": float(mse_gt_mean),
            "mse_residual_argmax_path": float(mse_argmax_mean),
            "mse_residual_argmax_when_correct": float(mse_argmax_when_correct),
            "n_total": float(total),
            "n_argmax_match": float(n_argmax_match),
        }
    finally:
        if was_training:
            model.train()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SiglipConditionModel on LIBERO RLDS + step1 clusters.")
    p.add_argument(
        "--dataset_root",
        type=Path,
        default=ROOT_DIR / "dataset",
        help="TFDS / RLDS 根目录",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default="libero_object_no_noops",
        help="OXE 数据集名，如 libero_object_no_noops",
    )
    p.add_argument(
        "--clusters_path",
        type=Path,
        default=ROOT_DIR / "step1_embedding" / "libero_object_per_instruction_clusters_proj_all.pt",
        help="step1 聚类与 episode_id 监督",
    )
    p.add_argument(
        "--ckpt_path",
        type=Path,
        default=ROOT_DIR / "step3_encoder_training" / "siglip_condition_model_sto_0.pt",
        help="SFT checkpoint",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    p.add_argument(
        "--rlds_eval_split",
        action="store_true",
        help="使用 make_single_dataset(train=False)。默认 train=True（与 train_condition_model 一致）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="全局随机种子（Python / NumPy / PyTorch；DataLoader worker 由此派生）。",
    )
    return p.parse_args()


def build_episode_supervision(clusters_path: Path) -> Tuple[Dict[str, Tuple[int, np.ndarray]], int, int]:
    clusters = torch.load(clusters_path, map_location="cpu")
    labels: torch.Tensor = clusters["labels"]
    projected_embeds: torch.Tensor = clusters["projected_embeds"]
    cluster_centers: torch.Tensor = clusters["cluster_centers"]
    episode_ids: List = clusters["episode_ids"]
    assert projected_embeds.shape == cluster_centers.shape

    residuals = (projected_embeds.float() - cluster_centers.float()).cpu().numpy()
    residual_dim = projected_embeds.shape[-1]
    num_classes = int(labels.max().item()) + 1

    episode_to_supervision: Dict[str, Tuple[int, np.ndarray]] = {}
    for idx, eid in enumerate(episode_ids):
        episode_to_supervision[str(eid)] = (int(labels[idx].item()), residuals[idx])

    return episode_to_supervision, num_classes, residual_dim


def _build_rlds_first_frame_samples_split(
    dataset_root: Path,
    dataset_name: str,
    episode_to_supervision: Dict[str, Tuple[int, np.ndarray]],
    train: bool,
) -> List[dict]:
    """与 train_condition_model.build_rlds_first_frame_samples 相同，增加 train 开关；返回前稳定排序以利复现。"""
    OPENVLA_COND_DIR = ROOT_DIR / "step2_warmup" / "openvla-oft-conditioned"
    if str(OPENVLA_COND_DIR) not in sys.path:
        sys.path.insert(0, str(OPENVLA_COND_DIR))
    from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs  # type: ignore
    from prismatic.vla.datasets.rlds.dataset import make_single_dataset  # type: ignore

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
        window_size=1,
        future_action_window_size=0,
        subsample_length=None,
        skip_unlabeled=True,
        max_action=None,
        max_proprio=None,
        task_augment_strategy=None,
        task_augment_kwargs={},
    )
    frame_transform_kwargs = dict(
        image_augment_kwargs={},
        resize_size=(384, 384),
        depth_resize_size={},
    )
    dataset, _, _ = make_single_dataset(
        dataset_kwargs=dataset_kwargs,
        train=train,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
    )

    samples: List[dict] = []
    for frame in dataset.as_numpy_iterator():
        obs = frame["observation"]
        task = frame["task"]
        timesteps = obs["timestep"]
        if int(timesteps[0]) != 0:
            continue
        episode_ids = obs.get("episode_id", None)
        if episode_ids is None:
            continue
        episode_id = str(episode_ids[0][0])
        if episode_id not in episode_to_supervision:
            continue
        image_primary = obs["image_primary"][0][0]
        lang_arr = task["language_instruction"]
        instruction = str(
            lang_arr[0].decode("utf-8") if isinstance(lang_arr[0], (bytes, bytearray)) else lang_arr[0]
        )
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
    return _sort_rlds_samples_for_repro(samples)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"seed={args.seed} (device={device})")

    episode_to_supervision, num_classes, residual_dim = build_episode_supervision(args.clusters_path)
    print(f"Clusters: {len(episode_to_supervision)} episodes, num_classes={num_classes}, residual_dim={residual_dim}")

    rlds_train = True
    samples = _build_rlds_first_frame_samples_split(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        episode_to_supervision=episode_to_supervision,
        train=rlds_train,
    )
    if not samples:
        raise RuntimeError(
            f"未找到对齐样本 dataset={args.dataset_name} train={rlds_train}。"
            "若当前为空，可尝试去掉 --rlds_eval_split 或改用与训练相同的数据集名。"
        )

    dataset = LiberoFirstFrameDataset(samples=samples)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=_dataloader_worker_init_fn if args.num_workers > 0 else None,
        generator=loader_generator,
    )
    print(f"Eval samples: {len(dataset)} (dataset_name={args.dataset_name}, rlds_train={rlds_train})")

    if not args.ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt_path}")

    ckpt = torch.load(args.ckpt_path, map_location=device)
    print(f"ckpt path: {args.ckpt_path}")
    state_dict = ckpt.get("model_state_dict", ckpt)
    siglip_model_path = ckpt.get("siglip_model_path", "/data/users/kongyilun/models/siglip-so400m-patch14-384")#"/data/users/kongyilun/code/rlinf-condition/step3_encoder_training/siglip_condition_model_sto_0.pt"
    #
    center_embed_dim = int(ckpt.get("center_embed_dim", 512))
    print(f"siglip_model_path: {siglip_model_path}")
    model = SiglipConditionRLModel(
        model_path=siglip_model_path,
        num_classes=num_classes,
        residual_dim=residual_dim,
        center_embed_dim=center_embed_dim,
    )
    # import ipdb; ipdb.set_trace()
    tmp=torch.load("/data/users/kongyilun/code/rlinf-condition/logs/20260406-16:12:22-libero_object_grpo_openvlaoft_opt.yaml/libero_object_z_generator_konly_refined_new/checkpoints/global_step_25/condition_policy.pt", map_location="cpu")
    model.load_state_dict(tmp['model'], strict=False)
    model.to(device)

    evaluate_siglip_condition_model(model, loader, device, log_prefix="")


__all__ = ["evaluate_siglip_condition_model"]


if __name__ == "__main__":
    main()
