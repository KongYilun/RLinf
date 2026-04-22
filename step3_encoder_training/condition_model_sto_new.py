import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
from transformers import AutoModel
from transformers import AutoProcessor
import math


class SiglipConditionModel(nn.Module):
    """
    SFT 用：SigLIP + 分类头 + cluster_embed(id) + 残差头（确定性残差）。

    用于 ``residual_head`` 的聚类下标由 ``forward`` 的 ``cluster_ids_for_residual`` 指定；
    若为 ``None``，则使用 ``logits_c`` 的 argmax（与旧行为一致）。
    """

    def __init__(
        self,
        model_path: str,
        num_classes: int,
        residual_dim: int,
        center_embed_dim: int = 512,
    ):
        super().__init__()

        self.model_path = model_path
        self.siglip = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self._processor = None

        self.embed_dim = self.siglip.config.vision_config.hidden_size
        self.num_classes = num_classes
        self.residual_dim = residual_dim
        self.center_embed_dim = center_embed_dim

        hidden_dim = 2048
        self.classifier_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.cluster_embed = nn.Embedding(num_classes, center_embed_dim)
        # 默认 N(0,1) 过大，残差头初值与 reg_loss 会爆；与常见 transformer 词表 init 一致用较小 std
        nn.init.normal_(self.cluster_embed.weight, mean=0.0, std=0.02)

        self.residual_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + center_embed_dim, hidden_dim ),
            nn.LayerNorm(hidden_dim ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, residual_dim),
        )
        # self.residual_head = nn.Linear(self.embed_dim * 2 + center_embed_dim, residual_dim)
        self._init_weights()

    def _init_weights(self):
        # 残差头：首层用小高斯；最后一层输出维很大时，即使 std=0.02 也会使 ||pred||^2 ~ O(D*fan_in*σ²)，
        # 初始 residual_l2 上千。最后一层零初始化使 pred≈0，显著降低初始 residual_l2 / reg_loss。
        res_linears = [m for m in self.residual_head.modules() if isinstance(m, nn.Linear)]
        for i, m in enumerate(res_linears):
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            if i == len(res_linears) - 1:
                nn.init.zeros_(m.weight)
            else:
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        for m in self.residual_head:
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    #     for m in self.classifier_head:
    #         if isinstance(m, nn.Linear):
    #             # 1. 针对 ReLU 激活函数的权重初始化 (Kaiming Normal)
    #             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #             # 2. 偏置通常初始化为 0
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.LayerNorm):
            #     # 3. LayerNorm 的 weight (gamma) 设为 1，bias (beta) 设为 0
            #     nn.init.constant_(m.weight, 1.0)
            #     nn.init.constant_(m.bias, 0)


    def _get_processor(self) -> AutoProcessor:
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_path, torch_dtype=torch.bfloat16)
        return self._processor

    def _align_siglip_embedding_dtype(self, embedding: torch.Tensor) -> torch.Tensor:
        """HF SigLIP may return fp32 embeds while heads are bf16 after ``model.to(bfloat16)``."""
        w = next(self.classifier_head.parameters())
        return embedding.to(dtype=w.dtype)

    def forward(
        self,
        images,
        instructions,
        cluster_ids_for_residual: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        images: List[PIL.Image] 或 ndarray(uint8, HxWx3) 的 batch
        instructions: List[str]，与 images 对齐
        cluster_ids_for_residual: 可选 ``[B]`` long，用于 ``cluster_embed`` 与残差头；
            若为 ``None``，则用 ``logits_c.argmax(-1)``。
        """
        processor = self._get_processor()

        proc = processor(images=images, text=instructions, padding="max_length", return_tensors="pt")
        device = next(self.siglip.parameters()).device
        dtype = next(self.siglip.parameters()).dtype
        proc = {k: v.to(device) if torch.is_tensor(v) else v for k, v in proc.items()}

        if "pixel_values" in proc and torch.is_tensor(proc["pixel_values"]):
            proc["pixel_values"] = proc["pixel_values"].to(dtype=dtype)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and dtype == torch.bfloat16
            else torch.autocast(device_type=device.type, enabled=False)
        )
        with autocast_ctx:
            outputs = self.siglip(**proc, **kwargs)

        embedding = self._align_siglip_embedding_dtype(
            torch.cat((outputs.image_embeds, outputs.text_embeds), dim=1)
        )

        logits_c = self.classifier_head(embedding)

        if cluster_ids_for_residual is None:
            pred_idx = logits_c.argmax(dim=-1)
        else:
            pred_idx = cluster_ids_for_residual.to(device=device, dtype=torch.long).view(-1)
            if pred_idx.shape[0] != embedding.shape[0]:
                raise ValueError(
                    f"cluster_ids_for_residual 长度 {pred_idx.shape[0]} 与 batch {embedding.shape[0]} 不一致"
                )

        center_emb = self.cluster_embed(pred_idx)
        combined_input = torch.cat((embedding, center_emb), dim=1)
        residual_embedding = self.residual_head(combined_input)

        return {
            "embedding": embedding,
            "logits_c": logits_c,
            "residual_embedding": residual_embedding,
            "pred_cluster_idx": pred_idx,
        }


class SiglipConditionRLModel(SiglipConditionModel):
    """
    RL / rollout：在 SFT 结构上加对角高斯残差策略，输出离散聚类与连续残差的 log_prob，
    供 REINFORCE 等算法使用。

    - 离散：``Categorical(softmax(logits_c / temperature))``
    - 连续：``Independent(Normal(residual_mean, exp(log_residual_std)), 1)``
    """

    def __init__(
        self,
        model_path: str,
        num_classes: int,
        residual_dim: int,
        center_embed_dim: int = 512,
    ):
        super().__init__(model_path, num_classes, residual_dim, center_embed_dim)
        # self.log_residual_std = nn.Parameter(torch.zeros(residual_dim))
        self.log_residual_std = nn.Parameter(
            torch.ones(residual_dim) * math.log(0.001)
        )
        # Set via ``init_condition_policy_sampling_generator`` for reproducible sampling.
        self._condition_policy_sampling_generator: Optional[torch.Generator] = None
        self._condition_policy_sampling_seed: Optional[int] = None

    def init_condition_policy_sampling_generator(
        self, device: Union[torch.device, int, str], seed: int
    ) -> None:
        """
        Use a dedicated ``torch.Generator`` for cluster / residual sampling in ``forward``.

        Call after ``.to(device)``. Per-rank seeds (e.g. ``base_seed + rank``) avoid
        identical samples across distributed rollout ranks.

        ``device`` may be a ``torch.device``, or a CUDA index ``int`` (as from
        ``torch.cuda.current_device()``), or a device string.

        Omit this call to keep using the global PyTorch RNG (legacy behavior).
        """
        self._condition_policy_sampling_seed = int(seed) #% (2**63)
        if isinstance(device, torch.device):
            dev = device
        elif isinstance(device, int):
            dev = torch.device("cuda", device)
        else:
            dev = torch.device(device)
        if dev.type == "cuda":
            gen = torch.Generator(device=dev)
        else:
            gen = torch.Generator()
        gen.manual_seed(self._condition_policy_sampling_seed)
        self._condition_policy_sampling_generator = gen

    def load_sft_state_dict(self, state_dict: dict, strict: bool = False) -> None:
        """加载 SFT 训练得到的权重；``log_residual_std`` 可不在 ckpt 中。"""
        self.load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        images,
        instructions,
        cluster_sample_temperature: float = 1.0,
        deterministic_cluster: bool = False,
        deterministic_residual: bool = False,
        **kwargs,
    ):
        """
        cluster_sample_temperature: 离散策略 softmax 温度
        deterministic_cluster: 若为 True，聚类用 argmax 而非采样
        deterministic_residual: 若为 True，残差用均值（不采样），log_prob_residual 对「当前输出」仍按高斯计
        """
        processor = self._get_processor()

        proc = processor(images=images, text=instructions, padding="max_length", return_tensors="pt")
        device = next(self.siglip.parameters()).device
        dtype = next(self.siglip.parameters()).dtype
        proc = {k: v.to(device) if torch.is_tensor(v) else v for k, v in proc.items()}

        if "pixel_values" in proc and torch.is_tensor(proc["pixel_values"]):
            proc["pixel_values"] = proc["pixel_values"].to(dtype=dtype)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and dtype == torch.bfloat16
            else torch.autocast(device_type=device.type, enabled=False)
        )
        with autocast_ctx:
            outputs = self.siglip(**proc, **kwargs)

        embedding = self._align_siglip_embedding_dtype(
            torch.cat((outputs.image_embeds, outputs.text_embeds), dim=1)
        )
        logits_c = self.classifier_head(embedding)

        temp = max(float(cluster_sample_temperature), 1e-8)
        probs = F.softmax(logits_c.float() / temp, dim=-1)
        cat_dist = torch.distributions.Categorical(probs=probs)

        gen = self._condition_policy_sampling_generator

        if deterministic_cluster:
            pred_idx = logits_c.argmax(dim=-1)
        else:
            if gen is not None:
                pred_idx = torch.multinomial(
                    probs, num_samples=1, replacement=True, generator=gen
                ).squeeze(-1)
            else:
                pred_idx = cat_dist.sample()

        log_prob_cluster = cat_dist.log_prob(pred_idx)

        center_emb = self.cluster_embed(pred_idx)
        combined_input = torch.cat((embedding, center_emb), dim=1)
        residual_mean = self.residual_head(combined_input)

        std = self.log_residual_std.exp().view(1, -1).expand_as(residual_mean)
        normal = torch.distributions.Normal(residual_mean.float(), std.float())
        r_dist = torch.distributions.Independent(normal, 1)

        if deterministic_residual:
            residual_sample = residual_mean
        else:
            if gen is not None:
                eps = torch.randn(
                    residual_mean.shape,
                    generator=gen,
                    device=residual_mean.device,
                    dtype=torch.float32,
                )
                residual_sample = (residual_mean.float() + std.float() * eps).to(
                    dtype=residual_mean.dtype
                )
            else:
                residual_sample = r_dist.rsample() if self.training else r_dist.sample()

        log_prob_residual = r_dist.log_prob(residual_sample.float())

        return {
            "embedding": embedding,
            "logits_c": logits_c,
            "residual_mean": residual_mean,
            "residual_embedding": residual_sample,
            "pred_cluster_idx": pred_idx,
            "log_prob_cluster": log_prob_cluster,
            "log_prob_residual": log_prob_residual,
            "log_prob_joint": log_prob_cluster + log_prob_residual,
            "residual_std": std,
        }

    def forward_z_for_actor_grpo(
        self,
        images: list,
        instructions: list[str],
        pred_cluster_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        VLA GRPO 反传用：SigLIP 前向无梯度，``embedding`` detach 后仅 ``cluster_embed`` 与
        ``residual_head`` 参与计算图，得到与 rollout 同形状的连续条件向量 ``[B, residual_dim]``。
        """
        processor = self._get_processor()

        proc = processor(images=images, text=instructions, padding="max_length", return_tensors="pt")
        device = next(self.siglip.parameters()).device
        dtype = next(self.siglip.parameters()).dtype
        proc = {k: v.to(device) if torch.is_tensor(v) else v for k, v in proc.items()}

        if "pixel_values" in proc and torch.is_tensor(proc["pixel_values"]):
            proc["pixel_values"] = proc["pixel_values"].to(dtype=dtype)

        pred_cluster_idx = pred_cluster_idx.to(device=device, dtype=torch.long).view(-1)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and dtype == torch.bfloat16
            else torch.autocast(device_type=device.type, enabled=False)
        )
        with autocast_ctx:
            with torch.no_grad():
                outputs = self.siglip(**proc)
            embedding = self._align_siglip_embedding_dtype(
                torch.cat((outputs.image_embeds, outputs.text_embeds), dim=1).detach()
            )
            center_emb = self.cluster_embed(pred_cluster_idx)
            combined_input = torch.cat((embedding, center_emb), dim=-1)
            z = self.residual_head(combined_input)
        return z

    def evaluate_log_prob(
        self,
        images,
        instructions,
        pred_cluster_idx: torch.Tensor,
        residual_sample: torch.Tensor,
        cluster_sample_temperature: float = 1.0,
        reinforce_logprob: str = "joint",
    ) -> dict[str, torch.Tensor]:
        """
        Log-probability of fixed discrete cluster indices and residual samples under the
        current policy (for REINFORCE). Does not resample.

        Args:
            images: Same format as ``forward`` (list of HWC uint8 / ndarray).
            instructions: List of strings, batch-aligned with ``pred_cluster_idx``.
            pred_cluster_idx: ``[B]`` long on the same device as model outputs after forward.
            residual_sample: ``[B, residual_dim]`` (float), the sampled residuals from rollout.
            cluster_sample_temperature: Softmax temperature for the categorical.
            reinforce_logprob: ``"joint"`` | ``"joint_per_dim"`` | ``"cluster"``.
                ``joint`` uses the true factorized log-density sum (residual term is O(D)
                and often dominates gradients). ``joint_per_dim`` uses
                ``log_prob_cluster + log_prob_residual / D`` as a surrogate so both
                branches receive comparable REINFORCE signal (not the exact product policy
                log-prob).
        """
        processor = self._get_processor()

        proc = processor(images=images, text=instructions, padding="max_length", return_tensors="pt")
        device = next(self.siglip.parameters()).device
        dtype = next(self.siglip.parameters()).dtype
        proc = {k: v.to(device) if torch.is_tensor(v) else v for k, v in proc.items()}

        if "pixel_values" in proc and torch.is_tensor(proc["pixel_values"]):
            proc["pixel_values"] = proc["pixel_values"].to(dtype=dtype)

        pred_cluster_idx = pred_cluster_idx.to(device=device, dtype=torch.long).view(-1)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and dtype == torch.bfloat16
            else torch.autocast(device_type=device.type, enabled=False)
        )
        with autocast_ctx:
            outputs = self.siglip(**proc)

        embedding = self._align_siglip_embedding_dtype(
            torch.cat((outputs.image_embeds, outputs.text_embeds), dim=1)
        )
        logits_c = self.classifier_head(embedding)

        temp = max(float(cluster_sample_temperature), 1e-8)
        probs = F.softmax(logits_c.float() / temp, dim=-1)
        cat_dist = torch.distributions.Categorical(probs=probs)
        log_prob_cluster = cat_dist.log_prob(pred_cluster_idx)

        center_emb = self.cluster_embed(pred_cluster_idx)
        combined_input = torch.cat((embedding, center_emb), dim=1)
        residual_mean = self.residual_head(combined_input)

        std = self.log_residual_std.exp().view(1, -1).expand_as(residual_mean)
        normal = torch.distributions.Normal(residual_mean.float(), std.float())
        r_dist = torch.distributions.Independent(normal, 1)
        log_prob_residual = r_dist.log_prob(residual_sample.float())

        log_prob_joint = log_prob_cluster + log_prob_residual
        d_res = residual_sample.shape[-1]
        if reinforce_logprob == "cluster":
            train_log_prob = log_prob_cluster
        elif reinforce_logprob == "joint":
            train_log_prob = log_prob_joint
        elif reinforce_logprob == "joint_per_dim":
            train_log_prob = log_prob_cluster + log_prob_residual / float(d_res)
        else:
            raise ValueError(f"Unknown reinforce_logprob={reinforce_logprob!r}")

        return {
            "log_prob_cluster": log_prob_cluster,
            "log_prob_residual": log_prob_residual,
            "log_prob_joint": log_prob_joint,
            "train_log_prob": train_log_prob,
            "residual_mean": residual_mean,
            "cluster_entropy": cat_dist.entropy(),
        }


__all__ = ["SiglipConditionModel", "SiglipConditionRLModel"]
