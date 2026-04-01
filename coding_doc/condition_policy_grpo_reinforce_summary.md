# SigLIP 条件策略：GRPO + REINFORCE 双优化与实现说明

本文档整理相关讨论与当前代码中的实现要点，便于对照配置与排错。

---

## 1. 背景：离散 log prob 与残差 log prob 的量级差

在因子化策略 \(\pi(c, r \mid s) = \pi_c(c\mid s)\,\pi_r(r\mid c,s)\) 下：

- **分类**：`log_prob_cluster` 为单项，量级约在 \(\log(1/K)\) 到 \(0\) 之间（例如约 \(-2 \sim -4\)）。
- **对角高斯残差**：`log_prob_residual` 是 **\(D\) 维** log 密度之和。在样本接近均值时，每维约 \(-\log\sigma - \text{const}\)；\(\sigma\) 较小或 \(D\) 较大时，总和可达 **\(10^3 \sim 10^4\)** 量级。

因此 **联合 log prob** `log_prob_cluster + log_prob_residual` 几乎由残差项主导；若用同一标量 advantage 做 REINFORCE，**有效梯度往往严重偏向残差分支**（`residual_head`、`log_residual_std`），离散聚类头信号偏弱。

**与初值 `log_residual_std = log(0.001)` 的关系**：小 \(\sigma\) 会抬高峰值处的 log 密度（每维约 \(-\log\sigma\)），再对 \(D\) 维求和，会进一步放大上述现象。这不一定是“初始化错误”，而是 **高维 log 密度求和与单项分类 log prob 的天然尺度差异**。

**曾讨论的缓解方式**（在仅使用联合 REINFORCE 时）：例如 `joint_per_dim`（用 `log_prob_cluster + log_prob_residual / D` 作为 surrogate）以平衡梯度。当前主线实现已改为 **分拆训练目标与双优化器**（见下文），不再依赖在 actor 端对联合 log prob 做 per-dim 缩放。

---

## 2. 当前设计：两条更新路径 + 两个优化器

### 2.1 GRPO（VLA actor loss）→ 只动 `cluster_embed` 与 `residual_head`

- 在 **`run_training`** 的 micro-batch 内，当满足条件时，用 **`SiglipConditionRLModel.forward_z_for_actor_grpo`** 重算条件向量 \(z\)：
  - **SigLIP** 在 **`torch.no_grad()`** 下前向，**`embedding` 再 detach**，使梯度 **不** 回传到 SigLIP。
  - **`cluster_embed(pred_cluster_idx)`** 与 **`residual_head`** 在计算图中，梯度随 **GRPO / policy loss** 回传。
- **适用模型**：当前在 actor 中仅对 **OpenVLA / OpenVLA_OFT** 开启该路径。
- **Batch 依赖**：需要 rollout 提供与 optimizable embedding 一致的字段（如 `z_ids` 为二维连续向量、以及 `cond_initial_image_hwc`、`cond_pred_cluster_idx` 等），详见 `EmbodiedFSDPActor._micro_batch_has_cm_grpo_inputs`。

**优化器**：**`siglip_condition_grpo_optimizer`**（AdamW），参数仅为 **`cluster_embed` + `residual_head`**。  
**步进**：在 **`EmbodiedFSDPActor.optimizer_step`** 中，在主 FSDP actor **`grad_scaler.step(optimizer)`** 之后，对 GRPO 优化器执行 **`unscale` →（可选）`grpo_cm_grad_clip` → `step`**，最后 **`grad_scaler.update()`** 一次。

**时间窗口**（配置）：

- `grpo_cm_start_step`：从 runner 写入的 **`_runner_global_step`** 起算，**大于等于** 该步才做上述 GRPO→CM 更新。
- `grpo_cm_end_step`：若为 **`null`/未设**，表示无上界；若为非负整数，**闭区间**上界（`global_step > end` 则关闭）。

其他常用项：`grpo_cm_lr`、`grpo_cm_weight_decay`、`grpo_cm_grad_clip`。

### 2.2 REINFORCE → 只动 `siglip` 与 `classifier_head`

- **损失**：仅 **`log_prob_cluster`**（`evaluate_log_prob(..., reinforce_logprob="cluster")`），**不再**把残差 L2 合进 REINFORCE 损失。
- **优化器**：**`siglip_condition_reinforce_optimizer`**，参数为 **`siglip` + `classifier_head`**。
- **调度**：仍由 **`update_interval_vla_steps`** 与 **`_vla_step_counter`** 控制；在 **`train_condition_policy_if_due`** 中执行。
- **分布式**：**`all_reduce`** 与 **grad clip** 仅针对 **REINFORCE 参数子集**，避免与 GRPO 子模块混淆。

### 2.3 `log_residual_std`

两条优化器路径均未将其纳入参数组时，其在训练中 **不被更新**（除非后续改参数组）；rollout 仍可按 RL 头采样残差。

---

## 3. Residual L2：不并入 VLA 标量 loss

**讨论结论**：即使把 \(\|z\|^2\) 加进总 loss，对 **VLA 权重** 的梯度在数学上通常仍为 0（\(z\) 由条件子网络产生、不依赖 VLA 参数），但为 **指标与实现清晰**，不应把 residual L2 **合并进** 用于记录与主反向的 **VLA policy loss**。

**实现**：

1. **`policy_loss` + entropy 等** 得到的 **`loss`** **不包含** `grpo_residual_l2_coef * mean(||z||^2)`。
2. 先 **`grad_scaler.scale(loss).backward(retain_graph=True)`**（当存在待做的 L2 项时）。
3. 再单独 **`grad_scaler.scale(l2_term).backward()`**，其中  
   `l2_term = grpo_residual_l2_coef * mean(||z||^2) / gradient_accumulation`。

这样 **日志里的 `metrics_data["loss"]`** 对应 **纯 VLA 侧主损失**（已除 accumulation）；**`cond/grpo_residual_l2`** 仍记录未乘系数的 \(\|z\|^2\) 均值，便于监控。

---

## 4. 模块与入口文件

| 角色 | 路径 |
|------|------|
| 条件模型（SFT + RL） | `step3_encoder_training/condition_model_sto_new.py`（`SiglipConditionModel` / `SiglipConditionRLModel`） |
| Actor：GRPO、REINFORCE、存盘 | `rlinf/workers/actor/fsdp_actor_worker.py` |
| Rollout：加载与推理同构模型 | `rlinf/workers/rollout/hf/huggingface_worker.py`（从 `condition_model_sto_new` 导入） |
| 保存时机说明 | `rlinf/runners/embodied_runner.py`（注释） |

**`forward_z_for_actor_grpo`**（`condition_model_sto_new.py`）：输入首帧图像列表、指令字符串列表、rollout 记录的 **`pred_cluster_idx`**，输出 **`[B, residual_dim]`** 的 \(z\)，供 VLA 前向与 GRPO 反传至 **`cluster_embed` / `residual_head`**。

---

## 5. 检查点：与 VLA `global_step` 对齐

- 在每个 **`checkpoints/global_step_{k}/`** 目录下保存 **`condition_policy.pt`**（不再使用按 REINFORCE 间隔命名的主文件名；加载时若无新文件可回退 **`condition_policy_vla_step_{tag}.pt`** 与旧键 **`optimizer`**）。
- **内容**（典型）：`model`、`reinforce_optimizer`、`grpo_optimizer`、`baseline`、`vla_step_counter`。
- **语义**：磁盘上的条件模型权重对应 **`run_training` 结束时刻**（已包含当步内的 **GRPO→CM** 更新），**尚未** 执行当步在 save 之后触发的 **REINFORCE**（与 `embodied_runner` 中「先 save 再 `train_condition_policy_if_due`」的顺序一致）。

---

## 6. 配置示例（`algorithm.condition_policy`）

以下为常见键名（默认值以代码为准）：

```yaml
condition_policy:
  enable_rl: true
  checkpoint_path: null   # 默认路径见 fsdp_actor_worker._init_condition_policy
  lr: 1.0e-5              # REINFORCE：siglip + classifier_head
  weight_decay: 0.01
  grad_clip: null
  update_interval_vla_steps: 5
  micro_batch_size: 32
  cluster_sample_temperature: 1.0
  baseline_momentum: 0.95

  grpo_cm_start_step: 0
  grpo_cm_end_step: null
  grpo_cm_lr: 1.0e-4
  grpo_cm_weight_decay: 0.01
  grpo_residual_l2_coef: 0.5
  grpo_cm_grad_clip: null
```

**前置条件**：`use_optimizable_embedding: true`；`enable_rl: true` 时 rollout 侧会校验该开关。

---

## 7. 小结表

| 项目 | GRPO→CM | REINFORCE |
|------|---------|-----------|
| 信号 | VLA actor loss（经 \(z\) 连到 CM） | `log_prob_cluster` × advantage |
| 更新参数 | `cluster_embed`, `residual_head` | `siglip`, `classifier_head` |
| 优化器 | `siglip_condition_grpo_optimizer` | `siglip_condition_reinforce_optimizer` |
| 何时 | 每个满足窗口与数据条件的训练步（micro-batch 内） | 按 `update_interval_vla_steps` |
| L2 | 单独 backward，**不** 并入 VLA `loss` 标量 | 不使用残差 L2 |

---

## 8. 修订记录（文档层面）

- 整理自对话：联合 log prob 量级、双优化器设计、存盘顺序、L2 与 VLA loss 分离等。
- 若代码中默认 checkpoint 文件名或 `evaluate_logprob` 选项有变，请以仓库当前实现为准，并同步更新本节与第 4、6 节中的路径/表项。
