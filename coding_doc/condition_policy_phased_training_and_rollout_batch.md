# 条件策略近期改动：`group_size`、rollout batch 与分阶段训练

本文档记录 **2026 年前后** 在条件策略（SigLIP + 可优化 embedding）与 GRPO 联训上的增量实现与讨论结论。与 GRPO/REINFORCE 双优化器、残差 L2 等 **基础设计** 请对照 [`condition_policy_grpo_reinforce_summary.md`](condition_policy_grpo_reinforce_summary.md)。

---

## 1. Rollout batch `B` 与 `algorithm.group_size`

### 1.1 `B` 从哪里来

每个 **rollout 进程**、每个 **pipeline stage** 在单步里看到的观测 batch 大小为：

- `train_num_envs_per_stage = env.train.total_num_envs / env_world_size / pipeline_stage_num`
- `gather_num = rollout_world_size / env_world_size`
- **`B = train_num_envs_per_stage // gather_num`**（env 侧按 `gather_num` 切分后发给单个 rollout rank 的条数）

实现参考：`rlinf/workers/env/env_worker.py`（`gather_num`、`train_num_envs_per_stage` 与按 `gather_num` 的 `chunk`）。

### 1.2 代码是否强制 `B == group_size`

**没有。** `algorithm.group_size` 在 actor 的 GRPO 里用于 `batch_size % group_size == 0` 等 reshape；**不会**在配置层校验 `B` 与 `group_size` 相等。

### 1.3 `use_optimizable_embedding=True` 时 `z` 的语义

Rollout 在 **每个 chunk 步、每个 stage** 首次需要 latent 时，用 **第一条 env** 的图像与指令跑一次 SigLIP，得到一条 `z_vec`，再 **`expand` 到整批 `B`**：

```631:632:rlinf/workers/rollout/hf/huggingface_worker.py
                            # broadcast same z-embedding to all envs in the group
                            z_batch = z_vec.unsqueeze(0).expand(B, -1).contiguous()
```

**讨论结论：**

- 若该 batch 在数据布局上对应 **恰好一个 GRPO 组**（常见做法：`B == group_size`，且同组共享同一 conditioning 上下文），则「一组共用一个 `z`」与 GRPO 设计一致。
- 若 **`B = k * group_size` 且 `k > 1`**，当前实现仍只算 **一个** `z` 并广播到全部 `B` 条 → **不同 GRPO 组会错误共用同一 `z`**，与按组归一化的 advantage 不一致。
- 若 **`B % group_size != 0`**，actor 侧 GRPO 可能 **直接 assert 失败**；即便不失败，与「按组」语义也不对齐。
- **后续若需支持 `B` 含多组**：应在 rollout 侧按 **连续 `group_size` 块**（或按组索引）分别算 SigLIP / `z`，而不是单次 `expand(B, …)`。

### 1.4 `use_optimizable_embedding=False`（离散 `z_ids`）

非 optimizable 路径里 `group_tensor` 长度为 **`group_size`**，VLA 离散插入路径会把 `z_ids` 与 `task_descriptions` 等 **按 batch 维 zip**；因此实际上也要求 **batch 维与离散 id 条数一致**，通常仍应 **`B == group_size`**，否则长度或语义会错配。Eval 配置里常见 **`env.eval.group_size: 1`**，使每条 env 自有 id。

---

## 2. 分阶段：`cluster_only` 时间窗与 GRPO→CM 的互斥

当配置了 `cluster_only_schedule_end_step`（及可选 `cluster_only_schedule_start_step`）时：

- **Rollout**：`_effective_cluster_only()` 在 runner 步落入 `[start, end]` 时使用「仅聚类中心 / 关残差」等策略（与静态 `cluster_only` 二选一，由调度覆盖静态）。
- **Actor**：`_scheduled_cluster_only_phase_active()` 为真时，**关闭** `_cm_grpo_training_enabled()` 中的 CM GRPO 更新（该窗口内不通过 `forward_z_for_actor_grpo` 走 `cluster_embed`/`residual_head` 的 GRPO 优化器步）。

这样可以在 **早期步数** 固定为聚类主导、**之后** 再打开对条件子模块的 GRPO 微调。相关逻辑：`rlinf/workers/rollout/hf/huggingface_worker.py`（`_effective_cluster_only`）、`rlinf/workers/actor/fsdp_actor_worker.py`（`_scheduled_cluster_only_phase_active`、`_cm_grpo_training_enabled`）。

---

## 3. `cond_residual_applied` 与 actor 侧 `z` 混合

Rollout 在条件策略 RL 开启时，把 **`cond_residual_applied`**（0/1 标量，表示本步 `z` 是否含残差项）写入 chunk meta，经 `io_struct` 等到 actor。

训练 micro-batch 中，若启用 CM GRPO 且具备 `cond_initial_image_hwc`、`cond_pred_cluster_idx` 等输入，actor 会：

1. 用 `_build_grpo_z_for_microbatch` 重算 **`z_grpo`**；
2. 若存在 **`cond_residual_applied`**，则  
   **`z_mix = z_grpo * m + z_roll.detach() * (1 - m)`**，其中 `m` 为 `cond_residual_applied` 广播；否则直接用 **`z_grpo`**。

这样 **GRPO 反传路径** 与 **rollout 当时是否把残差打进 `z`** 对齐，避免在「仅中心」阶段用全残差 `z` 算 policy loss。见 `rlinf/workers/actor/fsdp_actor_worker.py` 中 `z_mix` 一段。

---

## 4. 其它配置与工程项（YAML / runner）

| 项 | 作用 |
|----|------|
| `reinforce_disable_from_runner_step` | 达到指定 runner 全局步后，跳过 SigLIP+classifier 的 REINFORCE 更新（GRPO 窗口可与之一致或单独配）。 |
| `residual_when_sample_matches_argmax_only`（rollout） | 仅在采样聚类与 argmax 一致时对残差应用某种约束（与 `_z_from_siglip_row` 配合）。 |
| `_vla_micro_batch_effective` | 在条件策略 RL 开启等条件下 **减半** VLA micro-batch，缓解显存。 |
| `prune_previous_actor_dcp` + `prune_previous_actor_dcp_keep_steps` | 保存新 checkpoint 后删除 **上一份** actor 的 `dcp_checkpoint` 以省盘；`keep_steps` 中的全局步 **不删**（例如保护 step 50 的 DCP）。`rlinf/runners/embodied_runner.py`。 |
| `sampling_seed` / `sampling_seed_add_rollout_rank` | 仅在 **rollout** 侧为条件模型采样初始化 **`torch.Generator`**，保证可复现且各 rollout rank 可区分；**不在 actor** 上挂采样 seed（避免与训练图重复/不一致）。`rlinf/workers/rollout/hf/huggingface_worker.py`。 |

示例配置：`examples/embodiment/config/libero_object_grpo_openvlaoft_opt.yaml` 等。

---

## 5. 条件模型与 Rollout 实现细节（摘要）

- **`argmax_cluster_idx`**：与 `pred_cluster_idx` 一并产出，供 rollout/actor 在「采样 vs argmax」分支上使用。
- **HuggingFace rollout**：`_maybe_init_condition_policy_sampling_generator` 须为 **类内普通方法**（曾误嵌套在 `__init__` 内，已修正）。

---

## 6. 指标与监控

- 主训练日志中 **`train/cond/grpo_residual_l2`**（或项目内等价 key）用于看 \(\|z\|^2\) 尺度；**CM GRPO 子优化器**的 grad norm 若需对齐排查，可视需求再加日志（当前主线以 residual L2 与 loss 为主）。

---

## 7. 涉及文件（便于 code review）

| 区域 | 路径 |
|------|------|
| 条件模型前向 / GRPO z | `step3_encoder_training/condition_model_sto_new.py` |
| Rollout：调度、`z`、`cond_residual_applied`、采样器 | `rlinf/workers/rollout/hf/huggingface_worker.py` |
| Actor：分阶段 GRPO、REINFORCE 关闭、`z_mix`、micro-batch | `rlinf/workers/actor/fsdp_actor_worker.py` |
| 数据结构 | `rlinf/data/io_struct.py` |
| Checkpoint / DCP 修剪 | `rlinf/runners/embodied_runner.py` |
| VLA 侧 `z` 使用 | `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py` |
| 配置样例 | `examples/embodiment/config/libero_object_grpo_openvlaoft_opt*.yaml`、`debug_opt_new.yaml` |

---

## 8. 待办 / 可选增强

1. 若产品上要 **`B > group_size` 且多组同批**：实现 **按组** SigLIP 或按块 `z`，替换单次 `expand(B, …)`。  
2. 配置校验（可选）：在启动时检查 `B` 与 `group_size` 关系并 **告警或 assert**，避免静默错训。  
3. 按需补充 **CM GRPO grad norm** 或 `grpo_cm_active` 指标，便于与 `grpo_residual_l2` 对照。
