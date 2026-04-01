# SigLIP 条件策略 REINFORCE 实现说明

本文说明在 Libero Object + GRPO + OpenVLA-OFT 训练中，为 **SigLIP 条件模型** 增加的 **on-policy REINFORCE** 训练路径及相关工程改动。设计讨论见同目录下的 `libero_object_grpo_openvlaoft_siglip_training_discussion copy.md`。

> **行号说明**：下文「关键代码索引」中的 **行号** 对应文档更新时仓库中的版本；若你本地代码已改动，请以 IDE 跳转为准，并酌情同步修改本文。

---

## 1. 目标与原则

- **主目标**：在继续训练 VLA（GRPO actor）的同时，用 **REINFORCE** 更新 `SiglipConditionRLModel`，使条件编码（离散 cluster + 连续残差）与回报对齐。
- **不通过 VLA loss 反传**：SigLIP 的梯度只来自条件策略的 `log_prob` 与（可选）残差均值 L2，不经过 OpenVLA 的计算图。
- **调度**：每累计 **`update_interval_vla_steps`（默认 5）次 VLA 训练步** 后触发 **1 次** SigLIP 更新；该次更新 **不增加** `runner` 的 `global_step` / `max_epochs` 计数（仍在同一轮 embodied 外层循环内完成）。
- **组内共享 z 的回报**：同一 GRPO group 共用一次条件采样，REINFORCE 使用的标量回报 **R** 为该 group 内各条 rollout **回报之和再对组内 env 取平均**（与配置中的 `group_size` 一致）。
- **断点续训**：SigLIP 相关状态单独写入 **`condition_policy_vla_step_{N}.pt`**（与 `actor/` 目录并列），`N` 为 **最近一次已完成 REINFORCE 的 VLA 步**（例如在第 25 步保存时，文件名为 `condition_policy_vla_step_20.pt`，表示盘上的权重对应第 20 步训练后的 condition；保存发生在当步 SigLIP 更新之前）。仍兼容旧名 **`condition_policy.pt`** 的加载。
- **兼容旧配置**：`algorithm.condition_policy.enable_rl` 默认为 **`false`** 时，行为与未加本功能前一致（仍可仅用 `use_optimizable_embedding` 做连续 z 注入，不做条件策略 RL）。

---

## 2. 配置（全部在 `algorithm` 下）

在 YAML 的 `algorithm` 块中增加嵌套表 **`condition_policy`**，例如 `examples/embodiment/config/libero_object_grpo_openvlaoft.yaml`。

| 字段 | 含义 |
|------|------|
| `enable_rl` | 是否启用 SigLIP REINFORCE；`false` 时不初始化第二套优化器、不第二次 send/recv 权重、不写 `condition_policy.pt`。 |
| `checkpoint_path` | SigLIP 初始权重；`null` 时用默认路径 `step3_encoder_training/siglip_condition_model_sto.pt`。 |
| `cluster_sample_temperature` | Rollout 与 `evaluate_log_prob` 中离散策略的温度；若省略则回退 `algorithm.siglip_cluster_sample_temperature`。 |
| `deterministic_cluster_train` / `deterministic_residual_train` | 训练 rollout 时是否对 cluster / 残差用确定性前向（探索一般设为 `false`）。 |
| `deterministic_cluster_eval` / `deterministic_residual_eval` | 预留；**当前 eval rollout 路径仍未接 SigLIP**，需在后续若把 eval 改为走条件模型时再接线。 |
| `reinforce_logprob` | `joint`（cluster+残差）或 `cluster`（仅离散部分）。 |
| `baseline_momentum` | EMA baseline 动量（对 **跨 rank 平均后的** 批次回报均值更新）。 |
| `residual_l2_coef` | 对 **`evaluate_log_prob` 内部算出的 `residual_mean`** 的 L2 系数（有梯度，用于约束残差头）。 |
| `update_interval_vla_steps` | 每多少次 **VLA `run_training` 完成**（内部计数器 +1）触发一次 SigLIP 更新。 |
| `lr` / `weight_decay` / `grad_clip` / `micro_batch_size` | SigLIP 专用 AdamW 与微批大小。 |

约束：**`enable_rl: true` 时必须 `use_optimizable_embedding: true`**，否则在 rollout worker 初始化时会直接报错。

---

## 3. 数据结构与 Rollout 张量

### 3.1 `rlinf/data/io_struct.py`

- **`ChunkStepResult`** 增加可选字段：`cond_log_prob_*`、`cond_residual`、`cond_initial_image_hwc`（uint8 HWC）、`cond_pred_cluster_idx` 等；仅在「本 group 第一次算 z」的那一步填入。
- **`EmbodiedRolloutResult`** 按 **每个 `rollout_epoch` 一条** 追加上述信息（与「每个 epoch 内第一次 chunk」对齐），在 **`to_dict()`** 里沿 chunk 维 **重复展开** 到与 `prev_logprobs` 相同的时间长度，从而 **无需改** `process_nested_dict_for_adv` / `process_nested_dict_for_train` 的通用 reshape 逻辑。

### 3.2 `rlinf/workers/rollout/hf/huggingface_worker.py`

- 读取 `algorithm.condition_policy`；加载 SigLIP 的 ckpt 路径可由 `checkpoint_path` 覆盖。
- 每个 GRPO group **首次**调用 `SiglipConditionRLModel` 时：根据配置选择 `deterministic_*_train` 与温度；若 `enable_rl`，把 **采样得到的** log prob、残差、cluster id、首帧图（在组内 broadcast 成 `[B, ...]`）写入 `ChunkStepResult`。
- 使用 **`pending_cond_meta[stage_id]`** 仅在「刚算完 z 的那一步」把 meta 交给 `ChunkStepResult`，避免与 `group_latents` 列表混淆。
- **`sync_model_from_actor`**：在 `enable_rl` 时，在收完 VLA 权重后 **再 recv 一次** SigLIP 的 `state_dict`；若 `enable_rl` 为 false，**不能**多发一次，否则会对侧死等。

---

## 4. Actor 侧逻辑

### 4.1 `rlinf/workers/actor/fsdp_actor_worker.py`（`EmbodiedFSDPActor`）

- **`enable_rl` 时**：初始化 `SiglipConditionRLModel`、`AdamW`、on-policy **缓冲**（按步累积 `task_ids`、图像、残差、cluster id、回报等）、EMA baseline、`vla_step_counter`；从仓库根目录加载 **`libero_object_instruction_to_task_id_map.pt`** 构造 `task_id → instruction` 字符串，供 `evaluate_log_prob` 使用。
- **`compute_advantages_and_returns` 之后**（`rollout_batch` 已是 adv 用的形状）：若存在 cond 相关 key，计算 **`cond_group_reward`**：对 `rewards` 在时间与 action-chunk 维求和得到每条 env 的总回报，再 reshape 为 `[n_prompts, group_size]` 对 **组维取均值**，并 `repeat_interleave` 回 `[RB]`，再 broadcast 到与 `rewards` 相同的前两维 `[S, RB]`。
- **`run_training`**：
  1. 在 **`process_nested_dict_for_train`（shuffle）之前** 调用 **`_snapshot_condition_policy_buffer`**：在 batch 维上 **每隔 `group_size` 取一个代表**（与「一组一个 z」一致），把本步的图像、残差、cluster id、`cond_group_reward` 等写入缓冲。
  2. 照常完成 VLA 训练与 `lr_scheduler.step()`。
  3. **`_vla_step_counter += 1`**（不在此函数内更新 condition model）。
- **Runner（`embodied_runner.py`）**：在 **`global_step` 自增且按需 `_save_checkpoint()` 之后** 调用 **`train_condition_policy_if_due()`**。这样 **checkpoint 里的 `condition_policy.pt` 与当步保存的 VLA 一致**：均为「本步 rollout 所依据的、尚未做本周期 REINFORCE 更新」的 condition model；保存完成后再做 SigLIP 更新，接下来最多 **`update_interval_vla_steps` 步** VLA 仍消费 **上一版** condition 的 rollout 数据直至下一轮更新（与「每 5 步 VLA 基于上一版 condition 采样，再更新 condition」一致）。
- **`train_condition_policy_if_due()`**（actor 对外接口）：内部即 **`_train_condition_policy_if_due`**：若 `counter % update_interval_vla_steps == 0` 且缓冲非空，拼接样本，用 **`evaluate_log_prob`** 算 `train_log_prob`，损失大致为  
  `-(R - baseline) * logp + residual_l2_coef * ||residual_mean||^2`，**各 rank 梯度 `all_reduce` 平均** 后 `optimizer.step()`。
- **`sync_model_to_rollout`**：在发完 VLA `state_dict` 后，若 `enable_rl`，再 **send** 一份 SigLIP 的 CPU `state_dict`（与 rollout 第二次 recv 成对）。
- **`save_condition_policy_pt(checkpoint_dir)` / `load_condition_policy_pt(checkpoint_dir, global_step)`**：在目录下读写 **`condition_policy_vla_step_{tag}.pt`**，`tag = max(0, ((step - 1) // interval) * interval)`，与保存时 actor 上 `_vla_step_counter` 一致；**仅 rank 0 写盘**；checkpoint 内另存 `cond_policy_trained_at_vla_step` 便于核对。若无新文件名则尝试 **`condition_policy.pt`**。

### 4.2 `step3_encoder_training/condition_model_sto.py`

- 新增 **`evaluate_log_prob`**：给定图像、指令、**固定的** `pred_cluster_idx` 与 **rollout 采样残差**，在当前网络参数下计算各 log prob 及用于训练的 **`train_log_prob`**，并返回 **`residual_mean`**（供 L2 正则项反传）。

---

## 5. Runner 与 Checkpoint

### `rlinf/runners/embodied_runner.py`

- **`init_workers`**：在 actor 主 checkpoint 加载并设置 **`global_step`** 之后，调用 **`load_condition_policy_pt(resume_dir, global_step)`**（按上式解析文件名，或回退旧文件）。
- **`_save_checkpoint`**：在保存 `actor/` 后，若 `enable_rl`，再保存同级 **`condition_policy_vla_step_{tag}.pt`**（含义见上文）。
- **`run()` 主循环**：在 **`run_training()` 与 `global_step` 自增之后**，若本步需要存盘则先 **`_save_checkpoint()`**，再调用 **`train_condition_policy_if_due()`**，保证 checkpoint 与「本步 rollout 所用 condition」一致。

---

## 6. 行为对比小结

| 场景 | 行为 |
|------|------|
| `use_optimizable_embedding: false` | 与原先离散 z / 无 SigLIP 连续注入一致（不涉及 `condition_policy`）。 |
| `use_optimizable_embedding: true`，`enable_rl: false` | 与「仅 SigLIP 生成连续 z、无第二套优化器」的先前实现一致；无第二次权重同步、无 `condition_policy.pt`。 |
| `enable_rl: true` | 上述 + 缓冲 + 按间隔 REINFORCE + 单独 checkpoint + rollout/actor 双份权重同步。 |

---

## 7. 使用与注意

1. 打开条件策略 RL：设置 **`use_optimizable_embedding: true`** 与 **`condition_policy.enable_rl: true`**，并按需调整温度、`reinforce_logprob`、间隔与学习率。
2. 训练日志中可能出现 **`cond/*`** 指标（policy loss、residual L2、log prob 均值、回报均值、baseline 等）。
3. **Eval**：当前 `evaluate()` 仍使用原随机 z 逻辑；若要在验证时也对 SigLIP 使用 deterministic 前向，需在 eval rollout 中单独接线（配置项已预留）。
4. 运行需能从项目根目录找到 **`step3_encoder_training`**、**`libero_object_instruction_to_task_id_map.pt`**、**`libero_object_per_instruction_centers.pt`** 等路径（与现有 embodiment 脚本一致）。

---

## 8. 涉及文件清单

| 文件 | 角色 |
|------|------|
| `step3_encoder_training/condition_model_sto.py` | `evaluate_log_prob` |
| `rlinf/data/io_struct.py` | Chunk / EmbodiedRollout 条件字段与 `to_dict` 展开 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | 条件采样、张量打包、SigLIP 权重二次同步 |
| `rlinf/workers/actor/fsdp_actor_worker.py` | SigLIP 优化器、缓冲、REINFORCE、cond 回报、二次 send、存盘接口 |
| `rlinf/runners/embodied_runner.py` | `condition_policy.pt` 的保存与恢复 |
| `examples/embodiment/config/libero_object_grpo_openvlaoft.yaml` 等 | `algorithm.condition_policy` 默认值 |

---

## 9. 关键代码索引（行号与摘录）

### 9.1 总览表

| 主题 | 文件 | 行号（约） |
|------|------|------------|
| YAML `condition_policy` 默认值 | `examples/embodiment/config/libero_object_grpo_openvlaoft.yaml` | 45–60 |
| `evaluate_log_prob`（REINFORCE 用 log prob + `residual_mean`） | `step3_encoder_training/condition_model_sto.py` | 215–287 |
| `ChunkStepResult` 条件字段 | `rlinf/data/io_struct.py` | 1245–1295 |
| `EmbodiedRolloutResult` 收集、`append_result`、`to_dict` 展开 | `rlinf/data/io_struct.py` | 1298–1455 |
| Rollout：`condition_policy` 校验、SigLIP ckpt 路径 | `rlinf/workers/rollout/hf/huggingface_worker.py` | 45–53，103–105 |
| Rollout：二次 `recv` SigLIP | `rlinf/workers/rollout/hf/huggingface_worker.py` | 237–257 |
| Rollout：`pending_cond_meta`、SigLIP 前向、`ChunkStepResult` 打包 | `rlinf/workers/rollout/hf/huggingface_worker.py` | 306–521 |
| Actor：`EmbodiedFSDPActor` 读配置 | `rlinf/workers/actor/fsdp_actor_worker.py` | 744–746 |
| Actor：`_init_condition_policy` | `rlinf/workers/actor/fsdp_actor_worker.py` | 778–821 |
| Actor：二次 `send` SigLIP | `rlinf/workers/actor/fsdp_actor_worker.py` | 834–861 |
| Actor：`cond_group_reward`、snapshot、`_train_condition_policy_if_due`、`run_training` 尾部仅 `vla_step_counter++`、`train_condition_policy_if_due` | `rlinf/workers/actor/fsdp_actor_worker.py` | 987–1174，1352–1366，1368–1380 |
| Runner：save 后调用 `train_condition_policy_if_due`；`save/load_condition_policy_pt` 传目录与 `global_step` | `rlinf/runners/embodied_runner.py` | 约 100–102，196–202，217–218，244–246 |

与轨迹张量一起 reshape 的通用逻辑仍在 `process_nested_dict_for_adv` / `process_nested_dict_for_train`（`rlinf/workers/actor/fsdp_actor_worker.py` 约 70–104），条件相关 key 与普通 tensor 走同一分支。

### 9.2 配置示例（YAML）

```45:60:examples/embodiment/config/libero_object_grpo_openvlaoft.yaml
  condition_policy:
    enable_rl: False
    checkpoint_path: null # default: step3_encoder_training/siglip_condition_model_sto.pt
    cluster_sample_temperature: 2.0 # falls back to siglip_cluster_sample_temperature if omitted
    deterministic_cluster_train: false
    deterministic_residual_train: false
    deterministic_cluster_eval: true
    deterministic_residual_eval: true
    reinforce_logprob: joint # joint | cluster
    baseline_momentum: 0.95
    residual_l2_coef: 0.5
    update_interval_vla_steps: 5
    lr: 1.0e-5
    weight_decay: 0.01
    grad_clip: null
    micro_batch_size: 32
```

### 9.3 `evaluate_log_prob`（固定 rollout 采样下的 log prob）

签名与返回（中间为 SigLIP 前向、Categorical / Normal 与 `reinforce_logprob` 分支，见 **215–279 行**）：

```215:223:step3_encoder_training/condition_model_sto.py
    def evaluate_log_prob(
        self,
        images,
        instructions,
        pred_cluster_idx: torch.Tensor,
        residual_sample: torch.Tensor,
        cluster_sample_temperature: float = 1.0,
        reinforce_logprob: str = "joint",
    ) -> dict[str, torch.Tensor]:
```

```273:287:step3_encoder_training/condition_model_sto.py
        log_prob_joint = log_prob_cluster + log_prob_residual
        if reinforce_logprob == "cluster":
            train_log_prob = log_prob_cluster
        elif reinforce_logprob == "joint":
            train_log_prob = log_prob_joint
        else:
            raise ValueError(f"Unknown reinforce_logprob={reinforce_logprob!r}")

        return {
            "log_prob_cluster": log_prob_cluster,
            "log_prob_residual": log_prob_residual,
            "log_prob_joint": log_prob_joint,
            "train_log_prob": train_log_prob,
            "residual_mean": residual_mean,
        }
```

### 9.4 数据结构：`ChunkStepResult` 与 `EmbodiedRolloutResult.to_dict` 展开

```1256:1263:rlinf/data/io_struct.py
    # Optional: one row per env, filled on the first chunk of each rollout epoch when RL
    # condition policy is enabled (see EmbodiedRolloutResult).
    cond_log_prob_cluster: Optional[torch.Tensor] = None
    cond_log_prob_residual: Optional[torch.Tensor] = None
    cond_log_prob_joint: Optional[torch.Tensor] = None
    cond_residual: Optional[torch.Tensor] = None
    cond_initial_image_hwc: Optional[torch.Tensor] = None  # [B, H, W, 3] uint8
    cond_pred_cluster_idx: Optional[torch.Tensor] = None  # [B] long
```

```1419:1433:rlinf/data/io_struct.py
        if len(self.cond_log_prob_cluster) > 0:
            n_ep = self.rollout_epoch
            n_chunks_total = len(self.prev_logprobs)
            assert n_ep > 0 and n_chunks_total % n_ep == 0, (
                f"cond policy: bad chunk count {n_chunks_total=} vs {n_ep=}"
            )
            n_chunk = n_chunks_total // n_ep

            def _expand_along_chunks(stacked: torch.Tensor) -> torch.Tensor:
                # stacked: [n_ep, B, ...]
                return (
                    stacked.unsqueeze(1)
                    .expand(-1, n_chunk, *([-1] * (stacked.ndim - 1)))
                    .reshape(n_ep * n_chunk, *stacked.shape[1:])
                )
```

```1435:1455:rlinf/data/io_struct.py
            rollout_result_dict["cond_log_prob_cluster"] = _expand_along_chunks(
                torch.stack(self.cond_log_prob_cluster, dim=0).cpu().contiguous()
            )
            rollout_result_dict["cond_log_prob_residual"] = _expand_along_chunks(
                torch.stack(self.cond_log_prob_residual, dim=0).cpu().contiguous()
            )
            rollout_result_dict["cond_log_prob_joint"] = _expand_along_chunks(
                torch.stack(self.cond_log_prob_joint, dim=0).cpu().contiguous()
            )
            rollout_result_dict["cond_residual"] = _expand_along_chunks(
                torch.stack(self.cond_residual, dim=0).cpu().contiguous()
            )
            rollout_result_dict["cond_initial_image_hwc"] = _expand_along_chunks(
                torch.stack(self.cond_initial_image_hwc, dim=0).cpu().contiguous()
            )
            rollout_result_dict["cond_task_ids"] = _expand_along_chunks(
                torch.stack(self.cond_task_ids, dim=0).cpu().contiguous()
            )
            rollout_result_dict["cond_pred_cluster_idx"] = _expand_along_chunks(
                torch.stack(self.cond_pred_cluster_idx, dim=0).cpu().contiguous()
            )
```

### 9.5 Rollout：配置、`pending_cond_meta`、二次同步

```45:53:rlinf/workers/rollout/hf/huggingface_worker.py
        self.condition_policy_cfg = cfg.algorithm.get("condition_policy") or {}
        self.condition_policy_enable_rl = bool(
            self.condition_policy_cfg.get("enable_rl", False)
        )
        if self.condition_policy_enable_rl and not self.use_optimizable_embedding:
            raise ValueError(
                "algorithm.condition_policy.enable_rl requires "
                "algorithm.use_optimizable_embedding=True."
            )
```

```237:257:rlinf/workers/rollout/hf/huggingface_worker.py
    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name, src_rank=self.actor_weight_src_rank, async_op=True
        ).async_wait()

        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

        if self.condition_policy_enable_rl and self.siglip_condition_model is not None:
            cp_state = await self.recv(
                self.actor_group_name, src_rank=self.actor_weight_src_rank, async_op=True
            ).async_wait()
            self.siglip_condition_model.load_state_dict(cp_state)
            del cp_state
            self.siglip_condition_model.to(self.device)
            self.siglip_condition_model.eval()
            gc.collect()
            torch.cuda.empty_cache()
```

### 9.6 Actor：组内平均回报、缓冲快照、REINFORCE 步、`run_training` 衔接

```991:1005:rlinf/workers/actor/fsdp_actor_worker.py
    def _attach_cond_group_reward_for_condition_policy(self) -> None:
        if not self.condition_policy_enable_rl:
            return
        rb = self.rollout_batch
        if "cond_log_prob_cluster" not in rb or "rewards" not in rb:
            return
        rewards = rb["rewards"]
        gs = self.cfg.algorithm.group_size
        s0, rb_sz = rewards.shape[0], rewards.shape[1]
        per_env = rewards.sum(dim=(0, 2))
        n_prompts = rb_sz // gs
        rmat = per_env.reshape(n_prompts, gs)
        rz = rmat.mean(dim=1)
        cond_r = rz.repeat_interleave(gs)
        rb["cond_group_reward"] = cond_r.unsqueeze(0).expand(s0, -1).contiguous()
```

```1105:1141:rlinf/workers/actor/fsdp_actor_worker.py
            with torch.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
                dtype=torch.bfloat16,
            ):
                out_lp = self.siglip_condition_model.evaluate_log_prob(
                    images=mb_img,
                    instructions=mb_instr,
                    pred_cluster_idx=mb_idx,
                    residual_sample=mb_res,
                    cluster_sample_temperature=temp,
                    reinforce_logprob=reinforce_mode,
                )
                logp = out_lp["train_log_prob"]
                pg = -(mb_adv * logp).mean()
                res_mean = out_lp["residual_mean"]
                l2 = res_mean.float().pow(2).sum(dim=-1).mean()
                loss_mb = pg + l2_coef * l2

            loss_mb.backward()
            total_pg += pg.detach()
            total_l2 += l2.detach()
            total_lp += logp.detach().mean()
            n_mb += 1

        if dist.is_initialized() and self._world_size > 1:
            for p in self.siglip_condition_model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        grad_clip = self.condition_policy_cfg.get("grad_clip", None)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.siglip_condition_model.parameters(), float(grad_clip)
            )

        self.siglip_condition_optimizer.step()
```

```1182:1195:rlinf/workers/actor/fsdp_actor_worker.py
        self.model.train()
        self._snapshot_condition_policy_buffer()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )
```

```1352:1366:rlinf/workers/actor/fsdp_actor_worker.py
        self._vla_step_counter += 1

        return mean_metric_dict

    def train_condition_policy_if_due(self) -> dict[str, float]:
        """
        REINFORCE update for SigLIP (if ``vla_step_counter`` hits the interval).

        Call from the runner **after** ``_save_checkpoint`` when saving, so the
        checkpoint's condition model matches the policy that produced the rollout
        for the saved VLA step; then this method applies the deferred update.
        """
        if not self.condition_policy_enable_rl:
            return {}
        return self._train_condition_policy_if_due()
```

（保存路径为 ``checkpoint_dir/condition_policy_vla_step_{tag}.pt``，详见 ``_condition_policy_train_tag`` 与 ``save_condition_policy_pt`` / ``load_condition_policy_pt`` 源码。）

### 9.7 Runner：先 save 再 `train_condition_policy_if_due`；恢复与保存 `condition_policy.pt`

```196:202:rlinf/runners/embodied_runner.py
                # Save VLA + condition_policy.pt **before** SigLIP REINFORCE so the
                # checkpoint matches rollouts from the pre-update condition model.
                if save_model:
                    self._save_checkpoint()

                with self.timer("condition_policy_training"):
                    cond_policy_metrics = self.actor.train_condition_policy_if_due().wait()
```

```217:219:rlinf/runners/embodied_runner.py
            train_dict = dict(actor_training_metrics[0])
            train_dict.update(cond_policy_metrics[0] or {})
            training_metrics = {f"train/{k}": v for k, v in train_dict.items()}
```

```100:102:rlinf/runners/embodied_runner.py
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])
        self.actor.load_condition_policy_pt(resume_dir, self.global_step).wait()
```

```244:246:rlinf/runners/embodied_runner.py
        if cp_cfg.get("enable_rl", False):
            self.actor.save_condition_policy_pt(base_output_dir).wait()
```

---

*文档版本：与仓库中 condition policy REINFORCE 实现同步；若代码有后续迭代，请同步更新第 9 节行号与摘录。*
