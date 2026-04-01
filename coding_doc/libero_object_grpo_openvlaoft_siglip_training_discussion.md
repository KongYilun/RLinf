# Libero Object GRPO + OpenVLA-OFT + SigLIP 条件编码训练记录

## 背景与目标

你希望在 `examples/embodiment/run_embodiment.sh` 跑 `examples/embodiment/config/libero_object_grpo_openvlaoft.yaml` 时：

1. 检测 GRPO 算法在一个 `group` 内的初始 state（第一帧观测图像）是否相同。
2. 若相同，使用 `siglip_condition_model_new.pt` 根据该首帧观测图像和任务指令预测聚类中心选择与 embedding 残差，随后：
   - 从 `libero_object_per_instruction_centers.pt` 中按“任务指令 -> 聚类中心 embedding”取出 center embedding
   - 将预测残差加入 center embedding 得到最终 `z_embedding`
   - 把 `z_embedding` 注入到 `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py` 的多模态 embedding 流程
3. 在 `libero_object_grpo_openvlaoft.yaml` 中加入参数开关：是否采用可优化的 embedding；`False` 保持现状，`True` 执行上述功能。

在后续阶段，你进一步提出：想在训练 VLA 的同时训练 `SiglipConditionModel`，并讨论训练目标设计、同步 vs 间隔训练策略。

---

## 已完成的代码改动（当前落地）

### 1) 配置开关：`use_optimizable_embedding`

在 `examples/embodiment/config/libero_object_grpo_openvlaoft.yaml` 的 `rollout` 下新增：

```yaml
rollout:
  use_optimizable_embedding: False
```

当为 `False` 时保持原逻辑（使用离散的 `z_ids` 进入 per-instruction centers 查表）。

当为 `True` 时启用 SigLIP 条件编码生成连续 embedding，并把它作为 `z_ids` 的连续值传入模型（见下文）。

---

### 2) Rollout 侧：按 group 检测首帧相同性 + 生成 `z_embedding`

文件：`rlinf/workers/rollout/hf/huggingface_worker.py`

主要逻辑：

1. 读取配置：`self.use_optimizable_embedding = cfg.rollout.get("use_optimizable_embedding", False)`
2. 若启用：
   - 初始化 `SiglipConditionModel`（从 `step3_encoder_training/siglip_condition_model_new.pt` 加载 checkpoint）
   - 加载 `libero_object_per_instruction_centers.pt`（instruction -> centers）
3. 在 `generate()` 的每个 `stage_id` 中维护一个缓存 `group_latents[stage_id]`，用于让同一个 GRPO group 重用同一个 z_embedding。
4. 第一次遇到该 group 的 step 时（即 `group_latents[stage_id] is None`）：
   - 取 `env_output["obs"]["main_images"]` 作为输入（必要时取第 0 个时间维）
   - 将首帧图像展平为向量并做 group 内一致性检测（按元素逐一完全相等；不一致时打印 warning）
   - 用 `img0` + 指令 `instr0` 调用 `siglip_condition_model_new.pt`：
     - `cluster_id = argmax(logits_c)`
     - `z_vec = instruction_centers[instr0][cluster_id] + residual_embedding`
   - 将 `z_vec` broadcast 为 `[B, D]`，作为该 group 的 `z_ids`（连续向量）缓存起来
5. 后续步复用缓存的 `z_ids`，不重复推理 SigLIP。

---

### 3) VLA 模型侧：让 `z_ids` 同时支持“离散 id”与“连续 embedding”

文件：`rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py`

关键变更点：

1. 在 `predict_action_batch`（rollout 推理路径）与 `default_forward`（训练路径）两处，新增分支：
   - `z_ids.dim() == 2`：视为已经是连续 `z_embedding`，直接作为插入 token 的 embedding
   - 否则：视为旧逻辑离散 id，通过 per-instruction centers 查表得到 embedding
2. 保持旧路径的兼容性：`use_optimizable_embedding=False` 时行为应与之前一致。

---

## 讨论：SiglipConditionModel 的训练目标如何设计？

你想在训练 VLA 的同时训练 SigLIP 条件编码模型，讨论训练目标时主要分为两类：

### A) 纯 RL-through-z 的目标（最端到端）

将 SigLIP 视作 VLA 策略的一部分：

- `z = SiglipConditionModel(img0, instruction)`
- VLA 使用 `z` 来预测动作
- GRPO 的 actor loss 会对 `z` 有梯度，因此也能更新 SigLIP

总结为：

1. 让 SigLIP 参数参与反向传播（即在 actor 训练阶段需要重新计算 `z`，或让计算图可回传）
2. 用轻量正则稳定训练（比如对 residual 做 L2，避免 residual 过大）

可能的组合形式：

```text
L_total = L_actor(RL-through-z) + lambda_res * ||residual||^2
```

### B) 多任务/弱监督锚定（更稳）

保留原先 step3 的监督形式：

- 分类头：预测 cluster id（交叉熵）
- residual 回归头：回归 residual（MSE）

在 RL 阶段可以引入弱权重的监督：

```text
L_total = L_actor + alpha * L_cls + beta * L_reg_sup + lambda_res * ||residual||^2
```

该做法通常更稳健，因为 RL-through-z 的信号噪声更大，弱监督可以防止 SigLIP 表示崩掉或遗忘原本聚类结构。

---

## 讨论：同步训练 vs 间隔训练（8 step VLA / 2 step SigLIP）

你提出要实现并对比两种策略：

### 1) Joint：同步训练（每个 actor update 同时更新 VLA 与 SigLIP）

优点：

- 最端到端，可能学习到更适配当前策略分布的条件编码

缺点：

- 联合优化更不稳定；RL 信号大噪声，会让 SigLIP 参数漂移
- 需要更仔细的学习率/正则调参

### 2) Alternating：交替训练（例如 8:2）

一种可选范式：

- 前 8 步：冻结 SigLIP，只更新 VLA
- 后 2 步：冻结 VLA，只更新 SigLIP（或 SigLIP 强一些）

优点：

- 每段训练问题聚焦，便于 debug 和做消融

缺点：

- 实现需要额外的训练调度逻辑（全局 step % 周期）
- 计算开销可能略增加（取决于实现方式）

你要的“都实现方便实验测试”，本质上意味着 actor 训练 loop 必须支持训练模式切换与冻结策略。

---

## 讨论：residual L2 正则 vs z 范数/偏移约束的区别

设最终用于 VLA 的条件向量为：

- `z = center + residual`

### residual L2 正则

- 约束的是 residual：`||residual||^2`
- 直观含义：让 z 相对 center 的“增量”不要太大，尽量保留 center + 小残差的语义

### z 范数 / z 偏移约束

- 约束的是最终 z 本身：如 `||z||^2` 或 `||z - z0||^2`
- 直观含义：
  - 范数约束：防止最终 z 幅值过大（避免发散/激活爆炸）
  - 偏移约束：把 z 锚定到参考点附近（比如离线 center 或上一次 z 的 EMA）

区别总结：

- 只约束 residual 不一定控制 z 的整体幅值（因为 center 本身可能很大或分布差异较大）
- 只约束 z 幅值可能允许残差与 center 互相抵消/绕路，从而削弱“center + small residual”的结构先验

常见组合：`residual L2`（保结构）+ 轻量 `z 范数/偏移`（防漂移/防爆炸）。

---

## 还没落地但需要的改动（若继续推进“训练 SigLIP”）

当前已完成的是 rollout 侧用 SigLIP 生成连续 `z_embedding` 并注入 VLA。

若要“在训练 VLA 的同时训练 SigLIP”，通常还需要 actor 侧做结构性调整，例如：

1. 让 actor 训练阶段能重新计算 `z = SigLIP(img0, instruction)`，使 RL loss 能对 SigLIP 参数回传梯度
2. buffer 或数据管线需要保存足够信息以重建 SigLIP 输入（首帧图像与指令）
3. actor 训练 loop 增加训练模式调度（joint / alternating 8:2），对 VLA 与 SigLIP 分别冻结/更新
4. 配置新增：SigLIP 训练目标权重与交替周期参数（例如 `vla_steps`, `siglip_steps`）

---

## 基于 `condition_model_sto.py` 的实现方案（REINFORCE + 5:1 调度）

你当前的 `step3_encoder_training/condition_model_sto.py` 已经提供了非常关键的 RL 接口：

- `SiglipConditionRLModel.forward(...)` 可输出：
  - `pred_cluster_idx`
  - `log_prob_cluster`
  - `log_prob_residual`
  - `log_prob_joint = log_prob_cluster + log_prob_residual`
  - `residual_embedding`（采样值）与 `residual_mean`、`residual_std`
- 且支持：
  - `cluster_sample_temperature`
  - `deterministic_cluster`
  - `deterministic_residual`

这意味着可以直接把 `SiglipConditionRLModel` 作为 “condition policy” 来做 on-policy REINFORCE。

### 一、训练目标（按你当前要求）

按你最新设定：

1. 不使用 “VLA loss 对 SigLIP 端到端反传” 作为主训练方式；
2. 使用 REINFORCE 最大化 condition 对应回报；
3. 保留 residual L2 正则。

建议 loss 写成：

```text
A = R - b
L_pg = - E[ A * log_prob_condition ]
L_res_l2 = E[ ||residual_embedding||^2 ]
L_total = L_pg + lambda_res_l2 * L_res_l2
```

其中：

- `log_prob_condition` 可选 `log_prob_joint`（离散+连续都优化）或仅 `log_prob_cluster`（只优化 behavior mode）。
- `R` 为该样本（建议 group 粒度）对应的实际回报。
- `b` 为 baseline（建议先用 EMA baseline，后续可扩展 value baseline）。

### 二、5 步 VLA + 1 次 SigLIP 更新（on-policy）

你希望 “每训练 5 步 VLA，进行一次 SigLIPConditionModel 更新”，并使用这 5 步产生的 on-policy 数据。

推荐流程：

1. **Step t..t+4（5步）**：
   - 正常 rollout + VLA 训练；
   - rollout 侧用 `SiglipConditionRLModel` 采样 condition；
   - 将以下字段存入 “condition on-policy buffer”：
     - `initial_image`
     - `instruction`
     - `pred_cluster_idx`
     - `residual_embedding`（或 residual_sample）
     - `log_prob_cluster/log_prob_residual/log_prob_joint`（至少一种）
     - `reward`（用于构造 `R`）
2. **Step t+5**：
   - 冻结 VLA，仅更新 SigLIP；
   - 用最近 5 步样本计算 `L_total` 做一次（或少量 epoch）更新；
   - 清空该 5-step on-policy buffer；
   - 将更新后的 SigLIP 参数同步到 rollout worker。

注意：

- 为保持 on-policy，SigLIP 更新不建议重复多 epoch（1~2 次即可）。
- rollout 继续可以使用 `deterministic_cluster=False` 进行探索；eval 时改为 deterministic。

### 三、需要改动的代码模块（按当前工程组织）

#### 1) 配置层（`examples/embodiment/config/libero_object_grpo_openvlaoft.yaml`）

新增 condition-policy 相关配置（建议）：

```yaml
rollout:
  use_optimizable_embedding: true
  condition_policy:
    enable_rl: true
    model_type: siglip_condition_rl
    cluster_sample_temperature: 1.0
    deterministic_cluster_train: false
    deterministic_residual_train: false
    deterministic_cluster_eval: true
    deterministic_residual_eval: true

actor:
  condition_policy_optim:
    enable: true
    update_interval_vla_steps: 5
    lr: 1.0e-5
    weight_decay: 1.0e-2
    reinforce_logprob: joint   # joint | cluster
    baseline: ema
    baseline_momentum: 0.95
    residual_l2_coef: 0.5
```

#### 2) rollout worker（`rlinf/workers/rollout/hf/huggingface_worker.py`）

把当前 `SiglipConditionModel` 切换/兼容为 `SiglipConditionRLModel`：

- 初始化时加载 RL 版本模型（优先支持从 SFT ckpt warm-start）；
- 训练模式调用：
  - `deterministic_cluster=False`
  - `deterministic_residual=False`
- 评估模式调用：
  - `deterministic_cluster=True`
  - `deterministic_residual=True`
- 生成 `z = center[instruction][pred_cluster_idx] + residual_embedding` 后注入 VLA；
- 额外把 condition-policy 的采样信息塞进 rollout 结果（至少 group 粒度存一次）。

#### 3) 数据结构（`rlinf/data/io_struct.py`）

在 `ChunkStepResult` / `EmbodiedRolloutResult` 增加 condition-policy 字段容器：

- `cond_initial_images`（或可重建首帧的轻量引用）
- `cond_instructions`
- `cond_cluster_idx`
- `cond_log_prob_cluster`
- `cond_log_prob_residual`
- `cond_log_prob_joint`
- `cond_residual`
- `cond_group_reward`（或先存 step reward，actor 端聚合）

并在 `append_result()/to_dict()/split` 路径里透传。

#### 4) actor 训练逻辑（actor worker / trainer）

新增 `SiglipConditionRLModel` 的优化器与调度器：

- 维护 `vla_step_counter`；
- 每步先做 VLA 常规训练；
- 当 `counter % 5 == 0` 时触发 `train_condition_policy()`：
  - 从最近5步 rollout 汇总 condition on-policy 数据；
  - 计算 `R`（建议 group return）；
  - 更新 baseline `b`；
  - 计算 `L_total = -E[(R-b)*log_prob] + lambda*E[||residual||^2]`；
  - `optimizer_siglip.step()`；
  - 清空 condition on-policy buffer。

#### 5) 权重同步与 checkpoint

- rollout 侧需要能接收 actor 端最新的 SigLIP 权重；
- checkpoint 需保存：
  - `siglip_condition_model.state_dict()`
  - `siglip_optimizer.state_dict()`
  - baseline 状态（EMA值）
  - 调度计数器（用于 resume 后保持 5:1 节奏）

#### 6) 日志监控（建议必须加）

新增指标：

- `cond/reward_mean`, `cond/reward_std`
- `cond/log_prob_mean`
- `cond/policy_loss`, `cond/residual_l2`
- `cond/baseline`
- `cond/cluster_entropy`
- `cond/cluster_histogram`

用于判断：

- 是否 mode collapse；
- REINFORCE 是否高方差；
- residual 是否过大（正则是否有效）。

### 四、实现取舍建议（基于当前模型接口）

由于 `SiglipConditionRLModel` 已同时提供离散+连续概率：

- 若你要“更强调 behavior mode”，推荐先用：
  - `reinforce_logprob = cluster`
  - 即仅用 `log_prob_cluster` 参与 REINFORCE；
  - `residual` 先只保留 L2 正则。
- 稳定后再切到：
  - `reinforce_logprob = joint`
  - 同时优化 cluster 与 residual 的随机策略。

这样实验会更可解释，且更符合你提出的目标导向。

