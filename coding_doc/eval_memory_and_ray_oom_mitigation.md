# 评估阶段内存优化与 Ray 节点 OOM 处理（讨论与修改纪要）

本文档汇总 embodied **仅评测**（`only_eval` / `eval_embodied_agent.py`）及训练过程中 **val** 路径上，针对 **CPU 内存峰值** 与 **Ray 节点内存压力 OOM** 的讨论结论，并**完整记录评测相关代码修改**（含调用链与关键片段）。

---

## 1. 评测代码路径与数据流

### 1.1 入口与 Runner

- **入口**：`examples/embodiment/eval_embodied_agent.py`  
  - 创建 `Cluster`、`MultiStepRolloutWorker`、`EnvWorker` 组，`EmbodiedEvalRunner.init_workers()` 后调用 `runner.run()`。
- **评测编排**：`rlinf/runners/embodied_eval_runner.py` 的 `EmbodiedEvalRunner.evaluate()`：
  - 同时启动 **`EnvWorker.evaluate`** 与 **`MultiStepRolloutWorker.evaluate`**（异步协同，经 `Channel` 交换观测与动作）。
  - `env_handle.wait()` 得到各 env 进程返回的 **per-rank 指标字典**（内含 `torch.Tensor`）。
  - 在 driver 侧调用 **`compute_evaluate_metrics(eval_metrics_list)`** 做跨 rank 聚合。

```python
# rlinf/runners/embodied_eval_runner.py — EmbodiedEvalRunner.evaluate()
def evaluate(self):
    env_handle: Handle = self.env.evaluate(
        input_channel=self.rollout_channel,
        output_channel=self.env_channel,
    )
    rollout_handle: Handle = self.rollout.evaluate(
        input_channel=self.env_channel,
        output_channel=self.rollout_channel,
    )
    env_results = env_handle.wait()
    rollout_handle.wait()
    eval_metrics_list = [results for results in env_results if results is not None]
    eval_metrics = compute_evaluate_metrics(eval_metrics_list)
    return eval_metrics
```

### 1.2 Env 侧（指标如何产生）

- **`EnvWorker.evaluate`**（`rlinf/workers/env/env_worker.py`）：  
  - 循环中 `env_evaluate_step` 将 `env_info` 里各 key 的 tensor **`append` 到 `eval_metrics[key]` 列表**。  
  - 评测结束、关闭 eval 环境后，将每个 key 的列表 **合并为单个 tensor** 并返回（供 `compute_evaluate_metrics` 使用）。

### 1.3 Rollout 侧（策略推理）

- **`MultiStepRolloutWorker.evaluate`**（`rlinf/workers/rollout/hf/huggingface_worker.py`）：  
  - `enable_offload` 时先 **`reload_model()`** 把 VLA（及 SigLIP 条件模型）加载到 GPU。  
  - 进度条文案为 **`Evaluating Rollout Epochs`**。  
  - 每步：`recv_env_output(..., mode="eval")` → `preprocess_env_obs` → `predict(..., mode="eval")` → `send_chunk_actions(..., mode="eval")`。

### 1.4 Driver 侧聚合

- **`compute_evaluate_metrics`**（`rlinf/utils/metric_utils.py`）：  
  - 对每个 metric key，将多个 env rank 上的 tensor **聚合成标量均值**（并写入 `num_trajectories`）。

---

## 2. 背景与问题

### 2.1 `EnvWorker.evaluate` 末尾的指标合并

- **原写法**：`torch.cat(value, dim=0).contiguous().cpu()`。  
- **风险**：`cat` + 强制 `contiguous` 易产生「所有 chunk + 一整块新缓冲」的短峰值；列表在合并完成前一直持有全部 chunk 引用。

### 2.2 Ray 报节点内存 OOM

- **日志特征**：`Evaluating Rollout Epochs` 完成后，raylet 报 **Workers killed due to memory pressure**。  
- **主要根因**：**`env.eval.total_num_envs` 过大**时，单个 EnvWorker 进程内并行大量 LIBERO/MuJoCo 实例，**节点 RSS** 暴涨。指标合并是次要因素。  
- **配置原则**：避免单进程 **数百路** 并行仿真；需要更大吞吐时通过 **多 env 进程 / 多节点** 分摊（使 `total_num_envs // env_world_size // stage_num` 落在机器可承受范围）。

---

## 3. 评测相关代码修改详录

### 3.1 `rlinf/workers/env/env_worker.py`

**新增**模块级函数 **`_concat_metric_tensors_lowmem`**：

- 各 chunk 的 `device` / `dtype` / `shape[1:]` 一致时：`torch.empty` 预分配，`while chunks: c = chunks.pop()` 从后往前 **`copy_`**，减少同时存活的对象引用。  
- 不一致时回退 **`torch.cat`**。  
- 单元素时 **`chunks.clear()`** 后返回（必要时再 `contiguous()`）。

**`evaluate` 末尾**（在 `close` / `stop_env` 之后）：

- 先 **`gc.collect()`**，再对每个 key 调用 **`_concat_metric_tensors_lowmem(value).cpu()`**，最后再 **`gc.collect()`** 与 **`torch.cuda.empty_cache()`**（若可用）。

**`interact`（训练 rollout）** 末尾合并 `env_metrics` 时同样使用 **`_concat_metric_tensors_lowmem(...).cpu()`**（与评测路径一致的低峰值合并策略）。

### 3.2 `rlinf/utils/metric_utils.py`（driver 聚合，评测结果必经）

**新增** **`_tensor_list_global_mean_numpy(tensors)`**：

- 对每个 tensor 做 `.float()`，累加 **`sum().item()`** 与 **`numel()`**，最后 **`total / count`** 得到与 **`torch.concat(...).mean()`** 等价的全局标量均值（全体元素平均），**避免**跨 rank 再做大 **`torch.concat`**。

**`compute_evaluate_metrics`** 中每个 `env_info_key`：

- 构建 **`per_rank = [...]`** 后立刻 **`_tensor_list_global_mean_numpy(per_rank)`**，并 **`del per_rank`**，降低峰值引用。

### 3.3 `rlinf/workers/rollout/hf/huggingface_worker.py`（评测 rollout）

**`evaluate` 内层循环**（每步 `send_chunk_actions` 之后）：

- **`del env_output, extracted_obs, actions`**，便于尽快释放观测与动作相关的大对象引用。

**`evaluate` 结束后**：

- **`enable_offload == True`**：调用 **`_release_rollout_memory_after_eval()`**（将 `hf_model`、可选的 `siglip_condition_model` 迁到 CPU，`gc`、`cuda.synchronize`、`empty_cache`、`ipc_collect` 等）。  
- **否则**：仅 **`gc.collect()`** + **`cuda.synchronize` / `empty_cache`**，减轻评测结束后的缓存占用。

说明：**`_release_rollout_memory_after_eval`** 与 **`offload_model`** 类似，但多 **`synchronize` / `ipc_collect`** 与二次 **`gc`**，更适合评测收尾。

### 3.4 配置文件（评测并行环境数）

- 文件示例：`examples/embodiment/config/libero_object_grpo_openvlaoft_opt_eval.yaml`  
- 关注 **`env.eval.total_num_envs`**：曾出现 **400** 导致 Ray OOM；需按机器内存调小，或增加 env worker 数分摊。  
- 仓库中该示例当前可能为 **64 / 200** 等，以你本地 yaml 为准；原则是 **单进程并行 env 数与内存线性相关，过大必炸**。

---

## 4. 使用与调参建议

1. **`env.eval.total_num_envs`**：评测 OOM 时**首先**下调该值或增加 env 并行度（world size）。  
2. **Ray**：可参考 [Ray OOM 预防](https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html) 调整阈值；**不能**替代减少仿真并发。  
3. 若使用**自定义 eval yaml**，请同步检查是否仍配置过大的 **`total_num_envs`**。

---

## 5. 相关文件一览

| 文件 | 评测相关变更 |
|------|----------------|
| `examples/embodiment/eval_embodied_agent.py` | 评测入口（无内存专项改动，属调用链） |
| `rlinf/runners/embodied_eval_runner.py` | `evaluate()` 调度 env + rollout + `compute_evaluate_metrics` |
| `rlinf/workers/env/env_worker.py` | `evaluate` 末尾低峰值 concat、`gc` 时机 |
| `rlinf/utils/metric_utils.py` | `compute_evaluate_metrics` 无大 concat 聚合 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | `evaluate` 步内 `del`、结束 `_release` / `empty_cache` |
| `examples/embodiment/config/*.yaml` | `env.eval.total_num_envs` 等 |

---

*文档用于对照「评测全链路」上的内存相关修改；若后续改动上述函数，请同步更新本节。*
