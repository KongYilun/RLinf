# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict, masked_normalization
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    cat_list_of_dict_tensor,
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper
from step3_encoder_training.condition_model_sto_new import SiglipConditionRLModel


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


def get_nested_k_split_for_specific_keys(nested_dict, num_splits, key_list):
    """
    Get k-split iterator for some keys in nested_dict.
    """
    extra_dict = {}
    for key in key_list:
        if key not in nested_dict.keys():
            continue
        value = nested_dict[key]
        if isinstance(value, dict):
            extra_dict[key] = split_dict_to_chunk(value, num_splits)
        elif isinstance(value, torch.Tensor):
            continue
        else:
            raise NotImplementedError(
                f"Only support dict and tensor type, but got {type(value)}"
            )
    # {key1: [d1, d2, ...], key2: [d1, d2, ...]} -> [{key1: d1, key2: d1}, {key1: d2, key2: d2}, ...]
    extra_list = [
        {k: extra_dict[k][i] for k in extra_dict.keys()} for i in range(num_splits)
    ]
    return extra_list


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            self.cfg.actor.model.encoder_seq_length - self.cfg.data.max_prompt_length
        )
        self.calculate_entropy = self.cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            self.cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = self.cfg.algorithm.kl_beta
        self.kl_penalty_type = self.cfg.algorithm.kl_penalty_type

        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.group_size
            // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(self.cfg.algorithm.loss_agg_func)
        self.enable_offload = (
            self.cfg.actor.get("enable_offload", False) and not self.is_pipeline
        )
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        self.n_mini_batches = self.cfg.algorithm.n_minibatches
        self.task_type = self.cfg.runner.task_type
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "liger_kernel")

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if self.cfg.algorithm.kl_beta > 0 and self.cfg.actor.get(
            "combine_reference_model", True
        ):
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        if hasattr(self, "rollout_state_dict"):
            del self.rollout_state_dict
        clear_memory(sync=False)

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device, True)

        self.rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=True
        )

        has_visual = any("visual." in k for k in self.rollout_state_dict.keys())

        state_dict = {}

        if self._weight_dst_rank_in_rollout is not None:
            for k, v in self.rollout_state_dict.items():
                name = k
                if has_visual:
                    if name.startswith("model.language_model."):
                        name = "model." + name[21:]
                    # NOTE:
                    # if transformers version is 4.56.1 or older(not tested),
                    # the following line should be uncommented

                    # elif name.startswith("model."):
                    #     name = name[6:]
                state_dict[name] = reduce_tensor(v) if not self.is_pipeline else v
            if not self.is_pipeline:
                self.send(
                    state_dict,
                    self._rollout_group_name,
                    self._weight_dst_rank_in_rollout,
                )
            else:
                for weight_dst_rank in self._weight_dst_rank_in_rollout:
                    self.send(
                        state_dict,
                        self._rollout_group_name,
                        weight_dst_rank,
                    )

        state_dict.clear()
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    @torch.no_grad()
    def inference_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in batch.keys():
            for key in batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in batch["multi_modal_inputs"]],
                    dim=0,
                ).cuda()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **multi_modal_inputs,
        )

        logits = outputs.logits
        logits = logits[:, -self.response_len - 1 : -1, :]
        logits = logits / self.cfg.algorithm.sampling_params.temperature

        responses = input_ids[:, -self.response_len :]
        logprobs = compute_logprobs_from_logits(
            logits=logits, target=responses, op_type=self.entropy_op_type
        )
        return logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
    ) -> None:
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            recv_batch_size += rollout_result.num_sequence
            self._load_weight_and_optimizer()

            num_splits = (
                rollout_result.num_sequence
                // self.cfg.algorithm.logprob_forward_micro_batch_size
            )
            micro_batches_iter = get_iterator_k_split(
                batch,
                num_splits=num_splits,
            )
            micro_batches = list(micro_batches_iter)

            prev_logprobs = []
            with self.worker_timer():
                for micro_batch in micro_batches:
                    prev_logprobs.append(self.inference_step(micro_batch).cpu())

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = torch.cat(prev_logprobs)
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = torch.cat(prev_logprobs)

            if compute_ref_logprobs:
                assert self.ref_policy_state_dict is not None, (
                    "Reference policy state dict is None but compute_ref_logprobs is True"
                )
                ref_logprobs = []
                with cpu_weight_swap(self.model, self.ref_policy_state_dict):
                    for micro_batch in micro_batches:
                        ref_logprobs.append(self.inference_step(micro_batch).cpu())
                    rollout_result.ref_logprobs = torch.cat(ref_logprobs)

            output_channel.put(rollout_result)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt
            micro_batches = get_iterator_k_split(batch, micro_batch_cnt)
            micro_batches_iter = iter(micro_batches)
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
            )
            for k, v in m_batch.items():
                m_batch[k] = v.cuda() if isinstance(v, torch.Tensor) else v

            multi_modal_inputs = {}
            if "multi_modal_inputs" in m_batch.keys():
                for key in m_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                        dim=0,
                    ).cuda()

            input_ids = m_batch["input_ids"]
            attention_mask = m_batch["attention_mask"]
            position_ids = m_batch["position_ids"]
            prev_logprobs = m_batch["prev_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            with self.amp_context:
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                logits: torch.Tensor = output.logits

                logits.div_(self.cfg.algorithm.sampling_params.temperature)

                responses = input_ids[:, -self.response_len :]
                logits = logits[
                    :, -self.response_len - 1 : -1, :
                ]  # (bsz, response_length, vocab_size)
                logprobs = compute_logprobs_from_logits(
                    logits, responses, self.entropy_op_type
                )

                if self.cfg.algorithm.get("importance_sampling_fix", False):
                    rollout_prev_logprobs = prev_logprobs
                    recompute_prev_logprobs = m_batch["recompute_prev_logprobs"]
                    advantages = advantages * torch.clamp(
                        (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                        min=self.cfg.algorithm.importance_sampling_clip,
                    )

                loss, mbs_metrics_data = policy_loss(
                    loss_type=self.cfg.algorithm.loss_type,
                    loss_agg_func=self.loss_agg_func,
                    logprobs=logprobs,
                    old_logprobs=prev_logprobs,
                    advantages=advantages,
                    clip_ratio_low=clip_ratio_low,
                    clip_ratio_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_mask=loss_mask,
                    task_type=self.task_type,
                )

                entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.calculate_entropy:
                    entropy = compute_entropy_from_logits(
                        logits,
                    )

                    entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                    if self.calculate_entropy_loss:
                        loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

                kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.kl_beta > 0 and ref_logprobs is not None:
                    kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                    kl_loss = self.loss_agg_func(kld, loss_mask)
                    loss = loss + kl_loss * self.kl_beta

                # add to log
                # scale loss for gradient accumulation and backprop
                loss = loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": loss.detach(),
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()
        return mbs_metrics_list, grad_norm, lr_list

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer():
            for _ in range(self.n_mini_batches):
                metrics, grad_norm, lr_list = self.training_step(
                    batch=train_batch_iterator
                )

                # aggregate metrics across micro-batches
                mean_metric_dict = {
                    key: torch.mean(torch.stack(value))
                    for key, value in metrics.items()
                }
                mean_metric_dict = all_reduce_dict(
                    mean_metric_dict, op=torch.distributed.ReduceOp.AVG
                )

                mean_metric_dict["actor/grad_norm"] = float(grad_norm)
                mean_metric_dict["actor/lr"] = lr_list[0]
                training_metrics_list.append(mean_metric_dict)

        # put lr scheduler step here
        self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def run_training(self, input_channel: Channel) -> tuple[dict, list]:
        # Get all batches for this DP
        if self.is_pipeline:
            with self.worker_timer():
                return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                metrics, grad_norm, lr_list = self.training_step(batch=mini_batch)

                # aggregate metrics across micro-batches
                mean_metric_dict = {
                    key: torch.mean(torch.stack(value))
                    for key, value in metrics.items()
                }
                mean_metric_dict = all_reduce_dict(
                    mean_metric_dict, op=torch.distributed.ReduceOp.AVG
                )

                mean_metric_dict["actor/grad_norm"] = float(grad_norm)
                mean_metric_dict["actor/lr"] = lr_list[0]
                training_metrics_list.append(mean_metric_dict)

        # put lr scheduler step here
        self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].cuda(),
                    loss_mask=mask.cuda(),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.cfg.algorithm.get("reinpp_kl_beta", 0.0),
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=batch["prev_logprobs"].cuda()
                    if "prev_logprobs" in batch
                    else None,
                    ref_logprob=batch["ref_logprobs"].cuda()
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        self.condition_policy_cfg = self.cfg.algorithm.get("condition_policy") or {}
        self.condition_policy_enable_rl = bool(
            self.condition_policy_cfg.get("enable_rl", False)
        )
        # SigLIP REINFORCE scheduling; always defined so run_training can stay simple.
        # Incremented only when condition_policy_enable_rl (see run_training).
        self._vla_step_counter = 0
        # Set each iteration before run_training (runner global_step).
        self._runner_global_step = 0

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """
        Setup destination ranks for weight communication.
        It can support any topology between actor and rollout workers.
        Assuming there are M actor ranks and N rollout ranks, each actor rank
        will send weights to most ceil(N/M) rollout ranks according to the modulo rule.
        """
        rollout_world_size = self._component_placement.get_world_size("rollout")
        actor_world_size = self._world_size
        rank = self._rank
        self._weight_dst_rank_in_rollout = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            if i * actor_world_size + rank < rollout_world_size:
                self._weight_dst_rank_in_rollout.append(i * actor_world_size + rank)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()
        if self.condition_policy_enable_rl:
            self._init_condition_policy()

    def _init_condition_policy(self) -> None:
        device_str = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        ckpt_path = self.condition_policy_cfg.get(
            "checkpoint_path", None
        ) or "step3_encoder_training/siglip_condition_model_sto_0.pt"
        ckpt = torch.load(ckpt_path, map_location=device_str)
        num_classes: int = ckpt["num_classes"]
        residual_dim: int = ckpt["residual_dim"]
        siglip_model_path: str = ckpt["siglip_model_path"]
        center_embed_dim: int = int(ckpt.get("center_embed_dim", 512))

        model = SiglipConditionRLModel(
            model_path=siglip_model_path,
            num_classes=num_classes,
            residual_dim=residual_dim,
            center_embed_dim=center_embed_dim,
        )
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        device = torch.device(device_str)
        model.to(device)
        # if device.type == "cuda" and torch.cuda.is_bf16_supported():
        #     model.to(dtype=torch.bfloat16)
        model.train()

        reinforce_lr = float(self.condition_policy_cfg.get("lr", 1e-5))
        reinforce_wd = float(self.condition_policy_cfg.get("weight_decay", 0.01))
        grpo_lr = float(self.condition_policy_cfg.get("grpo_cm_lr", 1e-4))
        grpo_wd = float(self.condition_policy_cfg.get("grpo_cm_weight_decay", 0.01))
        self._siglip_reinforce_trainable_params = list(model.siglip.parameters()) + list(
            model.classifier_head.parameters()
        )
        self._siglip_grpo_trainable_params = list(model.cluster_embed.parameters()) + list(
            model.residual_head.parameters()
        )
        self.siglip_condition_reinforce_optimizer = torch.optim.AdamW(
            self._siglip_reinforce_trainable_params,
            lr=reinforce_lr,
            weight_decay=reinforce_wd,
        )
        self.siglip_condition_grpo_optimizer = torch.optim.AdamW(
            self._siglip_grpo_trainable_params,
            lr=grpo_lr,
            weight_decay=grpo_wd,
        )
        self.siglip_condition_model = model
        self._cond_baseline = torch.zeros(1, device=device)
        self._cond_policy_buffer: dict[str, list[torch.Tensor]] = {
            "task_ids": [],
            "residual": [],
            "reward": [],
            "images": [],
            "pred_cluster_idx": [],
        }

        instr_map = torch.load("libero_object_instruction_to_task_id_map.pt")
        self._task_id_to_instruction = {int(v): k for k, v in instr_map.items()}

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device)

        state_dict = self.get_model_state_dict(cpu_offload=False, full_state_dict=True)
        for rank in self._weight_dst_rank_in_rollout:
            self.send(
                state_dict,
                self._rollout_group_name,
                rank,
                async_op=True,
            )
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        if self.condition_policy_enable_rl:
            cp_sd = {
                k: v.detach().cpu()
                for k, v in self.siglip_condition_model.state_dict().items()
            }
            for rank in self._weight_dst_rank_in_rollout:
                self.send(cp_sd, self._rollout_group_name, rank, async_op=True)

    def recv_rollout_batch(self, input_channel: Channel) -> None:
        """
        Receive rollout batch from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        self.rollout_batch = {}
        recv_list = []
        for _ in range(split_num):
            recv_list.append(input_channel.get())

        # shape [num_chunk, bsz, chunk_size], cat dim 1
        self.rollout_batch = cat_list_of_dict_tensor(recv_list, dim=1)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        self._attach_cond_group_reward_for_condition_policy()
        return rollout_metrics

    def _attach_cond_group_reward_for_condition_policy(self) -> None:
        if not self.condition_policy_enable_rl:
            return
        rb = self.rollout_batch
        if "cond_log_prob_cluster" not in rb or "rewards" not in rb:
            return
        rewards = rb["rewards"]
        # from ray.util import pdb; pdb.set_trace()
        gs = self.cfg.algorithm.group_size
        s0, rb_sz = rewards.shape[0], rewards.shape[1]
        per_env = rewards.sum(dim=(0, 2))
        n_prompts = rb_sz // gs
        rmat = per_env.reshape(n_prompts, gs)
        rz = rmat.mean(dim=1)
        cond_r = rz.repeat_interleave(gs)
        rb["cond_group_reward"] = cond_r.unsqueeze(0).expand(s0, -1).contiguous()

    def _snapshot_condition_policy_buffer(self) -> None:
        if not self.condition_policy_enable_rl:
            return
        rb = self.rollout_batch
        need = (
            "cond_residual",
            "cond_initial_image_hwc",
            "cond_task_ids",
            "cond_group_reward",
            "cond_pred_cluster_idx",
        )
        if not all(k in rb for k in need):
            return
        gs = self.cfg.algorithm.group_size
        # from ray.util import pdb; pdb.set_trace()
        _, rb_sz = rb["cond_log_prob_cluster"].shape[:2]
        idx = torch.arange(0, rb_sz, gs, device=rb["cond_task_ids"].device)

        self._cond_policy_buffer["task_ids"].append(
            rb["cond_task_ids"][0, idx].detach().cpu().clone()
        )
        self._cond_policy_buffer["residual"].append(
            rb["cond_residual"][0, idx].detach().cpu().clone()
        )
        self._cond_policy_buffer["reward"].append(
            rb["cond_group_reward"][0, idx].detach().cpu().clone()
        )
        self._cond_policy_buffer["images"].append(
            rb["cond_initial_image_hwc"][0, idx].detach().cpu().clone()
        )
        self._cond_policy_buffer["pred_cluster_idx"].append(
            rb["cond_pred_cluster_idx"][0, idx].detach().cpu().clone()
        )

    def _scheduled_cluster_only_phase_active(self) -> bool:
        """Whether ``cluster_only_schedule_end_step`` is set and current runner step is inside it."""
        cp = self.condition_policy_cfg
        end = cp.get("cluster_only_schedule_end_step", None)
        if end is None:
            return False
        start = int(cp.get("cluster_only_schedule_start_step", 0))
        g = int(getattr(self, "_runner_global_step", 0))
        return int(start) <= g <= int(end)

    def _cm_grpo_training_enabled(self) -> bool:
        if not self.condition_policy_enable_rl:
            return False
        if not self.cfg.algorithm.get("use_optimizable_embedding", False):
            return False
        if self._scheduled_cluster_only_phase_active():
            return False
        start = int(self.condition_policy_cfg.get("grpo_cm_start_step", 0))
        end = self.condition_policy_cfg.get("grpo_cm_end_step", None)
        g = int(getattr(self, "_runner_global_step", 0))
        if g < start:
            return False
        if end is not None and int(end) >= 0 and g > int(end):
            return False
        return True

    def _micro_batch_has_cm_grpo_inputs(self, data: dict) -> bool:
        if "z_ids" not in data or data["z_ids"] is None:
            return False
        if data["z_ids"].dim() != 2:
            return False
        if "cond_initial_image_hwc" not in data:
            return False
        if "cond_pred_cluster_idx" not in data:
            return False
        return True

    def _build_grpo_z_for_microbatch(self, data: dict) -> torch.Tensor:
        mb = int(data["prev_logprobs"].shape[0])
        imgs = data["cond_initial_image_hwc"]
        pidx = data["cond_pred_cluster_idx"].to(dtype=torch.long).view(-1)[:mb]
        tid = data.get("cond_task_ids", data["task_ids"]).view(-1)[:mb]
        images_list = [imgs[i].detach().cpu().numpy() for i in range(mb)]
        instructions = [
            self._task_id_to_instruction[int(tid[i].item())] for i in range(mb)
        ]
        return self.siglip_condition_model.forward_z_for_actor_grpo(
            images_list, instructions, pidx
        )

    def _should_step_siglip_grpo_optimizer(self) -> bool:
        if not self.condition_policy_enable_rl:
            return False
        if not getattr(self, "_cm_grpo_enabled_this_train", False):
            return False
        for p in self._siglip_grpo_trainable_params:
            if p.grad is not None:
                return True
        return False

    def optimizer_step(self) -> tuple[float, list[float]]:
        self.optimizer_steps += 1
        self.grad_scaler.unscale_(optimizer=self.optimizer)
        grad_norm = self._strategy.clip_grad_norm_(model=self.model)

        main_finite = torch.isfinite(torch.as_tensor(grad_norm))
        if not main_finite:
            self._logger.warning(
                f"[FSDP] Non-finite grad norm {grad_norm} detected. Skipping optimizer step."
            )
        else:
            self.grad_scaler.step(optimizer=self.optimizer)

        if main_finite and self._should_step_siglip_grpo_optimizer():
            self.grad_scaler.unscale_(optimizer=self.siglip_condition_grpo_optimizer)
            gclip = self.condition_policy_cfg.get("grpo_cm_grad_clip", None)
            max_norm = float(gclip) if gclip is not None else float("inf")
            cm_gn = torch.nn.utils.clip_grad_norm_(
                self._siglip_grpo_trainable_params, max_norm
            )
            if torch.isfinite(torch.as_tensor(cm_gn)):
                self.grad_scaler.step(optimizer=self.siglip_condition_grpo_optimizer)

        self.grad_scaler.update()

        if self.critic_warmup_steps > 0:
            lr_list = [0.0 for _ in self.optimizer.param_groups]
            if self.optimizer_steps >= self.critic_warmup_steps:
                self.optimizer = self.build_optimizer(model=self.model)
                self.critic_warmup_steps = 0
        else:
            lr_list = [group["lr"] for group in self.optimizer.param_groups]

        return grad_norm, lr_list

    def _cond_reinforce_advantages(
        self,
        rewards_dev: torch.Tensor,
        task_ids_cpu: torch.Tensor,
        device: torch.device,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """REINFORCE advantages for condition policy: global z-score or per-task stats."""
        mode = str(
            self.condition_policy_cfg.get("reinforce_adv_mode", "global")
        ).lower()
        standardize_pt = bool(
            self.condition_policy_cfg.get("reinforce_per_task_adv_standardize", True)
        )

        with torch.no_grad():
            if dist.is_initialized() and self._world_size > 1:
                n_loc = torch.tensor(
                    float(rewards_dev.numel()),
                    device=device,
                    dtype=torch.float32,
                )
                sum_loc = rewards_dev.sum()
                sumsq_loc = (rewards_dev * rewards_dev).sum()
                t = torch.stack([sum_loc, sumsq_loc, n_loc])
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                n_g = t[2].clamp(min=1.0)
                mean_r = t[0] / n_g
                var = t[1] / n_g - mean_r * mean_r
                var = var.clamp(min=0.0)
                std_r = torch.sqrt(var) + eps
            else:
                mean_r = rewards_dev.mean()
                std_r = rewards_dev.std(unbiased=False) + eps

            adv_global = (rewards_dev - mean_r) / std_r

            if mode != "per_task":
                return adv_global.detach(), mean_r, std_r

            tid = task_ids_cpu.long().to(device=device)
            tid_max = int(tid.max().item()) if tid.numel() > 0 else 0
            if self._task_id_to_instruction:
                map_max = max(self._task_id_to_instruction.keys())
                num_tasks = max(tid_max, int(map_max)) + 1
            else:
                num_tasks = tid_max + 1

            sum_t = torch.zeros(num_tasks, device=device, dtype=torch.float32)
            cnt_t = torch.zeros(num_tasks, device=device, dtype=torch.float32)
            sumsq_t = torch.zeros(num_tasks, device=device, dtype=torch.float32)
            sum_t.scatter_add_(0, tid, rewards_dev)
            cnt_t.scatter_add_(0, tid, torch.ones_like(rewards_dev))
            sumsq_t.scatter_add_(0, tid, rewards_dev * rewards_dev)

            if dist.is_initialized() and self._world_size > 1:
                dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(sumsq_t, op=dist.ReduceOp.SUM)

            mean_t = sum_t / cnt_t.clamp(min=1.0)
            var_t = sumsq_t / cnt_t.clamp(min=1.0) - mean_t * mean_t
            var_t = var_t.clamp(min=0.0)
            std_t = torch.sqrt(var_t) + eps

            mean_i = mean_t[tid]
            cnt_i = cnt_t[tid]

            if standardize_pt:
                std_i = std_t[tid]
                adv_pt = (rewards_dev - mean_i) / std_i
            else:
                adv_pt = rewards_dev - mean_i

            low_cnt = cnt_i < 2.0
            adv = torch.where(low_cnt, adv_global, adv_pt)
            return adv.detach(), mean_r, std_r

    def _vla_micro_batch_effective(self) -> int:
        """Optionally halve VLA actor micro-batch from ``half_micro_batch_from_runner_step``."""
        base = int(self.cfg.actor.micro_batch_size)
        if not bool(self.cfg.actor.get("half_micro_batch_enable", False)):
            return base
        thr = self.cfg.actor.get("half_micro_batch_from_runner_step", None)
        if thr is None:
            return base
        g = int(getattr(self, "_runner_global_step", 0))
        if g < int(thr):
            return base
        half = base // 2
        assert half * 2 == base, (
            "actor.micro_batch_size must be even when actor.half_micro_batch_enable is true"
        )
        return max(half, 1)

    def _train_condition_policy_if_due(self) -> dict[str, float]:
        if not self.condition_policy_enable_rl:
            return {}
        rds = self.condition_policy_cfg.get("reinforce_disable_from_runner_step", None)
        if rds is not None and int(getattr(self, "_runner_global_step", 0)) >= int(
            rds
        ):
            return {}
        interval = int(self.condition_policy_cfg.get("update_interval_vla_steps", 5))
        if self._vla_step_counter == 0 or self._vla_step_counter % interval != 0:
            return {}
        buf = self._cond_policy_buffer
        if len(buf["task_ids"]) == 0:
            return {}
        task_ids = torch.cat(buf["task_ids"], dim=0)
        residual = torch.cat(buf["residual"], dim=0)
        rewards = torch.cat(buf["reward"], dim=0)
        images_u8 = torch.cat(buf["images"], dim=0)
        pred_cluster_idx_all = torch.cat(buf["pred_cluster_idx"], dim=0)
        for k in buf:
            buf[k].clear()

        device = next(self.siglip_condition_model.parameters()).device
        temp = float(
            self.condition_policy_cfg.get(
                "cluster_sample_temperature",
                self.cfg.algorithm.get("siglip_cluster_sample_temperature", 1.0),
            )
        )

        n = int(task_ids.shape[0])
        if n == 0:
            return {}

        rewards_dev = rewards.to(device=device, dtype=torch.float32)

        self.siglip_condition_model.train()
        eps = 1e-8
        adv, mean_r, std_r = self._cond_reinforce_advantages(
            rewards_dev, task_ids, device, eps
        )
        ent_coef = float(self.condition_policy_cfg.get("cluster_entropy_coef", 0.0))

        self._cond_baseline.copy_(mean_r.reshape_as(self._cond_baseline))

        images_list = [images_u8[i].numpy() for i in range(n)]
        instructions = [
            self._task_id_to_instruction[int(task_ids[i].item())] for i in range(n)
        ]
        pred_cluster_stored = pred_cluster_idx_all.to(device=device, dtype=torch.long)
        residual_sample = residual.to(device=device, dtype=torch.float32)

        micro_bs = min(int(self.condition_policy_cfg.get("micro_batch_size", 32)), n)
        reinforce_epochs = max(
            1, int(self.condition_policy_cfg.get("reinforce_epochs", 2))
        )
        grad_clip = self.condition_policy_cfg.get("grad_clip", None)

        total_pg = torch.zeros(1, device=device)
        total_lp = torch.zeros(1, device=device)
        total_ent = torch.zeros(1, device=device)
        n_steps = 0

        for _ in range(reinforce_epochs):
            for start in range(0, n, micro_bs):
                end = min(start + micro_bs, n)
                mb_img = images_list[start:end]
                mb_instr = instructions[start:end]
                mb_idx = pred_cluster_stored[start:end]
                mb_res = residual_sample[start:end]
                mb_adv = adv[start:end]

                self.siglip_condition_reinforce_optimizer.zero_grad(set_to_none=True)

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
                        reinforce_logprob="cluster",
                    )
                    logp = out_lp["log_prob_cluster"]
                    pg = -(mb_adv * logp).mean()
                    cluster_ent = out_lp["cluster_entropy"]
                    if ent_coef != 0.0:
                        loss_mb = pg - ent_coef * cluster_ent.mean()
                    else:
                        loss_mb = pg

                loss_mb.backward()

                if dist.is_initialized() and self._world_size > 1:
                    for p in self._siglip_reinforce_trainable_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._siglip_reinforce_trainable_params, float(grad_clip)
                    )

                self.siglip_condition_reinforce_optimizer.step()

                total_pg += pg.detach()
                total_lp += logp.detach().mean()
                total_ent += cluster_ent.detach().float().mean()
                n_steps += 1

        denom = max(n_steps, 1)
        metrics = {
            "cond/policy_loss": float((total_pg / denom).item()),
            "cond/log_prob_cluster_mean": float((total_lp / denom).item()),
            "cond/reward_mean": float(mean_r.item()),
            "cond/baseline": float(mean_r.item()),
            "cond/reward_std": float(std_r.item()),
            "cond/cluster_entropy_mean": float((total_ent / denom).item()),
        }
        if dist.is_initialized() and self._world_size > 1:
            t = torch.tensor(
                [
                    metrics["cond/policy_loss"],
                    metrics["cond/log_prob_cluster_mean"],
                    metrics["cond/reward_mean"],
                    metrics["cond/baseline"],
                    metrics["cond/reward_std"],
                    metrics["cond/cluster_entropy_mean"],
                ],
                device=device,
                dtype=torch.float32,
            )
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            metrics = {
                "cond/policy_loss": t[0].item(),
                "cond/log_prob_cluster_mean": t[1].item(),
                "cond/reward_mean": t[2].item(),
                "cond/baseline": t[3].item(),
                "cond/reward_std": t[4].item(),
                "cond/cluster_entropy_mean": t[5].item(),
            }
            del t

        self.siglip_condition_reinforce_optimizer.zero_grad(set_to_none=True)
        del (
            rewards_dev,
            adv,
            pred_cluster_stored,
            residual_sample,
            images_list,
            instructions,
            task_ids,
            residual,
            rewards,
            images_u8,
            pred_cluster_idx_all,
            mean_r,
            std_r,
            total_pg,
            total_lp,
            total_ent,
        )
        if device.type == "cuda":
            clear_memory(sync=True)

        return metrics

    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        self._cm_grpo_enabled_this_train = False
        if self.condition_policy_enable_rl and getattr(
            self, "siglip_condition_model", None
        ) is not None:
            self._cm_grpo_enabled_this_train = self._cm_grpo_training_enabled()
            if self._cm_grpo_enabled_this_train:
                self.siglip_condition_model.train()
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

        vla_mb = self._vla_micro_batch_effective()
        assert (
            self.cfg.actor.global_batch_size % (vla_mb * self._world_size) == 0
        ), "global_batch_size is not divisible by effective micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size // vla_mb // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = get_iterator_k_split(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % vla_mb == 0, (
                    f"{train_global_batch_size=}, {vla_mb=}"
                )

                train_micro_batch = get_iterator_k_split(
                    train_global_batch,
                    train_global_batch_size // vla_mb,
                )

                self.optimizer.zero_grad()
                if self.condition_policy_enable_rl and getattr(
                    self, "_cm_grpo_enabled_this_train", False
                ):
                    self.siglip_condition_grpo_optimizer.zero_grad(set_to_none=True)
                for idx, data in enumerate(train_micro_batch):
                    data = put_tensor_device(
                        data, f"cuda:{int(os.environ['LOCAL_RANK'])}"
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    advantages = data["advantages"]
                    prev_logprobs = data["prev_logprobs"]
                    returns = data.get("returns", None)
                    prev_values = data.get("prev_values", None)
                    loss_mask = data.get("loss_mask", None)
                    loss_mask_sum = data.get("loss_mask_sum", None)

                    z_grpo = None
                    data_for_model = data
                    _cm_grpo_this_mb = (
                        self.condition_policy_enable_rl
                        and getattr(self, "_cm_grpo_enabled_this_train", False)
                        and self._micro_batch_has_cm_grpo_inputs(data)
                        and SupportedModel(self.cfg.actor.model.model_type)
                        in (SupportedModel.OPENVLA, SupportedModel.OPENVLA_OFT)
                    )
                    if _cm_grpo_this_mb:
                        mb = int(data["prev_logprobs"].shape[0])
                        z_grpo = self._build_grpo_z_for_microbatch(data)
                        z_roll = data["z_ids"][:mb]
                        cra = data.get("cond_residual_applied")
                        if (
                            cra is not None
                            and isinstance(z_roll, torch.Tensor)
                            and z_roll.dim() == 2
                        ):
                            m = cra.reshape(-1)[:mb].to(
                                device=z_grpo.device, dtype=z_grpo.dtype
                            )
                            m = m.unsqueeze(1)
                            z_roll_d = z_roll.to(
                                device=z_grpo.device, dtype=z_grpo.dtype
                            ).detach()
                            z_mix = z_grpo * m + z_roll_d * (1.0 - m)
                            data_for_model = {**data, "z_ids": z_mix}
                        else:
                            data_for_model = {**data, "z_ids": z_grpo}

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        data_for_model["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        data_for_model["top_k"] = self.cfg.algorithm.sampling_params.top_k

                    compute_values = (
                        True if self.cfg.algorithm.adv_type == "gae" else False
                    )
                    

                    with self.amp_context:
                        output_dict = self.model(
                            data=data_for_model,
                            compute_logprobs=True,
                            compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                            compute_values=compute_values,
                            use_cache=False,
                        )

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.GR00T
                    ]:
                        prev_logprobs = output_dict["prev_logprobs"]

                    kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": output_dict["logprobs"],
                        "values": output_dict.get("values", None),
                        "old_logprobs": prev_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": prev_values,
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }
                    loss, metrics_data = policy_loss(**kwargs)

                    grpo_l2_coef = float(
                        self.condition_policy_cfg.get("grpo_residual_l2_coef", 0.5)
                    )
                    l2_cm_for_bwd = None
                    if z_grpo is not None and grpo_l2_coef > 0:
                        l2_cm_for_bwd = z_grpo.float().pow(2).sum(dim=-1).mean()
                        metrics_data["cond/grpo_residual_l2"] = (
                            l2_cm_for_bwd.detach().item()
                        )

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not kwargs["critic_warmup"]
                    ):
                        entropy = output_dict["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=output_dict["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                    metrics_data["entropy_loss"] = entropy_loss.detach().item()

                    loss /= self.gradient_accumulation
                    retain_for_l2 = l2_cm_for_bwd is not None
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward(
                            retain_graph=retain_for_l2
                        )
                        if retain_for_l2:
                            l2_term = (
                                grpo_l2_coef
                                * l2_cm_for_bwd
                                / self.gradient_accumulation
                            )
                            self.grad_scaler.scale(l2_term).backward()

                    if (
                        dist.is_initialized()
                        and self._world_size > 1
                        and getattr(self, "_cm_grpo_enabled_this_train", False)
                    ):
                        for p in self._siglip_grpo_trainable_params:
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

                    metrics_data["loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        if self.condition_policy_enable_rl:
            self._vla_step_counter += 1

        return mean_metric_dict

    def train_condition_policy_if_due(self) -> dict[str, float]:
        """
        REINFORCE on ``log_prob_cluster`` for ``siglip`` + ``classifier_head`` (if interval).

        Call from the runner **after** ``_save_checkpoint``. The saved
        ``condition_policy.pt`` reflects state **after** GRPO updates inside
        ``run_training`` and **before** this REINFORCE step.
        """
        if not self.condition_policy_enable_rl:
            return {}
        return self._train_condition_policy_if_due()

    def save_condition_policy_pt(self, checkpoint_dir: str) -> None:
        if not self.condition_policy_enable_rl or self._rank != 0:
            return
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, "condition_policy.pt")
        payload = {
            "model": self.siglip_condition_model.state_dict(),
            "reinforce_optimizer": self.siglip_condition_reinforce_optimizer.state_dict(),
            "grpo_optimizer": self.siglip_condition_grpo_optimizer.state_dict(),
            "baseline": self._cond_baseline.detach().cpu(),
            "vla_step_counter": self._vla_step_counter,
        }
        torch.save(payload, path)

    def load_condition_policy_pt(self, checkpoint_dir: str, global_step: int) -> None:
        if not self.condition_policy_enable_rl:
            return
        path = os.path.join(checkpoint_dir, "condition_policy.pt")
        if not os.path.isfile(path):
            interval = int(self.condition_policy_cfg.get("update_interval_vla_steps", 5))
            gs = int(global_step)
            tag = max(0, ((gs - 1) // interval) * interval) if gs > 0 else 0
            legacy_tagged = os.path.join(
                checkpoint_dir, f"condition_policy_vla_step_{tag}.pt"
            )
            if os.path.isfile(legacy_tagged):
                path = legacy_tagged
            else:
                return
        ck = torch.load(path, map_location="cpu")
        dev = next(self.siglip_condition_model.parameters()).device
        self.siglip_condition_model.load_state_dict(ck["model"])
        self.siglip_condition_model.to(dev)
        if "reinforce_optimizer" in ck:
            self.siglip_condition_reinforce_optimizer.load_state_dict(
                ck["reinforce_optimizer"]
            )
        elif "optimizer" in ck:
            self.siglip_condition_reinforce_optimizer.load_state_dict(ck["optimizer"])
        if "grpo_optimizer" in ck:
            self.siglip_condition_grpo_optimizer.load_state_dict(ck["grpo_optimizer"])
        if "baseline" in ck:
            self._cond_baseline = ck["baseline"].to(
                device=dev, dtype=self._cond_baseline.dtype
            )
        if "vla_step_counter" in ck:
            self._vla_step_counter = int(ck["vla_step_counter"])

    def set_global_step(self, global_step) -> None:
        """
        Set the global step for the model, if needed.
        """
        self._runner_global_step = int(global_step)
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
