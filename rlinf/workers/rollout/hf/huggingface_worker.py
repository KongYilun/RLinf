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

import copy
import gc
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.rollout.hf.utils import init_real_obs
from step3_encoder_training.condition_model_sto_new import SiglipConditionRLModel


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        # whether to use learned / optimizable embedding instead of discrete z_ids
        self.use_optimizable_embedding = cfg.algorithm.get(
            "use_optimizable_embedding", False
        )
        self.condition_policy_cfg = cfg.algorithm.get("condition_policy") or {}
        self.condition_policy_enable_rl = bool(
            self.condition_policy_cfg.get("enable_rl", False)
        )
        if self.condition_policy_enable_rl and not self.use_optimizable_embedding:
            raise ValueError(
                "algorithm.condition_policy.enable_rl requires "
                "algorithm.use_optimizable_embedding=True."
            )

        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        self.hf_model.eval()

        self.setup_sample_params()

        # lazily initialize SigLIP condition model and per-instruction centers
        # when using optimizable embeddings
        self.siglip_condition_model: SiglipConditionRLModel | None = None
        self.instruction_centers: dict[str, torch.Tensor] | None = None
        if self.use_optimizable_embedding:
            self._init_condition_model_and_centers()

        if self.enable_offload:
            self.offload_model()

    def _init_condition_model_and_centers(self):
        """
        Load SigLIP-based condition model and per-instruction cluster centers.

        Loads ``SiglipConditionRLModel`` (stochastic cluster + Gaussian residual).
        ``pred_cluster_idx`` matches the sampled cluster; ``residual_embedding`` is
        the sampled residual. z = geometric_center(cluster_id) + residual.
        """
        device = self.device
        device_str = f"cuda:{device}" if isinstance(device, int) else device

        ckpt_path = self.condition_policy_cfg.get(
            "checkpoint_path", None
        ) or "step3_encoder_training/siglip_condition_model_sto.pt"
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
        model.to(device)
        # if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        #     model.to(dtype=torch.bfloat16)
        model.eval()
        self.siglip_condition_model = model

        # per-instruction cluster centers: instruction -> [num_clusters, D]
        centers_path = "libero_object_per_instruction_centers.pt"
        instruction_centers = torch.load(centers_path, map_location="cpu")
        # keep tensors on CPU; move slices to CUDA on demand
        self.instruction_centers = instruction_centers

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"]
            if self._sampling_params["do_sample"]
            else 1.0,
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        self._eval_sampling_params = {
            "do_sample": True
            if self._sampling_params.get("temperature_eval", -1) > 0
            else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def predict(self, env_obs, z_ids, mode="train"):
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            kwargs = {"mode": mode}

        kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                z_ids=z_ids,
                **kwargs,
            )

        return actions, result

    def get_dones_and_rewards(
        self, env_output: dict[str, torch.Tensor], extracted_obs: dict[str, Any]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
        """
        Get dones and rewards from environment batch, handling auto_reset if needed.

        Args:
            env_output: Environment batch containing dones, rewards, and optionally final_obs

        Returns:
            Tuple of (dones, rewards, real_extracted_obs). dones and rewards are tensors.
        """
        # First step: no rewards yet, only dones
        real_extracted_obs = None
        if env_output["rewards"] is None:
            if hasattr(self.hf_model, "q_head"):
                real_extracted_obs = init_real_obs(extracted_obs)
            return (
                env_output["dones"].bool().cpu().contiguous(),
                None,
                real_extracted_obs,
            )

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()

        # Handle auto_reset: add bootstrap value to rewards for done episodes
        # Note: currently this is not correct for chunk-size>1 with partial reset
        if dones.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head"):
                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    final_extracted_obs = self.hf_model.preprocess_env_obs(final_obs)
                    if hasattr(self.hf_model, "q_head"):
                        real_extracted_obs = init_real_obs(final_extracted_obs)
                    actions, result = self.predict(final_extracted_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                # Add bootstrap value to the last step of done episodes
                rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()

        if real_extracted_obs is None and hasattr(self.hf_model, "q_head"):
            real_extracted_obs = init_real_obs(extracted_obs)
        return dones, rewards, real_extracted_obs

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

    def update_intervene_actions(self, env_output, forward_inputs):
        intervene_actions = env_output["intervene_actions"]
        intervene_flags = env_output["intervene_flags"]
        if intervene_actions is not None:
            if "action" in forward_inputs:
                policy_action = forward_inputs["action"].to(intervene_actions.device)
                policy_action = policy_action.reshape(
                    policy_action.shape[0], self.hf_model.num_action_chunks, -1
                )
                intervene_actions = intervene_actions.reshape(
                    intervene_actions.shape[0], self.hf_model.num_action_chunks, -1
                )
                action = intervene_actions * intervene_flags[
                    ..., None
                ] + policy_action * (~intervene_flags[..., None])
                action = action.reshape(action.shape[0], -1)
                forward_inputs["action"] = action
            else:
                raise NotImplementedError(f"{forward_inputs.keys()=}")
        return forward_inputs

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        if self.enable_offload:
            self.reload_model()

        self.buffer_list = [
            EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            for _ in range(self.num_pipeline_stages)
        ]

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            last_extracted_obs = [None for _ in range(self.num_pipeline_stages)]
            last_forward_inputs = [
                None for _ in range(self.num_pipeline_stages)
            ]  # save actions

            # per-GRPO-group latent: either discrete id or continuous embedding
            if self.use_optimizable_embedding:
                group_latents = [None for _ in range(self.num_pipeline_stages)]
                pending_cond_meta = (
                    [None for _ in range(self.num_pipeline_stages)]
                    if self.condition_policy_enable_rl
                    else None
                )
            else:
                val = torch.randint(0, 10, (1,)).item()
                group_tensor = torch.full(
                    (self.cfg.algorithm.group_size,),
                    fill_value=val,
                    dtype=torch.long,
                    device=self.device,
                )
            # from ray.util import pdb; pdb.set_trace()
            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)

                    if last_forward_inputs[stage_id] is not None:
                        last_forward_inputs[stage_id] = self.update_intervene_actions(
                            env_output, last_forward_inputs[stage_id]
                        )

                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                        env_output, extracted_obs
                    )
                    # initialize learned z-embedding from the first-step observation
                    if self.use_optimizable_embedding:
                        if group_latents[stage_id] is None:
                            obs = env_output["obs"]
                            main_images = obs.get("main_images", None)
                            task_desc = obs.get("task_descriptions", None)

                            if main_images is None or task_desc is None:
                                raise RuntimeError(
                                    "main_images or task_descriptions missing in env obs "
                                    "while use_optimizable_embedding=True."
                                )

                            # main_images: [B, ..., H, W, C] or [B, ..., C, H, W]
                            # take the first camera / time dimension if present
                            imgs = main_images
                            if imgs.ndim == 5:
                                # [B, T, H, W, C] or [B, T, C, H, W] -> use first T
                                imgs = imgs[:, 0]
                            B = imgs.shape[0]

                            # detect whether all initial states in this GRPO group share
                            # the same first-frame image
                            # flat = imgs.reshape(B, -1)
                            # same_mask = torch.all(
                            #     flat == flat[0:1], dim=1
                            # ).detach().cpu()
                            # if not bool(same_mask.all()):
                            #     # only print once per group to avoid log spam
                            #     print(
                            #         "[MultiStepRolloutWorker] Warning: initial states in "
                            #         "a GRPO group are not identical when using "
                            #         "optimizable embeddings."
                            #     )

                            # build SigLIP inputs from the first env in the group
                            img0 = imgs[0].detach().cpu()
                            if img0.ndim != 3:
                                raise RuntimeError(
                                    f"Unexpected image rank for SigLIP: {img0.shape}"
                                )
                            # HWC or CHW -> convert to HWC uint8
                            if img0.shape[-1] == 3:
                                img0_hwc = img0
                            elif img0.shape[0] == 3:
                                img0_hwc = img0.permute(1, 2, 0)
                            else:
                                raise RuntimeError(
                                    f"Cannot infer channel dimension from shape {img0.shape}"
                                )
                            if img0_hwc.dtype.is_floating_point:
                                img0_hwc = (img0_hwc.clamp(0.0, 1.0) * 255.0).to(
                                    torch.uint8
                                )
                            elif img0_hwc.dtype != torch.uint8:
                                img0_hwc = img0_hwc.to(torch.uint8)

                            instr0 = (
                                task_desc[0]
                                if isinstance(task_desc, (list, tuple))
                                else task_desc
                            )
                            if isinstance(instr0, bytes):
                                instr0 = instr0.decode("utf-8")

                            if self.siglip_condition_model is None:
                                raise RuntimeError(
                                    "SiglipConditionRLModel not initialized while "
                                    "use_optimizable_embedding=True."
                                )
                            if self.instruction_centers is None:
                                raise RuntimeError(
                                    "instruction_centers not initialized while "
                                    "use_optimizable_embedding=True."
                                )

                            siglip_temp = float(
                                self.condition_policy_cfg.get(
                                    "cluster_sample_temperature",
                                    self.cfg.algorithm.get(
                                        "siglip_cluster_sample_temperature", 1.0
                                    ),
                                )
                            )
                            if self.condition_policy_enable_rl:
                                det_c = bool(
                                    self.condition_policy_cfg.get(
                                        "deterministic_cluster_train", False
                                    )
                                )
                                det_r = bool(
                                    self.condition_policy_cfg.get(
                                        "deterministic_residual_train", False
                                    )
                                )
                            else:
                                det_c, det_r = False, False
                            siglip_outputs = self.siglip_condition_model(
                                images=[img0_hwc.numpy()],
                                instructions=[instr0],
                                cluster_sample_temperature=siglip_temp,
                                deterministic_cluster=det_c,
                                deterministic_residual=det_r,
                            )
                            residual = siglip_outputs["residual_embedding"][0]
                            cluster_id = int(
                                siglip_outputs["pred_cluster_idx"][0].item()
                            )
                            centers_per_instr = self.instruction_centers[instr0]
                            center_vec = centers_per_instr[cluster_id].to(
                                residual.device, dtype=residual.dtype
                            )
                            z_vec = center_vec + residual
                            # broadcast same z-embedding to all envs in the group
                            z_batch = z_vec.unsqueeze(0).expand(B, -1).contiguous()
                            group_latents[stage_id] = z_batch.to(self.device)
                            if self.condition_policy_enable_rl:
                                pending_cond_meta[stage_id] = {
                                    "siglip_outputs": siglip_outputs,
                                    "img0_hwc": img0_hwc,
                                    "B": B,
                                }

                        z_ids = group_latents[stage_id]
                        cond_meta = None
                        if self.condition_policy_enable_rl and pending_cond_meta is not None:
                            cond_meta = pending_cond_meta[stage_id]
                            pending_cond_meta[stage_id] = None
                    else:
                        z_ids = group_tensor
                        cond_meta = None

                    actions, result = self.predict(extracted_obs, z_ids)

                    cond_kwargs = {}
                    if cond_meta is not None:
                        so = cond_meta["siglip_outputs"]
                        img0_hwc = cond_meta["img0_hwc"]
                        Bm = cond_meta["B"]
                        lp_c = (
                            so["log_prob_cluster"].reshape(-1)[0].repeat(Bm).detach().cpu()
                        )
                        lp_r = (
                            so["log_prob_residual"].reshape(-1)[0].repeat(Bm).detach().cpu()
                        )
                        lp_j = (
                            so["log_prob_joint"].reshape(-1)[0].repeat(Bm).detach().cpu()
                        )
                        res_b = (
                            so["residual_embedding"]
                            .reshape(1, -1)
                            .expand(Bm, -1)
                            .detach()
                            .cpu()
                        )
                        img_b = (
                            img0_hwc.unsqueeze(0)
                            .expand(Bm, -1, -1, -1)
                            .contiguous()
                            .cpu()
                        )
                        pc = (
                            so["pred_cluster_idx"]
                            .reshape(-1)[0]
                            .repeat(Bm)
                            .detach()
                            .cpu()
                        )
                        cond_kwargs = {
                            "cond_log_prob_cluster": lp_c,
                            "cond_log_prob_residual": lp_r,
                            "cond_log_prob_joint": lp_j,
                            "cond_residual": res_b,
                            "cond_initial_image_hwc": img_b,
                            "cond_pred_cluster_idx": pc,
                        }

                    chunk_step_result = ChunkStepResult(
                        prev_logprobs=result["prev_logprobs"],
                        prev_values=result["prev_values"],
                        dones=dones,
                        truncations=env_output["truncations"],
                        terminations=env_output["terminations"],
                        rewards=rewards,  # the first step is reset step, reward is none, which will not be appended to the buffer
                        forward_inputs=last_forward_inputs[stage_id],
                        z_ids=z_ids,
                        task_ids=result["task_ids"],
                        **cond_kwargs,
                    )
                    self.buffer_list[stage_id].append_result(chunk_step_result)
                    if last_extracted_obs[stage_id] is not None and hasattr(
                        self.hf_model, "q_head"
                    ):
                        self.buffer_list[stage_id].add_transition(
                            last_extracted_obs[stage_id], real_extracted_obs
                        )
                    last_extracted_obs[stage_id] = extracted_obs
                    last_forward_inputs[stage_id] = result["forward_inputs"]

                    self.send_chunk_actions(output_channel, actions)

            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                last_forward_inputs[stage_id] = self.update_intervene_actions(
                    env_output, last_forward_inputs[stage_id]
                )

                extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                # Get dones and rewards from environment batch (final step of epoch)
                dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                    env_output, extracted_obs
                )
                self.buffer_list[stage_id].dones.append(dones)
                self.buffer_list[stage_id].truncations.append(env_output["truncations"])
                self.buffer_list[stage_id].terminations.append(
                    env_output["terminations"]
                )
                self.buffer_list[stage_id].rewards.append(rewards)
                self.buffer_list[stage_id].forward_inputs.append(
                    put_tensor_device(last_forward_inputs[stage_id], "cpu")
                )

                with self.worker_timer():
                    actions, result = self.predict(extracted_obs, z_ids)
                # For the final step, we only need prev_values for bootstrapping
                # This is a special case that doesn't create a full ChunkStepResult
                if "prev_values" in result:
                    self.buffer_list[stage_id].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )
                if hasattr(self.hf_model, "q_head"):
                    self.buffer_list[stage_id].add_transition(
                        last_extracted_obs[stage_id], real_extracted_obs
                    )

        for i in range(self.num_pipeline_stages):
            self.send_rollout_batch(actor_channel, i)

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            random_nums = torch.tensor([0, 1, 2], dtype=torch.long).to(self.device)
            random_nums = random_nums.repeat(self.cfg.env.eval.total_num_envs//self.placement.get_world_size("actor")//3)
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    actions, _ = self.predict(extracted_obs, random_nums, mode="eval")
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)

    async def recv_env_output(
        self, input_channel: Channel, mode="train"
    ) -> dict[str, torch.Tensor]:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        # Use asyncio so that it can run alongside async weight syncing
        env_output = await input_channel.get(
            key=f"{self._rank}_{mode}", async_op=True
        ).async_wait()
        return env_output

    def send_chunk_actions(self, output_channel: Channel, chunk_actions, mode="train"):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        output_channel.put(
            item=chunk_actions, key=f"{self._rank}_{mode}", async_op=True
        )

    def send_rollout_batch(self, actor_channel: Channel, stage_id: int):
        # send rollout_batch to actor
        split_num = self.get_actor_split_num()
        splitted_rollout_result = self.buffer_list[stage_id].to_splitted_dict(split_num)
        for i in range(split_num):
            actor_channel.put(item=splitted_rollout_result[i], async_op=True)

    def get_actor_split_num(self):
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def set_global_step(self, global_step):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
