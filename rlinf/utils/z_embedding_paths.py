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

"""Resolve z-embedding artifact filenames ({suite}_per_instruction_centers.pt, etc.)."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def default_siglip_condition_checkpoint_path(cfg: DictConfig) -> str:
    """
    Default SigLIP condition RL checkpoint when ``algorithm.condition_policy.checkpoint_path``
    is unset. Uses the same suite as :func:`resolve_z_embedding_suite`, e.g.
    ``step3_encoder_training/libero_10_siglip_condition_model.pt``.
    """
    suite = resolve_z_embedding_suite(cfg)
    return f"step3_encoder_training/{suite}_siglip_condition_model.pt"


def resolve_z_embedding_suite(cfg: DictConfig) -> str:
    """
    Return the filename prefix for per-suite z embedding maps.

    Precedence:
        1. ``algorithm.z_embedding_suite`` (global override)
        2. ``actor.model.z_embedding_suite`` (per-run model block)
        3. ``env.train.task_suite_name`` (e.g. ``libero_object``, ``libero_10``)
        4. ``"libero_object"`` (legacy default)

    Expected files (cwd or launch dir): ``{suite}_per_instruction_centers.pt`` and
    ``{suite}_instruction_to_task_id_map.pt``.
    """
    for key in (
        "algorithm.z_embedding_suite",
        "actor.model.z_embedding_suite",
        "env.train.task_suite_name",
    ):
        v = OmegaConf.select(cfg, key)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return "libero_object"
