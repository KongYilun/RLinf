import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
import numpy as np

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    # task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


# Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
cfg = GenerateConfig(
    pretrained_checkpoint = "/data/yujiale/ckpts/openvla-7b-base+robotwin_9_tasks+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--05_30_sft--10000_chkpt",
    use_l1_regression = True,
    use_diffusion = False,
    use_film = False,
    num_images_in_input = 3,
    use_proprio = True,
    load_in_8bit = False,
    load_in_4bit = False,
    center_crop = True,
    num_open_loop_steps = NUM_ACTIONS_CHUNK,
    unnorm_key = "robotwin_9_tasks",
)

# Load OpenVLA-OFT policy and inputs processor
vla = get_vla(cfg)
processor = get_processor(cfg)

# Load MLP action head to generate continuous actions (via L1 regression)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

# Load proprio projector to map proprio to language embedding space
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# Load sample observation:
#   observation (dict): {
#     "full_image": primary third-person image,
#     "wrist_image": wrist-mounted camera image,
#     "state": robot proprioceptive state,
#     "task_description": task description,
#   }
with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
    observation = pickle.load(file)
    observation['state'] = np.pad(observation['state'], (0, 6), mode='constant', constant_values=0)
    observation['left_wrist_image']=observation['wrist_image']

# Generate robot action chunk (sequence of future actions)
actions = get_vla_action(cfg, vla, processor, observation, observation["task_description"], action_head, proprio_projector)
print("Generated action chunk:")
for act in actions:
    print(act)