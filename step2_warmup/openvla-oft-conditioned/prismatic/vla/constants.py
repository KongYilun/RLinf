"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

ALOHA_CONSTANTS_12CHUCK = {
    "NUM_ACTIONS_CHUNK": 12,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

ALOHA_CONSTANTS_8CHUCK = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

ALOHA_CONSTANTS_6CHUCK = {
    "NUM_ACTIONS_CHUNK": 6,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

ALOHA_CONSTANTS_3CHUCK = {
    "NUM_ACTIONS_CHUNK": 3,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}
import os

# Function to detect robot platform from command line arguments
def detect_robot_platform():
    
    robot_env = os.environ.get('ROBOT_PLATFORM', '').upper()
    if robot_env:
        # 环境变量映射到平台
        env_mapping = {
            'LIBERO': 'LIBERO',
            'ALOHA': 'ALOHA',
            'ALOHA_12': 'ALOHA_12',
            'ALOHA_8': 'ALOHA_8',
            'ALOHA_6': 'ALOHA_6',
            'BRIDGE': 'BRIDGE',
        }
        if robot_env in env_mapping:
            print(f"Detected robot platform from environment: {env_mapping[robot_env]}")
            return env_mapping[robot_env]
    
    
    
    cmd_args = " ".join(sys.argv).lower()

    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha_12chunk" in cmd_args:
        return "ALOHA_12"
    elif "aloha_8chunk" in cmd_args:
        return "ALOHA_8"
    elif "aloha_6chunk" in cmd_args:
        return "ALOHA_6"
    elif "aloha_3chunk" in cmd_args:
        return "ALOHA_3"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    else:
        # Default to LIBERO if unclear
        return "ALOHA"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA_12":
    constants = ALOHA_CONSTANTS_12CHUCK
elif ROBOT_PLATFORM == "ALOHA_8":
    constants = ALOHA_CONSTANTS_8CHUCK
elif ROBOT_PLATFORM == "ALOHA_6":
    constants = ALOHA_CONSTANTS_6CHUCK
elif ROBOT_PLATFORM == "ALOHA_3":
    constants = ALOHA_CONSTANTS_3CHUCK
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
