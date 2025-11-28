import sys
from pathlib import Path
import pickle
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from transformers import LlamaTokenizer, LlamaForCausalLM
repo_root = Path(__file__).resolve().parent.parent
print("REPO ROOT:",repo_root)
sys.path.append(str(repo_root))

#UNDER HERE IS A LIBERO SET UP IMPORT, in the called run_libero_eval I removed two libero.libero imports
#in addition I edited the get_action_head to only use gpu(I dont have nvidia)
#If needed, manually set the correct constants in`prismatic/vla/constants.py`
from experiments.robot.libero.run_libero_eval import GenerateConfig
print("GOODBYE")

from experiments.robot.openvla_utils import get_action_head,get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
cfg = GenerateConfig(
    
    pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-spatial",#'openlm-research/open_llama_3b',## 
    use_l1_regression = True,
    use_diffusion = False,
    use_film = False,
    num_images_in_input = 1,
    use_proprio = True,
    load_in_8bit = False,
    load_in_4bit = False,
    center_crop = True,
    num_open_loop_steps = NUM_ACTIONS_CHUNK,
    unnorm_key = "libero_spatial_no_noops",
)
print("HELLO")
# Load OpenVLA-OFT policy and inputs processor
vla = get_vla(cfg)
processor = get_processor(cfg)
# Load MLP action head to generate continuous actions (via L1 regression)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

# Load proprio projector to map proprio to language embedding space
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=8)
print("WE HERE")

image=Image.open("D:/CodingProjects/Calibration/FineTunedDeepLabV3-/calibration_pics/pic_13.jpg").convert("RGB")
image_np = np.array(image, dtype=np.uint8)  # shape (H, W, 3)

curr = torch.zeros((1,8))
# Load sample observation:
observation = {
"full_image": image_np,
"state": curr,
}

#TODO what is the expectation data type of observation?
# Generate robot action chunk (sequence of future actions)
actions = get_vla_action(cfg=cfg, vla=vla, processor=processor, obs=observation, task_label="",action_head=action_head, proprio_projector=proprio_projector)
print("Generated action chunk:")
for act in actions:
    print(act)