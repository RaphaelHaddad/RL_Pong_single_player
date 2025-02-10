# game_env.py
import numpy as np
import torch
import random
import os
from multipleTest import MultipleTest
from agent import R2D2Agent
from config import CONFIG


os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

################################################################################################

# Options: "train", "test", "retrain"
MODE = "multiple_test"  


EPOCH_TEST = "checkpoints/model_episode_200.pth"
EPOCH_RETRAIN = "checkpoints/model_episode_200.pth"
CHECKPOINTS = {
    "checkpoints/model_episode_0.pth": "0",
    "checkpoints/model_episode_100.pth": "100",
    "checkpoints/model_episode_200.pth": "200",
}

################################################################################################

agent = R2D2Agent(CONFIG)
            
if __name__ == "__main__":
    if MODE == "train":
        agent.save_initial_agent(CONFIG)
        agent.train()
    elif MODE == "test":
        agent.load_initial_agent(EPOCH_TEST)
        agent.play()
    elif MODE == "retrain":
        agent.load_initial_agent(EPOCH_RETRAIN)
        agent.train()
    elif MODE == "multiple_test":
        multi_test = MultipleTest(CONFIG, CHECKPOINTS)
        multi_test.run()
